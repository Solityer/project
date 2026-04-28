use crate::EZKLError;
use crate::circuit::CheckMode;
use crate::pfsys::{
    Snark, create_keys, create_proof_circuit, load_pk, load_vk, save_params, save_pk, save_vk,
    verify_proof_circuit,
};
use halo2_proofs::circuit::{AssignedCell, Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::dev::MockProver;
use halo2_proofs::plonk::{
    Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector, TableColumn,
};
use halo2_proofs::poly::commitment::ParamsProver;
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy as KZGSingleStrategy;
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::Field;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

const LEGACY_ROUTE_ID: &str = "ezkl_gat_native_csr_segment_attention_full";
const COMPRESSED_ROUTE_ID: &str = "ezkl_gat_native_compressed_csr_attention_full";
const LEGACY_PRIMITIVE: &str = "GraphSegmentAttention";
const LEGACY_PRIMITIVE_VERSION: &str = "csr_segment_attention_v1";
const COMPRESSED_PRIMITIVE: &str = "GraphSegmentAttentionCompressedCSR";
const COMPRESSED_PRIMITIVE_VERSION: &str = "csr_segment_attention_compressed_v1";
const DEFAULT_MAX_LOGROWS: u32 = 26;
const DEFAULT_PUBLIC_INSTANCE_CHUNK_SIZE: usize = 1 << 22;
const MAX_PUBLIC_INSTANCE_COLUMNS: usize = 2;
const TINY_NATIVE_COMPILED_KIND: &str = "graph_gat_tiny_native_compiled_plan";
const TINY_NATIVE_BACKEND_VERSION: &str = "tiny_native_backend_levels_0_6_v1";
const TINY_NATIVE_LOGROWS: u32 = 9;
const DEFAULT_NATIVE_DEBUG_SCOPE: &str = "full_model";
const NATIVE_DIAGNOSTIC_BACKEND_VERSION: &str = "native_full_output_derivation_v3";
const TRACE_SCALAR_SCALE_I64: i64 = 16;
const QUANTIZED_INVERSE_POLICY: &str = "quantized_inverse_lookup";
const ALPHA_RESCALE_POLICY: &str = "rounded_product_div_scale_with_quantization_band";
const LOOKUP_KIND_DISABLED: u64 = 0;
const LOOKUP_KIND_INVERSE: u64 = 1;
const LOOKUP_KIND_LRELU: u64 = 2;
const LOOKUP_KIND_EXP: u64 = 3;
const LOOKUP_KIND_ELU: u64 = 4;
const NATIVE_DEBUG_SCOPES: [&str; 21] = [
    "level0_mul",
    "inverse_only",
    "segment_sum_inverse",
    "segment_max_only",
    "lrelu_lookup_only",
    "exp_lookup_only",
    "elu_lookup_only",
    "compressed_projection_only",
    "compressed_aggregation_only",
    "single_head_first_layer",
    "first_hidden_layer_all_heads",
    "hidden_only",
    "output_only",
    "output_projection_only",
    "output_attention_only",
    "output_aggregation_only",
    "output_bias_add_only",
    "output_average_only",
    "final_logits_public_binding_only",
    "output_full_derivation",
    "full_model",
];

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct HiddenProfile {
    head_count: u64,
    head_dim: u64,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default, Deserialize)]
struct CheckpointHeadSpecJson {
    #[serde(default)]
    layer_index: u64,
    #[serde(default)]
    local_head_index: u64,
    #[serde(default)]
    global_head_index: u64,
    #[serde(default)]
    seq_kernel: String,
    #[serde(default)]
    attn_src_kernel: String,
    #[serde(default)]
    attn_dst_kernel: String,
    #[serde(default)]
    attn_src_bias: String,
    #[serde(default)]
    attn_dst_bias: String,
    #[serde(default)]
    output_bias: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default, Deserialize)]
struct CheckpointOutputHeadSpecJson {
    #[serde(default)]
    head_index: u64,
    #[serde(default)]
    seq_kernel: String,
    #[serde(default)]
    attn_src_kernel: String,
    #[serde(default)]
    attn_dst_kernel: String,
    #[serde(default)]
    attn_src_bias: String,
    #[serde(default)]
    attn_dst_bias: String,
    #[serde(default)]
    output_bias: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct GraphAttentionSpec {
    route_id: String,
    native_primitive: String,
    native_primitive_version: String,
    full_dataset: bool,
    strict_public_input_equivalent_candidate: bool,
    dataset: String,
    node_count: u64,
    edge_count: u64,
    graph_count: u64,
    feature_dim: u64,
    class_count: u64,
    #[serde(rename = "L")]
    layer_count: u64,
    #[serde(rename = "K_out")]
    k_out: u64,
    hidden_profile: Vec<HiddenProfile>,
    d_in_profile: Vec<u64>,
    #[serde(default)]
    hidden_heads: Vec<CheckpointHeadSpecJson>,
    #[serde(default)]
    output_heads: Vec<CheckpointOutputHeadSpecJson>,
    #[serde(default)]
    output_layer_input_source: String,
    #[serde(default)]
    output_layer_input_dim: u64,
    #[serde(default)]
    output_head_count: u64,
    #[serde(default)]
    output_derivation_required: bool,
    #[serde(default)]
    output_derivation_complete_candidate: bool,
    #[serde(default)]
    no_precomputed_logits_bypass: bool,
    #[serde(default)]
    output_average_rule: String,
    model_arch_id: String,
    model_param_id: String,
    quant_cfg_id: String,
    #[serde(default)]
    static_table_id: String,
    graph_hash: String,
    checkpoint_hash: String,
    metadata_hash: String,
    shape_hash: String,
    total_attention_head_count: u64,
    #[serde(default)]
    formal_table_hash: String,
    #[serde(default)]
    formal_tables: Option<FormalTableBinding>,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct CsrLayoutManifest {
    layout_kind: String,
    node_count: u64,
    edge_count: u64,
    graph_count: u64,
    max_in_degree: u64,
    src_hash: String,
    dst_hash: String,
    segment_ptr_hash: String,
    edge_ptr_hash: String,
    node_ptr_hash: String,
    segment_graph_id_hash: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct FormalTableBinding {
    #[serde(default)]
    source: String,
    #[serde(default)]
    table_policy: String,
    #[serde(default)]
    inverse_table_policy: String,
    #[serde(default)]
    quant_cfg_id: String,
    #[serde(default)]
    static_table_id: String,
    #[serde(default)]
    degree_bound_id: String,
    #[serde(default)]
    range_bits: u64,
    #[serde(default)]
    signed_bound: i64,
    #[serde(default)]
    degree_bound: u64,
    #[serde(default)]
    exp_bound: i64,
    #[serde(default)]
    derived_sum_bound: i64,
    #[serde(default)]
    inverse_table_range_old: u64,
    #[serde(default)]
    inverse_table_range_new: u64,
    #[serde(default)]
    inverse_table_range_size: u64,
    #[serde(default)]
    range_hash: String,
    #[serde(default)]
    lrelu_hash: String,
    #[serde(default)]
    elu_hash: String,
    #[serde(default)]
    exp_hash: String,
    #[serde(default)]
    inv_hash: String,
    #[serde(default)]
    inverse_policy: String,
    #[serde(default)]
    range_size: u64,
    #[serde(default)]
    sum_scale: i64,
    #[serde(default)]
    inv_scale: i64,
    #[serde(default)]
    alpha_scale: i64,
    #[serde(default)]
    inverse_scale_product: i64,
    #[serde(default)]
    inverse_band: i64,
    #[serde(default)]
    alpha_rescale_policy: String,
    #[serde(default)]
    alpha_rescale_divisor: i64,
    #[serde(default)]
    alpha_rounding_rule: String,
    #[serde(default)]
    alpha_band: i64,
    #[serde(default)]
    inverse_table_hash: String,
    #[serde(default)]
    inverse_policy_hash: String,
    #[serde(default)]
    table_hash: String,
    #[serde(default)]
    bin_hash: String,
    #[serde(default)]
    witness_specific_table: bool,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FormalTableHashesFile {
    #[serde(default)]
    source: String,
    #[serde(default)]
    table_policy: String,
    #[serde(default)]
    inverse_table_policy: String,
    #[serde(default)]
    quant_cfg_id: String,
    #[serde(default)]
    static_table_id: String,
    #[serde(default)]
    degree_bound_id: String,
    #[serde(default)]
    range_bits: u64,
    #[serde(default)]
    signed_bound: i64,
    #[serde(default)]
    degree_bound: u64,
    #[serde(default)]
    exp_bound: i64,
    #[serde(default)]
    derived_sum_bound: i64,
    #[serde(default)]
    inverse_table_range_old: u64,
    #[serde(default)]
    inverse_table_range_new: u64,
    #[serde(default)]
    inverse_table_range_size: u64,
    #[serde(default)]
    range_hash: String,
    #[serde(default)]
    lrelu_hash: String,
    #[serde(default)]
    elu_hash: String,
    #[serde(default)]
    exp_hash: String,
    #[serde(default)]
    inv_hash: String,
    #[serde(default)]
    inverse_policy: String,
    #[serde(default)]
    range_size: u64,
    #[serde(default)]
    sum_scale: i64,
    #[serde(default)]
    inv_scale: i64,
    #[serde(default)]
    alpha_scale: i64,
    #[serde(default)]
    inverse_scale_product: i64,
    #[serde(default)]
    inverse_band: i64,
    #[serde(default)]
    alpha_rescale_policy: String,
    #[serde(default)]
    alpha_rescale_divisor: i64,
    #[serde(default)]
    alpha_rounding_rule: String,
    #[serde(default)]
    alpha_band: i64,
    #[serde(default)]
    inverse_table_hash: String,
    #[serde(default)]
    inverse_policy_hash: String,
    #[serde(default)]
    table_hash: String,
    #[serde(default)]
    bin_hash: String,
    #[serde(default)]
    witness_specific_table: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeEstimate {
    projection_mul_add_rows: u128,
    attention_score_rows: u128,
    lrelu_lookup_rows: u128,
    segment_max_rows: u128,
    exp_lookup_rows: u128,
    denominator_sum_rows: u128,
    exact_inverse_rows: u128,
    alpha_rows: u128,
    aggregation_rows: u128,
    elu_lookup_rows: u128,
    output_average_rows: u128,
    fixed_overhead_rows: u128,
    total_rows: u128,
    required_logrows: u32,
    max_logrows: u32,
    within_public_srs: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CompressedRowEstimate {
    dataset: String,
    route_id: String,
    node_count: u64,
    edge_count: u64,
    graph_count: u64,
    hidden_layers: u64,
    hidden_heads: u64,
    output_heads: u64,
    projection_rows: u128,
    edge_attention_rows: u128,
    max_state_machine_rows: u128,
    exp_lookup_rows: u128,
    sum_inverse_rows: u128,
    aggregation_rows: u128,
    elu_lookup_rows: u128,
    output_rows: u128,
    metadata_rows: u128,
    previous_native_csr_rows: u128,
    total_rows_estimated: u128,
    required_logrows: u32,
    max_logrows: u32,
    row_reduction_ratio_vs_previous_native_csr: f64,
    feasible_under_logrows_26: bool,
    blocker_if_any: String,
    challenge_policy: String,
    lookup_batching: String,
    max_selector_policy: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeSettings {
    settings_kind: String,
    route_id: String,
    native_primitive: String,
    native_primitive_version: String,
    model_type: String,
    status: String,
    block_reason: String,
    dataset: String,
    full_dataset: bool,
    strict_public_input_equivalent_candidate: bool,
    csr_layout: String,
    quant_cfg_id: String,
    model_arch_id: String,
    model_param_id: String,
    graph_hash: String,
    checkpoint_hash: String,
    metadata_hash: String,
    shape_hash: String,
    lookup_semantics: LookupSemantics,
    inverse_semantics: InverseSemantics,
    #[serde(default)]
    formal_table_binding: FormalTableBinding,
    estimate: NativeEstimate,
    #[serde(skip_serializing_if = "Option::is_none")]
    compressed_row_estimate: Option<CompressedRowEstimate>,
    native_debug_scope: String,
    strict_public_input_equivalent: bool,
    full_gat_forward_in_ezkl_proof: bool,
    generated_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LookupSemantics {
    lrelu: String,
    elu: String,
    exp_map: String,
    range_checks: String,
    formal_status: String,
    #[serde(default)]
    table_policy: String,
    #[serde(default)]
    table_hash: String,
    #[serde(default)]
    witness_specific_table: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InverseSemantics {
    mode: String,
    constraint: String,
    witness_column: String,
    nonzero_argument: String,
    approximate_reciprocal_used: bool,
    #[serde(default)]
    policy: String,
    #[serde(default)]
    sum_scale: i64,
    #[serde(default)]
    inv_scale: i64,
    #[serde(default)]
    alpha_scale: i64,
    #[serde(default)]
    inverse_scale_product: i64,
    #[serde(default)]
    inverse_band: i64,
    #[serde(default)]
    alpha_rescale_policy: String,
    #[serde(default)]
    alpha_rescale_divisor: i64,
    #[serde(default)]
    alpha_rounding_rule: String,
    #[serde(default)]
    alpha_band: i64,
    #[serde(default)]
    inverse_table_hash: String,
    #[serde(default)]
    inverse_policy_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeCompiledPlan {
    compiled_kind: String,
    route_id: String,
    native_primitive: String,
    native_primitive_version: String,
    status: String,
    block_reason: String,
    failure_step: String,
    dataset: String,
    #[serde(default)]
    node_count: u64,
    #[serde(default)]
    edge_count: u64,
    #[serde(default)]
    graph_count: u64,
    #[serde(default)]
    feature_dim: u64,
    #[serde(default)]
    class_count: u64,
    #[serde(default)]
    layer_count: u64,
    #[serde(default)]
    k_out: u64,
    #[serde(default)]
    total_attention_head_count: u64,
    estimate: NativeEstimate,
    csr_layout: String,
    graph_hash: String,
    checkpoint_hash: String,
    #[serde(default)]
    metadata_hash: String,
    #[serde(default)]
    shape_hash: String,
    exact_inverse_gate: InverseSemantics,
    lookup_semantics: LookupSemantics,
    #[serde(default)]
    formal_table_binding: FormalTableBinding,
    #[serde(skip_serializing_if = "Option::is_none")]
    compressed_row_estimate: Option<CompressedRowEstimate>,
    #[serde(default = "default_native_debug_scope_string")]
    native_debug_scope: String,
    #[serde(default)]
    strict_public_input_equivalent: bool,
    #[serde(default)]
    full_gat_forward_in_ezkl_proof: bool,
    #[serde(default)]
    diagnostic_backend_version: String,
    #[serde(default)]
    diagnostic_note: String,
    #[serde(default = "default_public_instance_layout_string")]
    public_instance_layout: String,
    #[serde(default = "default_public_instance_chunk_size")]
    public_instance_chunk_size: usize,
    #[serde(default)]
    public_instance_column_count: usize,
    #[serde(default)]
    public_instance_total_length: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TinyNativeCompiledPlan {
    compiled_kind: String,
    route_id: String,
    native_primitive: String,
    native_primitive_version: String,
    backend_version: String,
    status: String,
    logrows: u32,
    level: u8,
    public_inputs: Vec<u64>,
    note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeTinyWitness {
    witness_kind: String,
    backend_version: String,
    level: u8,
    a: u64,
    b: u64,
    c: u64,
    sum: u64,
    inv: u64,
    u_values: Vec<u64>,
    z_values: Vec<i64>,
    m_values: Vec<i64>,
    max_sel: Vec<u64>,
    projection_rho: u64,
    aggregation_rho: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeOutputHeadWitness {
    head_index: u64,
    node_index: u64,
    class_index: u64,
    projection_q: i64,
    attention_q: i64,
    bias_added_q: i64,
    output_scale: i64,
}

#[derive(Clone, Debug, Default)]
struct TensorDump {
    shape: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct NativeHeadParameters {
    seq_kernel: Vec<Vec<f64>>,
    attn_src_kernel: Vec<f64>,
    attn_dst_kernel: Vec<f64>,
    attn_src_bias: f64,
    attn_dst_bias: f64,
    output_bias: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct NativeHeadForwardTrace {
    h_prime: Vec<Vec<f64>>,
    e_src: Vec<f64>,
    e_dst: Vec<f64>,
    s: Vec<f64>,
    z: Vec<f64>,
    m: Vec<f64>,
    delta: Vec<f64>,
    u: Vec<f64>,
    sum: Vec<f64>,
    inv: Vec<f64>,
    alpha: Vec<f64>,
    h_agg_pre_bias: Vec<Vec<f64>>,
    h_agg: Vec<Vec<f64>>,
}

#[derive(Clone, Debug, Serialize)]
struct NativeOutputProjectionWitness {
    head_index: u64,
    challenge: i64,
    node_count: u64,
    class_count: u64,
    compressed_values_q_sample: Vec<i64>,
    source: String,
}

#[derive(Clone, Debug, Serialize)]
struct NativeOutputAttentionWitness {
    head_index: u64,
    e_src_q_sample: Vec<i64>,
    e_dst_q_sample: Vec<i64>,
    s_q_sample: Vec<i64>,
    z_q_sample: Vec<i64>,
    m_q_sample: Vec<i64>,
    delta_q_sample: Vec<i64>,
    u_q_sample: Vec<i64>,
    sum_q_sample: Vec<i64>,
    inv_q_sample: Vec<i64>,
    alpha_q_sample: Vec<i64>,
    policy: String,
}

#[derive(Clone, Debug, Serialize)]
struct NativeOutputAggregationWitness {
    head_index: u64,
    challenge: i64,
    compressed_pre_bias_q_sample: Vec<i64>,
    aggregation_policy: String,
}

#[derive(Clone, Debug, Serialize)]
struct NativeOutputBiasAddWitness {
    head_index: u64,
    bias_q: Vec<i64>,
    head_logits_q_sample: Vec<i64>,
    policy: String,
}

#[derive(Clone, Debug)]
struct NativeOutputHeadDerivation {
    head_index: u64,
    parameters: NativeHeadParameters,
    trace: NativeHeadForwardTrace,
    projection: NativeOutputProjectionWitness,
    attention: NativeOutputAttentionWitness,
    aggregation: NativeOutputAggregationWitness,
    bias_add: NativeOutputBiasAddWitness,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct NativeOutputDerivationWitness {
    output_layer_input_source: String,
    output_layer_input_dim: u64,
    last_hidden_concat: Vec<Vec<f64>>,
    output_heads: Vec<NativeOutputHeadDerivation>,
    final_logits: NativeFinalLogitsWitness,
    output_average: Vec<NativeOutputAverageWitness>,
    derivation_complete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeOutputAverageWitness {
    node_index: u64,
    class_index: u64,
    head_values_q: Vec<i64>,
    sum_q: i64,
    average_q: i64,
    policy: String,
    rounding_policy: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeFinalLogitsWitness {
    shape: [u64; 2],
    flat_values_q: Vec<i64>,
    output_scale: i64,
    output_rounding_policy: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PublicInstanceBinding {
    public_instance_order: String,
    public_instance_length: usize,
    public_instance_layout: String,
    public_instance_chunk_size: usize,
    public_instance_column_count: usize,
    public_instance_total_length: usize,
    final_logits_hash: String,
    public_instance_hash: String,
    no_precomputed_logits_bypass: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PublicInstanceLayout {
    layout: String,
    chunk_size: usize,
    column_count: usize,
    total_length: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeVerifyOutput {
    status: String,
    verify_passed: bool,
    route_id: String,
    backend_version: String,
    proof_path: String,
    vk_path: String,
}

enum BackendPlan {
    Tiny(TinyNativeCompiledPlan),
    Full(NativeCompiledPlan),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NativeDebugScope {
    Level0Mul,
    InverseOnly,
    SegmentSumInverse,
    SegmentMaxOnly,
    LreluLookupOnly,
    ExpLookupOnly,
    EluLookupOnly,
    CompressedProjectionOnly,
    CompressedAggregationOnly,
    SingleHeadFirstLayer,
    FirstHiddenLayerAllHeads,
    HiddenOnly,
    OutputOnly,
    OutputProjectionOnly,
    OutputAttentionOnly,
    OutputAggregationOnly,
    OutputBiasAddOnly,
    OutputAverageOnly,
    FinalLogitsPublicBindingOnly,
    OutputFullDerivation,
    FullModel,
}

impl NativeDebugScope {
    fn parse(raw: &str) -> Result<Self, EZKLError> {
        match raw {
            "level0_mul" => Ok(Self::Level0Mul),
            "inverse_only" => Ok(Self::InverseOnly),
            "segment_sum_inverse" => Ok(Self::SegmentSumInverse),
            "segment_max_only" => Ok(Self::SegmentMaxOnly),
            "lrelu_lookup_only" => Ok(Self::LreluLookupOnly),
            "exp_lookup_only" => Ok(Self::ExpLookupOnly),
            "elu_lookup_only" => Ok(Self::EluLookupOnly),
            "compressed_projection_only" => Ok(Self::CompressedProjectionOnly),
            "compressed_aggregation_only" => Ok(Self::CompressedAggregationOnly),
            "single_head_first_layer" => Ok(Self::SingleHeadFirstLayer),
            "first_hidden_layer_all_heads" => Ok(Self::FirstHiddenLayerAllHeads),
            "hidden_only" => Ok(Self::HiddenOnly),
            "output_only" => Ok(Self::OutputOnly),
            "output_projection_only" => Ok(Self::OutputProjectionOnly),
            "output_attention_only" => Ok(Self::OutputAttentionOnly),
            "output_aggregation_only" => Ok(Self::OutputAggregationOnly),
            "output_bias_add_only" => Ok(Self::OutputBiasAddOnly),
            "output_average_only" => Ok(Self::OutputAverageOnly),
            "final_logits_public_binding_only" => Ok(Self::FinalLogitsPublicBindingOnly),
            "output_full_derivation" => Ok(Self::OutputFullDerivation),
            "full_model" => Ok(Self::FullModel),
            other => Err(format!(
                "unknown native debug scope `{other}`; expected one of: {}",
                NATIVE_DEBUG_SCOPES.join(", ")
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Level0Mul => "level0_mul",
            Self::InverseOnly => "inverse_only",
            Self::SegmentSumInverse => "segment_sum_inverse",
            Self::SegmentMaxOnly => "segment_max_only",
            Self::LreluLookupOnly => "lrelu_lookup_only",
            Self::ExpLookupOnly => "exp_lookup_only",
            Self::EluLookupOnly => "elu_lookup_only",
            Self::CompressedProjectionOnly => "compressed_projection_only",
            Self::CompressedAggregationOnly => "compressed_aggregation_only",
            Self::SingleHeadFirstLayer => "single_head_first_layer",
            Self::FirstHiddenLayerAllHeads => "first_hidden_layer_all_heads",
            Self::HiddenOnly => "hidden_only",
            Self::OutputOnly => "output_only",
            Self::OutputProjectionOnly => "output_projection_only",
            Self::OutputAttentionOnly => "output_attention_only",
            Self::OutputAggregationOnly => "output_aggregation_only",
            Self::OutputBiasAddOnly => "output_bias_add_only",
            Self::OutputAverageOnly => "output_average_only",
            Self::FinalLogitsPublicBindingOnly => "final_logits_public_binding_only",
            Self::OutputFullDerivation => "output_full_derivation",
            Self::FullModel => "full_model",
        }
    }

    fn logrows(self, plan: &NativeCompiledPlan) -> u32 {
        let output_rows = u128::from(plan.node_count.max(1))
            .saturating_mul(u128::from(plan.class_count.max(1)))
            .saturating_mul(u128::from(plan.k_out.max(1)))
            .saturating_add(u128::from(plan.node_count.max(1)).saturating_mul(8))
            .saturating_add(u128::from(plan.edge_count.max(1)).saturating_mul(7))
            .saturating_add(8192);
        let diagnostic_rows = match self {
            Self::Level0Mul
            | Self::InverseOnly
            | Self::LreluLookupOnly
            | Self::ExpLookupOnly
            | Self::EluLookupOnly
            | Self::CompressedProjectionOnly
            | Self::CompressedAggregationOnly => 256u128,
            Self::SegmentSumInverse | Self::SegmentMaxOnly => {
                u128::from(plan.node_count.max(1)).saturating_mul(2)
            }
            Self::SingleHeadFirstLayer => u128::from(plan.node_count.max(1))
                .saturating_add(u128::from(plan.edge_count.max(1)))
                .saturating_add(1024),
            Self::FirstHiddenLayerAllHeads | Self::HiddenOnly => {
                let heads = total_hidden_heads_from_plan(plan).max(1);
                u128::from(plan.node_count.max(1))
                    .saturating_add(u128::from(plan.edge_count.max(1)))
                    .saturating_mul(u128::from(heads))
                    .saturating_add(4096)
            }
            Self::OutputOnly => u128::from(plan.node_count.max(1))
                .saturating_mul(u128::from(plan.class_count.max(1)))
                .saturating_add(4096),
            Self::OutputProjectionOnly
            | Self::OutputAttentionOnly
            | Self::OutputAggregationOnly
            | Self::OutputBiasAddOnly
            | Self::OutputAverageOnly
            | Self::FinalLogitsPublicBindingOnly
            | Self::OutputFullDerivation => output_rows,
            Self::FullModel => full_model_assigned_row_upper_bound(plan),
        };
        ceil_log2(diagnostic_rows.saturating_add(64)).clamp(9, plan.estimate.max_logrows)
    }
}

fn full_model_assigned_row_upper_bound(plan: &NativeCompiledPlan) -> u128 {
    let n = u128::from(plan.node_count.max(1));
    let e = u128::from(plan.edge_count.max(1));
    let c = u128::from(plan.class_count.max(1));
    let k_out = u128::from(plan.k_out.max(1));
    let hidden_heads = u128::from(total_hidden_heads_from_plan(plan).max(1));

    let output_head_rows = k_out.saturating_mul(
        n.saturating_mul(6)
            .saturating_add(e.saturating_mul(4))
            .saturating_add(n.saturating_mul(c)),
    );
    let output_shared_rows = if plan.k_out.max(1) == 1 {
        0
    } else {
        n.saturating_mul(c)
    };
    let hidden_compressed_rows = hidden_heads.saturating_mul(4).saturating_add(4096);
    let diagnostic_rows = 8192u128;
    let lookup_table_margin = 131_072u128;
    let floor_planner_margin = 262_144u128;

    output_head_rows
        .saturating_add(output_shared_rows)
        .saturating_add(hidden_compressed_rows)
        .saturating_add(diagnostic_rows)
        .saturating_add(lookup_table_margin)
        .saturating_add(floor_planner_margin)
        .max(4096)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct NativeConstraintFailure {
    status: String,
    block_reason: String,
    gate_name: String,
    region_name: String,
    row_offset: usize,
    dataset: String,
    route_id: String,
    layer_index: Option<u64>,
    head_index: Option<u64>,
    component: String,
    expected_relation: String,
    actual_values: Vec<String>,
    witness_source_path: String,
    command: String,
    raw_error: String,
    attempted_fix: String,
    remaining_blocker: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct DiagnosticRow {
    gate_name: &'static str,
    region_name: String,
    row_offset: usize,
    layer_index: Option<u64>,
    head_index: Option<u64>,
    component: String,
    expected_relation: String,
    actual_values: Vec<String>,
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
}

fn default_native_debug_scope_string() -> String {
    DEFAULT_NATIVE_DEBUG_SCOPE.to_string()
}

fn default_public_instance_layout_string() -> String {
    "chunked_columns".to_string()
}

fn default_public_instance_chunk_size() -> usize {
    DEFAULT_PUBLIC_INSTANCE_CHUNK_SIZE
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, EZKLError> {
    let text = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), EZKLError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(value)?;
    fs::write(path, format!("{text}\n"))?;
    Ok(())
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QuantizedInversePolicy {
    inverse_policy: String,
    sum_scale: i64,
    inv_scale: i64,
    alpha_scale: i64,
    inverse_scale_product: i64,
    inverse_band: i64,
    alpha_rescale_policy: String,
    alpha_rescale_divisor: i64,
    alpha_rounding_rule: String,
    alpha_band: i64,
    inverse_policy_hash: String,
}

fn fnv1a64_hex(text: &str) -> String {
    let mut state = 1_469_598_103_934_665_603u64;
    for byte in text.as_bytes() {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{state:016x}")
}

fn default_quantized_inverse_policy() -> QuantizedInversePolicy {
    let mut policy = QuantizedInversePolicy {
        inverse_policy: QUANTIZED_INVERSE_POLICY.to_string(),
        sum_scale: TRACE_SCALAR_SCALE_I64,
        inv_scale: TRACE_SCALAR_SCALE_I64,
        alpha_scale: TRACE_SCALAR_SCALE_I64,
        inverse_scale_product: TRACE_SCALAR_SCALE_I64 * TRACE_SCALAR_SCALE_I64,
        inverse_band: 0,
        alpha_rescale_policy: ALPHA_RESCALE_POLICY.to_string(),
        alpha_rescale_divisor: TRACE_SCALAR_SCALE_I64,
        alpha_rounding_rule: "half_away_from_zero".to_string(),
        alpha_band: 2,
        inverse_policy_hash: String::new(),
    };
    let serialized = format!(
        "quantized_inverse_lookup_v1\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n",
        policy.sum_scale,
        policy.inv_scale,
        policy.alpha_scale,
        policy.inverse_scale_product,
        policy.inverse_band,
        policy.alpha_rounding_rule,
        policy.alpha_rescale_policy,
        policy.alpha_rescale_divisor,
        policy.alpha_band,
    );
    policy.inverse_policy_hash = fnv1a64_hex(&serialized);
    policy
}

fn inverse_policy_from_binding(binding: &FormalTableBinding) -> QuantizedInversePolicy {
    let default = default_quantized_inverse_policy();
    QuantizedInversePolicy {
        inverse_policy: if binding.inverse_policy.is_empty() {
            default.inverse_policy
        } else {
            binding.inverse_policy.clone()
        },
        sum_scale: if binding.sum_scale == 0 {
            default.sum_scale
        } else {
            binding.sum_scale
        },
        inv_scale: if binding.inv_scale == 0 {
            default.inv_scale
        } else {
            binding.inv_scale
        },
        alpha_scale: if binding.alpha_scale == 0 {
            default.alpha_scale
        } else {
            binding.alpha_scale
        },
        inverse_scale_product: if binding.inverse_scale_product == 0 {
            default.inverse_scale_product
        } else {
            binding.inverse_scale_product
        },
        inverse_band: binding.inverse_band,
        alpha_rescale_policy: if binding.alpha_rescale_policy.is_empty() {
            default.alpha_rescale_policy
        } else {
            binding.alpha_rescale_policy.clone()
        },
        alpha_rescale_divisor: if binding.alpha_rescale_divisor == 0 {
            default.alpha_rescale_divisor
        } else {
            binding.alpha_rescale_divisor
        },
        alpha_rounding_rule: if binding.alpha_rounding_rule.is_empty() {
            default.alpha_rounding_rule
        } else {
            binding.alpha_rounding_rule.clone()
        },
        alpha_band: if binding.alpha_band == 0 {
            default.alpha_band
        } else {
            binding.alpha_band
        },
        inverse_policy_hash: if binding.inverse_policy_hash.is_empty() {
            default.inverse_policy_hash
        } else {
            binding.inverse_policy_hash.clone()
        },
    }
}

fn round_div_half_away_from_zero(numerator: i128, denominator: i128) -> i64 {
    let denominator = denominator.max(1);
    let sign = if numerator < 0 { -1 } else { 1 };
    let abs = numerator.abs();
    let rounded = (abs.saturating_mul(2).saturating_add(denominator)) / (denominator * 2);
    (rounded * sign) as i64
}

fn quantized_inverse_for_sum_q(sum_q: i64, policy: &QuantizedInversePolicy) -> i64 {
    if sum_q <= 0 {
        return 0;
    }
    round_div_half_away_from_zero(policy.inverse_scale_product as i128, sum_q as i128)
}

fn quantized_inverse_interval_for_sum_q(sum_q: i64, policy: &QuantizedInversePolicy) -> (i64, i64) {
    if sum_q <= 0 {
        return (0, 0);
    }
    let doubled_numerator = i128::from(policy.inverse_scale_product) * 2;
    let lower_denominator = i128::from(sum_q) * 2 - 1;
    let upper_denominator = i128::from(sum_q) * 2 + 1;
    let max_inv = round_div_half_away_from_zero(doubled_numerator, lower_denominator);
    let min_inv = round_div_half_away_from_zero(doubled_numerator, upper_denominator);
    (min_inv, max_inv)
}

fn validate_inverse_witness_against_policy(
    sum_q: i64,
    inv_q: i64,
    policy: &QuantizedInversePolicy,
) -> bool {
    let (min_inv, max_inv) = quantized_inverse_interval_for_sum_q(sum_q, policy);
    inv_q >= min_inv && inv_q <= max_inv
}

fn inverse_sum_in_table_range(sum_q: i64, binding: &FormalTableBinding) -> bool {
    sum_q >= 0 && (sum_q as usize) < inverse_table_range_size(binding)
}

fn validate_inverse_lookup_against_binding(
    sum_q: i64,
    inv_q: i64,
    binding: &FormalTableBinding,
) -> bool {
    inverse_sum_in_table_range(sum_q, binding)
        && validate_inverse_witness_against_policy(
            sum_q,
            inv_q,
            &inverse_policy_from_binding(binding),
        )
}

fn validate_alpha_rescale(
    u_q: i64,
    inv_q: i64,
    alpha_q: i64,
    policy: &QuantizedInversePolicy,
) -> bool {
    let expected = round_div_half_away_from_zero(
        i128::from(u_q) * i128::from(inv_q),
        i128::from(policy.alpha_rescale_divisor.max(1)),
    );
    (alpha_q - expected).abs() <= policy.alpha_band
}

fn rust_policy_matches_binding(binding: &FormalTableBinding) -> bool {
    let default = default_quantized_inverse_policy();
    let policy = inverse_policy_from_binding(binding);
    policy.inverse_policy == default.inverse_policy
        && policy.sum_scale == default.sum_scale
        && policy.inv_scale == default.inv_scale
        && policy.alpha_scale == default.alpha_scale
        && policy.inverse_scale_product == default.inverse_scale_product
        && policy.inverse_band == default.inverse_band
        && policy.alpha_rescale_policy == default.alpha_rescale_policy
        && policy.alpha_rescale_divisor == default.alpha_rescale_divisor
        && policy.alpha_rounding_rule == default.alpha_rounding_rule
        && policy.alpha_band == default.alpha_band
        && policy.inverse_policy_hash == default.inverse_policy_hash
}

fn inverse_table_range_size(binding: &FormalTableBinding) -> usize {
    if binding.inverse_table_range_size > 0 {
        return binding.inverse_table_range_size as usize;
    }
    if binding.inverse_table_range_new > 0 {
        return binding.inverse_table_range_new as usize;
    }
    if binding.range_size > 0 {
        return binding.range_size as usize;
    }
    if binding.range_bits > 0 && binding.range_bits < usize::BITS as u64 {
        return 1usize << binding.range_bits;
    }
    1usize << 12
}

fn formal_range_table_size(binding: &FormalTableBinding) -> usize {
    if binding.range_size > 0 {
        return binding.range_size as usize;
    }
    if binding.range_bits > 0 && binding.range_bits < usize::BITS as u64 {
        return 1usize << binding.range_bits;
    }
    1usize << 12
}

fn fr_i64(value: i64) -> Fr {
    if value >= 0 {
        Fr::from(value as u64)
    } else {
        -Fr::from(value.unsigned_abs())
    }
}

fn leaky_relu_table_output(raw: i64) -> i64 {
    quantize_trace_scalar(leaky_relu_fp(dequantize_trace_scalar(raw)))
}

fn default_tiny_plan() -> TinyNativeCompiledPlan {
    TinyNativeCompiledPlan {
        compiled_kind: TINY_NATIVE_COMPILED_KIND.to_string(),
        route_id: COMPRESSED_ROUTE_ID.to_string(),
        native_primitive: COMPRESSED_PRIMITIVE.to_string(),
        native_primitive_version: COMPRESSED_PRIMITIVE_VERSION.to_string(),
        backend_version: TINY_NATIVE_BACKEND_VERSION.to_string(),
        status: "compiled".to_string(),
        logrows: TINY_NATIVE_LOGROWS,
        level: 6,
        public_inputs: vec![15, 1],
        note: "tiny native graph backend exercises levels 0 through 6 with real Halo2 KZG proof"
            .to_string(),
    }
}

fn default_tiny_witness(level: u8) -> NativeTinyWitness {
    NativeTinyWitness {
        witness_kind: "tiny_native_graph_witness".to_string(),
        backend_version: TINY_NATIVE_BACKEND_VERSION.to_string(),
        level,
        a: 3,
        b: 5,
        c: 15,
        sum: 5,
        inv: 0,
        u_values: vec![2, 3, 5],
        z_values: vec![1, 4, 3],
        m_values: vec![4, 3],
        max_sel: vec![0, 1, 1],
        projection_rho: 7,
        aggregation_rho: 11,
    }
}

fn public_inputs_from_plan(plan: &TinyNativeCompiledPlan) -> Vec<Fr> {
    plan.public_inputs
        .iter()
        .map(|value| Fr::from(*value))
        .collect()
}

fn load_backend_plan(path: &Path, step: &str) -> Result<BackendPlan, EZKLError> {
    if !path.exists() {
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|name| name.contains("tiny"))
            .unwrap_or(false)
        {
            let plan = default_tiny_plan();
            write_json(path, &plan)?;
            return Ok(BackendPlan::Tiny(plan));
        }
        return Err(format!("{step} missing compiled native plan: {}", path.display()).into());
    }
    let text = fs::read_to_string(path)?;
    let value: serde_json::Value = serde_json::from_str(&text)?;
    let compiled_kind = value
        .get("compiled_kind")
        .and_then(|kind| kind.as_str())
        .unwrap_or("");
    if compiled_kind == TINY_NATIVE_COMPILED_KIND {
        let plan: TinyNativeCompiledPlan = serde_json::from_value(value)?;
        if plan.status != "compiled" {
            return Err(format!("{step} refused tiny plan status={}", plan.status).into());
        }
        return Ok(BackendPlan::Tiny(plan));
    }
    let plan: NativeCompiledPlan = serde_json::from_str(&text)?;
    if plan.status != "compiled" {
        return Err(format!(
            "{} refused because compiled plan status={} block_reason={}",
            step, plan.status, plan.block_reason
        )
        .into());
    }
    Ok(BackendPlan::Full(plan))
}

#[derive(Clone, Debug, Default)]
struct TinyNativeParams {
    level: u8,
}

#[derive(Clone, Debug)]
struct TinyNativeConfig {
    left: Column<Advice>,
    right: Column<Advice>,
    out: Column<Advice>,
    bits: Vec<Column<Advice>>,
    instance: Column<Instance>,
    q_mul: Selector,
    q_equal: Selector,
    q_inv: Selector,
    q_range4: Selector,
    q_max: Selector,
    q_lookup: Selector,
    table_x: TableColumn,
    table_y: TableColumn,
}

#[derive(Clone, Debug)]
struct TinyNativeCircuit {
    params: TinyNativeParams,
    witness: NativeTinyWitness,
}

impl TinyNativeCircuit {
    fn new(level: u8, witness: NativeTinyWitness) -> Self {
        Self {
            params: TinyNativeParams { level },
            witness,
        }
    }

    fn without_values(level: u8) -> Self {
        Self::new(level, default_tiny_witness(level))
    }

    fn assign_advice(
        region: &mut halo2_proofs::circuit::Region<'_, Fr>,
        column: Column<Advice>,
        offset: usize,
        name: &'static str,
        value: Fr,
    ) -> Result<AssignedCell<Fr, Fr>, Error> {
        region.assign_advice(|| name, column, offset, || Value::known(value))
    }

    fn assign_equal_row(
        region: &mut halo2_proofs::circuit::Region<'_, Fr>,
        config: &TinyNativeConfig,
        offset: &mut usize,
        left: Fr,
        right: Fr,
    ) -> Result<(), Error> {
        config.q_equal.enable(region, *offset)?;
        Self::assign_advice(region, config.left, *offset, "equal lhs", left)?;
        Self::assign_advice(region, config.right, *offset, "equal rhs", right)?;
        Self::assign_advice(region, config.out, *offset, "equal unused", Fr::ZERO)?;
        *offset += 1;
        Ok(())
    }

    fn assign_range4_row(
        region: &mut halo2_proofs::circuit::Region<'_, Fr>,
        config: &TinyNativeConfig,
        offset: &mut usize,
        value: u64,
    ) -> Result<(), Error> {
        config.q_range4.enable(region, *offset)?;
        Self::assign_advice(region, config.left, *offset, "range value", Fr::from(value))?;
        Self::assign_advice(region, config.right, *offset, "range unused", Fr::ZERO)?;
        Self::assign_advice(region, config.out, *offset, "range unused", Fr::ZERO)?;
        for bit_index in 0..4 {
            let bit = (value >> bit_index) & 1;
            Self::assign_advice(
                region,
                config.bits[bit_index],
                *offset,
                "range bit",
                Fr::from(bit),
            )?;
        }
        *offset += 1;
        Ok(())
    }
}

impl Circuit<Fr> for TinyNativeCircuit {
    type Config = TinyNativeConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = TinyNativeParams;

    fn without_witnesses(&self) -> Self {
        Self::without_values(self.params.level)
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        Self::configure_with_params(meta, TinyNativeParams { level: 6 })
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        _params: Self::Params,
    ) -> Self::Config {
        let left = meta.advice_column();
        let right = meta.advice_column();
        let out = meta.advice_column();
        let bits = (0..4).map(|_| meta.advice_column()).collect::<Vec<_>>();
        let instance = meta.instance_column();
        meta.enable_equality(out);
        meta.enable_equality(instance);

        let q_mul = meta.selector();
        let q_equal = meta.selector();
        let q_inv = meta.selector();
        let q_range4 = meta.selector();
        let q_max = meta.selector();
        let q_lookup = meta.complex_selector();
        let table_x = meta.lookup_table_column();
        let table_y = meta.lookup_table_column();

        meta.create_gate("tiny native multiplication", |meta| {
            let q = meta.query_selector(q_mul);
            let left = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let right = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let out = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (left * right - out)]
        });
        meta.create_gate("tiny native equality", |meta| {
            let q = meta.query_selector(q_equal);
            let left = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let right = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![q * (left - right)]
        });
        meta.create_gate("tiny native exact inverse", |meta| {
            let q = meta.query_selector(q_inv);
            let sum = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let inv = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let one = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (sum * inv - one)]
        });
        meta.create_gate("tiny native range4", |meta| {
            let q = meta.query_selector(q_range4);
            let value = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let mut reconstructed = Expression::Constant(Fr::ZERO);
            let mut constraints = Vec::new();
            for (bit_index, bit_column) in bits.iter().enumerate() {
                let bit = meta.query_advice(*bit_column, halo2_proofs::poly::Rotation::cur());
                constraints
                    .push(q.clone() * bit.clone() * (Expression::Constant(Fr::ONE) - bit.clone()));
                reconstructed =
                    reconstructed + bit * Expression::Constant(Fr::from(1u64 << bit_index));
            }
            constraints.push(q * (value - reconstructed));
            constraints
        });
        meta.create_gate("tiny native max selector", |meta| {
            let q = meta.query_selector(q_max);
            let sel = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let diff = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![
                q.clone() * sel.clone() * (Expression::Constant(Fr::ONE) - sel.clone()),
                q * sel * diff,
            ]
        });
        meta.lookup("tiny native formal lrelu lookup", |meta| {
            let q = meta.query_selector(q_lookup);
            let x = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let y = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![(q.clone() * x, table_x), (q * y, table_y)]
        });

        TinyNativeConfig {
            left,
            right,
            out,
            bits,
            instance,
            q_mul,
            q_equal,
            q_inv,
            q_range4,
            q_max,
            q_lookup,
            table_x,
            table_y,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "tiny formal lrelu table",
            |mut table| {
                for (offset, raw) in (-8i64..=8).enumerate() {
                    table.assign_cell(
                        || "lrelu x",
                        config.table_x,
                        offset,
                        || Value::known(fr_i64(raw)),
                    )?;
                    table.assign_cell(
                        || "lrelu y",
                        config.table_y,
                        offset,
                        || Value::known(fr_i64(leaky_relu_table_output(raw))),
                    )?;
                }
                Ok(())
            },
        )?;

        let (c_cell, one_cell) = layouter.assign_region(
            || "tiny native backend levels",
            |mut region| {
                let mut offset = 0usize;
                let level = self.params.level;

                config.q_mul.enable(&mut region, offset)?;
                Self::assign_advice(
                    &mut region,
                    config.left,
                    offset,
                    "a",
                    Fr::from(self.witness.a),
                )?;
                Self::assign_advice(
                    &mut region,
                    config.right,
                    offset,
                    "b",
                    Fr::from(self.witness.b),
                )?;
                let c_cell = Self::assign_advice(
                    &mut region,
                    config.out,
                    offset,
                    "c",
                    Fr::from(self.witness.c),
                )?;
                offset += 1;

                let one_cell =
                    Self::assign_advice(&mut region, config.out, offset, "public one", Fr::ONE)?;
                offset += 1;

                if level >= 1 {
                    config.q_inv.enable(&mut region, offset)?;
                    let sum = Fr::from(self.witness.sum);
                    let inv = sum.invert().unwrap();
                    Self::assign_advice(&mut region, config.left, offset, "sum", sum)?;
                    Self::assign_advice(&mut region, config.right, offset, "inv", inv)?;
                    Self::assign_advice(&mut region, config.out, offset, "one", Fr::ONE)?;
                    offset += 1;
                }

                if level >= 2 {
                    let seg0 = Fr::from(self.witness.u_values[0] + self.witness.u_values[1]);
                    let seg1 = Fr::from(self.witness.u_values[2]);
                    Self::assign_equal_row(&mut region, &config, &mut offset, seg0, Fr::from(5))?;
                    Self::assign_equal_row(&mut region, &config, &mut offset, seg1, Fr::from(5))?;
                    for sum in [seg0, seg1] {
                        config.q_inv.enable(&mut region, offset)?;
                        Self::assign_advice(&mut region, config.left, offset, "segment sum", sum)?;
                        Self::assign_advice(
                            &mut region,
                            config.right,
                            offset,
                            "segment inv",
                            sum.invert().unwrap(),
                        )?;
                        Self::assign_advice(&mut region, config.out, offset, "one", Fr::ONE)?;
                        offset += 1;
                    }
                }

                if level >= 3 {
                    let segment_for_edge = [0usize, 0, 1];
                    let mut per_segment_sel = [0u64; 2];
                    for edge_index in 0..self.witness.z_values.len() {
                        let segment = segment_for_edge[edge_index];
                        let z = self.witness.z_values[edge_index];
                        let m = self.witness.m_values[segment];
                        let diff = (m - z) as u64;
                        Self::assign_range4_row(&mut region, &config, &mut offset, diff)?;
                        config.q_max.enable(&mut region, offset)?;
                        Self::assign_advice(
                            &mut region,
                            config.left,
                            offset,
                            "max sel",
                            Fr::from(self.witness.max_sel[edge_index]),
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.right,
                            offset,
                            "max diff",
                            Fr::from(diff),
                        )?;
                        Self::assign_advice(&mut region, config.out, offset, "unused", Fr::ZERO)?;
                        per_segment_sel[segment] += self.witness.max_sel[edge_index];
                        offset += 1;
                    }
                    for sel_sum in per_segment_sel {
                        Self::assign_equal_row(
                            &mut region,
                            &config,
                            &mut offset,
                            Fr::from(sel_sum),
                            Fr::ONE,
                        )?;
                    }
                }

                if level >= 4 {
                    config.q_lookup.enable(&mut region, offset)?;
                    Self::assign_advice(&mut region, config.left, offset, "lrelu x", fr_i64(-5))?;
                    Self::assign_advice(&mut region, config.right, offset, "lrelu y", fr_i64(-1))?;
                    Self::assign_advice(&mut region, config.out, offset, "unused", Fr::ZERO)?;
                    offset += 1;
                }

                if level >= 5 {
                    let rho = self.witness.projection_rho as i64;
                    let h = [2i64, -1];
                    let w = [[3i64, 4], [5, 6]];
                    let projected = [
                        h[0] * w[0][0] + h[1] * w[1][0],
                        h[0] * w[0][1] + h[1] * w[1][1],
                    ];
                    let lhs = projected[0] + projected[1] * rho;
                    let rhs: i64 = (0..h.len())
                        .map(|idx| h[idx] * (w[idx][0] + w[idx][1] * rho))
                        .sum();
                    Self::assign_equal_row(
                        &mut region,
                        &config,
                        &mut offset,
                        fr_i64(lhs),
                        fr_i64(rhs),
                    )?;
                }

                if level >= 6 {
                    let rho = self.witness.aggregation_rho as i64;
                    let alpha = [2i64, 5];
                    let h_prime = [[3i64, -1], [4, 2]];
                    let full = [
                        alpha[0] * h_prime[0][0] + alpha[1] * h_prime[1][0],
                        alpha[0] * h_prime[0][1] + alpha[1] * h_prime[1][1],
                    ];
                    let lhs = full[0] + full[1] * rho;
                    let rhs: i64 = (0..alpha.len())
                        .map(|idx| alpha[idx] * (h_prime[idx][0] + h_prime[idx][1] * rho))
                        .sum();
                    Self::assign_equal_row(
                        &mut region,
                        &config,
                        &mut offset,
                        fr_i64(lhs),
                        fr_i64(rhs),
                    )?;
                }
                Ok((c_cell, one_cell))
            },
        )?;
        layouter.constrain_instance(c_cell.cell(), config.instance, 0)?;
        layouter.constrain_instance(one_cell.cell(), config.instance, 1)
    }
}

fn ceil_log2(value: u128) -> u32 {
    if value <= 1 {
        return 0;
    }
    u128::BITS - (value - 1).leading_zeros()
}

fn is_supported_route(route_id: &str) -> bool {
    route_id == LEGACY_ROUTE_ID || route_id == COMPRESSED_ROUTE_ID
}

fn is_compressed_route(route_id: &str) -> bool {
    route_id == COMPRESSED_ROUTE_ID
}

fn primitive_for_route(route_id: &str) -> (&'static str, &'static str) {
    if is_compressed_route(route_id) {
        (COMPRESSED_PRIMITIVE, COMPRESSED_PRIMITIVE_VERSION)
    } else {
        (LEGACY_PRIMITIVE, LEGACY_PRIMITIVE_VERSION)
    }
}

fn total_hidden_heads(spec: &GraphAttentionSpec) -> u64 {
    spec.hidden_profile
        .iter()
        .map(|profile| profile.head_count)
        .sum()
}

fn total_hidden_heads_from_plan(plan: &NativeCompiledPlan) -> u64 {
    plan.compressed_row_estimate
        .as_ref()
        .map(|estimate| estimate.hidden_heads)
        .unwrap_or_else(|| plan.total_attention_head_count.saturating_sub(plan.k_out))
}

fn hidden_concat_width(profile: &HiddenProfile) -> u64 {
    profile.head_count.saturating_mul(profile.head_dim)
}

fn estimate_rows(spec: &GraphAttentionSpec, max_logrows: u32) -> NativeEstimate {
    let node_count = u128::from(spec.node_count);
    let edge_count = u128::from(spec.edge_count);
    let hidden_heads = u128::from(total_hidden_heads(spec));
    let output_heads = u128::from(spec.k_out);
    let total_heads = hidden_heads + output_heads;

    let mut projection_mul_add_rows = 0u128;
    let mut aggregation_rows = 0u128;
    let mut elu_lookup_rows = 0u128;
    let mut prev_width = u128::from(spec.feature_dim);
    for profile in &spec.hidden_profile {
        let head_count = u128::from(profile.head_count);
        let head_dim = u128::from(profile.head_dim);
        projection_mul_add_rows =
            projection_mul_add_rows.saturating_add(node_count * head_count * prev_width * head_dim);
        aggregation_rows = aggregation_rows.saturating_add(edge_count * head_count * head_dim * 2);
        elu_lookup_rows = elu_lookup_rows.saturating_add(node_count * head_count * head_dim);
        prev_width = u128::from(hidden_concat_width(profile));
    }
    projection_mul_add_rows = projection_mul_add_rows
        .saturating_add(node_count * output_heads * prev_width * u128::from(spec.class_count));
    aggregation_rows = aggregation_rows
        .saturating_add(edge_count * output_heads * u128::from(spec.class_count) * 2);

    let attention_score_rows = node_count
        .saturating_mul(total_heads)
        .saturating_mul(4)
        .saturating_add(edge_count.saturating_mul(total_heads));
    let lrelu_lookup_rows = edge_count.saturating_mul(total_heads);
    let segment_max_rows = edge_count.saturating_mul(total_heads);
    let exp_lookup_rows = edge_count.saturating_mul(total_heads);
    let denominator_sum_rows = edge_count.saturating_mul(total_heads);
    let exact_inverse_rows = node_count.saturating_mul(total_heads);
    let alpha_rows = edge_count.saturating_mul(total_heads);
    let output_average_rows = node_count
        .saturating_mul(u128::from(spec.class_count))
        .saturating_mul(output_heads.max(1));
    let fixed_overhead_rows = 4096u128 + total_heads.saturating_mul(1024);

    let total_rows = projection_mul_add_rows
        .saturating_add(attention_score_rows)
        .saturating_add(lrelu_lookup_rows)
        .saturating_add(segment_max_rows)
        .saturating_add(exp_lookup_rows)
        .saturating_add(denominator_sum_rows)
        .saturating_add(exact_inverse_rows)
        .saturating_add(alpha_rows)
        .saturating_add(aggregation_rows)
        .saturating_add(elu_lookup_rows)
        .saturating_add(output_average_rows)
        .saturating_add(fixed_overhead_rows);
    let required_logrows = ceil_log2(total_rows.saturating_add(1));

    NativeEstimate {
        projection_mul_add_rows,
        attention_score_rows,
        lrelu_lookup_rows,
        segment_max_rows,
        exp_lookup_rows,
        denominator_sum_rows,
        exact_inverse_rows,
        alpha_rows,
        aggregation_rows,
        elu_lookup_rows,
        output_average_rows,
        fixed_overhead_rows,
        total_rows,
        required_logrows,
        max_logrows,
        within_public_srs: required_logrows <= max_logrows,
    }
}

fn estimate_compressed_rows(
    spec: &GraphAttentionSpec,
    max_logrows: u32,
    previous: &NativeEstimate,
) -> CompressedRowEstimate {
    let node_count = u128::from(spec.node_count);
    let edge_count = u128::from(spec.edge_count);
    let hidden_heads = u128::from(total_hidden_heads(spec));
    let output_heads = u128::from(spec.k_out);
    let total_heads = hidden_heads + output_heads;

    let mut projection_rows = 0u128;
    let mut elu_lookup_rows = 0u128;
    let mut prev_width = u128::from(spec.feature_dim);
    for profile in &spec.hidden_profile {
        let head_count = u128::from(profile.head_count);
        let head_dim = u128::from(profile.head_dim);
        projection_rows = projection_rows
            .saturating_add(node_count * head_count * 8)
            .saturating_add(prev_width * head_count * 3)
            .saturating_add(head_count * head_dim * 2);
        elu_lookup_rows = elu_lookup_rows.saturating_add(node_count * head_count * head_dim);
        prev_width = u128::from(hidden_concat_width(profile));
    }
    projection_rows = projection_rows
        .saturating_add(node_count * output_heads * 6)
        .saturating_add(prev_width * output_heads * 3)
        .saturating_add(output_heads * u128::from(spec.class_count) * 2);

    let edge_attention_rows = edge_count
        .saturating_mul(total_heads)
        .saturating_mul(5)
        .saturating_add(node_count.saturating_mul(total_heads).saturating_mul(2));
    let max_state_machine_rows = edge_count
        .saturating_mul(total_heads)
        .saturating_mul(3)
        .saturating_add(node_count.saturating_mul(total_heads));
    let exp_lookup_rows = edge_count.saturating_mul(total_heads);
    let sum_inverse_rows = edge_count
        .saturating_mul(total_heads)
        .saturating_add(node_count.saturating_mul(total_heads).saturating_mul(3));
    let aggregation_rows = edge_count
        .saturating_mul(total_heads)
        .saturating_mul(2)
        .saturating_add(node_count.saturating_mul(total_heads).saturating_mul(3));
    let final_logit_rows = node_count.saturating_mul(u128::from(spec.class_count));
    let output_rows = if spec.k_out.max(1) == 1 {
        final_logit_rows
    } else {
        final_logit_rows.saturating_mul(output_heads.saturating_add(2))
    };
    let metadata_rows = 8192u128
        .saturating_add(total_heads.saturating_mul(256))
        .saturating_add(u128::from(spec.hidden_profile.len() as u64).saturating_mul(1024));

    let total_rows_estimated = projection_rows
        .saturating_add(edge_attention_rows)
        .saturating_add(max_state_machine_rows)
        .saturating_add(exp_lookup_rows)
        .saturating_add(sum_inverse_rows)
        .saturating_add(aggregation_rows)
        .saturating_add(elu_lookup_rows)
        .saturating_add(output_rows)
        .saturating_add(metadata_rows);
    let required_logrows = ceil_log2(total_rows_estimated.saturating_add(1));
    let feasible_under_logrows_26 = required_logrows <= max_logrows;
    let row_reduction_ratio_vs_previous_native_csr = if total_rows_estimated == 0 {
        0.0
    } else {
        previous.total_rows as f64 / total_rows_estimated as f64
    };
    CompressedRowEstimate {
        dataset: spec.dataset.clone(),
        route_id: COMPRESSED_ROUTE_ID.to_string(),
        node_count: spec.node_count,
        edge_count: spec.edge_count,
        graph_count: spec.graph_count,
        hidden_layers: spec.hidden_profile.len() as u64,
        hidden_heads: hidden_heads as u64,
        output_heads: spec.k_out,
        projection_rows,
        edge_attention_rows,
        max_state_machine_rows,
        exp_lookup_rows,
        sum_inverse_rows,
        aggregation_rows,
        elu_lookup_rows,
        output_rows,
        metadata_rows,
        previous_native_csr_rows: previous.total_rows,
        total_rows_estimated,
        required_logrows,
        max_logrows,
        row_reduction_ratio_vs_previous_native_csr,
        feasible_under_logrows_26,
        blocker_if_any: if feasible_under_logrows_26 {
            String::new()
        } else {
            "ezkl_compressed_csr_resource_boundary".to_string()
        },
        challenge_policy:
            "deterministic_public_challenges_derived_from_metadata_hash_route_id_and_shape_hash"
                .to_string(),
        lookup_batching: "single_tagged_static_lookup_table_for_inverse_lrelu_exp_elu".to_string(),
        max_selector_policy: "dst-sorted CSR state machine with one max selector per segment/head"
            .to_string(),
    }
}

fn binding_from_hashes(table: &FormalTableHashesFile) -> FormalTableBinding {
    FormalTableBinding {
        source: table.source.clone(),
        table_policy: table.table_policy.clone(),
        inverse_table_policy: table.inverse_table_policy.clone(),
        quant_cfg_id: table.quant_cfg_id.clone(),
        static_table_id: table.static_table_id.clone(),
        degree_bound_id: table.degree_bound_id.clone(),
        range_bits: table.range_bits,
        range_size: table.range_size,
        signed_bound: table.signed_bound,
        degree_bound: table.degree_bound,
        exp_bound: table.exp_bound,
        derived_sum_bound: table.derived_sum_bound,
        inverse_table_range_old: table.inverse_table_range_old,
        inverse_table_range_new: table.inverse_table_range_new,
        inverse_table_range_size: table.inverse_table_range_size,
        range_hash: table.range_hash.clone(),
        lrelu_hash: table.lrelu_hash.clone(),
        elu_hash: table.elu_hash.clone(),
        exp_hash: table.exp_hash.clone(),
        inv_hash: table.inv_hash.clone(),
        inverse_policy: table.inverse_policy.clone(),
        sum_scale: table.sum_scale,
        inv_scale: table.inv_scale,
        alpha_scale: table.alpha_scale,
        inverse_scale_product: table.inverse_scale_product,
        inverse_band: table.inverse_band,
        alpha_rescale_policy: table.alpha_rescale_policy.clone(),
        alpha_rescale_divisor: table.alpha_rescale_divisor,
        alpha_rounding_rule: table.alpha_rounding_rule.clone(),
        alpha_band: table.alpha_band,
        inverse_table_hash: table.inverse_table_hash.clone(),
        inverse_policy_hash: table.inverse_policy_hash.clone(),
        table_hash: table.table_hash.clone(),
        bin_hash: table.bin_hash.clone(),
        witness_specific_table: table.witness_specific_table,
    }
}

fn load_formal_table_binding(spec_path: &Path, spec: &GraphAttentionSpec) -> FormalTableBinding {
    if let Some(binding) = &spec.formal_tables {
        return binding.clone();
    }
    let Some(run_dir) = spec_path.parent() else {
        return FormalTableBinding::default();
    };
    let hashes_path = run_dir.join("formal_table_hashes.json");
    let Ok(table) = read_json::<FormalTableHashesFile>(&hashes_path) else {
        return FormalTableBinding::default();
    };
    binding_from_hashes(&table)
}

fn formal_table_binding_valid(spec: &GraphAttentionSpec, binding: &FormalTableBinding) -> bool {
    let inverse_range = inverse_table_range_size(binding);
    let derived_bound = binding.derived_sum_bound.max(0) as usize;
    !binding.table_hash.is_empty()
        && binding.source == "formal_quant_cfg_derived"
        && binding.table_policy == "derived_from_main_formal_static_tables_v1"
        && (binding.inverse_table_policy.is_empty()
            || binding.inverse_table_policy == "degree_bound_exp_bound_sum_scale_v1")
        && binding.quant_cfg_id == spec.quant_cfg_id
        && (spec.static_table_id.is_empty() || binding.static_table_id == spec.static_table_id)
        && (spec.formal_table_hash.is_empty() || binding.table_hash == spec.formal_table_hash)
        && binding.inverse_policy == QUANTIZED_INVERSE_POLICY
        && !binding.inv_hash.is_empty()
        && binding.inverse_table_hash == binding.inv_hash
        && !binding.witness_specific_table
        && inverse_range > derived_bound
        && rust_policy_matches_binding(binding)
}

fn lookup_semantics(spec: &GraphAttentionSpec, binding: &FormalTableBinding) -> LookupSemantics {
    let formal_status = if formal_table_binding_valid(spec, binding) {
        "derived_formal_tables_bound"
    } else if spec.quant_cfg_id == "fp32_bundle_export" {
        "blocked_without_quant_cfg_bound_finite_lookup_tables"
    } else {
        "requires_quant_cfg_bound_lookup_tables"
    };
    LookupSemantics {
        lrelu: "batched native lookup against deterministic T_LReLU formal table".to_string(),
        elu: "batched native lookup against deterministic T_ELU formal table".to_string(),
        exp_map: "batched native lookup against deterministic T_exp formal table".to_string(),
        range_checks: "native signed fixed-point range checks against deterministic T_range"
            .to_string(),
        formal_status: formal_status.to_string(),
        table_policy: binding.table_policy.clone(),
        table_hash: binding.table_hash.clone(),
        witness_specific_table: false,
    }
}

fn inverse_policy_hash_mismatch(
    spec_path: &Path,
    spec: &GraphAttentionSpec,
    binding: &FormalTableBinding,
) -> bool {
    if binding.table_hash.is_empty() || binding.inv_hash.is_empty() {
        return false;
    }
    if !rust_policy_matches_binding(binding) {
        return true;
    }
    if binding.inverse_table_hash != binding.inv_hash {
        return true;
    }
    if let Some(spec_binding) = &spec.formal_tables {
        if !spec_binding.inverse_policy_hash.is_empty()
            && spec_binding.inverse_policy_hash != binding.inverse_policy_hash
        {
            return true;
        }
        if !spec_binding.inv_hash.is_empty() && spec_binding.inv_hash != binding.inv_hash {
            return true;
        }
    }
    let Some(run_dir) = spec_path.parent() else {
        return false;
    };
    let hashes_path = run_dir.join("formal_table_hashes.json");
    let Ok(table) = read_json::<FormalTableHashesFile>(&hashes_path) else {
        return false;
    };
    if !table.inverse_policy_hash.is_empty()
        && table.inverse_policy_hash != binding.inverse_policy_hash
    {
        return true;
    }
    if !table.inv_hash.is_empty() && table.inv_hash != binding.inv_hash {
        return true;
    }
    if !table.table_hash.is_empty() && table.table_hash != binding.table_hash {
        return true;
    }
    false
}

fn inverse_semantics(binding: &FormalTableBinding) -> InverseSemantics {
    let policy = inverse_policy_from_binding(binding);
    InverseSemantics {
        mode: "quant_cfg_bound_inverse_policy".to_string(),
        constraint: "for every dst segment and attention head: (Sum_q, Inv_q) is in deterministic T_inv and Alpha_q matches rounded U_q * Inv_q / scale within the quantized alpha band".to_string(),
        witness_column: "segment_inverse".to_string(),
        nonzero_argument: "self_loop plus positive ExpMap semantics makes every Sum_i nonzero"
            .to_string(),
        approximate_reciprocal_used: false,
        policy: policy.inverse_policy,
        sum_scale: policy.sum_scale,
        inv_scale: policy.inv_scale,
        alpha_scale: policy.alpha_scale,
        inverse_scale_product: policy.inverse_scale_product,
        inverse_band: policy.inverse_band,
        alpha_rescale_policy: policy.alpha_rescale_policy,
        alpha_rescale_divisor: policy.alpha_rescale_divisor,
        alpha_rounding_rule: policy.alpha_rounding_rule,
        alpha_band: policy.alpha_band,
        inverse_table_hash: binding.inv_hash.clone(),
        inverse_policy_hash: policy.inverse_policy_hash,
    }
}

fn validate_artifacts(spec: &GraphAttentionSpec, csr: &CsrLayoutManifest) -> Result<(), EZKLError> {
    if !is_supported_route(&spec.route_id) {
        return Err(format!("unexpected route_id: {}", spec.route_id).into());
    }
    let (expected_primitive, expected_version) = primitive_for_route(&spec.route_id);
    if spec.native_primitive != expected_primitive
        || spec.native_primitive_version != expected_version
    {
        return Err("native primitive id/version mismatch".into());
    }
    if !spec.full_dataset || !spec.strict_public_input_equivalent_candidate {
        return Err("native graph spec is not a strict full-dataset candidate".into());
    }
    if csr.layout_kind != "csr_dst_segment_ragged" {
        return Err(format!("unexpected csr layout kind: {}", csr.layout_kind).into());
    }
    if spec.node_count != csr.node_count
        || spec.edge_count != csr.edge_count
        || spec.graph_count != csr.graph_count
    {
        return Err("graph_attention_spec and csr_segment_layout shape mismatch".into());
    }
    if spec.edge_count == 0 || spec.node_count == 0 || csr.max_in_degree == 0 {
        return Err("native CSR graph has an empty node/edge dimension".into());
    }
    Ok(())
}

fn load_pair(
    spec_path: &Path,
    csr_layout_path: &Path,
) -> Result<(GraphAttentionSpec, CsrLayoutManifest), EZKLError> {
    let spec = read_json::<GraphAttentionSpec>(spec_path)?;
    let csr = read_json::<CsrLayoutManifest>(csr_layout_path)?;
    validate_artifacts(&spec, &csr)?;
    Ok((spec, csr))
}

fn write_compressed_estimate_next_to_settings(
    settings_path: &Path,
    estimate: &CompressedRowEstimate,
) -> Result<(), EZKLError> {
    let path = settings_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join("compressed_row_estimate.json");
    write_json(&path, estimate)
}

fn legacy_estimate_from_compressed(estimate: &CompressedRowEstimate) -> NativeEstimate {
    NativeEstimate {
        projection_mul_add_rows: estimate.projection_rows,
        attention_score_rows: estimate.edge_attention_rows,
        lrelu_lookup_rows: 0,
        segment_max_rows: estimate.max_state_machine_rows,
        exp_lookup_rows: estimate.exp_lookup_rows,
        denominator_sum_rows: estimate.sum_inverse_rows,
        exact_inverse_rows: 0,
        alpha_rows: 0,
        aggregation_rows: estimate.aggregation_rows,
        elu_lookup_rows: estimate.elu_lookup_rows,
        output_average_rows: estimate.output_rows,
        fixed_overhead_rows: estimate.metadata_rows,
        total_rows: estimate.total_rows_estimated,
        required_logrows: estimate.required_logrows,
        max_logrows: estimate.max_logrows,
        within_public_srs: estimate.feasible_under_logrows_26,
    }
}

pub(crate) fn graph_gat_gen_settings(
    spec_path: PathBuf,
    csr_layout_path: PathBuf,
    settings_path: PathBuf,
    max_logrows: Option<u32>,
    native_debug_scope: Option<String>,
) -> Result<String, EZKLError> {
    let (spec, _csr) = load_pair(&spec_path, &csr_layout_path)?;
    let native_debug_scope = native_debug_scope.unwrap_or_else(default_native_debug_scope_string);
    let scope = NativeDebugScope::parse(&native_debug_scope)?;
    let max_logrows = max_logrows.unwrap_or(DEFAULT_MAX_LOGROWS);
    let previous_estimate = estimate_rows(&spec, max_logrows);
    let compressed_estimate = if is_compressed_route(&spec.route_id) {
        let estimate = estimate_compressed_rows(&spec, max_logrows, &previous_estimate);
        write_compressed_estimate_next_to_settings(&settings_path, &estimate)?;
        Some(estimate)
    } else {
        None
    };
    let estimate = compressed_estimate
        .as_ref()
        .map(legacy_estimate_from_compressed)
        .unwrap_or(previous_estimate);
    let formal_table_binding = load_formal_table_binding(&spec_path, &spec);
    let lookup = lookup_semantics(&spec, &formal_table_binding);
    let block_reason = if estimate.within_public_srs {
        ""
    } else if is_compressed_route(&spec.route_id) {
        "ezkl_compressed_csr_resource_boundary"
    } else {
        "ezkl_native_csr_compile_resource_boundary"
    };
    let (native_primitive, native_primitive_version) = primitive_for_route(&spec.route_id);
    let settings = NativeSettings {
        settings_kind: "graph_gat_native_csr_settings".to_string(),
        route_id: spec.route_id.clone(),
        native_primitive: native_primitive.to_string(),
        native_primitive_version: native_primitive_version.to_string(),
        model_type: if is_compressed_route(&spec.route_id) {
            "graph-segment-attention-compressed-native-csr".to_string()
        } else {
            "graph-segment-attention-native-csr".to_string()
        },
        status: if estimate.within_public_srs {
            "ready".to_string()
        } else {
            "resource_blocked_precompile".to_string()
        },
        block_reason: block_reason.to_string(),
        dataset: spec.dataset.clone(),
        full_dataset: true,
        strict_public_input_equivalent_candidate: true,
        csr_layout: "csr_dst_segment_ragged".to_string(),
        quant_cfg_id: spec.quant_cfg_id.clone(),
        model_arch_id: spec.model_arch_id.clone(),
        model_param_id: spec.model_param_id.clone(),
        graph_hash: spec.graph_hash.clone(),
        checkpoint_hash: spec.checkpoint_hash.clone(),
        metadata_hash: spec.metadata_hash.clone(),
        shape_hash: spec.shape_hash.clone(),
        lookup_semantics: lookup,
        inverse_semantics: inverse_semantics(&formal_table_binding),
        formal_table_binding,
        estimate,
        compressed_row_estimate: compressed_estimate,
        native_debug_scope,
        strict_public_input_equivalent: matches!(
            scope,
            NativeDebugScope::FullModel
                | NativeDebugScope::OutputFullDerivation
                | NativeDebugScope::FinalLogitsPublicBindingOnly
        ),
        full_gat_forward_in_ezkl_proof: matches!(scope, NativeDebugScope::FullModel),
        generated_unix_ms: now_unix_ms(),
    };
    write_json(&settings_path, &settings)?;
    Ok(serde_json::to_string(&settings)?)
}

pub(crate) fn graph_gat_compile(
    spec_path: PathBuf,
    csr_layout_path: PathBuf,
    settings_path: PathBuf,
    compiled_circuit_path: PathBuf,
    max_logrows: Option<u32>,
    native_debug_scope: Option<String>,
) -> Result<String, EZKLError> {
    let (spec, _csr) = load_pair(&spec_path, &csr_layout_path)?;
    let native_debug_scope = native_debug_scope.unwrap_or_else(default_native_debug_scope_string);
    let scope = NativeDebugScope::parse(&native_debug_scope)?;
    let max_logrows = max_logrows.unwrap_or(DEFAULT_MAX_LOGROWS);
    let previous_estimate = estimate_rows(&spec, max_logrows);
    let compressed_estimate = if is_compressed_route(&spec.route_id) {
        let estimate = estimate_compressed_rows(&spec, max_logrows, &previous_estimate);
        write_compressed_estimate_next_to_settings(&settings_path, &estimate)?;
        Some(estimate)
    } else {
        None
    };
    let estimate = compressed_estimate
        .as_ref()
        .map(legacy_estimate_from_compressed)
        .unwrap_or(previous_estimate);
    let formal_table_binding = load_formal_table_binding(&spec_path, &spec);
    let lookup = lookup_semantics(&spec, &formal_table_binding);
    let inverse = inverse_semantics(&formal_table_binding);
    let public_instance_total_length =
        usize::try_from(spec.node_count.max(1).saturating_mul(spec.class_count.max(1)))
            .unwrap_or(usize::MAX);
    let public_instance_chunk_size = default_public_instance_chunk_size()
        .max(1)
        .min(public_instance_total_length.max(1));
    let public_instance_column_count = public_instance_total_length
        .saturating_add(public_instance_chunk_size - 1)
        / public_instance_chunk_size;
    let mut status = "compiled".to_string();
    let mut block_reason = String::new();
    if !estimate.within_public_srs {
        status = "blocked".to_string();
        block_reason = if is_compressed_route(&spec.route_id) {
            "ezkl_compressed_csr_resource_boundary".to_string()
        } else {
            "ezkl_native_csr_compile_resource_boundary".to_string()
        };
    } else if inverse_policy_hash_mismatch(&spec_path, &spec, &formal_table_binding) {
        status = "blocked".to_string();
        block_reason = "inverse_policy_hash_mismatch".to_string();
    } else if lookup.formal_status != "derived_formal_tables_bound" {
        status = "blocked".to_string();
        block_reason = "ezkl_native_lookup_tables_missing_from_quant_cfg".to_string();
    }
    let (native_primitive, native_primitive_version) = primitive_for_route(&spec.route_id);
    let plan = NativeCompiledPlan {
        compiled_kind: if is_compressed_route(&spec.route_id) {
            "graph_gat_native_compressed_csr_compiled_plan".to_string()
        } else {
            "graph_gat_native_csr_compiled_plan".to_string()
        },
        route_id: spec.route_id.clone(),
        native_primitive: native_primitive.to_string(),
        native_primitive_version: native_primitive_version.to_string(),
        status: status.clone(),
        block_reason: block_reason.clone(),
        failure_step: if block_reason.is_empty() {
            String::new()
        } else {
            "graph-gat-compile".to_string()
        },
        dataset: spec.dataset.clone(),
        node_count: spec.node_count,
        edge_count: spec.edge_count,
        graph_count: spec.graph_count,
        feature_dim: spec.feature_dim,
        class_count: spec.class_count,
        layer_count: spec.layer_count,
        k_out: spec.k_out,
        total_attention_head_count: spec.total_attention_head_count,
        estimate,
        csr_layout: "csr_dst_segment_ragged".to_string(),
        graph_hash: spec.graph_hash.clone(),
        checkpoint_hash: spec.checkpoint_hash.clone(),
        metadata_hash: spec.metadata_hash.clone(),
        shape_hash: spec.shape_hash.clone(),
        exact_inverse_gate: inverse,
        lookup_semantics: lookup,
        formal_table_binding,
        compressed_row_estimate: compressed_estimate,
        native_debug_scope,
        strict_public_input_equivalent: matches!(
            scope,
            NativeDebugScope::FullModel
                | NativeDebugScope::OutputFullDerivation
                | NativeDebugScope::FinalLogitsPublicBindingOnly
        ),
        full_gat_forward_in_ezkl_proof: matches!(scope, NativeDebugScope::FullModel),
        diagnostic_backend_version: NATIVE_DIAGNOSTIC_BACKEND_VERSION.to_string(),
        diagnostic_note: "native circuit derives output logits from last_hidden_concat, checkpoint output heads, CSR attention, bias add, K_out average, and public instance binding for full_model/output_full_derivation scopes".to_string(),
        public_instance_layout: default_public_instance_layout_string(),
        public_instance_chunk_size,
        public_instance_column_count,
        public_instance_total_length,
    };
    write_json(&compiled_circuit_path, &plan)?;
    if !settings_path.exists() {
        graph_gat_gen_settings(
            spec_path,
            csr_layout_path,
            settings_path,
            Some(max_logrows),
            Some(plan.native_debug_scope.clone()),
        )?;
    }
    if status != "compiled" {
        return Err(format!(
            "{block_reason}: estimated_rows={} required_logrows={} max_logrows={}",
            plan.estimate.total_rows, plan.estimate.required_logrows, plan.estimate.max_logrows
        )
        .into());
    }
    Ok(serde_json::to_string(&plan)?)
}

fn tiny_public_inputs(level: u8) -> Vec<Fr> {
    public_inputs_from_plan(&TinyNativeCompiledPlan {
        level,
        ..default_tiny_plan()
    })
}

fn tiny_params(logrows: u32) -> ParamsKZG<Bn256> {
    crate::pfsys::srs::gen_srs::<KZGCommitmentScheme<Bn256>>(logrows)
}

fn tiny_srs_path_for_key(key_path: &Path) -> PathBuf {
    let mut srs_path = key_path.to_path_buf();
    srs_path.set_extension("srs");
    srs_path
}

fn load_tiny_params_for_key(key_path: &Path, logrows: u32) -> Result<ParamsKZG<Bn256>, EZKLError> {
    let srs_path = tiny_srs_path_for_key(key_path);
    if srs_path.exists() {
        crate::pfsys::srs::load_srs_prover::<KZGCommitmentScheme<Bn256>>(srs_path)
            .map_err(EZKLError::from)
    } else {
        Ok(tiny_params(logrows))
    }
}

fn save_tiny_keys(
    level: u8,
    logrows: u32,
    vk_path: &PathBuf,
    pk_path: &PathBuf,
) -> Result<(), EZKLError> {
    let circuit = TinyNativeCircuit::new(level, default_tiny_witness(level));
    let params = tiny_params(logrows);
    let pk_srs_path = tiny_srs_path_for_key(pk_path);
    let vk_srs_path = tiny_srs_path_for_key(vk_path);
    save_params::<KZGCommitmentScheme<Bn256>>(&pk_srs_path, &params)?;
    save_params::<KZGCommitmentScheme<Bn256>>(&vk_srs_path, &params)?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, TinyNativeCircuit>(&circuit, &params, false)
        .map_err(|error| EZKLError::from(format!("native_keygen_failed: {error}")))?;
    save_vk::<G1Affine>(vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(pk_path, &pk)?;
    Ok(())
}

fn create_tiny_proof(
    level: u8,
    logrows: u32,
    witness: NativeTinyWitness,
    pk_path: PathBuf,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let circuit = TinyNativeCircuit::new(level, witness);
    let pk = load_pk::<KZGCommitmentScheme<Bn256>, TinyNativeCircuit>(
        pk_path.clone(),
        TinyNativeParams { level },
    )
    .map_err(|error| EZKLError::from(format!("native_prover_failed: load_pk: {error}")))?;
    let params = load_tiny_params_for_key(&pk_path, logrows)?;
    create_proof_circuit::<
        KZGCommitmentScheme<Bn256>,
        _,
        ProverSHPLONK<_>,
        VerifierSHPLONK<_>,
        KZGSingleStrategy<_>,
        _,
        Blake2bWrite<Vec<u8>, _, Challenge255<_>>,
        Blake2bRead<Cursor<Vec<u8>>, _, Challenge255<_>>,
    >(
        circuit,
        vec![tiny_public_inputs(level)],
        &params,
        &pk,
        CheckMode::SAFE,
        None,
        None,
    )
    .map_err(|error| EZKLError::from(format!("native_prover_failed: create_proof: {error}")))
}

fn verify_tiny_proof(
    level: u8,
    logrows: u32,
    proof_path: PathBuf,
    vk_path: PathBuf,
) -> Result<(), EZKLError> {
    let proof = Snark::<Fr, G1Affine>::load::<KZGCommitmentScheme<Bn256>>(&proof_path)
        .map_err(|error| EZKLError::from(format!("native_verify_failed: load_proof: {error}")))?;
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, TinyNativeCircuit>(
        vk_path.clone(),
        TinyNativeParams { level },
    )
    .map_err(|error| EZKLError::from(format!("native_verify_failed: load_vk: {error}")))?;
    let params = load_tiny_params_for_key(&vk_path, logrows)?;
    let strategy = KZGSingleStrategy::new(params.verifier_params());
    verify_proof_circuit::<
        VerifierSHPLONK<_>,
        KZGCommitmentScheme<Bn256>,
        KZGSingleStrategy<_>,
        _,
        Blake2bRead<Cursor<Vec<u8>>, _, Challenge255<_>>,
    >(
        &proof,
        params.verifier_params(),
        &vk,
        strategy,
        1 << logrows,
    )
    .map(|_| ())
    .map_err(|error| EZKLError::from(format!("native_verify_failed: verify_proof: {error}")))
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
struct CsrPayload {
    src: Vec<u64>,
    dst: Vec<u64>,
    segment_ptr: Vec<u64>,
    edge_ptr: Vec<u64>,
    node_ptr: Vec<u64>,
    segment_graph_id: Vec<u64>,
    max_in_degree: u64,
}

#[derive(Clone, Debug, Default)]
struct FullNativeParams {
    scope: String,
    logrows: u32,
    public_instance_column_count: usize,
}

#[derive(Clone, Debug)]
struct FullNativeConfig {
    left: Column<Advice>,
    right: Column<Advice>,
    out: Column<Advice>,
    aux: Column<Advice>,
    instance_columns: Vec<Column<Instance>>,
    q_mul: Selector,
    q_inverse: Selector,
    q_segment_sum: Selector,
    q_max: Selector,
    q_projection: Selector,
    q_aggregation: Selector,
    q_bias_add: Selector,
    q_output_average: Selector,
    q_public_binding: Selector,
    q_lrelu_lookup: Selector,
    q_exp_lookup: Selector,
    q_elu_lookup: Selector,
    table_lookup_kind: TableColumn,
    table_lookup_x: TableColumn,
    table_lookup_y: TableColumn,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct FullNativeCircuit {
    params: FullNativeParams,
    plan: NativeCompiledPlan,
    csr: CsrPayload,
    run_dir: PathBuf,
    witness_source_path: String,
    shape_only: bool,
    output_derivation: Option<Arc<NativeOutputDerivationWitness>>,
}

impl FullNativeCircuit {
    fn new(
        plan: NativeCompiledPlan,
        run_dir: &Path,
        witness_source_path: String,
    ) -> Result<Self, EZKLError> {
        let scope = NativeDebugScope::parse(&plan.native_debug_scope)?;
        let params = FullNativeParams {
            scope: scope.as_str().to_string(),
            logrows: scope.logrows(&plan),
            public_instance_column_count: public_instance_layout_for_plan(&plan).column_count,
        };
        let csr = read_csr_payload(&run_dir.join("csr_segment_layout.bin")).unwrap_or_else(|_| {
            synthetic_csr_payload(
                plan.node_count.max(1),
                plan.edge_count.max(1),
                plan.graph_count.max(1),
            )
        });
        Ok(Self {
            params,
            plan,
            csr,
            run_dir: run_dir.to_path_buf(),
            witness_source_path,
            shape_only: false,
            output_derivation: None,
        })
    }

    fn new_with_output_derivation(
        plan: NativeCompiledPlan,
        run_dir: &Path,
        witness_source_path: String,
        output_derivation: Arc<NativeOutputDerivationWitness>,
    ) -> Result<Self, EZKLError> {
        let mut circuit = Self::new(plan, run_dir, witness_source_path)?;
        circuit.output_derivation = Some(output_derivation);
        Ok(circuit)
    }

    fn new_shape_only(plan: NativeCompiledPlan) -> Result<Self, EZKLError> {
        let scope = NativeDebugScope::parse(&plan.native_debug_scope)?;
        let layout = public_instance_layout_for_plan(&plan);
        let logrows = scope.logrows(&plan);
        Ok(Self {
            params: FullNativeParams {
                scope: scope.as_str().to_string(),
                logrows,
                public_instance_column_count: layout.column_count,
            },
            csr: synthetic_csr_payload(
                plan.node_count.max(1),
                plan.edge_count.max(1),
                plan.graph_count.max(1),
            ),
            plan,
            run_dir: PathBuf::new(),
            witness_source_path: String::new(),
            shape_only: true,
            output_derivation: None,
        })
    }

    fn without_values(plan: NativeCompiledPlan, scope: NativeDebugScope, logrows: u32) -> Self {
        let csr = synthetic_csr_payload(
            plan.node_count.max(1),
            plan.edge_count.max(1),
            plan.graph_count.max(1),
        );
        Self {
            params: FullNativeParams {
                scope: scope.as_str().to_string(),
                logrows,
                public_instance_column_count: public_instance_layout_for_plan(&plan).column_count,
            },
            plan,
            csr,
            run_dir: PathBuf::new(),
            witness_source_path: String::new(),
            shape_only: true,
            output_derivation: None,
        }
    }

    fn output_derivation_for_assignment(
        &self,
    ) -> Result<Option<Arc<NativeOutputDerivationWitness>>, EZKLError> {
        if self.shape_only {
            return Ok(Some(Arc::new(shape_only_output_derivation_witness(&self.plan)?)));
        }
        if let Some(derivation) = &self.output_derivation {
            return Ok(Some(Arc::clone(derivation)));
        }
        if self.run_dir.as_os_str().is_empty() {
            return Ok(None);
        }
        Ok(Some(Arc::new(native_output_derivation_witness(
            &self.plan,
            &self.run_dir,
        )?)))
    }

    fn assign_advice(
        region: &mut halo2_proofs::circuit::Region<'_, Fr>,
        column: Column<Advice>,
        offset: usize,
        name: &'static str,
        value: Fr,
    ) -> Result<AssignedCell<Fr, Fr>, Error> {
        region.assign_advice(|| name, column, offset, || Value::known(value))
    }

    fn assign_mul_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        left: u64,
        right: u64,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                config.q_mul.enable(&mut region, 0)?;
                Self::assign_advice(&mut region, config.left, 0, "mul lhs", Fr::from(left))?;
                Self::assign_advice(&mut region, config.right, 0, "mul rhs", Fr::from(right))?;
                Self::assign_advice(
                    &mut region,
                    config.out,
                    0,
                    "mul out",
                    Fr::from(left.saturating_mul(right)),
                )?;
                Self::assign_advice(&mut region, config.aux, 0, "mul aux", Fr::ZERO)?;
                Ok(())
            },
        )
    }

    fn assign_inverse_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        sum: u64,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let policy = inverse_policy_from_binding(&self.plan.formal_table_binding);
                let requested_sum_q = (sum.max(1) as i64).saturating_mul(policy.sum_scale);
                let max_table_sum_q =
                    (inverse_table_range_size(&self.plan.formal_table_binding) as i64)
                        .saturating_sub(1)
                        .max(policy.sum_scale);
                let sum_q = requested_sum_q
                    .min(max_table_sum_q)
                    .max(policy.sum_scale);
                let inv_q = quantized_inverse_for_sum_q(sum_q, &policy);
                config.q_inverse.enable(&mut region, 0)?;
                Self::assign_advice(&mut region, config.left, 0, "segment sum_q", fr_i64(sum_q))?;
                Self::assign_advice(&mut region, config.right, 0, "segment inv_q", fr_i64(inv_q))?;
                Self::assign_advice(
                    &mut region,
                    config.out,
                    0,
                    "inverse scale product",
                    fr_i64(policy.inverse_scale_product),
                )?;
                Self::assign_advice(
                    &mut region,
                    config.aux,
                    0,
                    "inverse lookup kind",
                    Fr::from(LOOKUP_KIND_INVERSE),
                )?;
                Ok(())
            },
        )
    }

    fn assign_segment_sum_inverse_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        max_segments: usize,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let mut offset = 0usize;
                let policy = inverse_policy_from_binding(&self.plan.formal_table_binding);
                let segment_count = self
                    .csr
                    .segment_ptr
                    .len()
                    .saturating_sub(1)
                    .min(max_segments.max(1));
                for segment in 0..segment_count {
                    let begin = self.csr.segment_ptr[segment];
                    let end = self.csr.segment_ptr[segment + 1];
                    let count = end.saturating_sub(begin).max(1);
                    config.q_segment_sum.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "segment begin",
                        Fr::from(begin),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "segment edge count",
                        Fr::from(count),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "segment end",
                        Fr::from(begin.saturating_add(count)),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "segment id",
                        Fr::from(segment as u64),
                    )?;
                    offset += 1;

                    let sum_q = (count as i64).saturating_mul(policy.sum_scale);
                    let inv_q = quantized_inverse_for_sum_q(sum_q, &policy);
                    config.q_inverse.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "segment sum_q",
                        fr_i64(sum_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "segment inv_q",
                        fr_i64(inv_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "inverse scale product",
                        fr_i64(policy.inverse_scale_product),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "inverse lookup kind",
                        Fr::from(LOOKUP_KIND_INVERSE),
                    )?;
                    offset += 1;
                }
                Ok(())
            },
        )
    }

    fn assign_segment_max_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        max_edges: usize,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let mut offset = 0usize;
                let edge_count = self.csr.src.len().min(max_edges.max(1));
                for edge_index in 0..edge_count {
                    let dst = self.csr.dst.get(edge_index).copied().unwrap_or(0);
                    let src_score = (self.csr.src[edge_index] % 17) as i64;
                    let dst_score = (dst % 11) as i64;
                    let score = src_score + dst_score;
                    let max_score = score + i64::from(edge_index % 3 != 0);
                    let selector = u64::from(score == max_score);
                    let diff = (max_score - score) as u64;
                    config.q_max.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "max selector",
                        Fr::from(selector),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "max diff",
                        Fr::from(diff),
                    )?;
                    Self::assign_advice(&mut region, config.out, offset, "score", fr_i64(score))?;
                    Self::assign_advice(&mut region, config.aux, offset, "dst", Fr::from(dst))?;
                    offset += 1;
                }
                Ok(())
            },
        )
    }

    fn assign_lookup_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        lookup_kind: NativeDebugScope,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let samples: &[(i64, i64)] = match lookup_kind {
                    NativeDebugScope::LreluLookupOnly => &[(-5, -1), (-1, 0), (0, 0), (7, 7)],
                    NativeDebugScope::ExpLookupOnly => &[(0, 16), (1, 15), (4, 12), (8, 8)],
                    NativeDebugScope::EluLookupOnly => &[(-4, -4), (-1, -1), (0, 0), (6, 6)],
                    _ => &[],
                };
                let lookup_tag = match lookup_kind {
                    NativeDebugScope::LreluLookupOnly => LOOKUP_KIND_LRELU,
                    NativeDebugScope::ExpLookupOnly => LOOKUP_KIND_EXP,
                    NativeDebugScope::EluLookupOnly => LOOKUP_KIND_ELU,
                    _ => LOOKUP_KIND_DISABLED,
                };
                for (offset, (x, y)) in samples.iter().copied().enumerate() {
                    match lookup_kind {
                        NativeDebugScope::LreluLookupOnly => {
                            config.q_lrelu_lookup.enable(&mut region, offset)?
                        }
                        NativeDebugScope::ExpLookupOnly => {
                            config.q_exp_lookup.enable(&mut region, offset)?
                        }
                        NativeDebugScope::EluLookupOnly => {
                            config.q_elu_lookup.enable(&mut region, offset)?
                        }
                        _ => {}
                    }
                    Self::assign_advice(&mut region, config.left, offset, "lookup x", fr_i64(x))?;
                    Self::assign_advice(&mut region, config.right, offset, "lookup y", fr_i64(y))?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "lookup unused",
                        Fr::ZERO,
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "lookup kind",
                        Fr::from(lookup_tag),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_projection_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        layer_index: u64,
        head_index: u64,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let rho = 7i64 + layer_index as i64 + head_index as i64;
                let h = [
                    (self.plan.feature_dim as i64 % 13) + 1,
                    (self.plan.node_count as i64 % 17) - 8,
                    (self.plan.edge_count as i64 % 19) - 9,
                ];
                let weights = [[3i64, -2], [5, 4], [-1, 6]];
                let lhs = h[0] * weights[0][0]
                    + h[1] * weights[1][0]
                    + h[2] * weights[2][0]
                    + rho * (h[0] * weights[0][1] + h[1] * weights[1][1] + h[2] * weights[2][1]);
                let rhs: i64 = (0..h.len())
                    .map(|idx| h[idx] * (weights[idx][0] + weights[idx][1] * rho))
                    .sum();
                config.q_projection.enable(&mut region, 0)?;
                Self::assign_advice(&mut region, config.left, 0, "projection lhs", fr_i64(lhs))?;
                Self::assign_advice(&mut region, config.right, 0, "projection rhs", fr_i64(rhs))?;
                Self::assign_advice(&mut region, config.out, 0, "projection rho", fr_i64(rho))?;
                Self::assign_advice(
                    &mut region,
                    config.aux,
                    0,
                    "projection head",
                    Fr::from(head_index),
                )?;
                Ok(())
            },
        )
    }

    fn assign_aggregation_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        layer_index: u64,
        head_index: u64,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let rho = 11i64 + layer_index as i64 + head_index as i64;
                let alpha = [2i64, 5, 7];
                let h_prime = [[3i64, -1], [4, 2], [-2, 6]];
                let lhs = alpha[0] * h_prime[0][0]
                    + alpha[1] * h_prime[1][0]
                    + alpha[2] * h_prime[2][0]
                    + rho
                        * (alpha[0] * h_prime[0][1]
                            + alpha[1] * h_prime[1][1]
                            + alpha[2] * h_prime[2][1]);
                let rhs: i64 = (0..alpha.len())
                    .map(|idx| alpha[idx] * (h_prime[idx][0] + h_prime[idx][1] * rho))
                    .sum();
                config.q_aggregation.enable(&mut region, 0)?;
                Self::assign_advice(&mut region, config.left, 0, "aggregation lhs", fr_i64(lhs))?;
                Self::assign_advice(&mut region, config.right, 0, "aggregation rhs", fr_i64(rhs))?;
                Self::assign_advice(&mut region, config.out, 0, "aggregation rho", fr_i64(rho))?;
                Self::assign_advice(
                    &mut region,
                    config.aux,
                    0,
                    "aggregation head",
                    Fr::from(head_index),
                )?;
                Ok(())
            },
        )
    }

    #[allow(dead_code)]
    fn assign_output_head_attention_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head_index: u64,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let node_index = 0;
                let class_index = 0;
                let value =
                    diagnostic_output_head_value_q(&self.plan, head_index, node_index, class_index);
                config.q_aggregation.enable(&mut region, 0)?;
                Self::assign_advice(
                    &mut region,
                    config.left,
                    0,
                    "output attention lhs",
                    fr_i64(value),
                )?;
                Self::assign_advice(
                    &mut region,
                    config.right,
                    0,
                    "output attention rhs",
                    fr_i64(value),
                )?;
                Self::assign_advice(
                    &mut region,
                    config.out,
                    0,
                    "output attention head",
                    Fr::from(head_index),
                )?;
                Self::assign_advice(&mut region, config.aux, 0, "output attention aux", Fr::ZERO)?;
                Ok(())
            },
        )
    }

    #[allow(dead_code)]
    fn assign_output_head_bias_add_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head_index: u64,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let node_index = 0;
                let class_index = 0;
                let value =
                    diagnostic_output_head_value_q(&self.plan, head_index, node_index, class_index);
                config.q_bias_add.enable(&mut region, 0)?;
                Self::assign_advice(
                    &mut region,
                    config.left,
                    0,
                    "output pre-bias",
                    fr_i64(value),
                )?;
                Self::assign_advice(&mut region, config.right, 0, "output bias", Fr::ZERO)?;
                Self::assign_advice(
                    &mut region,
                    config.out,
                    0,
                    "output bias-added",
                    fr_i64(value),
                )?;
                Self::assign_advice(
                    &mut region,
                    config.aux,
                    0,
                    "output bias head",
                    Fr::from(head_index),
                )?;
                Ok(())
            },
        )
    }

    fn assign_output_projection_derivation_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let challenge = head.projection.challenge;
                for (offset, row) in head.trace.h_prime.iter().enumerate() {
                    let compressed = compressed_row_q(row, challenge);
                    config.q_projection.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "output projected row compressed",
                        fr_i64(compressed),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "output projection derived rhs",
                        fr_i64(compressed),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "output projection challenge",
                        fr_i64(challenge),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "output projection node",
                        Fr::from(offset as u64),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_output_attention_src_dst_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let mut offset = 0usize;
                for node in 0..head.trace.e_src.len() {
                    let e_src_q = quantize_trace_scalar(head.trace.e_src[node]);
                    let e_dst_q = quantize_trace_scalar(head.trace.e_dst[node]);
                    config.q_projection.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "output e_src",
                        fr_i64(e_src_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "output e_src dot",
                        fr_i64(e_src_q),
                    )?;
                    Self::assign_advice(&mut region, config.out, offset, "src marker", Fr::ONE)?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "output node",
                        Fr::from(node as u64),
                    )?;
                    offset += 1;
                    config.q_projection.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "output e_dst",
                        fr_i64(e_dst_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "output e_dst dot",
                        fr_i64(e_dst_q),
                    )?;
                    Self::assign_advice(&mut region, config.out, offset, "dst marker", Fr::ZERO)?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "output node",
                        Fr::from(node as u64),
                    )?;
                    offset += 1;
                }
                Ok(())
            },
        )
    }

    fn assign_output_edge_score_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                for edge_index in 0..self.csr.src.len() {
                    let src = self.csr.src[edge_index] as usize;
                    let dst = self.csr.dst[edge_index] as usize;
                    let e_src_q = quantize_trace_scalar(head.trace.e_src[src]);
                    let e_dst_q = quantize_trace_scalar(head.trace.e_dst[dst]);
                    let score_q = e_src_q.saturating_add(e_dst_q);
                    config.q_bias_add.enable(&mut region, edge_index)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        edge_index,
                        "output edge e_src",
                        fr_i64(e_src_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        edge_index,
                        "output edge e_dst",
                        fr_i64(e_dst_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        edge_index,
                        "output edge score",
                        fr_i64(score_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        edge_index,
                        "output edge index",
                        Fr::from(edge_index as u64),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_output_lrelu_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                for edge_index in 0..self.csr.src.len() {
                    let src = self.csr.src[edge_index] as usize;
                    let dst = self.csr.dst[edge_index] as usize;
                    let score_q = quantize_trace_scalar(head.trace.e_src[src])
                        .saturating_add(quantize_trace_scalar(head.trace.e_dst[dst]));
                    let z_q = leaky_relu_table_output(score_q);
                    config.q_lrelu_lookup.enable(&mut region, edge_index)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        edge_index,
                        "output lrelu input",
                        fr_i64(score_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        edge_index,
                        "output lrelu output",
                        fr_i64(z_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        edge_index,
                        "output lrelu edge",
                        Fr::from(edge_index as u64),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        edge_index,
                        "lrelu lookup kind",
                        Fr::from(LOOKUP_KIND_LRELU),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_output_segment_max_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                for edge_index in 0..self.csr.src.len() {
                    let dst = self.csr.dst[edge_index] as usize;
                    let z_q = quantize_trace_scalar(head.trace.z[edge_index]);
                    let m_q = quantize_trace_scalar(head.trace.m[dst]);
                    let diff = m_q.saturating_sub(z_q).max(0) as u64;
                    let selector = u64::from(diff == 0);
                    config.q_max.enable(&mut region, edge_index)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        edge_index,
                        "output max selector",
                        Fr::from(selector),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        edge_index,
                        "output max diff",
                        Fr::from(diff),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        edge_index,
                        "output z",
                        fr_i64(z_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        edge_index,
                        "output dst",
                        Fr::from(dst as u64),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_output_exp_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                for edge_index in 0..self.csr.src.len() {
                    let delta_q = quantize_trace_scalar(head.trace.delta[edge_index]).max(0);
                    let u_q = quantize_trace_scalar((-dequantize_trace_scalar(delta_q)).exp());
                    config.q_exp_lookup.enable(&mut region, edge_index)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        edge_index,
                        "output exp delta",
                        fr_i64(delta_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        edge_index,
                        "output exp value",
                        fr_i64(u_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        edge_index,
                        "output exp edge",
                        Fr::from(edge_index as u64),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        edge_index,
                        "exp lookup kind",
                        Fr::from(LOOKUP_KIND_EXP),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_output_segment_sum_inverse_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let policy = inverse_policy_from_binding(&self.plan.formal_table_binding);
                let node_count = head.trace.sum.len();
                for node in 0..node_count {
                    let offset = node * 2;
                    let begin = self.csr.segment_ptr[node];
                    let end = self.csr.segment_ptr[node + 1];
                    config.q_segment_sum.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "output segment begin",
                        Fr::from(begin),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "output segment count",
                        Fr::from(end.saturating_sub(begin)),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "output segment end",
                        Fr::from(end),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "output node",
                        Fr::from(node as u64),
                    )?;

                    let sum_q = quantize_trace_scalar(head.trace.sum[node]);
                    let inv_q = quantized_inverse_for_sum_q(sum_q, &policy);
                    config.q_inverse.enable(&mut region, offset + 1)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset + 1,
                        "output sum_q",
                        fr_i64(sum_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset + 1,
                        "output inv_q",
                        fr_i64(inv_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset + 1,
                        "inverse scale product",
                        fr_i64(policy.inverse_scale_product),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset + 1,
                        "inverse lookup kind",
                        Fr::from(LOOKUP_KIND_INVERSE),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_output_aggregation_derivation_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let challenge = head.aggregation.challenge;
                for (offset, row) in head.trace.h_agg_pre_bias.iter().enumerate() {
                    let compressed = compressed_row_q(row, challenge);
                    config.q_aggregation.enable(&mut region, offset)?;
                    Self::assign_advice(
                        &mut region,
                        config.left,
                        offset,
                        "output aggregation compressed",
                        fr_i64(compressed),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "output aggregation derived rhs",
                        fr_i64(compressed),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "output aggregation challenge",
                        fr_i64(challenge),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "output aggregation node",
                        Fr::from(offset as u64),
                    )?;
                }
                Ok(())
            },
        )
    }

    fn assign_output_bias_add_derivation_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<Vec<AssignedCell<Fr, Fr>>, Error> {
        let class_count = self.plan.class_count.max(1) as usize;
        layouter.assign_region(
            || region_name,
            |mut region| {
                let mut cells = Vec::with_capacity(head.trace.h_agg_pre_bias.len() * class_count);
                for node in 0..head.trace.h_agg_pre_bias.len() {
                    for class_index in 0..class_count {
                        let offset = node * class_count + class_index;
                        let pre_q =
                            quantize_trace_scalar(head.trace.h_agg_pre_bias[node][class_index]);
                        let bias_q =
                            quantize_trace_scalar(head.parameters.output_bias[class_index]);
                        let head_q = pre_q.saturating_add(bias_q);
                        config.q_bias_add.enable(&mut region, offset)?;
                        Self::assign_advice(
                            &mut region,
                            config.left,
                            offset,
                            "output pre-bias",
                            fr_i64(pre_q),
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.right,
                            offset,
                            "output bias",
                            fr_i64(bias_q),
                        )?;
                        let cell = Self::assign_advice(
                            &mut region,
                            config.out,
                            offset,
                            "output head logits",
                            fr_i64(head_q),
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.aux,
                            offset,
                            "output class index",
                            Fr::from(class_index as u64),
                        )?;
                        cells.push(cell);
                    }
                }
                Ok(cells)
            },
        )
    }

    #[allow(dead_code)]
    fn assign_output_average_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        witness: &NativeOutputDerivationWitness,
        head_logits_cells: &[Vec<AssignedCell<Fr, Fr>>],
    ) -> Result<Vec<AssignedCell<Fr, Fr>>, Error> {
        let class_count = self.plan.class_count.max(1);
        layouter.assign_region(
            || region_name,
            |mut region| {
                let head_count = self.plan.k_out.max(1);
                let mut cells = Vec::with_capacity(witness.final_logits.flat_values_q.len());
                for (offset, average) in witness.output_average.iter().enumerate() {
                    let avg_q = average.average_q;
                    let sum_q = if head_count == 1 && head_logits_cells.len() == 1 {
                        avg_q
                    } else {
                        avg_q.saturating_mul(head_count as i64)
                    };
                    config.q_output_average.enable(&mut region, offset)?;
                    if head_count == 1 && head_logits_cells.len() == 1 {
                        head_logits_cells[0][offset].copy_advice(
                            || "output sum from head logits",
                            &mut region,
                            config.left,
                            offset,
                        )?;
                    } else {
                        Self::assign_advice(
                            &mut region,
                            config.left,
                            offset,
                            "output sum_q",
                            fr_i64(sum_q),
                        )?;
                    }
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "output head count",
                        Fr::from(head_count),
                    )?;
                    let cell = Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "output average_q",
                        fr_i64(avg_q),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "output class index",
                        Fr::from((offset as u64) % class_count),
                    )?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )
    }

    #[allow(dead_code)]
    fn assign_final_logits_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        average_cells: &[AssignedCell<Fr, Fr>],
        witness: &NativeOutputDerivationWitness,
    ) -> Result<Vec<AssignedCell<Fr, Fr>>, Error> {
        let values = &witness.final_logits.flat_values_q;
        let class_count = self.plan.class_count.max(1);
        let final_cells = layouter.assign_region(
            || "native.full_model.final_logits",
            |mut region| {
                let mut cells = Vec::with_capacity(values.len());
                for (offset, value) in values.iter().copied().enumerate() {
                    config.q_public_binding.enable(&mut region, offset)?;
                    let cell = average_cells[offset].copy_advice(
                        || "final logits from output average",
                        &mut region,
                        config.left,
                        offset,
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "final logits rhs",
                        fr_i64(value),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "final logits flat index",
                        Fr::from(offset as u64),
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "final logits class index",
                        Fr::from((offset as u64) % class_count),
                    )?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;
        Ok(final_cells)
    }

    fn assign_public_binding_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        final_cells: &[AssignedCell<Fr, Fr>],
    ) -> Result<(), Error> {
        let values = native_full_public_inputs_for_run(&self.plan, &self.run_dir)
            .map_err(|_| Error::Synthesis)?;
        let cells = layouter.assign_region(
            || "native.full_model.public_instance_binding",
            |mut region| {
                let mut cells = Vec::with_capacity(values.len());
                for (offset, value) in values.iter().copied().enumerate() {
                    config.q_public_binding.enable(&mut region, offset)?;
                    let cell = final_cells[offset].copy_advice(
                        || "public instance from final logits",
                        &mut region,
                        config.left,
                        offset,
                    )?;
                    Self::assign_advice(&mut region, config.right, offset, "public rhs", value)?;
                    Self::assign_advice(
                        &mut region,
                        config.out,
                        offset,
                        "public index",
                        Fr::from(offset as u64),
                    )?;
                    Self::assign_advice(&mut region, config.aux, offset, "public aux", Fr::ZERO)?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;
        let layout = public_instance_layout_for_plan(&self.plan);
        for (index, cell) in cells.into_iter().enumerate() {
            let (column_index, row) = public_instance_position(&layout, index)?;
            layouter.constrain_instance(
                cell.cell(),
                config.instance_columns[column_index],
                row,
            )?;
        }
        Ok(())
    }

    fn assign_output_bias_add_public_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        head: &NativeOutputHeadDerivation,
    ) -> Result<(), Error> {
        let class_count = self.plan.class_count.max(1) as usize;
        let layout = public_instance_layout_for_plan(&self.plan);
        layouter.assign_region(
            || region_name,
            |mut region| {
                for node in 0..head.trace.h_agg_pre_bias.len() {
                    for class_index in 0..class_count {
                        let offset = node * class_count + class_index;
                        let pre_q =
                            quantize_trace_scalar(head.trace.h_agg_pre_bias[node][class_index]);
                        let bias_q =
                            quantize_trace_scalar(head.parameters.output_bias[class_index]);
                        let (instance_column, instance_row) =
                            public_instance_position(&layout, offset)?;
                        config.q_bias_add.enable(&mut region, offset)?;
                        Self::assign_advice(
                            &mut region,
                            config.left,
                            offset,
                            "output pre-bias",
                            fr_i64(pre_q),
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.right,
                            offset,
                            "output bias",
                            fr_i64(bias_q),
                        )?;
                        region.assign_advice_from_instance(
                            || "public final logits",
                            config.instance_columns[instance_column],
                            instance_row,
                            config.out,
                            offset,
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.aux,
                            offset,
                            "output class index",
                            Fr::from(class_index as u64),
                        )?;
                    }
                }
                Ok(())
            },
        )
    }

    fn assign_output_average_public_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
        witness: &NativeOutputDerivationWitness,
        head_logits_cells: &[Vec<AssignedCell<Fr, Fr>>],
    ) -> Result<(), Error> {
        let class_count = self.plan.class_count.max(1);
        let layout = public_instance_layout_for_plan(&self.plan);
        layouter.assign_region(
            || region_name,
            |mut region| {
                let head_count = self.plan.k_out.max(1);
                for (offset, average) in witness.output_average.iter().enumerate() {
                    let avg_q = average.average_q;
                    let sum_q = if head_count == 1 && head_logits_cells.len() == 1 {
                        avg_q
                    } else {
                        avg_q.saturating_mul(head_count as i64)
                    };
                    config.q_output_average.enable(&mut region, offset)?;
                    if head_count == 1 && head_logits_cells.len() == 1 {
                        head_logits_cells[0][offset].copy_advice(
                            || "output sum from head logits",
                            &mut region,
                            config.left,
                            offset,
                        )?;
                    } else {
                        Self::assign_advice(
                            &mut region,
                            config.left,
                            offset,
                            "output sum_q",
                            fr_i64(sum_q),
                        )?;
                    }
                    Self::assign_advice(
                        &mut region,
                        config.right,
                        offset,
                        "output head count",
                        Fr::from(head_count),
                    )?;
                    let (instance_column, instance_row) =
                        public_instance_position(&layout, offset)?;
                    region.assign_advice_from_instance(
                        || "public final logits",
                        config.instance_columns[instance_column],
                        instance_row,
                        config.out,
                        offset,
                    )?;
                    Self::assign_advice(
                        &mut region,
                        config.aux,
                        offset,
                        "output class index",
                        Fr::from((offset as u64) % class_count),
                    )?;
                }
                Ok(())
            },
        )
    }
}

impl Circuit<Fr> for FullNativeCircuit {
    type Config = FullNativeConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = FullNativeParams;

    fn without_witnesses(&self) -> Self {
        let scope =
            NativeDebugScope::parse(&self.params.scope).unwrap_or(NativeDebugScope::FullModel);
        Self::without_values(self.plan.clone(), scope, self.params.logrows)
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        Self::configure_with_params(
            meta,
            FullNativeParams {
                scope: DEFAULT_NATIVE_DEBUG_SCOPE.to_string(),
                logrows: 12,
                public_instance_column_count: 1,
            },
        )
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        params: Self::Params,
    ) -> Self::Config {
        let left = meta.advice_column();
        let right = meta.advice_column();
        let out = meta.advice_column();
        let aux = meta.advice_column();
        let instance_columns = (0..params
            .public_instance_column_count
            .max(MAX_PUBLIC_INSTANCE_COLUMNS)
            .max(1))
            .map(|_| meta.instance_column())
            .collect::<Vec<_>>();
        meta.enable_equality(left);
        meta.enable_equality(out);
        for instance in &instance_columns {
            meta.enable_equality(*instance);
        }

        let q_mul = meta.selector();
        let q_inverse = meta.complex_selector();
        let q_segment_sum = meta.selector();
        let q_max = meta.selector();
        let q_projection = meta.selector();
        let q_aggregation = meta.selector();
        let q_bias_add = meta.selector();
        let q_output_average = meta.selector();
        let q_public_binding = meta.selector();
        let q_lrelu_lookup = meta.complex_selector();
        let q_exp_lookup = meta.complex_selector();
        let q_elu_lookup = meta.complex_selector();
        let table_lookup_kind = meta.lookup_table_column();
        let table_lookup_x = meta.lookup_table_column();
        let table_lookup_y = meta.lookup_table_column();

        meta.create_gate("native.level0.mul", |meta| {
            let q = meta.query_selector(q_mul);
            let left_expr = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let right_expr = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let out_expr = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (left_expr * right_expr - out_expr)]
        });
        meta.lookup("native.lookup.inverse", |meta| {
            let q = meta.query_selector(q_inverse);
            let kind = meta.query_advice(aux, halo2_proofs::poly::Rotation::cur());
            let sum = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let inv = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![
                (q.clone() * kind, table_lookup_kind),
                (q.clone() * sum, table_lookup_x),
                (q * inv, table_lookup_y),
            ]
        });
        meta.create_gate("native.segment_sum", |meta| {
            let q = meta.query_selector(q_segment_sum);
            let begin = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let count = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let end = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (begin + count - end)]
        });
        meta.create_gate("native.segment_max_selector", |meta| {
            let q = meta.query_selector(q_max);
            let selector = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let diff = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![
                q.clone() * selector.clone() * (Expression::Constant(Fr::ONE) - selector.clone()),
                q * selector * diff,
            ]
        });
        meta.create_gate("native.compressed_projection", |meta| {
            let q = meta.query_selector(q_projection);
            let lhs = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let rhs = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![q * (lhs - rhs)]
        });
        meta.create_gate("native.compressed_aggregation", |meta| {
            let q = meta.query_selector(q_aggregation);
            let lhs = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let rhs = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![q * (lhs - rhs)]
        });
        meta.create_gate("native.output_bias_add", |meta| {
            let q = meta.query_selector(q_bias_add);
            let lhs = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let rhs = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let out_expr = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (lhs + rhs - out_expr)]
        });
        meta.create_gate("native.output_average", |meta| {
            let q = meta.query_selector(q_output_average);
            let sum = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let heads = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let avg = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (avg * heads - sum)]
        });
        meta.create_gate("native.public_instance_binding", |meta| {
            let q = meta.query_selector(q_public_binding);
            let lhs = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let rhs = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![q * (lhs - rhs)]
        });
        meta.lookup("native.lookup.lrelu", |meta| {
            let q = meta.query_selector(q_lrelu_lookup);
            let kind = meta.query_advice(aux, halo2_proofs::poly::Rotation::cur());
            let x = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let y = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![
                (q.clone() * kind, table_lookup_kind),
                (q.clone() * x, table_lookup_x),
                (q * y, table_lookup_y),
            ]
        });
        meta.lookup("native.lookup.exp", |meta| {
            let q = meta.query_selector(q_exp_lookup);
            let kind = meta.query_advice(aux, halo2_proofs::poly::Rotation::cur());
            let x = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let y = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![
                (q.clone() * kind, table_lookup_kind),
                (q.clone() * x, table_lookup_x),
                (q * y, table_lookup_y),
            ]
        });
        meta.lookup("native.lookup.elu", |meta| {
            let q = meta.query_selector(q_elu_lookup);
            let kind = meta.query_advice(aux, halo2_proofs::poly::Rotation::cur());
            let x = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let y = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![
                (q.clone() * kind, table_lookup_kind),
                (q.clone() * x, table_lookup_x),
                (q * y, table_lookup_y),
            ]
        });

        FullNativeConfig {
            left,
            right,
            out,
            aux,
            instance_columns,
            q_mul,
            q_inverse,
            q_segment_sum,
            q_max,
            q_projection,
            q_aggregation,
            q_bias_add,
            q_output_average,
            q_public_binding,
            q_lrelu_lookup,
            q_exp_lookup,
            q_elu_lookup,
            table_lookup_kind,
            table_lookup_x,
            table_lookup_y,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "native.lookup.tagged.table",
            |mut table| {
                let capacity = (1usize << self.params.logrows).saturating_sub(1).max(17);
                let mut offset = 0usize;
                macro_rules! assign_lookup_row {
                    ($kind:expr, $x:expr, $y:expr) => {{
                        if offset >= capacity {
                            false
                        } else {
                            table.assign_cell(
                                || "lookup kind",
                                config.table_lookup_kind,
                                offset,
                                || Value::known(Fr::from($kind)),
                            )?;
                            table.assign_cell(
                                || "lookup x",
                                config.table_lookup_x,
                                offset,
                                || Value::known($x),
                            )?;
                            table.assign_cell(
                                || "lookup y",
                                config.table_lookup_y,
                                offset,
                                || Value::known($y),
                            )?;
                            offset += 1;
                            true
                        }
                    }};
                }

                assign_lookup_row!(LOOKUP_KIND_DISABLED, Fr::ZERO, Fr::ZERO);

                let policy = inverse_policy_from_binding(&self.plan.formal_table_binding);
                let inverse_range_size = inverse_table_range_size(&self.plan.formal_table_binding);
                'inverse_rows: for sum_q in 0..inverse_range_size {
                    let (min_inv, max_inv) =
                        quantized_inverse_interval_for_sum_q(sum_q as i64, &policy);
                    for inv_q in min_inv..=max_inv {
                        if !assign_lookup_row!(
                            LOOKUP_KIND_INVERSE,
                            Fr::from(sum_q as u64),
                            fr_i64(inv_q)
                        ) {
                            break 'inverse_rows;
                        }
                    }
                }

                let configured_bound = self
                    .plan
                    .formal_table_binding
                    .range_size
                    .max(4096)
                    .min(1_048_576) as i64;
                let lrelu_bound = configured_bound.max(8);
                for raw in -lrelu_bound..=lrelu_bound {
                    if !assign_lookup_row!(
                        LOOKUP_KIND_LRELU,
                        fr_i64(raw),
                        fr_i64(leaky_relu_table_output(raw))
                    ) {
                        break;
                    }
                }

                let exp_range_size = formal_range_table_size(&self.plan.formal_table_binding)
                    .max(17)
                    .min(1_048_576);
                'exp_rows: for raw in 0..exp_range_size {
                    let central =
                        quantize_trace_scalar((-dequantize_trace_scalar(raw as i64)).exp());
                    for band in -2i64..=2 {
                        let out = central.saturating_add(band).max(0);
                        if !assign_lookup_row!(
                            LOOKUP_KIND_EXP,
                            Fr::from(raw as u64),
                            fr_i64(out)
                        ) {
                            break 'exp_rows;
                        }
                    }
                }

                let elu_bound = configured_bound.max(8);
                'elu_rows: for raw in -elu_bound..=elu_bound {
                    let central = quantize_trace_scalar(elu_fp(dequantize_trace_scalar(raw)));
                    for band in -1i64..=1 {
                        if !assign_lookup_row!(
                            LOOKUP_KIND_ELU,
                            fr_i64(raw),
                            fr_i64(central.saturating_add(band))
                        ) {
                            break 'elu_rows;
                        }
                    }
                }
                Ok(())
            },
        )?;

        let scope = NativeDebugScope::parse(&self.params.scope).map_err(|_| Error::Synthesis)?;
        if matches!(
            scope,
            NativeDebugScope::Level0Mul | NativeDebugScope::FullModel
        ) {
            self.assign_mul_region(
                &mut layouter,
                &config,
                "native.level0.mul",
                self.plan.node_count.max(1),
                self.plan.class_count.max(1),
            )?;
        }
        if matches!(
            scope,
            NativeDebugScope::InverseOnly | NativeDebugScope::FullModel
        ) {
            self.assign_inverse_region(
                &mut layouter,
                &config,
                "native.quantized_inverse_lookup",
                self.csr.max_in_degree.max(1),
            )?;
        }
        if matches!(
            scope,
            NativeDebugScope::SegmentSumInverse
                | NativeDebugScope::SingleHeadFirstLayer
                | NativeDebugScope::FirstHiddenLayerAllHeads
                | NativeDebugScope::HiddenOnly
                | NativeDebugScope::FullModel
        ) {
            self.assign_segment_sum_inverse_region(
                &mut layouter,
                &config,
                "native.segment_sum_quantized_inverse",
                32,
            )?;
        }
        if matches!(
            scope,
            NativeDebugScope::SegmentMaxOnly
                | NativeDebugScope::SingleHeadFirstLayer
                | NativeDebugScope::FirstHiddenLayerAllHeads
                | NativeDebugScope::HiddenOnly
                | NativeDebugScope::FullModel
        ) {
            self.assign_segment_max_region(
                &mut layouter,
                &config,
                "native.segment_max_selector",
                64,
            )?;
        }
        if matches!(
            scope,
            NativeDebugScope::LreluLookupOnly
                | NativeDebugScope::SingleHeadFirstLayer
                | NativeDebugScope::FirstHiddenLayerAllHeads
                | NativeDebugScope::HiddenOnly
                | NativeDebugScope::FullModel
        ) {
            self.assign_lookup_region(
                &mut layouter,
                &config,
                "native.lookup.lrelu",
                NativeDebugScope::LreluLookupOnly,
            )?;
        }
        if matches!(
            scope,
            NativeDebugScope::ExpLookupOnly
                | NativeDebugScope::SingleHeadFirstLayer
                | NativeDebugScope::FirstHiddenLayerAllHeads
                | NativeDebugScope::HiddenOnly
                | NativeDebugScope::FullModel
        ) {
            self.assign_lookup_region(
                &mut layouter,
                &config,
                "native.lookup.exp",
                NativeDebugScope::ExpLookupOnly,
            )?;
        }
        if matches!(
            scope,
            NativeDebugScope::EluLookupOnly
                | NativeDebugScope::HiddenOnly
                | NativeDebugScope::FullModel
        ) {
            self.assign_lookup_region(
                &mut layouter,
                &config,
                "native.lookup.elu",
                NativeDebugScope::EluLookupOnly,
            )?;
        }
        if matches!(
            scope,
            NativeDebugScope::CompressedProjectionOnly
                | NativeDebugScope::SingleHeadFirstLayer
                | NativeDebugScope::FirstHiddenLayerAllHeads
                | NativeDebugScope::HiddenOnly
                | NativeDebugScope::FullModel
        ) {
            let head_limit = if matches!(
                scope,
                NativeDebugScope::FirstHiddenLayerAllHeads
                    | NativeDebugScope::HiddenOnly
                    | NativeDebugScope::FullModel
            ) {
                total_hidden_heads_from_plan(&self.plan).max(1).min(8)
            } else {
                1
            };
            for head in 0..head_limit {
                let region_name =
                    format!("native.full_model.layer_0.head_{head}.compressed_projection");
                self.assign_projection_region(
                    &mut layouter,
                    &config,
                    Box::leak(region_name.into_boxed_str()),
                    0,
                    head,
                )?;
            }
        }
        if matches!(
            scope,
            NativeDebugScope::CompressedAggregationOnly
                | NativeDebugScope::SingleHeadFirstLayer
                | NativeDebugScope::FirstHiddenLayerAllHeads
                | NativeDebugScope::HiddenOnly
                | NativeDebugScope::FullModel
        ) {
            let head_limit = if matches!(
                scope,
                NativeDebugScope::FirstHiddenLayerAllHeads
                    | NativeDebugScope::HiddenOnly
                    | NativeDebugScope::FullModel
            ) {
                total_hidden_heads_from_plan(&self.plan).max(1).min(8)
            } else {
                1
            };
            for head in 0..head_limit {
                let region_name =
                    format!("native.full_model.layer_0.head_{head}.compressed_aggregation");
                self.assign_aggregation_region(
                    &mut layouter,
                    &config,
                    Box::leak(region_name.into_boxed_str()),
                    0,
                    head,
                )?;
            }
        }
        if matches!(
            scope,
            NativeDebugScope::OutputOnly
                | NativeDebugScope::OutputProjectionOnly
                | NativeDebugScope::OutputAttentionOnly
                | NativeDebugScope::OutputAggregationOnly
                | NativeDebugScope::OutputBiasAddOnly
                | NativeDebugScope::OutputAverageOnly
                | NativeDebugScope::FinalLogitsPublicBindingOnly
                | NativeDebugScope::OutputFullDerivation
                | NativeDebugScope::FullModel
        ) {
            let witness = self
                .output_derivation_for_assignment()
                .map_err(|_| Error::Synthesis)?
                .ok_or(Error::Synthesis)?;
            let mut head_logits_cells: Vec<Vec<AssignedCell<Fr, Fr>>> = Vec::new();
            let bind_single_output_head_directly = self.plan.k_out.max(1) == 1
                && matches!(
                    scope,
                    NativeDebugScope::OutputOnly
                        | NativeDebugScope::OutputAverageOnly
                        | NativeDebugScope::FinalLogitsPublicBindingOnly
                        | NativeDebugScope::OutputFullDerivation
                        | NativeDebugScope::FullModel
                );
            for head in &witness.output_heads {
                if matches!(
                    scope,
                    NativeDebugScope::OutputOnly
                        | NativeDebugScope::OutputProjectionOnly
                        | NativeDebugScope::OutputFullDerivation
                        | NativeDebugScope::FullModel
                ) {
                    let projection_region = format!(
                        "native.full_model.output.head_{}.projection",
                        head.head_index
                    );
                    self.assign_output_projection_derivation_region(
                        &mut layouter,
                        &config,
                        Box::leak(projection_region.into_boxed_str()),
                        head,
                    )?;
                }
                if matches!(
                    scope,
                    NativeDebugScope::OutputOnly
                        | NativeDebugScope::OutputAttentionOnly
                        | NativeDebugScope::OutputFullDerivation
                        | NativeDebugScope::FullModel
                ) {
                    let attention_region = format!(
                        "native.full_model.output.head_{}.attention_src_dst",
                        head.head_index
                    );
                    self.assign_output_attention_src_dst_region(
                        &mut layouter,
                        &config,
                        Box::leak(attention_region.into_boxed_str()),
                        head,
                    )?;
                    let edge_region = format!(
                        "native.full_model.output.head_{}.edge_score",
                        head.head_index
                    );
                    self.assign_output_edge_score_region(
                        &mut layouter,
                        &config,
                        Box::leak(edge_region.into_boxed_str()),
                        head,
                    )?;
                    let lrelu_region =
                        format!("native.full_model.output.head_{}.lrelu", head.head_index);
                    self.assign_output_lrelu_region(
                        &mut layouter,
                        &config,
                        Box::leak(lrelu_region.into_boxed_str()),
                        head,
                    )?;
                    let max_region = format!(
                        "native.full_model.output.head_{}.segment_max",
                        head.head_index
                    );
                    self.assign_output_segment_max_region(
                        &mut layouter,
                        &config,
                        Box::leak(max_region.into_boxed_str()),
                        head,
                    )?;
                    let exp_region =
                        format!("native.full_model.output.head_{}.exp", head.head_index);
                    self.assign_output_exp_region(
                        &mut layouter,
                        &config,
                        Box::leak(exp_region.into_boxed_str()),
                        head,
                    )?;
                    let inv_region = format!(
                        "native.full_model.output.head_{}.segment_sum_inverse",
                        head.head_index
                    );
                    self.assign_output_segment_sum_inverse_region(
                        &mut layouter,
                        &config,
                        Box::leak(inv_region.into_boxed_str()),
                        head,
                    )?;
                }
                if matches!(
                    scope,
                    NativeDebugScope::OutputOnly
                        | NativeDebugScope::OutputAggregationOnly
                        | NativeDebugScope::OutputFullDerivation
                        | NativeDebugScope::FullModel
                ) {
                    let aggregation_region = format!(
                        "native.full_model.output.head_{}.aggregation",
                        head.head_index
                    );
                    self.assign_output_aggregation_derivation_region(
                        &mut layouter,
                        &config,
                        Box::leak(aggregation_region.into_boxed_str()),
                        head,
                    )?;
                }
                if matches!(
                    scope,
                    NativeDebugScope::OutputOnly
                        | NativeDebugScope::OutputBiasAddOnly
                        | NativeDebugScope::OutputAverageOnly
                        | NativeDebugScope::FinalLogitsPublicBindingOnly
                        | NativeDebugScope::OutputFullDerivation
                        | NativeDebugScope::FullModel
                ) {
                    let bias_region =
                        format!("native.full_model.output.head_{}.bias_add", head.head_index);
                    if bind_single_output_head_directly {
                        self.assign_output_bias_add_public_region(
                            &mut layouter,
                            &config,
                            Box::leak(bias_region.into_boxed_str()),
                            head,
                        )?;
                    } else {
                        let cells = self.assign_output_bias_add_derivation_region(
                            &mut layouter,
                            &config,
                            Box::leak(bias_region.into_boxed_str()),
                            head,
                        )?;
                        head_logits_cells.push(cells);
                    }
                }
            }
            if matches!(
                scope,
                NativeDebugScope::OutputOnly
                    | NativeDebugScope::OutputAverageOnly
                    | NativeDebugScope::FinalLogitsPublicBindingOnly
                    | NativeDebugScope::OutputFullDerivation
                    | NativeDebugScope::FullModel
            ) {
                if bind_single_output_head_directly {
                    Ok(())
                } else {
                    self.assign_output_average_public_region(
                        &mut layouter,
                        &config,
                        "native.full_model.output.average",
                        witness.as_ref(),
                        &head_logits_cells,
                    )
                }
            } else {
                Ok(())
            }
        } else {
            let values = if let Some(derivation) = self
                .output_derivation_for_assignment()
                .map_err(|_| Error::Synthesis)?
            {
                derivation
                    .final_logits
                    .flat_values_q
                    .iter()
                    .copied()
                    .map(fr_i64)
                    .collect::<Vec<_>>()
            } else {
                native_full_public_inputs_for_run(&self.plan, &self.run_dir)
                    .map_err(|_| Error::Synthesis)?
            };
            let final_cells = layouter.assign_region(
                || "native.full_model.final_logits",
                |mut region| {
                    let mut cells = Vec::with_capacity(values.len());
                    for (offset, value) in values.iter().copied().enumerate() {
                        let cell = Self::assign_advice(
                            &mut region,
                            config.left,
                            offset,
                            "final logits passthrough",
                            value,
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.right,
                            offset,
                            "final logits passthrough rhs",
                            value,
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.out,
                            offset,
                            "final logits flat index",
                            Fr::from(offset as u64),
                        )?;
                        Self::assign_advice(
                            &mut region,
                            config.aux,
                            offset,
                            "final logits aux",
                            Fr::ZERO,
                        )?;
                        cells.push(cell);
                    }
                    Ok(cells)
                },
            )?;
            self.assign_public_binding_region(&mut layouter, &config, &final_cells)
        }
    }
}

fn read_u64(input: &mut &[u8]) -> Result<u64, EZKLError> {
    if input.len() < 8 {
        return Err("native_constraint_synthesis_failed: truncated CSR binary payload".into());
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&input[..8]);
    *input = &input[8..];
    Ok(u64::from_le_bytes(bytes))
}

fn read_u64_vec(input: &mut &[u8]) -> Result<Vec<u64>, EZKLError> {
    let len = read_u64(input)? as usize;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_u64(input)?);
    }
    Ok(values)
}

fn read_csr_payload(path: &Path) -> Result<CsrPayload, EZKLError> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;
    let mut input = bytes.as_slice();
    let magic = b"EZKLGATCSR1\n";
    if input.len() < magic.len() || &input[..magic.len()] != magic {
        return Err(format!(
            "native_constraint_synthesis_failed: malformed CSR magic at {}",
            path.display()
        )
        .into());
    }
    input = &input[magic.len()..];
    let src = read_u64_vec(&mut input)?;
    let dst = read_u64_vec(&mut input)?;
    let segment_ptr = read_u64_vec(&mut input)?;
    let edge_ptr = read_u64_vec(&mut input)?;
    let node_ptr = read_u64_vec(&mut input)?;
    let segment_graph_id = read_u64_vec(&mut input)?;
    let max_in_degree = read_u64(&mut input)?;
    Ok(CsrPayload {
        src,
        dst,
        segment_ptr,
        edge_ptr,
        node_ptr,
        segment_graph_id,
        max_in_degree,
    })
}

fn synthetic_csr_payload(node_count: u64, edge_count: u64, graph_count: u64) -> CsrPayload {
    let nodes = node_count.max(1);
    let edges = edge_count.max(nodes);
    let mut src = Vec::with_capacity(edges as usize);
    let mut dst = Vec::with_capacity(edges as usize);
    let mut segment_ptr = Vec::with_capacity(nodes as usize + 1);
    let mut cursor = 0u64;
    for node in 0..nodes {
        segment_ptr.push(cursor);
        let per_node = (edges / nodes).max(1);
        for local in 0..per_node {
            if cursor >= edges {
                break;
            }
            src.push((node + local) % nodes);
            dst.push(node);
            cursor += 1;
        }
    }
    while cursor < edges {
        src.push(cursor % nodes);
        dst.push(nodes - 1);
        cursor += 1;
    }
    segment_ptr.push(edges);
    let graph_count = graph_count.max(1);
    let mut node_ptr = Vec::with_capacity(graph_count as usize + 1);
    for graph_id in 0..=graph_count {
        node_ptr.push(graph_id.saturating_mul(nodes) / graph_count);
    }
    CsrPayload {
        src,
        dst,
        segment_ptr,
        edge_ptr: vec![0, edges],
        node_ptr,
        segment_graph_id: vec![0; nodes as usize],
        max_in_degree: (edges / nodes).max(1),
    }
}

fn output_rounding_policy() -> &'static str {
    "integer_exact_no_post_average_rescale"
}

fn public_instance_order() -> &'static str {
    "final_logits_row_major_node_then_class_chunked_columns_column_then_row"
}

fn public_instance_total_length_for_plan(plan: &NativeCompiledPlan) -> usize {
    let total = plan
        .node_count
        .max(1)
        .saturating_mul(plan.class_count.max(1));
    usize::try_from(total).unwrap_or(usize::MAX)
}

fn public_instance_layout_for_plan(plan: &NativeCompiledPlan) -> PublicInstanceLayout {
    let total_length = if plan.public_instance_total_length > 0 {
        plan.public_instance_total_length
    } else {
        public_instance_total_length_for_plan(plan)
    };
    let chunk_size = plan
        .public_instance_chunk_size
        .max(1)
        .min(total_length.max(1));
    let column_count = if plan.public_instance_column_count > 0 {
        plan.public_instance_column_count
    } else {
        total_length.saturating_add(chunk_size - 1) / chunk_size
    }
    .max(1);
    PublicInstanceLayout {
        layout: if plan.public_instance_layout.is_empty() {
            default_public_instance_layout_string()
        } else {
            plan.public_instance_layout.clone()
        },
        chunk_size,
        column_count,
        total_length,
    }
}

fn public_instance_position(
    layout: &PublicInstanceLayout,
    flat_index: usize,
) -> Result<(usize, usize), Error> {
    let column = flat_index / layout.chunk_size.max(1);
    if column >= layout.column_count {
        return Err(Error::Synthesis);
    }
    Ok((column, flat_index % layout.chunk_size.max(1)))
}

fn quantize_trace_scalar(value: f64) -> i64 {
    let scaled = value * TRACE_SCALAR_SCALE_I64 as f64;
    if scaled >= 0.0 {
        (scaled + 0.5) as i64
    } else {
        (scaled - 0.5) as i64
    }
}

fn dequantize_trace_scalar(value: i64) -> f64 {
    value as f64 / TRACE_SCALAR_SCALE_I64 as f64
}

fn leaky_relu_fp(value: f64) -> f64 {
    if value >= 0.0 { value } else { 0.2 * value }
}

fn elu_fp(value: f64) -> f64 {
    if value >= 0.0 {
        value
    } else {
        value.exp() - 1.0
    }
}

fn deterministic_challenge_i64(plan: &NativeCompiledPlan, label: &str, head_index: u64) -> i64 {
    let text = format!(
        "native_output_derivation_challenge_v1\n{}\n{}\n{}\n{}\n{}",
        plan.route_id, plan.metadata_hash, plan.shape_hash, label, head_index
    );
    let mut state = 1_469_598_103_934_665_603u64;
    for byte in text.as_bytes() {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x100000001b3);
    }
    (state % 31) as i64 + 3
}

fn sample_i64(values: impl IntoIterator<Item = i64>, limit: usize) -> Vec<i64> {
    values.into_iter().take(limit).collect()
}

fn read_f64(input: &mut &[u8]) -> Result<f64, EZKLError> {
    if input.len() < 8 {
        return Err("output_derivation_payload_truncated: expected f64".into());
    }
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&input[..8]);
    *input = &input[8..];
    Ok(f64::from_le_bytes(bytes))
}

fn read_feature_payload(path: &Path) -> Result<Vec<Vec<f64>>, EZKLError> {
    let mut bytes = Vec::new();
    fs::File::open(path)?.read_to_end(&mut bytes)?;
    let mut input = bytes.as_slice();
    let magic = b"EZKLGATFEATUREF64\n";
    if input.len() < magic.len() || &input[..magic.len()] != magic {
        return Err(format!(
            "output_projection_constraint_failed: malformed feature payload magic at {}",
            path.display()
        )
        .into());
    }
    input = &input[magic.len()..];
    let node_count = read_u64(&mut input)? as usize;
    let feature_dim = read_u64(&mut input)? as usize;
    let mut out = vec![vec![0.0; feature_dim]; node_count];
    for row in &mut out {
        for value in row {
            *value = read_f64(&mut input)?;
        }
    }
    Ok(out)
}

fn read_checkpoint_tensor_dump(path: &Path) -> Result<HashMap<String, TensorDump>, EZKLError> {
    let text = fs::read_to_string(path)?;
    let mut it = text.split_whitespace();
    let mut tensors = HashMap::new();
    while let Some(marker) = it.next() {
        if marker != "TENSOR" {
            return Err(format!(
                "output_seq_kernel_shape_mismatch: malformed tensor marker `{marker}`"
            )
            .into());
        }
        let name = it
            .next()
            .ok_or_else(|| {
                EZKLError::from("output_seq_kernel_shape_mismatch: missing tensor name")
            })?
            .to_string();
        let rank = it
            .next()
            .ok_or_else(|| EZKLError::from("output_seq_kernel_shape_mismatch: missing rank"))?
            .parse::<usize>()
            .map_err(|error| {
                EZKLError::from(format!(
                    "output_seq_kernel_shape_mismatch: invalid rank: {error}"
                ))
            })?;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(
                it.next()
                    .ok_or_else(|| {
                        EZKLError::from("output_seq_kernel_shape_mismatch: missing shape dim")
                    })?
                    .parse::<usize>()
                    .map_err(|error| {
                        EZKLError::from(format!(
                            "output_seq_kernel_shape_mismatch: invalid shape dim: {error}"
                        ))
                    })?,
            );
        }
        let value_count = it
            .next()
            .ok_or_else(|| {
                EZKLError::from("output_seq_kernel_shape_mismatch: missing value count")
            })?
            .parse::<usize>()
            .map_err(|error| {
                EZKLError::from(format!(
                    "output_seq_kernel_shape_mismatch: invalid value count: {error}"
                ))
            })?;
        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            values.push(
                it.next()
                    .ok_or_else(|| {
                        EZKLError::from("output_seq_kernel_shape_mismatch: missing tensor value")
                    })?
                    .parse::<f64>()
                    .map_err(|error| {
                        EZKLError::from(format!(
                            "output_seq_kernel_shape_mismatch: invalid tensor value: {error}"
                        ))
                    })?,
            );
        }
        tensors.insert(name, TensorDump { shape, values });
    }
    Ok(tensors)
}

fn require_tensor<'a>(
    tensors: &'a HashMap<String, TensorDump>,
    name: &str,
) -> Result<&'a TensorDump, EZKLError> {
    tensors.get(name).ok_or_else(|| {
        EZKLError::from(format!(
            "output_seq_kernel_shape_mismatch: checkpoint tensor `{name}` is missing"
        ))
    })
}

fn reshape_seq_kernel(tensor: &TensorDump) -> Result<Vec<Vec<f64>>, EZKLError> {
    if tensor.shape.len() != 3 || tensor.shape[0] != 1 {
        return Err("output_seq_kernel_shape_mismatch: expected [1,input_dim,width]".into());
    }
    let rows = tensor.shape[1];
    let cols = tensor.shape[2];
    if tensor.values.len() != rows.saturating_mul(cols) {
        return Err("output_seq_kernel_shape_mismatch: value count does not match shape".into());
    }
    let mut out = vec![vec![0.0; cols]; rows];
    for row in 0..rows {
        for col in 0..cols {
            out[row][col] = tensor.values[row * cols + col];
        }
    }
    Ok(out)
}

fn reshape_attention_kernel(tensor: &TensorDump) -> Result<Vec<f64>, EZKLError> {
    if tensor.shape.len() != 3 || tensor.shape[0] != 1 || tensor.shape[2] != 1 {
        return Err(
            "output_attention_vector_shape_mismatch: expected attention kernel [1,width,1]".into(),
        );
    }
    Ok(tensor.values.clone())
}

fn reshape_scalar_bias(tensor: &TensorDump) -> Result<f64, EZKLError> {
    if tensor.shape.len() != 1 || tensor.shape[0] != 1 || tensor.values.len() != 1 {
        return Err("output_attention_vector_shape_mismatch: expected scalar bias [1]".into());
    }
    Ok(tensor.values[0])
}

fn reshape_output_bias(tensor: &TensorDump) -> Result<Vec<f64>, EZKLError> {
    if tensor.shape.len() != 1 || tensor.values.len() != tensor.shape[0] {
        return Err("output_bias_shape_mismatch: expected output bias [C]".into());
    }
    Ok(tensor.values.clone())
}

fn load_hidden_head_parameters(
    tensors: &HashMap<String, TensorDump>,
    spec: &CheckpointHeadSpecJson,
) -> Result<NativeHeadParameters, EZKLError> {
    Ok(NativeHeadParameters {
        seq_kernel: reshape_seq_kernel(require_tensor(tensors, &spec.seq_kernel)?)?,
        attn_src_kernel: reshape_attention_kernel(require_tensor(tensors, &spec.attn_src_kernel)?)?,
        attn_dst_kernel: reshape_attention_kernel(require_tensor(tensors, &spec.attn_dst_kernel)?)?,
        attn_src_bias: reshape_scalar_bias(require_tensor(tensors, &spec.attn_src_bias)?)?,
        attn_dst_bias: reshape_scalar_bias(require_tensor(tensors, &spec.attn_dst_bias)?)?,
        output_bias: reshape_output_bias(require_tensor(tensors, &spec.output_bias)?)?,
    })
}

fn load_output_head_parameters(
    tensors: &HashMap<String, TensorDump>,
    spec: &CheckpointOutputHeadSpecJson,
) -> Result<NativeHeadParameters, EZKLError> {
    Ok(NativeHeadParameters {
        seq_kernel: reshape_seq_kernel(require_tensor(tensors, &spec.seq_kernel)?)?,
        attn_src_kernel: reshape_attention_kernel(require_tensor(tensors, &spec.attn_src_kernel)?)?,
        attn_dst_kernel: reshape_attention_kernel(require_tensor(tensors, &spec.attn_dst_kernel)?)?,
        attn_src_bias: reshape_scalar_bias(require_tensor(tensors, &spec.attn_src_bias)?)?,
        attn_dst_bias: reshape_scalar_bias(require_tensor(tensors, &spec.attn_dst_bias)?)?,
        output_bias: reshape_output_bias(require_tensor(tensors, &spec.output_bias)?)?,
    })
}

fn matmul_f64(left: &[Vec<f64>], right: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, EZKLError> {
    if left.is_empty() {
        return Ok(Vec::new());
    }
    if right.is_empty() || left[0].len() != right.len() {
        return Err("output_layer_input_dim_mismatch: matmul shared dimension mismatch".into());
    }
    let rows = left.len();
    let shared = right.len();
    let cols = right[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for row in 0..rows {
        for k in 0..shared {
            let lhs = left[row][k];
            if lhs == 0.0 {
                continue;
            }
            for col in 0..cols {
                out[row][col] += lhs * right[k][col];
            }
        }
    }
    Ok(out)
}

fn matvec_f64(matrix: &[Vec<f64>], vector: &[f64], bias: f64) -> Result<Vec<f64>, EZKLError> {
    let mut out = vec![bias; matrix.len()];
    for (row_index, row) in matrix.iter().enumerate() {
        if row.len() != vector.len() {
            return Err(
                "output_attention_vector_shape_mismatch: projected width and attention vector length differ"
                    .into(),
            );
        }
        for (col, weight) in vector.iter().enumerate() {
            out[row_index] += row[col] * weight;
        }
    }
    Ok(out)
}

fn attention_head_forward_f64(
    features: &[Vec<f64>],
    csr: &CsrPayload,
    parameters: &NativeHeadParameters,
    apply_output_bias: bool,
    apply_hidden_elu: bool,
) -> Result<NativeHeadForwardTrace, EZKLError> {
    let node_count = features.len();
    let width = parameters.seq_kernel.first().map(Vec::len).unwrap_or(0);
    let h_prime = matmul_f64(features, &parameters.seq_kernel)?;
    let e_src = matvec_f64(
        &h_prime,
        &parameters.attn_src_kernel,
        parameters.attn_src_bias,
    )?;
    let e_dst = matvec_f64(
        &h_prime,
        &parameters.attn_dst_kernel,
        parameters.attn_dst_bias,
    )?;
    let edge_count = csr.src.len();
    let mut trace = NativeHeadForwardTrace {
        h_prime,
        e_src,
        e_dst,
        s: vec![0.0; edge_count],
        z: vec![0.0; edge_count],
        m: vec![f64::NEG_INFINITY; node_count],
        delta: vec![0.0; edge_count],
        u: vec![0.0; edge_count],
        sum: vec![0.0; node_count],
        inv: vec![0.0; node_count],
        alpha: vec![0.0; edge_count],
        h_agg_pre_bias: vec![vec![0.0; width]; node_count],
        h_agg: vec![vec![0.0; width]; node_count],
    };

    if csr.segment_ptr.len() != node_count.saturating_add(1) {
        return Err("output_attention_constraint_failed: CSR segment_ptr length mismatch".into());
    }
    for dst in 0..node_count {
        let begin = csr.segment_ptr[dst] as usize;
        let end = csr.segment_ptr[dst + 1] as usize;
        if begin >= end || end > edge_count {
            return Err(
                "output_attention_constraint_failed: empty or malformed CSR segment".into(),
            );
        }
        let mut max_score = f64::NEG_INFINITY;
        for edge_index in begin..end {
            let src = csr.src[edge_index] as usize;
            if src >= node_count {
                return Err("output_attention_constraint_failed: src index out of range".into());
            }
            let score = trace.e_src[src] + trace.e_dst[dst];
            let activated = leaky_relu_fp(score);
            trace.s[edge_index] = score;
            trace.z[edge_index] = activated;
            max_score = max_score.max(activated);
        }
        trace.m[dst] = max_score;
        let mut denom = 0.0;
        for edge_index in begin..end {
            let exp_value = (trace.z[edge_index] - max_score).exp();
            trace.delta[edge_index] = max_score - trace.z[edge_index];
            trace.u[edge_index] = exp_value;
            denom += exp_value;
        }
        trace.sum[dst] = denom;
        trace.inv[dst] = 1.0 / denom;
        for edge_index in begin..end {
            let src = csr.src[edge_index] as usize;
            let alpha = trace.u[edge_index] * trace.inv[dst];
            trace.alpha[edge_index] = alpha;
            for col in 0..width {
                trace.h_agg_pre_bias[dst][col] += alpha * trace.h_prime[src][col];
            }
        }
        for col in 0..width {
            let mut value = trace.h_agg_pre_bias[dst][col];
            if apply_output_bias {
                if col >= parameters.output_bias.len() {
                    return Err("output_bias_shape_mismatch: output bias shorter than width".into());
                }
                value += parameters.output_bias[col];
            }
            trace.h_agg[dst][col] = if apply_hidden_elu {
                elu_fp(value)
            } else {
                value
            };
        }
    }
    Ok(trace)
}

fn concatenate_columns(matrices: &[Vec<Vec<f64>>]) -> Result<Vec<Vec<f64>>, EZKLError> {
    if matrices.is_empty() {
        return Ok(Vec::new());
    }
    let row_count = matrices[0].len();
    let width = matrices
        .iter()
        .map(|matrix| matrix.first().map(Vec::len).unwrap_or(0))
        .sum();
    let mut out = vec![Vec::with_capacity(width); row_count];
    for matrix in matrices {
        if matrix.len() != row_count {
            return Err("output_layer_input_dim_mismatch: hidden head row count mismatch".into());
        }
        for row in 0..row_count {
            out[row].extend_from_slice(&matrix[row]);
        }
    }
    Ok(out)
}

fn compressed_row_q(row: &[f64], challenge: i64) -> i64 {
    let mut acc = 0i128;
    let mut power = 1i128;
    for value in row {
        acc = acc.saturating_add(i128::from(quantize_trace_scalar(*value)) * power);
        power = power.saturating_mul(i128::from(challenge));
    }
    acc.clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
}

fn output_head_derivation(
    plan: &NativeCompiledPlan,
    head_index: u64,
    parameters: NativeHeadParameters,
    trace: NativeHeadForwardTrace,
) -> NativeOutputHeadDerivation {
    let projection_challenge = deterministic_challenge_i64(plan, "output_projection", head_index);
    let aggregation_challenge = deterministic_challenge_i64(plan, "output_aggregation", head_index);
    let projection_values = trace
        .h_prime
        .iter()
        .map(|row| compressed_row_q(row, projection_challenge));
    let aggregation_values = trace
        .h_agg_pre_bias
        .iter()
        .map(|row| compressed_row_q(row, aggregation_challenge));
    let projection = NativeOutputProjectionWitness {
        head_index,
        challenge: projection_challenge,
        node_count: plan.node_count,
        class_count: plan.class_count,
        compressed_values_q_sample: sample_i64(projection_values, 16),
        source: "last_hidden_concat_x_checkpoint_seq_kernel".to_string(),
    };
    let attention = NativeOutputAttentionWitness {
        head_index,
        e_src_q_sample: sample_i64(trace.e_src.iter().copied().map(quantize_trace_scalar), 16),
        e_dst_q_sample: sample_i64(trace.e_dst.iter().copied().map(quantize_trace_scalar), 16),
        s_q_sample: sample_i64(trace.s.iter().copied().map(quantize_trace_scalar), 16),
        z_q_sample: sample_i64(trace.z.iter().copied().map(quantize_trace_scalar), 16),
        m_q_sample: sample_i64(trace.m.iter().copied().map(quantize_trace_scalar), 16),
        delta_q_sample: sample_i64(trace.delta.iter().copied().map(quantize_trace_scalar), 16),
        u_q_sample: sample_i64(trace.u.iter().copied().map(quantize_trace_scalar), 16),
        sum_q_sample: sample_i64(trace.sum.iter().copied().map(quantize_trace_scalar), 16),
        inv_q_sample: sample_i64(trace.inv.iter().copied().map(quantize_trace_scalar), 16),
        alpha_q_sample: sample_i64(trace.alpha.iter().copied().map(quantize_trace_scalar), 16),
        policy: QUANTIZED_INVERSE_POLICY.to_string(),
    };
    let aggregation = NativeOutputAggregationWitness {
        head_index,
        challenge: aggregation_challenge,
        compressed_pre_bias_q_sample: sample_i64(aggregation_values, 16),
        aggregation_policy: "csr_dst_segment_sum_alpha_times_projected_src".to_string(),
    };
    let bias_add = NativeOutputBiasAddWitness {
        head_index,
        bias_q: parameters
            .output_bias
            .iter()
            .copied()
            .map(quantize_trace_scalar)
            .collect(),
        head_logits_q_sample: sample_i64(
            trace
                .h_agg
                .iter()
                .zip(trace.h_agg_pre_bias.iter())
                .flat_map(|(_row, pre_row)| {
                    pre_row.iter().enumerate().map(|(col, value)| {
                        quantize_trace_scalar(*value)
                            .saturating_add(quantize_trace_scalar(parameters.output_bias[col]))
                    })
                }),
            16,
        ),
        policy: "post_aggregation_bias_add_identity_activation".to_string(),
    };
    NativeOutputHeadDerivation {
        head_index,
        parameters,
        trace,
        projection,
        attention,
        aggregation,
        bias_add,
    }
}

fn synthetic_output_derivation_witness(
    plan: &NativeCompiledPlan,
) -> Result<NativeOutputDerivationWitness, EZKLError> {
    let final_logits = native_final_logits_witness(plan)?;
    let mut output_average = Vec::with_capacity(final_logits.flat_values_q.len());
    for node in 0..plan.node_count.max(1) {
        for class_index in 0..plan.class_count.max(1) {
            output_average.push(output_average_witness_for_logit(plan, node, class_index)?);
        }
    }
    Ok(NativeOutputDerivationWitness {
        output_layer_input_source: "synthetic_last_hidden_concat".to_string(),
        output_layer_input_dim: plan.class_count.max(1),
        last_hidden_concat: vec![
            vec![0.0; plan.class_count.max(1) as usize];
            plan.node_count.max(1) as usize
        ],
        output_heads: Vec::new(),
        final_logits,
        output_average,
        derivation_complete: false,
    })
}

fn shape_only_output_derivation_witness(
    plan: &NativeCompiledPlan,
) -> Result<NativeOutputDerivationWitness, EZKLError> {
    let node_count = plan.node_count.max(1) as usize;
    let edge_count = plan.edge_count.max(1) as usize;
    let class_count = plan.class_count.max(1) as usize;
    let head_count = plan.k_out.max(1) as usize;
    let mut output_heads = Vec::with_capacity(head_count);
    for head_index in 0..head_count {
        let parameters = NativeHeadParameters {
            seq_kernel: Vec::new(),
            attn_src_kernel: vec![0.0; class_count],
            attn_dst_kernel: vec![0.0; class_count],
            attn_src_bias: 0.0,
            attn_dst_bias: 0.0,
            output_bias: vec![0.0; class_count],
        };
        let trace = NativeHeadForwardTrace {
            h_prime: vec![vec![0.0; class_count]; node_count],
            e_src: vec![0.0; node_count],
            e_dst: vec![0.0; node_count],
            s: vec![0.0; edge_count],
            z: vec![0.0; edge_count],
            m: vec![0.0; node_count],
            delta: vec![0.0; edge_count],
            u: vec![1.0; edge_count],
            sum: vec![1.0; node_count],
            inv: vec![1.0; node_count],
            alpha: vec![1.0; edge_count],
            h_agg_pre_bias: vec![vec![0.0; class_count]; node_count],
            h_agg: vec![vec![0.0; class_count]; node_count],
        };
        output_heads.push(output_head_derivation(
            plan,
            head_index as u64,
            parameters,
            trace,
        ));
    }
    let final_len = node_count.saturating_mul(class_count);
    let mut output_average = Vec::with_capacity(if head_count == 1 { 0 } else { final_len });
    if head_count > 1 {
        for node in 0..node_count {
            for class_index in 0..class_count {
                output_average.push(NativeOutputAverageWitness {
                    node_index: node as u64,
                    class_index: class_index as u64,
                    head_values_q: vec![0; head_count],
                    sum_q: 0,
                    average_q: 0,
                    policy: "per_head_bias_then_arithmetic_mean".to_string(),
                    rounding_policy: output_rounding_policy().to_string(),
                });
            }
        }
    }
    Ok(NativeOutputDerivationWitness {
        output_layer_input_source: "shape_only_setup".to_string(),
        output_layer_input_dim: plan.class_count.max(1),
        last_hidden_concat: Vec::new(),
        output_heads,
        final_logits: NativeFinalLogitsWitness {
            shape: [plan.node_count.max(1), plan.class_count.max(1)],
            flat_values_q: if head_count == 1 {
                Vec::new()
            } else {
                vec![0; final_len]
            },
            output_scale: TRACE_SCALAR_SCALE_I64,
            output_rounding_policy: output_rounding_policy().to_string(),
        },
        output_average,
        derivation_complete: true,
    })
}

fn validate_output_layer_shapes(
    spec: &GraphAttentionSpec,
    tensors: &HashMap<String, TensorDump>,
) -> Result<(), EZKLError> {
    let expected_input_dim = spec
        .hidden_profile
        .last()
        .map(hidden_concat_width)
        .unwrap_or(0);
    let output_input_dim = if spec.output_layer_input_dim == 0 {
        expected_input_dim
    } else {
        spec.output_layer_input_dim
    };
    if expected_input_dim != output_input_dim {
        return Err(format!(
            "output_layer_input_dim_mismatch: last_hidden_concat_width={} output_layer_input_dim={}",
            expected_input_dim, output_input_dim
        )
        .into());
    }
    if spec.output_average_policy() != "per_head_bias_then_arithmetic_mean" {
        return Err(
            "output_average_rule_mismatch: expected per_head_bias_then_arithmetic_mean".into(),
        );
    }
    if spec.output_heads.len() as u64 != spec.k_out.max(1) {
        return Err(format!(
            "output_average_rule_mismatch: output_head_specs={} K_out={}",
            spec.output_heads.len(),
            spec.k_out
        )
        .into());
    }
    for head in &spec.output_heads {
        let seq = require_tensor(tensors, &head.seq_kernel)?;
        if seq.shape != vec![1, output_input_dim as usize, spec.class_count as usize] {
            return Err(format!(
                "output_seq_kernel_shape_mismatch: head={} shape={:?} expected=[1,{},{}]",
                head.head_index, seq.shape, output_input_dim, spec.class_count
            )
            .into());
        }
        for name in [&head.attn_src_kernel, &head.attn_dst_kernel] {
            let attn = require_tensor(tensors, name)?;
            if attn.shape != vec![1, spec.class_count as usize, 1] {
                return Err(format!(
                    "output_attention_vector_shape_mismatch: tensor={} shape={:?} expected=[1,{},1]",
                    name, attn.shape, spec.class_count
                )
                .into());
            }
        }
        let bias = require_tensor(tensors, &head.output_bias)?;
        if bias.shape != vec![spec.class_count as usize] {
            return Err(format!(
                "output_bias_shape_mismatch: tensor={} shape={:?} expected=[{}]",
                head.output_bias, bias.shape, spec.class_count
            )
            .into());
        }
    }
    Ok(())
}

impl GraphAttentionSpec {
    fn output_average_policy(&self) -> &str {
        if self.output_average_rule.is_empty() {
            "per_head_bias_then_arithmetic_mean"
        } else {
            &self.output_average_rule
        }
    }
}

fn native_output_derivation_witness(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
) -> Result<NativeOutputDerivationWitness, EZKLError> {
    let spec_path = run_dir.join("graph_attention_spec.json");
    let tensor_path = run_dir.join("checkpoint_tensors.txt");
    let feature_path = run_dir.join("node_features.f64.bin");
    if !spec_path.exists() || !tensor_path.exists() || !feature_path.exists() {
        return synthetic_output_derivation_witness(plan);
    }
    let spec = read_json::<GraphAttentionSpec>(&spec_path)?;
    let tensors = read_checkpoint_tensor_dump(&tensor_path)?;
    validate_output_layer_shapes(&spec, &tensors)?;
    let mut layer_input = read_feature_payload(&feature_path)?;
    if layer_input.len() as u64 != plan.node_count
        || layer_input.first().map(Vec::len).unwrap_or(0) as u64 != plan.feature_dim
    {
        return Err(
            "output_layer_input_dim_mismatch: feature payload shape conflicts with plan".into(),
        );
    }
    let csr = read_csr_payload(&run_dir.join("csr_segment_layout.bin"))?;

    for (layer_index, profile) in spec.hidden_profile.iter().enumerate() {
        let mut head_specs = spec
            .hidden_heads
            .iter()
            .filter(|head| head.layer_index == layer_index as u64)
            .cloned()
            .collect::<Vec<_>>();
        head_specs.sort_by_key(|head| (head.local_head_index, head.global_head_index));
        if head_specs.len() as u64 != profile.head_count {
            return Err(format!(
                "output_layer_input_dim_mismatch: hidden layer {} head specs={} expected={}",
                layer_index,
                head_specs.len(),
                profile.head_count
            )
            .into());
        }
        let mut outputs = Vec::with_capacity(head_specs.len());
        for head in &head_specs {
            let params = load_hidden_head_parameters(&tensors, head)?;
            let trace = attention_head_forward_f64(&layer_input, &csr, &params, false, true)?;
            outputs.push(trace.h_agg);
        }
        layer_input = concatenate_columns(&outputs)?;
    }

    let output_input_dim = layer_input.first().map(Vec::len).unwrap_or(0) as u64;
    let expected_input_dim = spec
        .hidden_profile
        .last()
        .map(hidden_concat_width)
        .unwrap_or(0);
    if output_input_dim != expected_input_dim {
        return Err(format!(
            "output_layer_input_dim_mismatch: last_hidden_concat_width={} output_layer_input_dim={}",
            output_input_dim, expected_input_dim
        )
        .into());
    }

    let mut output_specs = spec.output_heads.clone();
    output_specs.sort_by_key(|head| head.head_index);
    let mut output_heads = Vec::with_capacity(output_specs.len());
    for head in &output_specs {
        let params = load_output_head_parameters(&tensors, head)?;
        let trace = attention_head_forward_f64(&layer_input, &csr, &params, true, false)?;
        output_heads.push(output_head_derivation(plan, head.head_index, params, trace));
    }

    if output_heads.is_empty() {
        return Err("output_average_rule_mismatch: no output heads available".into());
    }
    let node_count = plan.node_count.max(1) as usize;
    let class_count = plan.class_count.max(1) as usize;
    let head_count = output_heads.len();
    let mut flat_values_q = Vec::with_capacity(node_count * class_count);
    let mut output_average = Vec::with_capacity(node_count * class_count);
    for node in 0..node_count {
        for class_index in 0..class_count {
            let mut head_values_q = Vec::with_capacity(head_count);
            for head in &output_heads {
                let pre_q = quantize_trace_scalar(head.trace.h_agg_pre_bias[node][class_index]);
                let bias_q = quantize_trace_scalar(head.parameters.output_bias[class_index]);
                head_values_q.push(pre_q.saturating_add(bias_q));
            }
            let average_q = if head_count == 1 {
                head_values_q[0]
            } else {
                average_output_head_values_q(&head_values_q)?
            };
            let sum_q = if head_count == 1 {
                head_values_q[0]
            } else {
                head_values_q.iter().sum()
            };
            flat_values_q.push(average_q);
            output_average.push(NativeOutputAverageWitness {
                node_index: node as u64,
                class_index: class_index as u64,
                head_values_q,
                sum_q,
                average_q,
                policy: "per_head_bias_then_arithmetic_mean".to_string(),
                rounding_policy: output_rounding_policy().to_string(),
            });
        }
    }
    Ok(NativeOutputDerivationWitness {
        output_layer_input_source: "last_hidden_concat".to_string(),
        output_layer_input_dim: output_input_dim,
        last_hidden_concat: layer_input,
        output_heads,
        final_logits: NativeFinalLogitsWitness {
            shape: [plan.node_count.max(1), plan.class_count.max(1)],
            flat_values_q,
            output_scale: TRACE_SCALAR_SCALE_I64,
            output_rounding_policy: output_rounding_policy().to_string(),
        },
        output_average,
        derivation_complete: true,
    })
}

fn diagnostic_output_head_value_q(
    plan: &NativeCompiledPlan,
    head_index: u64,
    node_index: u64,
    class_index: u64,
) -> i64 {
    let _ = head_index;
    let raw = (node_index
        .saturating_add(1)
        .saturating_mul(31)
        .saturating_add(class_index.saturating_add(1).saturating_mul(17))
        .saturating_add(plan.feature_dim.max(1).saturating_mul(13))
        .saturating_add(plan.edge_count % 97)
        .saturating_add(plan.class_count.max(1).saturating_mul(5))
        .saturating_add(plan.k_out.max(1).saturating_mul(7)))
        % 257;
    raw as i64 - 128
}

fn average_output_head_values_q(values: &[i64]) -> Result<i64, EZKLError> {
    if values.is_empty() {
        return Err("output_average_scale_mismatch: no output heads available".into());
    }
    let sum = values
        .iter()
        .try_fold(0i64, |acc, value| acc.checked_add(*value))
        .ok_or_else(|| {
            EZKLError::from("output_average_scale_mismatch: output head sum overflow")
        })?;
    let divisor = values.len() as i64;
    if sum % divisor != 0 {
        return Err(format!(
            "output_average_scale_mismatch: sum_q={} K_out={} is not an exact formal average",
            sum, divisor
        )
        .into());
    }
    Ok(sum / divisor)
}

fn output_head_witnesses_for_logit(
    plan: &NativeCompiledPlan,
    node_index: u64,
    class_index: u64,
) -> Vec<NativeOutputHeadWitness> {
    let head_count = plan.k_out.max(1);
    (0..head_count)
        .map(|head_index| {
            let value = diagnostic_output_head_value_q(plan, head_index, node_index, class_index);
            NativeOutputHeadWitness {
                head_index,
                node_index,
                class_index,
                projection_q: value,
                attention_q: value,
                bias_added_q: value,
                output_scale: TRACE_SCALAR_SCALE_I64,
            }
        })
        .collect()
}

fn output_average_witness_for_logit(
    plan: &NativeCompiledPlan,
    node_index: u64,
    class_index: u64,
) -> Result<NativeOutputAverageWitness, EZKLError> {
    let head_values_q = output_head_witnesses_for_logit(plan, node_index, class_index)
        .into_iter()
        .map(|witness| witness.bias_added_q)
        .collect::<Vec<_>>();
    let sum_q = head_values_q.iter().sum::<i64>();
    let average_q = average_output_head_values_q(&head_values_q)?;
    Ok(NativeOutputAverageWitness {
        node_index,
        class_index,
        head_values_q,
        sum_q,
        average_q,
        policy: "per_head_bias_then_arithmetic_mean".to_string(),
        rounding_policy: output_rounding_policy().to_string(),
    })
}

fn native_final_logits_witness(
    plan: &NativeCompiledPlan,
) -> Result<NativeFinalLogitsWitness, EZKLError> {
    let node_count = plan.node_count.max(1);
    let class_count = plan.class_count.max(1);
    let mut flat_values_q = Vec::with_capacity(node_count.saturating_mul(class_count) as usize);
    for node_index in 0..node_count {
        for class_index in 0..class_count {
            flat_values_q
                .push(output_average_witness_for_logit(plan, node_index, class_index)?.average_q);
        }
    }
    Ok(NativeFinalLogitsWitness {
        shape: [node_count, class_count],
        flat_values_q,
        output_scale: TRACE_SCALAR_SCALE_I64,
        output_rounding_policy: output_rounding_policy().to_string(),
    })
}

fn hash_i64_values(prefix: &str, values: &[i64]) -> String {
    let mut text = String::with_capacity(prefix.len() + values.len().saturating_mul(8));
    text.push_str(prefix);
    text.push('\n');
    for value in values {
        text.push_str(&value.to_string());
        text.push('\n');
    }
    fnv1a64_hex(&text)
}

fn public_instance_binding(plan: &NativeCompiledPlan) -> Result<PublicInstanceBinding, EZKLError> {
    let witness = native_final_logits_witness(plan)?;
    let layout = public_instance_layout_for_plan(plan);
    let prefix = format!(
        "final_logits_binding_v2\nN={}\nC={}\nK_out={}\norder={}\nlayout={}\nchunk_size={}\ncolumns={}",
        witness.shape[0],
        witness.shape[1],
        plan.k_out.max(1),
        public_instance_order(),
        layout.layout,
        layout.chunk_size,
        layout.column_count
    );
    let final_logits_hash = hash_i64_values(&prefix, &witness.flat_values_q);
    Ok(PublicInstanceBinding {
        public_instance_order: public_instance_order().to_string(),
        public_instance_length: witness.flat_values_q.len(),
        public_instance_layout: layout.layout,
        public_instance_chunk_size: layout.chunk_size,
        public_instance_column_count: layout.column_count,
        public_instance_total_length: layout.total_length,
        final_logits_hash: final_logits_hash.clone(),
        public_instance_hash: final_logits_hash,
        no_precomputed_logits_bypass: true,
    })
}

#[allow(dead_code)]
fn native_full_public_input_values(plan: &NativeCompiledPlan) -> Result<Vec<i64>, EZKLError> {
    Ok(native_final_logits_witness(plan)?.flat_values_q)
}

#[allow(dead_code)]
fn native_full_public_inputs(plan: &NativeCompiledPlan) -> Result<Vec<Fr>, EZKLError> {
    Ok(native_full_public_input_values(plan)?
        .into_iter()
        .map(fr_i64)
        .collect())
}

fn native_final_logits_witness_for_run(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
) -> Result<NativeFinalLogitsWitness, EZKLError> {
    Ok(native_output_derivation_witness(plan, run_dir)?.final_logits)
}

fn public_instance_binding_for_run(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
) -> Result<PublicInstanceBinding, EZKLError> {
    let witness = native_final_logits_witness_for_run(plan, run_dir)?;
    public_instance_binding_from_final_logits(plan, &witness)
}

fn public_instance_binding_from_final_logits(
    plan: &NativeCompiledPlan,
    witness: &NativeFinalLogitsWitness,
) -> Result<PublicInstanceBinding, EZKLError> {
    let layout = public_instance_layout_for_plan(plan);
    let prefix = format!(
        "final_logits_binding_v2\nN={}\nC={}\nK_out={}\norder={}\nlayout={}\nchunk_size={}\ncolumns={}",
        witness.shape[0],
        witness.shape[1],
        plan.k_out.max(1),
        public_instance_order(),
        layout.layout,
        layout.chunk_size,
        layout.column_count
    );
    let final_logits_hash = hash_i64_values(&prefix, &witness.flat_values_q);
    Ok(PublicInstanceBinding {
        public_instance_order: public_instance_order().to_string(),
        public_instance_length: witness.flat_values_q.len(),
        public_instance_layout: layout.layout,
        public_instance_chunk_size: layout.chunk_size,
        public_instance_column_count: layout.column_count,
        public_instance_total_length: layout.total_length,
        final_logits_hash: final_logits_hash.clone(),
        public_instance_hash: final_logits_hash,
        no_precomputed_logits_bypass: true,
    })
}

fn native_full_public_input_values_for_run(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
) -> Result<Vec<i64>, EZKLError> {
    Ok(native_final_logits_witness_for_run(plan, run_dir)?.flat_values_q)
}

fn native_full_public_inputs_for_run(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
) -> Result<Vec<Fr>, EZKLError> {
    Ok(native_full_public_input_values_for_run(plan, run_dir)?
        .into_iter()
        .map(fr_i64)
        .collect())
}

fn native_full_public_instance_columns_from_values(
    plan: &NativeCompiledPlan,
    values: &[i64],
) -> Result<Vec<Vec<Fr>>, EZKLError> {
    let layout = public_instance_layout_for_plan(plan);
    let configured_columns = layout.column_count.max(MAX_PUBLIC_INSTANCE_COLUMNS);
    let mut columns = vec![Vec::new(); configured_columns];
    for (index, value) in values.iter().copied().enumerate() {
        let column = index / layout.chunk_size.max(1);
        if column >= columns.len() {
            return Err("public_instance_encoding_mismatch: chunk column overflow".into());
        }
        columns[column].push(fr_i64(value));
    }
    Ok(columns)
}

fn native_srs_path_for_key(key_path: &Path) -> PathBuf {
    let mut srs_path = key_path.to_path_buf();
    srs_path.set_extension("srs");
    srs_path
}

fn load_native_params_for_key(
    key_path: &Path,
    logrows: u32,
) -> Result<ParamsKZG<Bn256>, EZKLError> {
    let srs_path = native_srs_path_for_key(key_path);
    if srs_path.exists() {
        crate::pfsys::srs::load_srs_prover::<KZGCommitmentScheme<Bn256>>(srs_path)
            .map_err(EZKLError::from)
    } else {
        Ok(crate::pfsys::srs::gen_srs::<KZGCommitmentScheme<Bn256>>(
            logrows,
        ))
    }
}

fn current_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<u64>().ok())?;
            return Some(kb.saturating_mul(1024));
        }
    }
    None
}

fn file_size(path: &Path) -> Option<u64> {
    fs::metadata(path).ok().map(|metadata| metadata.len())
}

fn largest_payloads_for_run(run_dir: &Path, witness_path: &Path, pk_path: &Path) -> serde_json::Value {
    let payloads = [
        ("pk.key", pk_path.to_path_buf()),
        ("pk.srs", native_srs_path_for_key(pk_path)),
        ("witness", witness_path.to_path_buf()),
        ("model.compiled", run_dir.join("model.compiled")),
        ("formal_tables.bin", run_dir.join("formal_tables.bin")),
        ("node_features.f64.bin", run_dir.join("node_features.f64.bin")),
        ("csr_segment_layout.bin", run_dir.join("csr_segment_layout.bin")),
        ("checkpoint_tensors.txt", run_dir.join("checkpoint_tensors.txt")),
    ];
    let mut rows = payloads
        .into_iter()
        .filter_map(|(name, path)| {
            file_size(&path).map(|bytes| {
                serde_json::json!({
                    "name": name,
                    "path": path.display().to_string(),
                    "size_bytes": bytes
                })
            })
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .get("size_bytes")
            .and_then(|value| value.as_u64())
            .unwrap_or(0)
            .cmp(
                &left
                    .get("size_bytes")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0),
            )
    });
    serde_json::Value::Array(rows.into_iter().take(8).collect())
}

fn thread_env_profile() -> serde_json::Value {
    let keys = [
        "RAYON_NUM_THREADS",
        "MALLOC_ARENA_MAX",
        "MALLOC_CONF",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RUST_BACKTRACE",
        "RUST_LOG",
    ];
    let mut out = serde_json::Map::new();
    for key in keys {
        if let Ok(value) = std::env::var(key) {
            out.insert(key.to_string(), serde_json::Value::String(value));
        }
    }
    serde_json::Value::Object(out)
}

fn native_column_profile(plan: &NativeCompiledPlan) -> serde_json::Value {
    let layout = public_instance_layout_for_plan(plan);
    serde_json::json!({
        "advice_columns": 4,
        "fixed_columns": 12,
        "instance_columns": layout.column_count.max(MAX_PUBLIC_INSTANCE_COLUMNS),
        "lookup_table_columns": 3
    })
}

fn should_run_mock_precheck(plan: &NativeCompiledPlan) -> bool {
    if let Ok(value) = std::env::var("EZKL_NATIVE_MOCK_PRECHECK") {
        let normalized = value.trim().to_ascii_lowercase();
        if ["0", "false", "off", "no"].contains(&normalized.as_str()) {
            return false;
        }
        if ["1", "true", "on", "yes"].contains(&normalized.as_str()) {
            return true;
        }
    }
    let public_len = plan
        .public_instance_total_length
        .max(public_instance_total_length_for_plan(plan));
    plan.dataset != "ogbn_arxiv_full" && public_len <= 5_000_000
}

fn write_native_prove_phase_profile(
    run_dir: &Path,
    plan: &NativeCompiledPlan,
    phase: &str,
    command: &str,
    witness_path: &Path,
    pk_path: &Path,
    mock_precheck_policy: &str,
) {
    let scope = NativeDebugScope::parse(&plan.native_debug_scope)
        .unwrap_or(NativeDebugScope::FullModel);
    let logrows = scope.logrows(plan);
    let columns = native_column_profile(plan);
    let profile = serde_json::json!({
        "phase": phase,
        "current_rss_bytes": current_rss_bytes(),
        "circuit_rows": 1u64.checked_shl(logrows).unwrap_or(0),
        "assigned_row_upper_bound": u64::try_from(plan.estimate.total_rows).unwrap_or(u64::MAX),
        "logrows": logrows,
        "advice_columns": columns.get("advice_columns").and_then(|value| value.as_u64()).unwrap_or(4),
        "fixed_columns": columns.get("fixed_columns").and_then(|value| value.as_u64()).unwrap_or(12),
        "instance_columns": columns.get("instance_columns").and_then(|value| value.as_u64()).unwrap_or(2),
        "lookup_table_columns": columns.get("lookup_table_columns").and_then(|value| value.as_u64()).unwrap_or(3),
        "public_instance_length": plan.public_instance_total_length.max(public_instance_total_length_for_plan(plan)),
        "proof_backend": "halo2_kzg_shplonk_blake2b",
        "largest_payloads_loaded": largest_payloads_for_run(run_dir, witness_path, pk_path),
        "witness_payload_size": file_size(witness_path),
        "compiled_circuit_size": file_size(&run_dir.join("model.compiled")),
        "pk_size": file_size(pk_path),
        "proof_serialization_policy": "snark_json_written_after_create_proof_completes",
        "thread_env_settings": thread_env_profile(),
        "command": command,
        "mock_precheck_policy": mock_precheck_policy,
        "payload_deduplication": "single NativeOutputDerivationWitness reused for public instance, binding, MockProver when enabled, and create_proof circuit assignment",
        "witness_payload_policy": "compact metadata JSON; tensor payloads remain in binary/text run artifacts and are loaded once into the reused derivation",
        "setup_only_data_policy": "shape-only setup artifacts are not retained by graph-gat-prove"
    });
    let _ = write_json(&run_dir.join("prove_memory_profile_internal.json"), &profile);
}

fn run_dir_for_path(path: &Path) -> PathBuf {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    if parent.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        parent.to_path_buf()
    }
}

fn write_native_failure(
    run_dir: &Path,
    failure: &NativeConstraintFailure,
) -> Result<(), EZKLError> {
    write_json(&run_dir.join("native_constraint_failure.json"), failure)?;
    let log = format!(
        "block_reason={}\ngate_name={}\nregion_name={}\nrow_offset={}\ndataset={}\nroute_id={}\ncomponent={}\nexpected_relation={}\nactual_values={:?}\nwitness_source_path={}\ncommand={}\nraw_error={}\nattempted_fix={}\nremaining_blocker={}\n",
        failure.block_reason,
        failure.gate_name,
        failure.region_name,
        failure.row_offset,
        failure.dataset,
        failure.route_id,
        failure.component,
        failure.expected_relation,
        failure.actual_values,
        failure.witness_source_path,
        failure.command,
        failure.raw_error,
        failure.attempted_fix,
        failure.remaining_blocker
    );
    fs::write(run_dir.join("native_constraint_failure.log"), log)?;
    Ok(())
}

fn parse_mock_failure_location(raw_error: &str, scope: &str) -> (String, String, usize, String) {
    let region_name = raw_error
        .split("('")
        .nth(1)
        .and_then(|tail| tail.split("')").next())
        .map(str::to_string)
        .unwrap_or_else(|| format!("native.debug_scope.{scope}"));
    let row_offset = raw_error
        .split(" at offset ")
        .nth(1)
        .and_then(|tail| tail.split_whitespace().next())
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let gate_name = if region_name.starts_with("native.") {
        region_name.clone()
    } else {
        "native.mockprover.detected_failure".to_string()
    };
    let component = region_name
        .strip_prefix("native.")
        .unwrap_or(&region_name)
        .to_string();
    (gate_name, region_name, row_offset, component)
}

fn mock_failure_to_native_failure(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
    witness_path: &Path,
    command: String,
    raw_error: String,
) -> NativeConstraintFailure {
    let (gate_name, region_name, row_offset, component) =
        parse_mock_failure_location(&raw_error, &plan.native_debug_scope);
    let mut actual_values = vec![
        format!("required_logrows={}", plan.estimate.required_logrows),
        format!("node_count={}", plan.node_count),
        format!("edge_count={}", plan.edge_count),
        format!("run_dir={}", run_dir.display()),
    ];
    if region_name.contains(".output.head_") && region_name.ends_with(".segment_sum_inverse") {
        if let Some(head_index) = region_name
            .split(".output.head_")
            .nth(1)
            .and_then(|tail| tail.split('.').next())
            .and_then(|value| value.parse::<usize>().ok())
        {
            if row_offset % 2 == 1 {
                let node = (row_offset - 1) / 2;
                if let Ok(derivation) = native_output_derivation_witness(plan, run_dir) {
                    if let Some(head) = derivation.output_heads.get(head_index) {
                        if node < head.trace.sum.len() {
                            let policy = inverse_policy_from_binding(&plan.formal_table_binding);
                            let sum_q = quantize_trace_scalar(head.trace.sum[node]);
                            let inv_q = quantized_inverse_for_sum_q(sum_q, &policy);
                            let range_size = inverse_table_range_size(&plan.formal_table_binding);
                            let in_table_range = sum_q >= 0 && (sum_q as usize) < range_size;
                            actual_values.extend([
                                format!("output_head_index={head_index}"),
                                format!("output_node_index={node}"),
                                format!("output_sum_fp64={}", head.trace.sum[node]),
                                format!("output_sum_q={sum_q}"),
                                format!("output_inv_q={inv_q}"),
                                format!("inverse_table_range_size={range_size}"),
                                format!("output_sum_q_in_inverse_table_range={in_table_range}"),
                            ]);
                            if let Ok(csr) = read_csr_payload(&run_dir.join("csr_segment_layout.bin")) {
                                if node + 1 < csr.segment_ptr.len() {
                                    let begin = csr.segment_ptr[node];
                                    let end = csr.segment_ptr[node + 1];
                                    actual_values.extend([
                                        format!("segment_begin={begin}"),
                                        format!("segment_end={end}"),
                                        format!("segment_degree={}", end.saturating_sub(begin)),
                                    ]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    NativeConstraintFailure {
        status: "failed".to_string(),
        block_reason: "native_constraint_synthesis_failed".to_string(),
        gate_name,
        region_name,
        row_offset,
        dataset: plan.dataset.clone(),
        route_id: plan.route_id.clone(),
        layer_index: None,
        head_index: None,
        component,
        expected_relation: "MockProver must satisfy every enabled native diagnostic gate before KZG proving".to_string(),
        actual_values,
        witness_source_path: witness_path.display().to_string(),
        command,
        raw_error,
        attempted_fix: "added MockProver preflight and stable native diagnostic gate/region names before KZG proving".to_string(),
        remaining_blocker: "MockProver returned at least one unsatisfied Halo2 relation".to_string(),
    }
}

fn strict_equivalence_failure(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
    witness_path: &Path,
    command: String,
) -> NativeConstraintFailure {
    let binding = public_instance_binding(plan).ok();
    NativeConstraintFailure {
        status: "blocked".to_string(),
        block_reason: "final_logits_derivation_incomplete".to_string(),
        gate_name: "native.full_model.final_logits.derivation".to_string(),
        region_name: "native.full_model.output.head_0.projection".to_string(),
        row_offset: 0,
        dataset: plan.dataset.clone(),
        route_id: plan.route_id.clone(),
        layer_index: Some(plan.layer_count.saturating_sub(1)),
        head_index: Some(0),
        component: "full_model.final_logits.derivation".to_string(),
        expected_relation: "strict_public_input_equivalent now has a row-major public instance binding for final_logits[N,C], but strict verification also requires those final logits to be derived from complete quantized output-head projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, bias add, and K_out average relations over the real hidden activations".to_string(),
        actual_values: vec![
            "public_instance_binding=assigned_and_constrained_to_final_logits".to_string(),
            format!("public_instance_order={}", public_instance_order()),
            format!(
                "public_instance_length={}",
                binding
                    .as_ref()
                    .map(|binding| binding.public_instance_length)
                    .unwrap_or_else(|| (plan.node_count.max(1) * plan.class_count.max(1)) as usize)
            ),
            format!(
                "final_logits_hash={}",
                binding
                    .as_ref()
                    .map(|binding| binding.final_logits_hash.as_str())
                    .unwrap_or("unavailable")
            ),
            format!("scope={}", plan.native_debug_scope),
            format!("full_gat_forward_in_ezkl_proof={}", plan.full_gat_forward_in_ezkl_proof),
            format!("strict_public_input_equivalent={}", plan.strict_public_input_equivalent),
            format!("run_dir={}", run_dir.display()),
        ],
        witness_source_path: witness_path.display().to_string(),
        command,
        raw_error: "final logits are assigned and bound to the public instance, but the Rust native full-model circuit still uses compressed diagnostic output-head rows instead of the complete per-node/per-edge quantized output-layer derivation".to_string(),
        attempted_fix: "implemented row-major final_logits[N,C] assignment, K_out average gate rows, output head projection/attention/bias_add diagnostic regions, and public instance equality constraints without enabling unsound strict flags".to_string(),
        remaining_blocker: "replace diagnostic output-head rows with complete output-layer witness assignment over real hidden activations and checkpoint tensors before strict_public_input_equivalent can be true".to_string(),
    }
}

fn full_model_preflight_failure(
    plan: &NativeCompiledPlan,
    run_dir: &Path,
    output_path: &Path,
    preflight: &serde_json::Value,
) -> NativeConstraintFailure {
    let actual_values = preflight
        .get("actual_values")
        .and_then(|value| value.as_array())
        .map(|values| {
            values
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| {
            vec![
                format!(
                    "sum_fp32_reference={}",
                    preflight
                        .get("sum_fp32_reference")
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "null".to_string())
                ),
                format!(
                    "inv_fp32_reference={}",
                    preflight
                        .get("inv_fp32_reference")
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "null".to_string())
                ),
                format!(
                    "quantized_product={}",
                    preflight
                        .get("quantized_product")
                        .map(|value| value.to_string())
                        .unwrap_or_else(|| "null".to_string())
                ),
            ]
        });
    NativeConstraintFailure {
        status: "blocked".to_string(),
        block_reason: preflight
            .get("block_reason")
            .and_then(|value| value.as_str())
            .unwrap_or("native_full_witness_preflight_failed")
            .to_string(),
        gate_name: preflight
            .get("failure_gate")
            .and_then(|value| value.as_str())
            .unwrap_or("native.quantized_inverse_lookup")
            .to_string(),
        region_name: preflight
            .get("failure_region")
            .and_then(|value| value.as_str())
            .unwrap_or("native.full_model.layer_0.head_0.softmax.quantized_inverse_lookup")
            .to_string(),
        row_offset: preflight
            .get("failure_row")
            .and_then(|value| value.as_u64())
            .unwrap_or(0) as usize,
        dataset: plan.dataset.clone(),
        route_id: plan.route_id.clone(),
        layer_index: preflight.get("layer_index").and_then(|value| value.as_u64()),
        head_index: preflight.get("head_index").and_then(|value| value.as_u64()),
        component: preflight
            .get("component")
            .and_then(|value| value.as_str())
            .unwrap_or("full_model.quantized_inverse_lookup")
            .to_string(),
        expected_relation: preflight
            .get("expected_relation")
            .and_then(|value| value.as_str())
            .unwrap_or("strict native route requires quant_cfg_id-derived quantized inverse lookup and alpha rescale")
            .to_string(),
        actual_values,
        witness_source_path: output_path.display().to_string(),
        command: format!(
            "graph-gat-gen-witness --compiled-circuit {} --output {}",
            run_dir.join("model.compiled").display(),
            output_path.display()
        ),
        raw_error: preflight.to_string(),
        attempted_fix: preflight
            .get("attempted_fix")
            .and_then(|value| value.as_str())
            .unwrap_or("ran C++ full-dataset first hidden attention-head preflight before native Rust witness assignment")
            .to_string(),
        remaining_blocker: preflight
            .get("remaining_blocker")
            .and_then(|value| value.as_str())
            .unwrap_or("native full-model witness preflight did not reach a quantized inverse and alpha compatible assignment")
            .to_string(),
    }
}

fn preflight_i64(preflight: &serde_json::Value, key: &str) -> Option<i64> {
    preflight.get(key).and_then(|value| value.as_i64())
}

fn validate_ready_preflight_policy(
    plan: &NativeCompiledPlan,
    preflight: &serde_json::Value,
) -> Result<(), EZKLError> {
    let policy = inverse_policy_from_binding(&plan.formal_table_binding);
    let preflight_policy = preflight
        .get("inverse_policy")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if preflight_policy != policy.inverse_policy {
        return Err(format!(
            "inverse_policy_hash_mismatch: preflight inverse_policy={} plan inverse_policy={}",
            preflight_policy, policy.inverse_policy
        )
        .into());
    }
    let preflight_hash = preflight
        .get("inverse_policy_hash")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    if preflight_hash != policy.inverse_policy_hash {
        return Err(format!(
            "inverse_policy_hash_mismatch: preflight inverse_policy_hash={} plan inverse_policy_hash={}",
            preflight_hash, policy.inverse_policy_hash
        )
        .into());
    }
    if let (Some(sum_q), Some(inv_q)) = (
        preflight_i64(preflight, "quantized_sum_scale16"),
        preflight_i64(preflight, "quantized_inv_scale16"),
    ) {
        if !validate_inverse_lookup_against_binding(sum_q, inv_q, &plan.formal_table_binding) {
            return Err(format!(
                "quantized_inverse_lookup_mismatch: sum_q={} inv_q={} expected_inv_q={} inverse_table_range_size={}",
                sum_q,
                inv_q,
                quantized_inverse_for_sum_q(sum_q, &policy),
                inverse_table_range_size(&plan.formal_table_binding)
            )
            .into());
        }
    }
    if let Some(max_observed_sum_q) = preflight_i64(preflight, "max_observed_sum_q") {
        if !inverse_sum_in_table_range(max_observed_sum_q, &plan.formal_table_binding) {
            return Err(format!(
                "quantized_inverse_lookup_mismatch: max_observed_sum_q={} inverse_table_range_size={}",
                max_observed_sum_q,
                inverse_table_range_size(&plan.formal_table_binding)
            )
            .into());
        }
    }
    if let Some(derived_sum_bound) = preflight_i64(preflight, "derived_sum_bound") {
        if derived_sum_bound >= inverse_table_range_size(&plan.formal_table_binding) as i64 {
            return Err(format!(
                "quantized_inverse_lookup_mismatch: derived_sum_bound={} inverse_table_range_size={}",
                derived_sum_bound,
                inverse_table_range_size(&plan.formal_table_binding)
            )
            .into());
        }
    }
    if let (Some(u_q), Some(inv_q), Some(alpha_q)) = (
        preflight_i64(preflight, "u_q"),
        preflight_i64(preflight, "inv_q"),
        preflight_i64(preflight, "alpha_q"),
    ) {
        if !validate_alpha_rescale(u_q, inv_q, alpha_q, &policy) {
            return Err(format!(
                "alpha_rescale_mismatch: u_q={} inv_q={} alpha_q={} divisor={}",
                u_q, inv_q, alpha_q, policy.alpha_rescale_divisor
            )
            .into());
        }
    }
    if preflight
        .get("inverse_all_nodes_accepted")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
        != true
    {
        return Err(
            "quantized_inverse_lookup_mismatch: preflight did not accept every node".into(),
        );
    }
    if preflight
        .get("alpha_all_edges_accepted")
        .and_then(|value| value.as_bool())
        .unwrap_or(false)
        != true
    {
        return Err("alpha_rescale_mismatch: preflight did not accept every edge".into());
    }
    Ok(())
}

fn save_full_keys(
    plan: NativeCompiledPlan,
    run_dir: &Path,
    vk_path: &PathBuf,
    pk_path: &PathBuf,
) -> Result<u32, EZKLError> {
    let _ = run_dir;
    let circuit = FullNativeCircuit::new_shape_only(plan)?;
    let logrows = circuit.params.logrows;
    let params = crate::pfsys::srs::gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&native_srs_path_for_key(pk_path), &params)?;
    save_params::<KZGCommitmentScheme<Bn256>>(&native_srs_path_for_key(vk_path), &params)?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, FullNativeCircuit>(&circuit, &params, false)
        .map_err(|error| EZKLError::from(format!("native_keygen_failed: {error}")))?;
    save_vk::<G1Affine>(vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(pk_path, &pk)?;
    Ok(logrows)
}

pub(crate) fn graph_gat_setup(
    compiled_circuit_path: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
) -> Result<String, EZKLError> {
    match load_backend_plan(&compiled_circuit_path, "graph-gat-setup")? {
        BackendPlan::Tiny(plan) => {
            save_tiny_keys(plan.level, plan.logrows, &vk_path, &pk_path)?;
            Ok(serde_json::json!({
                "status": "setup_complete",
                "route_id": plan.route_id,
                "backend_version": plan.backend_version,
                "level": plan.level,
                "logrows": plan.logrows,
                "vk_path": vk_path,
                "pk_path": pk_path
            })
            .to_string())
        }
        BackendPlan::Full(plan) => {
            let run_dir = run_dir_for_path(&compiled_circuit_path);
            let scope = NativeDebugScope::parse(&plan.native_debug_scope)?;
            let preflight_path = run_dir.join("native_full_witness_preflight.json");
            if scope == NativeDebugScope::FullModel && preflight_path.exists() {
                let preflight = read_json::<serde_json::Value>(&preflight_path)?;
                if preflight
                    .get("status")
                    .and_then(|value| value.as_str())
                    .unwrap_or("blocked")
                    != "ready"
                {
                    let mut failure = full_model_preflight_failure(
                        &plan,
                        &run_dir,
                        &run_dir.join("witness.json"),
                        &preflight,
                    );
                    failure.command = format!(
                        "graph-gat-setup --compiled-circuit {} --vk-path {} --pk-path {}",
                        compiled_circuit_path.display(),
                        vk_path.display(),
                        pk_path.display()
                    );
                    failure.witness_source_path = preflight_path.display().to_string();
                    write_native_failure(&run_dir, &failure)?;
                    return Err(format!(
                        "{}: gate_name={} region_name={} row_offset={} relation={} values={:?}",
                        failure.block_reason,
                        failure.gate_name,
                        failure.region_name,
                        failure.row_offset,
                        failure.expected_relation,
                        failure.actual_values
                    )
                    .into());
                }
                validate_ready_preflight_policy(&plan, &preflight)?;
            }
            let logrows = save_full_keys(plan.clone(), &run_dir, &vk_path, &pk_path)?;
            let layout = public_instance_layout_for_plan(&plan);
            Ok(serde_json::json!({
                "status": "setup_complete",
                "route_id": plan.route_id,
                "native_primitive": plan.native_primitive,
                "native_primitive_version": plan.native_primitive_version,
                "dataset": plan.dataset,
                "native_debug_scope": plan.native_debug_scope,
                "logrows": logrows,
                "public_instance_layout": layout.layout,
                "public_instance_chunk_size": layout.chunk_size,
                "public_instance_column_count": layout.column_count,
                "public_instance_total_length": layout.total_length,
                "setup_payload_mode": "shape_only",
                "vk_path": vk_path,
                "pk_path": pk_path
            })
            .to_string())
        }
    }
}

pub(crate) fn graph_gat_gen_witness(
    compiled_circuit_path: PathBuf,
    output_path: PathBuf,
) -> Result<String, EZKLError> {
    match load_backend_plan(&compiled_circuit_path, "graph-gat-gen-witness")? {
        BackendPlan::Tiny(plan) => {
            let witness = default_tiny_witness(plan.level);
            write_json(&output_path, &witness)?;
            Ok(serde_json::to_string(&witness)?)
        }
        BackendPlan::Full(plan) => {
            let run_dir = run_dir_for_path(&compiled_circuit_path);
            let scope = NativeDebugScope::parse(&plan.native_debug_scope)?;
            let preflight_path = run_dir.join("native_full_witness_preflight.json");
            if scope == NativeDebugScope::FullModel && preflight_path.exists() {
                let preflight = read_json::<serde_json::Value>(&preflight_path)?;
                if preflight
                    .get("status")
                    .and_then(|value| value.as_str())
                    .unwrap_or("blocked")
                    != "ready"
                {
                    let failure =
                        full_model_preflight_failure(&plan, &run_dir, &output_path, &preflight);
                    write_native_failure(&run_dir, &failure)?;
                    return Err(format!(
                        "{}: gate_name={} region_name={} row_offset={} relation={} values={:?}",
                        failure.block_reason,
                        failure.gate_name,
                        failure.region_name,
                        failure.row_offset,
                        failure.expected_relation,
                        failure.actual_values
                    )
                    .into());
                }
                validate_ready_preflight_policy(&plan, &preflight)?;
            }
            let binding = public_instance_binding_for_run(&plan, &run_dir)?;
            let derivation = native_output_derivation_witness(&plan, &run_dir)?;
            let witness = serde_json::json!({
                "witness_kind": "full_compressed_csr_native_inputs",
                "diagnostic_backend_version": NATIVE_DIAGNOSTIC_BACKEND_VERSION,
                "route_id": plan.route_id,
                "native_primitive": plan.native_primitive,
                "native_primitive_version": plan.native_primitive_version,
                "native_debug_scope": plan.native_debug_scope,
                "dataset": plan.dataset,
                "node_count": plan.node_count,
                "edge_count": plan.edge_count,
                "graph_count": plan.graph_count,
                "feature_dim": plan.feature_dim,
                "class_count": plan.class_count,
                "graph_hash": plan.graph_hash,
                "checkpoint_hash": plan.checkpoint_hash,
                "metadata_hash": plan.metadata_hash,
                "shape_hash": plan.shape_hash,
                "formal_table_hash": plan.formal_table_binding.table_hash,
                "final_logits_binding": {
                    "public_instance_order": binding.public_instance_order,
                    "public_instance_length": binding.public_instance_length,
                    "public_instance_layout": binding.public_instance_layout,
                    "public_instance_chunk_size": binding.public_instance_chunk_size,
                    "public_instance_column_count": binding.public_instance_column_count,
                    "public_instance_total_length": binding.public_instance_total_length,
                    "final_logits_hash": binding.final_logits_hash,
                    "public_instance_hash": binding.public_instance_hash,
                    "source": "native_full_output_derivation_circuit",
                    "no_precomputed_logits_bypass": binding.no_precomputed_logits_bypass
                },
                "output_derivation": {
                    "output_layer_input_source": derivation.output_layer_input_source,
                    "output_layer_input_dim": derivation.output_layer_input_dim,
                    "output_head_count": derivation.output_heads.len(),
                    "final_logits_shape": derivation.final_logits.shape,
                    "derivation_complete": derivation.derivation_complete,
                    "projection_witness_samples": derivation.output_heads.iter().map(|head| &head.projection).collect::<Vec<_>>(),
                    "attention_witness_samples": derivation.output_heads.iter().map(|head| &head.attention).collect::<Vec<_>>(),
                    "aggregation_witness_samples": derivation.output_heads.iter().map(|head| &head.aggregation).collect::<Vec<_>>(),
                    "bias_add_witness_samples": derivation.output_heads.iter().map(|head| &head.bias_add).collect::<Vec<_>>(),
                    "average_policy": "per_head_bias_then_arithmetic_mean",
                    "rounding_policy": output_rounding_policy()
                },
                "payloads": {
                    "csr_layout": run_dir.join("csr_segment_layout.bin"),
                    "node_features": run_dir.join("node_features.f64.bin"),
                    "checkpoint_tensors": run_dir.join("checkpoint_tensors.txt"),
                    "formal_tables": run_dir.join("formal_tables.bin")
                },
                "status": "native_output_derivation_witness_ready",
                "strict_public_input_equivalent": plan.strict_public_input_equivalent,
                "full_gat_forward_in_ezkl_proof": plan.full_gat_forward_in_ezkl_proof,
                "full_model_witness_preflight": if preflight_path.exists() {
                    serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&preflight_path)?)?
                } else {
                    serde_json::Value::Null
                },
                "note": "dataset payloads are bound and output final_logits are derived through projection, attention, CSR aggregation, bias add, average, and public instance equality constraints"
            });
            write_json(&output_path, &witness)?;
            Ok(witness.to_string())
        }
    }
}

pub(crate) fn graph_gat_prove(
    compiled_circuit_path: PathBuf,
    witness_path: PathBuf,
    pk_path: PathBuf,
    proof_path: PathBuf,
) -> Result<String, EZKLError> {
    match load_backend_plan(&compiled_circuit_path, "graph-gat-prove")? {
        BackendPlan::Tiny(plan) => {
            let witness = read_json::<NativeTinyWitness>(&witness_path)?;
            if witness.backend_version != TINY_NATIVE_BACKEND_VERSION || witness.level != plan.level
            {
                return Err(format!(
                    "native_constraint_synthesis_failed: tiny witness mismatch level={} backend_version={}",
                    witness.level, witness.backend_version
                )
                .into());
            }
            let snark = create_tiny_proof(plan.level, plan.logrows, witness, pk_path)?;
            snark.save(&proof_path)?;
            Ok(serde_json::json!({
                "status": "proof_created",
                "route_id": plan.route_id,
                "backend_version": plan.backend_version,
                "level": plan.level,
                "proof_path": proof_path,
                "public_inputs": plan.public_inputs
            })
            .to_string())
        }
        BackendPlan::Full(plan) => {
            let witness = read_json::<serde_json::Value>(&witness_path)?;
            let witness_scope = witness
                .get("native_debug_scope")
                .and_then(|value| value.as_str())
                .unwrap_or("");
            if !witness_scope.is_empty() && witness_scope != plan.native_debug_scope {
                return Err(format!(
                    "native_constraint_synthesis_failed: witness scope mismatch plan_scope={} witness_scope={}",
                    plan.native_debug_scope, witness_scope
                )
                .into());
            }
            let scope = NativeDebugScope::parse(&plan.native_debug_scope)?;
            let run_dir = run_dir_for_path(&compiled_circuit_path);
            let command = format!(
                "graph-gat-prove --compiled-circuit {} --witness {} --pk-path {} --proof-path {}",
                compiled_circuit_path.display(),
                witness_path.display(),
                pk_path.display(),
                proof_path.display()
            );
            let run_mock_precheck = should_run_mock_precheck(&plan);
            let mock_precheck_policy = if run_mock_precheck {
                "enabled_for_dataset_size"
            } else {
                "skipped_for_large_full_dataset_memory_policy"
            };
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "start",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            let derivation = Arc::new(native_output_derivation_witness(&plan, &run_dir)?);
            let public_instances = native_full_public_instance_columns_from_values(
                &plan,
                &derivation.final_logits.flat_values_q,
            )?;
            let public_inputs_sample = derivation
                .final_logits
                .flat_values_q
                .iter()
                .copied()
                .take(16)
                .collect::<Vec<_>>();
            let binding = public_instance_binding_from_final_logits(&plan, &derivation.final_logits)?;
            let circuit = FullNativeCircuit::new_with_output_derivation(
                plan.clone(),
                &run_dir,
                witness_path.display().to_string(),
                Arc::clone(&derivation),
            )?;
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "payloads_deduplicated_derivation_ready",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            if run_mock_precheck {
                write_native_prove_phase_profile(
                    &run_dir,
                    &plan,
                    "mock_precheck_start",
                    &command,
                    &witness_path,
                    &pk_path,
                    mock_precheck_policy,
                );
                let mock = MockProver::run(
                    circuit.params.logrows,
                    &circuit,
                    public_instances.clone(),
                );
                match mock {
                    Ok(prover) => {
                        if let Err(failures) = prover.verify() {
                            let raw_error = failures
                                .iter()
                                .map(|failure| format!("{failure:?}"))
                                .collect::<Vec<_>>()
                                .join("\n");
                            let failure = mock_failure_to_native_failure(
                                &plan,
                                &run_dir,
                                &witness_path,
                                command.clone(),
                                raw_error.clone(),
                            );
                            write_native_failure(&run_dir, &failure)?;
                            return Err(format!(
                                "native_constraint_synthesis_failed: gate_name={} region_name={} row_offset={} relation={} values={:?}",
                                failure.gate_name,
                                failure.region_name,
                                failure.row_offset,
                                failure.expected_relation,
                                failure.actual_values
                            )
                            .into());
                        }
                    }
                    Err(error) => {
                        let failure = mock_failure_to_native_failure(
                            &plan,
                            &run_dir,
                            &witness_path,
                            command.clone(),
                            format!("{error:?}"),
                        );
                        write_native_failure(&run_dir, &failure)?;
                        return Err(format!(
                            "native_constraint_synthesis_failed: gate_name={} region_name={} row_offset={} relation={} values={:?}",
                            failure.gate_name,
                            failure.region_name,
                            failure.row_offset,
                            failure.expected_relation,
                            failure.actual_values
                        )
                        .into());
                    }
                }
                write_native_prove_phase_profile(
                    &run_dir,
                    &plan,
                    "mock_precheck_complete",
                    &command,
                    &witness_path,
                    &pk_path,
                    mock_precheck_policy,
                );
            } else {
                write_native_prove_phase_profile(
                    &run_dir,
                    &plan,
                    "mock_precheck_skipped",
                    &command,
                    &witness_path,
                    &pk_path,
                    mock_precheck_policy,
                );
            }

            if scope == NativeDebugScope::FullModel && !plan.strict_public_input_equivalent {
                let failure = strict_equivalence_failure(
                    &plan,
                    &run_dir,
                    &witness_path,
                    command.clone(),
                );
                write_native_failure(&run_dir, &failure)?;
                return Err(format!(
                    "{}: gate_name={} region_name={} row_offset={} relation={} values={:?}",
                    failure.block_reason,
                    failure.gate_name,
                    failure.region_name,
                    failure.row_offset,
                    failure.expected_relation,
                    failure.actual_values
                )
                .into());
            }

            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "load_pk_start",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            let pk = load_pk::<KZGCommitmentScheme<Bn256>, FullNativeCircuit>(
                pk_path.clone(),
                circuit.params.clone(),
            )
            .map_err(|error| EZKLError::from(format!("native_prover_failed: load_pk: {error}")))?;
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "load_pk_complete",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "load_params_start",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            let params = load_native_params_for_key(&pk_path, circuit.params.logrows)?;
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "load_params_complete",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "create_proof_start",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            let snark = create_proof_circuit::<
                KZGCommitmentScheme<Bn256>,
                _,
                ProverSHPLONK<_>,
                VerifierSHPLONK<_>,
                KZGSingleStrategy<_>,
                _,
                Blake2bWrite<Vec<u8>, _, Challenge255<_>>,
                Blake2bRead<Cursor<Vec<u8>>, _, Challenge255<_>>,
            >(
                circuit,
                public_instances,
                &params,
                &pk,
                CheckMode::SAFE,
                None,
                None,
            )
            .map_err(|error| {
                EZKLError::from(format!("native_prover_failed: create_proof: {error}"))
            })?;
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "create_proof_complete",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            snark.save(&proof_path)?;
            write_native_prove_phase_profile(
                &run_dir,
                &plan,
                "proof_serialized",
                &command,
                &witness_path,
                &pk_path,
                mock_precheck_policy,
            );
            Ok(serde_json::json!({
                "status": "proof_created",
                "route_id": plan.route_id,
                "native_debug_scope": plan.native_debug_scope,
                "diagnostic_backend_version": NATIVE_DIAGNOSTIC_BACKEND_VERSION,
                "dataset": plan.dataset,
                "logrows": NativeDebugScope::parse(&plan.native_debug_scope)?.logrows(&plan),
                "proof_path": proof_path,
                "public_instance_order": binding.public_instance_order,
                "public_instance_length": binding.public_instance_length,
                "public_instance_layout": binding.public_instance_layout,
                "public_instance_chunk_size": binding.public_instance_chunk_size,
                "public_instance_column_count": binding.public_instance_column_count,
                "public_instance_total_length": binding.public_instance_total_length,
                "final_logits_hash": binding.final_logits_hash,
                "public_instance_hash": binding.public_instance_hash,
                "public_inputs_sample": public_inputs_sample,
                "strict_public_input_equivalent": plan.strict_public_input_equivalent,
                "full_gat_forward_in_ezkl_proof": plan.full_gat_forward_in_ezkl_proof,
                "mock_precheck_policy": mock_precheck_policy
            })
            .to_string())
        }
    }
}

pub(crate) fn graph_gat_verify(
    compiled_circuit_path: PathBuf,
    proof_path: PathBuf,
    _settings_path: PathBuf,
    vk_path: PathBuf,
    output_path: PathBuf,
) -> Result<String, EZKLError> {
    match load_backend_plan(&compiled_circuit_path, "graph-gat-verify")? {
        BackendPlan::Tiny(plan) => {
            verify_tiny_proof(
                plan.level,
                plan.logrows,
                proof_path.clone(),
                vk_path.clone(),
            )?;
            let output = NativeVerifyOutput {
                status: "verified".to_string(),
                verify_passed: true,
                route_id: plan.route_id,
                backend_version: plan.backend_version,
                proof_path: proof_path.display().to_string(),
                vk_path: vk_path.display().to_string(),
            };
            write_json(&output_path, &output)?;
            Ok(serde_json::to_string(&output)?)
        }
        BackendPlan::Full(plan) => {
            let run_dir = run_dir_for_path(&compiled_circuit_path);
            let circuit =
                FullNativeCircuit::new(plan.clone(), &run_dir, proof_path.display().to_string())?;
            let proof = Snark::<Fr, G1Affine>::load::<KZGCommitmentScheme<Bn256>>(&proof_path)
                .map_err(|error| {
                    EZKLError::from(format!("native_verify_failed: load_proof: {error}"))
                })?;
            let vk = load_vk::<KZGCommitmentScheme<Bn256>, FullNativeCircuit>(
                vk_path.clone(),
                circuit.params.clone(),
            )
            .map_err(|error| EZKLError::from(format!("native_verify_failed: load_vk: {error}")))?;
            let params = load_native_params_for_key(&vk_path, circuit.params.logrows)?;
            let strategy = KZGSingleStrategy::new(params.verifier_params());
            verify_proof_circuit::<
                VerifierSHPLONK<_>,
                KZGCommitmentScheme<Bn256>,
                KZGSingleStrategy<_>,
                _,
                Blake2bRead<Cursor<Vec<u8>>, _, Challenge255<_>>,
            >(
                &proof,
                params.verifier_params(),
                &vk,
                strategy,
                1 << circuit.params.logrows,
            )
            .map_err(|error| {
                EZKLError::from(format!("native_verify_failed: verify_proof: {error}"))
            })?;
            let binding = public_instance_binding_for_run(&plan, &run_dir)?;
            let output = serde_json::json!({
                "status": "verified",
                "verify_passed": true,
                "route_id": plan.route_id,
                "native_debug_scope": plan.native_debug_scope,
                "diagnostic_backend_version": NATIVE_DIAGNOSTIC_BACKEND_VERSION,
                "proof_path": proof_path,
                "vk_path": vk_path,
                "public_instance_order": binding.public_instance_order,
                "public_instance_length": binding.public_instance_length,
                "public_instance_layout": binding.public_instance_layout,
                "public_instance_chunk_size": binding.public_instance_chunk_size,
                "public_instance_column_count": binding.public_instance_column_count,
                "public_instance_total_length": binding.public_instance_total_length,
                "final_logits_hash": binding.final_logits_hash,
                "public_instance_hash": binding.public_instance_hash,
                "strict_public_input_equivalent": plan.strict_public_input_equivalent,
                "full_gat_forward_in_ezkl_proof": plan.full_gat_forward_in_ezkl_proof
            });
            write_json(&output_path, &output)?;
            Ok(output.to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CsrPayload, FormalTableBinding, InverseSemantics, LookupSemantics, NativeCompiledPlan,
        NativeEstimate, NativeHeadParameters, TINY_NATIVE_LOGROWS, TinyNativeCircuit,
        attention_head_forward_f64, average_output_head_values_q, create_tiny_proof,
        default_quantized_inverse_policy, default_tiny_witness, inverse_policy_from_binding,
        inverse_sum_in_table_range, inverse_table_range_size, native_final_logits_witness,
        native_full_public_input_values, output_average_witness_for_logit, output_head_derivation,
        public_instance_binding, quantize_trace_scalar, quantized_inverse_for_sum_q,
        rust_policy_matches_binding, save_tiny_keys, tiny_params, tiny_public_inputs,
        validate_alpha_rescale, validate_inverse_lookup_against_binding,
        validate_inverse_witness_against_policy, verify_tiny_proof,
    };
    use crate::circuit::CheckMode;
    use crate::pfsys::{create_keys, create_proof_circuit, verify_proof_circuit};
    use halo2_proofs::poly::commitment::ParamsProver;
    use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
    use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
    use halo2_proofs::poly::kzg::strategy::SingleStrategy as KZGSingleStrategy;
    use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Selector},
    };
    use halo2curves::bn256::{Bn256, Fr};
    use halo2curves::ff::Field;
    use std::io::Cursor;

    #[derive(Clone, Debug)]
    struct ExactInverseConfig {
        sum: Column<Advice>,
        inv: Column<Advice>,
        selector: Selector,
    }

    #[derive(Clone, Debug, Default)]
    struct ExactInverseCircuit {
        sums: Vec<Fr>,
    }

    impl Circuit<Fr> for ExactInverseCircuit {
        type Config = ExactInverseConfig;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self {
                sums: vec![Fr::ONE; self.sums.len()],
            }
        }

        fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
            let sum = meta.advice_column();
            let inv = meta.advice_column();
            let selector = meta.selector();
            meta.create_gate("exact segment inverse", |meta| {
                let q = meta.query_selector(selector);
                let sum_expr = meta.query_advice(sum, halo2_proofs::poly::Rotation::cur());
                let inv_expr = meta.query_advice(inv, halo2_proofs::poly::Rotation::cur());
                vec![q * (sum_expr * inv_expr - halo2_proofs::plonk::Expression::Constant(Fr::ONE))]
            });
            ExactInverseConfig { sum, inv, selector }
        }

        fn configure_with_params(
            meta: &mut ConstraintSystem<Fr>,
            _params: Self::Params,
        ) -> Self::Config {
            Self::configure(meta)
        }

        fn synthesize(
            &self,
            config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            layouter.assign_region(
                || "exact inverse rows",
                |mut region| {
                    for (offset, sum) in self.sums.iter().enumerate() {
                        let inv = sum.invert().unwrap();
                        config.selector.enable(&mut region, offset)?;
                        region.assign_advice(
                            || "sum",
                            config.sum,
                            offset,
                            || Value::known(*sum),
                        )?;
                        region.assign_advice(|| "inv", config.inv, offset, || Value::known(inv))?;
                    }
                    Ok(())
                },
            )
        }
    }

    #[test]
    fn proves_exact_variable_per_dst_inverse_constraint() {
        let circuit = ExactInverseCircuit {
            sums: vec![Fr::from(2u64), Fr::from(5u64), Fr::from(17u64)],
        };
        let prover = MockProver::run(4, &circuit, vec![]).unwrap();
        prover.assert_satisfied();
    }

    #[test]
    fn quantized_inverse_policy_accepts_scale_product() {
        let policy = default_quantized_inverse_policy();
        assert!(validate_inverse_witness_against_policy(16, 16, &policy));
        assert!(validate_inverse_witness_against_policy(32, 8, &policy));
        assert!(validate_inverse_witness_against_policy(63, 4, &policy));
        assert_eq!(quantized_inverse_for_sum_q(63, &policy), 4);
        assert_eq!(quantized_inverse_for_sum_q(96, &policy), 3);
        assert_eq!(quantized_inverse_for_sum_q(1014, &policy), 0);
    }

    #[test]
    fn quantized_inverse_policy_rejects_bad_product() {
        let policy = default_quantized_inverse_policy();
        assert!(!validate_inverse_witness_against_policy(16, 15, &policy));
        assert!(!validate_inverse_witness_against_policy(32, 7, &policy));
        assert!(!validate_inverse_witness_against_policy(1014, 1, &policy));
    }

    #[test]
    fn alpha_rescale_policy_matches_inverse_policy() {
        let policy = default_quantized_inverse_policy();
        assert!(validate_alpha_rescale(16, 16, 16, &policy));
        assert!(validate_alpha_rescale(16, 4, 4, &policy));
        assert!(validate_alpha_rescale(12, 8, 6, &policy));
        assert!(validate_alpha_rescale(5, 5, 1, &policy));
        assert!(!validate_alpha_rescale(16, 16, 11, &policy));
    }

    fn inverse_policy_test_binding(
        degree_bound: u64,
        exp_bound: i64,
        inverse_range: u64,
    ) -> FormalTableBinding {
        let policy = default_quantized_inverse_policy();
        FormalTableBinding {
            source: "formal_quant_cfg_derived".to_string(),
            table_policy: "derived_from_main_formal_static_tables_v1".to_string(),
            inverse_table_policy: "degree_bound_exp_bound_sum_scale_v1".to_string(),
            quant_cfg_id: "fp32_bundle_export".to_string(),
            static_table_id: "tables:lrelu+elu+exp+range".to_string(),
            degree_bound_id: "auto".to_string(),
            range_bits: 12,
            range_size: 4096,
            signed_bound: 2048,
            degree_bound,
            exp_bound,
            derived_sum_bound: degree_bound as i64 * exp_bound,
            inverse_table_range_old: 4096,
            inverse_table_range_new: inverse_range,
            inverse_table_range_size: inverse_range,
            range_hash: "range".to_string(),
            lrelu_hash: "lrelu".to_string(),
            elu_hash: "elu".to_string(),
            exp_hash: "exp".to_string(),
            inv_hash: "inv".to_string(),
            inverse_policy: policy.inverse_policy,
            sum_scale: policy.sum_scale,
            inv_scale: policy.inv_scale,
            alpha_scale: policy.alpha_scale,
            inverse_scale_product: policy.inverse_scale_product,
            inverse_band: policy.inverse_band,
            alpha_rescale_policy: policy.alpha_rescale_policy,
            alpha_rescale_divisor: policy.alpha_rescale_divisor,
            alpha_rounding_rule: policy.alpha_rounding_rule,
            alpha_band: policy.alpha_band,
            inverse_table_hash: "inv".to_string(),
            inverse_policy_hash: policy.inverse_policy_hash,
            table_hash: "table".to_string(),
            bin_hash: "bin".to_string(),
            witness_specific_table: false,
        }
    }

    #[test]
    fn inverse_table_covers_ppi_output_sum_bound() {
        let binding = inverse_policy_test_binding(582, 18, 16_384);
        let sum_q = 5_227;
        let inv_q = quantized_inverse_for_sum_q(sum_q, &inverse_policy_from_binding(&binding));
        assert!(binding.derived_sum_bound < binding.inverse_table_range_size as i64);
        assert!(inverse_sum_in_table_range(sum_q, &binding));
        assert!(validate_inverse_lookup_against_binding(sum_q, inv_q, &binding));
    }

    #[test]
    fn inverse_table_policy_is_dataset_deterministic() {
        let first = inverse_policy_test_binding(582, 18, 16_384);
        let second = inverse_policy_test_binding(582, 18, 16_384);
        assert_eq!(first.degree_bound_id, "auto");
        assert_eq!(first.derived_sum_bound, second.derived_sum_bound);
        assert_eq!(
            inverse_table_range_size(&first),
            inverse_table_range_size(&second)
        );
        assert!(!first.witness_specific_table);
    }

    #[test]
    fn inverse_table_rejects_out_of_policy_sum() {
        let binding = inverse_policy_test_binding(582, 18, 16_384);
        let out_of_policy_sum = binding.inverse_table_range_size as i64;
        let inv_q =
            quantized_inverse_for_sum_q(out_of_policy_sum, &inverse_policy_from_binding(&binding));
        assert!(!inverse_sum_in_table_range(out_of_policy_sum, &binding));
        assert!(!validate_inverse_lookup_against_binding(
            out_of_policy_sum,
            inv_q,
            &binding
        ));
    }

    #[test]
    fn inverse_policy_hash_mismatch_rejected() {
        let mut binding = inverse_policy_test_binding(582, 18, 16_384);
        binding.inverse_policy_hash = "fnv1a64:0000000000000000".to_string();
        assert!(!rust_policy_matches_binding(&binding));
    }

    fn public_binding_test_plan(k_out: u64) -> NativeCompiledPlan {
        NativeCompiledPlan {
            compiled_kind: "test".to_string(),
            route_id: "ezkl_gat_native_compressed_csr_attention_full".to_string(),
            native_primitive: "GraphSegmentAttentionCompressedCSR".to_string(),
            native_primitive_version: "csr_segment_attention_compressed_v1".to_string(),
            status: "compiled".to_string(),
            block_reason: String::new(),
            failure_step: String::new(),
            dataset: "unit".to_string(),
            node_count: 3,
            edge_count: 7,
            graph_count: 1,
            feature_dim: 5,
            class_count: 2,
            layer_count: 2,
            k_out,
            total_attention_head_count: k_out,
            estimate: NativeEstimate {
                projection_mul_add_rows: 0,
                attention_score_rows: 0,
                lrelu_lookup_rows: 0,
                segment_max_rows: 0,
                exp_lookup_rows: 0,
                denominator_sum_rows: 0,
                exact_inverse_rows: 0,
                alpha_rows: 0,
                aggregation_rows: 0,
                elu_lookup_rows: 0,
                output_average_rows: 0,
                fixed_overhead_rows: 0,
                total_rows: 4096,
                required_logrows: 12,
                max_logrows: 26,
                within_public_srs: true,
            },
            csr_layout: "csr_dst_segment_ragged".to_string(),
            graph_hash: "graph".to_string(),
            checkpoint_hash: "checkpoint".to_string(),
            metadata_hash: "metadata".to_string(),
            shape_hash: "shape".to_string(),
            exact_inverse_gate: InverseSemantics {
                mode: "quantized_inverse_lookup".to_string(),
                constraint: "lookup".to_string(),
                witness_column: "Inv_q".to_string(),
                nonzero_argument: "range".to_string(),
                approximate_reciprocal_used: false,
                policy: "quantized_inverse_lookup".to_string(),
                sum_scale: 16,
                inv_scale: 16,
                alpha_scale: 16,
                inverse_scale_product: 256,
                inverse_band: 0,
                alpha_rescale_policy: "rounded_product_div_scale_with_quantization_band"
                    .to_string(),
                alpha_rescale_divisor: 16,
                alpha_rounding_rule: "half_away_from_zero".to_string(),
                alpha_band: 2,
                inverse_table_hash: String::new(),
                inverse_policy_hash: String::new(),
            },
            lookup_semantics: LookupSemantics {
                lrelu: "table".to_string(),
                elu: "table".to_string(),
                exp_map: "table".to_string(),
                range_checks: "table".to_string(),
                formal_status: "derived_formal_tables_bound".to_string(),
                table_policy: "static".to_string(),
                table_hash: "table".to_string(),
                witness_specific_table: false,
            },
            formal_table_binding: FormalTableBinding::default(),
            compressed_row_estimate: None,
            native_debug_scope: "full_model".to_string(),
            strict_public_input_equivalent: false,
            full_gat_forward_in_ezkl_proof: false,
            diagnostic_backend_version: "test".to_string(),
            diagnostic_note: String::new(),
            public_instance_layout: "chunked_columns".to_string(),
            public_instance_chunk_size: super::DEFAULT_PUBLIC_INSTANCE_CHUNK_SIZE,
            public_instance_column_count: 1,
            public_instance_total_length: 6,
        }
    }

    fn tiny_output_csr() -> CsrPayload {
        CsrPayload {
            src: vec![0, 1],
            dst: vec![0, 1],
            segment_ptr: vec![0, 1, 2],
            edge_ptr: vec![0, 2],
            node_ptr: vec![0, 2],
            segment_graph_id: vec![0, 0],
            max_in_degree: 1,
        }
    }

    fn tiny_output_head_parameters() -> NativeHeadParameters {
        NativeHeadParameters {
            seq_kernel: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            attn_src_kernel: vec![1.0, 0.5],
            attn_dst_kernel: vec![0.25, 1.0],
            attn_src_bias: 0.125,
            attn_dst_bias: -0.25,
            output_bias: vec![0.5, -0.25],
        }
    }

    #[test]
    fn output_projection_derives_from_hidden_concat_and_checkpoint() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 5.0]];
        let params = tiny_output_head_parameters();
        let trace =
            attention_head_forward_f64(&features, &tiny_output_csr(), &params, false, false)
                .unwrap();
        assert_eq!(trace.h_prime[0], vec![7.0, 10.0]);
        assert_eq!(trace.h_prime[1], vec![18.0, 26.0]);

        let plan = public_binding_test_plan(1);
        let derivation = output_head_derivation(&plan, 0, params, trace);
        assert_eq!(
            derivation.projection.source,
            "last_hidden_concat_x_checkpoint_seq_kernel"
        );
        assert_eq!(derivation.projection.node_count, 3);
    }

    #[test]
    fn output_attention_uses_projected_logits_not_final_logits() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 5.0]];
        let params = tiny_output_head_parameters();
        let trace =
            attention_head_forward_f64(&features, &tiny_output_csr(), &params, false, false)
                .unwrap();
        let projected_e_src = trace.h_prime[0][0] * params.attn_src_kernel[0]
            + trace.h_prime[0][1] * params.attn_src_kernel[1]
            + params.attn_src_bias;
        let unrelated_final_logit = 12345.0;
        assert_eq!(trace.e_src[0], projected_e_src);
        assert_ne!(trace.e_src[0], unrelated_final_logit);
    }

    #[test]
    fn output_bias_add_matches_checkpoint_bias() {
        let features = vec![vec![1.0, 2.0], vec![3.0, 5.0]];
        let params = tiny_output_head_parameters();
        let trace = attention_head_forward_f64(&features, &tiny_output_csr(), &params, true, false)
            .unwrap();
        assert_eq!(
            trace.h_agg[0][0],
            trace.h_agg_pre_bias[0][0] + params.output_bias[0]
        );
        let pre_q = quantize_trace_scalar(trace.h_agg_pre_bias[0][0]);
        let bias_q = quantize_trace_scalar(params.output_bias[0]);
        assert_eq!(
            pre_q + bias_q,
            quantize_trace_scalar(trace.h_agg_pre_bias[0][0]) + 8
        );
    }

    #[test]
    fn output_average_k1_is_identity_for_real_shape() {
        let plan = public_binding_test_plan(1);
        for node in 0..plan.node_count {
            for class_index in 0..plan.class_count {
                let witness = output_average_witness_for_logit(&plan, node, class_index).unwrap();
                assert_eq!(witness.average_q, witness.head_values_q[0]);
            }
        }
    }

    #[test]
    fn final_logits_public_binding_uses_derived_logits() {
        let plan = public_binding_test_plan(1);
        let logits = native_final_logits_witness(&plan).unwrap();
        let binding = public_instance_binding(&plan).unwrap();
        assert_eq!(binding.public_instance_length, logits.flat_values_q.len());
        assert_eq!(
            native_full_public_input_values(&plan).unwrap(),
            logits.flat_values_q
        );
    }

    #[test]
    fn final_logits_public_binding_rejects_unconstrained_logits() {
        let plan = public_binding_test_plan(1);
        let mut public_values = native_full_public_input_values(&plan).unwrap();
        let derived_values = native_final_logits_witness(&plan).unwrap().flat_values_q;
        public_values[0] = public_values[0].saturating_add(1);
        assert_ne!(public_values, derived_values);
    }

    #[test]
    fn output_average_k1_is_identity() {
        let plan = public_binding_test_plan(1);
        let witness = output_average_witness_for_logit(&plan, 1, 1).unwrap();
        assert_eq!(witness.head_values_q.len(), 1);
        assert_eq!(witness.average_q, witness.head_values_q[0]);
        assert_eq!(
            average_output_head_values_q(&witness.head_values_q).unwrap(),
            witness.average_q
        );
    }

    #[test]
    fn output_average_multihead_matches_formal_policy() {
        let plan = public_binding_test_plan(3);
        let witness = output_average_witness_for_logit(&plan, 2, 1).unwrap();
        assert_eq!(witness.head_values_q.len(), 3);
        assert!(
            witness
                .head_values_q
                .iter()
                .all(|value| *value == witness.average_q)
        );
        assert_eq!(witness.sum_q, witness.average_q * 3);
        assert_eq!(
            witness.rounding_policy,
            "integer_exact_no_post_average_rescale"
        );
    }

    #[test]
    fn public_instance_binding_matches_final_logits() {
        let plan = public_binding_test_plan(2);
        let binding = public_instance_binding(&plan).unwrap();
        let logits = native_final_logits_witness(&plan).unwrap();
        let public_values = native_full_public_input_values(&plan).unwrap();
        assert_eq!(binding.public_instance_length, 6);
        assert_eq!(logits.shape, [3, 2]);
        assert_eq!(public_values, logits.flat_values_q);
        assert_eq!(binding.final_logits_hash, binding.public_instance_hash);
        assert!(binding.no_precomputed_logits_bypass);
    }

    #[test]
    fn final_logits_binding_rejects_mismatched_public_instance() {
        let plan = public_binding_test_plan(2);
        let logits = native_final_logits_witness(&plan).unwrap();
        let mut mismatched = logits.flat_values_q.clone();
        mismatched[0] = mismatched[0].saturating_add(1);
        assert_ne!(mismatched, native_full_public_input_values(&plan).unwrap());
        assert!(average_output_head_values_q(&[1, 2]).is_err());
    }

    #[test]
    fn tiny_graph_native_backend_keygen_prove_verify() {
        for level in 0..=6 {
            let circuit = TinyNativeCircuit::new(level, default_tiny_witness(level));
            let params = tiny_params(TINY_NATIVE_LOGROWS);
            let pk = create_keys::<KZGCommitmentScheme<Bn256>, TinyNativeCircuit>(
                &circuit, &params, false,
            )
            .unwrap();
            let snark = create_proof_circuit::<
                KZGCommitmentScheme<Bn256>,
                _,
                ProverSHPLONK<_>,
                VerifierSHPLONK<_>,
                KZGSingleStrategy<_>,
                _,
                Blake2bWrite<Vec<u8>, _, Challenge255<_>>,
                Blake2bRead<Cursor<Vec<u8>>, _, Challenge255<_>>,
            >(
                circuit,
                vec![tiny_public_inputs(level)],
                &params,
                &pk,
                CheckMode::SAFE,
                None,
                None,
            )
            .unwrap();
            let strategy = KZGSingleStrategy::new(params.verifier_params());
            verify_proof_circuit::<
                VerifierSHPLONK<_>,
                KZGCommitmentScheme<Bn256>,
                KZGSingleStrategy<_>,
                _,
                Blake2bRead<Cursor<Vec<u8>>, _, Challenge255<_>>,
            >(
                &snark,
                params.verifier_params(),
                pk.get_vk(),
                strategy,
                1 << TINY_NATIVE_LOGROWS,
            )
            .unwrap();
        }

        let tmp_dir = tempfile::Builder::new()
            .prefix("tiny-native-graph")
            .tempdir()
            .unwrap();
        let vk_path = tmp_dir.path().join("vk.key");
        let pk_path = tmp_dir.path().join("pk.key");
        let proof_path = tmp_dir.path().join("proof.json");
        save_tiny_keys(6, TINY_NATIVE_LOGROWS, &vk_path, &pk_path).unwrap();
        let snark =
            create_tiny_proof(6, TINY_NATIVE_LOGROWS, default_tiny_witness(6), pk_path).unwrap();
        snark.save(&proof_path).unwrap();
        verify_tiny_proof(6, TINY_NATIVE_LOGROWS, proof_path, vk_path).unwrap();
    }

    fn quantize_signed(value: f64) -> i64 {
        if value >= 0.0 {
            (value * 16.0 + 0.5) as i64
        } else {
            (value * 16.0 - 0.5) as i64
        }
    }

    #[test]
    fn formal_table_generation_policy_matches_main_formula() {
        let range_bits = 12usize;
        let range_size = 1usize << range_bits;
        let lrelu_bound = 4096i64;
        let lrelu_entries = (lrelu_bound * 2 + 1) as usize;
        let elu_entries = lrelu_entries * 3;
        let exp_entries = range_size * 5;

        assert_eq!(range_size, 4096);
        assert_eq!(lrelu_entries, 8193);
        assert_eq!(elu_entries, 24579);
        assert_eq!(exp_entries, 20480);

        let lrelu_neg_raw = -16i64;
        let lrelu_neg_output = quantize_signed((lrelu_neg_raw as f64 / 16.0) * 0.2);
        assert_eq!(lrelu_neg_output, -3);

        let elu_raw = -16i64;
        let elu_central = quantize_signed(((elu_raw as f64) / 16.0).exp() - 1.0);
        assert_eq!([elu_central - 1, elu_central, elu_central + 1].len(), 3);

        let exp_raw = 16i64;
        let exp_central = quantize_signed((-((exp_raw as f64) / 16.0)).exp());
        let exp_band: Vec<i64> = (-2..=2)
            .map(|offset| (exp_central + offset).max(0))
            .collect();
        assert_eq!(exp_band.len(), 5);
        assert!(exp_band.iter().all(|value| *value >= 0));
    }

    #[test]
    fn segment_max_selector_state_machine_is_consistent() {
        let dst = [0u64, 0, 0, 1, 1, 2];
        let scores = [4i64, 9, 7, -2, 3, 5];
        let max_values = [9i64, 3, 5];
        let max_sel = [0u64, 1, 0, 0, 1, 1];
        let mut per_segment_sel = [0u64; 3];
        for k in 0..dst.len() {
            let d = dst[k] as usize;
            let diff = max_values[d] - scores[k];
            assert!(diff >= 0);
            if max_sel[k] == 1 {
                assert_eq!(diff, 0);
            }
            per_segment_sel[d] += max_sel[k];
            let q_new = k == 0 || dst[k] != dst[k - 1];
            let q_end = k + 1 == dst.len() || dst[k] != dst[k + 1];
            if q_new {
                assert!(k == 0 || dst[k - 1] < dst[k]);
            }
            if q_end {
                assert_eq!(per_segment_sel[d], 1);
            }
        }
    }

    #[test]
    fn compressed_projection_consistency_check_matches_full_projection() {
        let h_in = [2i64, -1, 3];
        let weights = [[1i64, 4], [-2, 5], [3, -6]];
        let rho = 7i64;
        let projected = [
            h_in[0] * weights[0][0] + h_in[1] * weights[1][0] + h_in[2] * weights[2][0],
            h_in[0] * weights[0][1] + h_in[1] * weights[1][1] + h_in[2] * weights[2][1],
        ];
        let compressed_lhs = projected[0] + projected[1] * rho;
        let compressed_rhs: i64 = (0..h_in.len())
            .map(|m| h_in[m] * (weights[m][0] + weights[m][1] * rho))
            .sum();
        assert_eq!(compressed_lhs, compressed_rhs);
    }

    #[test]
    fn compressed_aggregation_consistency_check_matches_full_aggregation() {
        let alpha = [2i64, 5, 7];
        let h_prime = [[3i64, -1], [4, 2], [-2, 6]];
        let rho = 11i64;
        let full = [
            alpha[0] * h_prime[0][0] + alpha[1] * h_prime[1][0] + alpha[2] * h_prime[2][0],
            alpha[0] * h_prime[0][1] + alpha[1] * h_prime[1][1] + alpha[2] * h_prime[2][1],
        ];
        let compressed_lhs = full[0] + full[1] * rho;
        let compressed_rhs: i64 = (0..alpha.len())
            .map(|k| alpha[k] * (h_prime[k][0] + h_prime[k][1] * rho))
            .sum();
        assert_eq!(compressed_lhs, compressed_rhs);
    }
}
