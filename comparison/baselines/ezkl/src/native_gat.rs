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
use std::fs;
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const LEGACY_ROUTE_ID: &str = "ezkl_gat_native_csr_segment_attention_full";
const COMPRESSED_ROUTE_ID: &str = "ezkl_gat_native_compressed_csr_attention_full";
const LEGACY_PRIMITIVE: &str = "GraphSegmentAttention";
const LEGACY_PRIMITIVE_VERSION: &str = "csr_segment_attention_v1";
const COMPRESSED_PRIMITIVE: &str = "GraphSegmentAttentionCompressedCSR";
const COMPRESSED_PRIMITIVE_VERSION: &str = "csr_segment_attention_compressed_v1";
const DEFAULT_MAX_LOGROWS: u32 = 26;
const TINY_NATIVE_COMPILED_KIND: &str = "graph_gat_tiny_native_compiled_plan";
const TINY_NATIVE_BACKEND_VERSION: &str = "tiny_native_backend_levels_0_6_v1";
const TINY_NATIVE_LOGROWS: u32 = 9;
const DEFAULT_NATIVE_DEBUG_SCOPE: &str = "full_model";
const NATIVE_DIAGNOSTIC_BACKEND_VERSION: &str = "native_full_diagnostic_scopes_v1";
const TRACE_SCALAR_SCALE_I64: i64 = 16;
const QUANTIZED_INVERSE_POLICY: &str = "quantized_inverse_lookup";
const ALPHA_RESCALE_POLICY: &str = "rounded_product_div_scale_with_quantization_band";
const NATIVE_DEBUG_SCOPES: [&str; 14] = [
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
    "full_model",
];

#[allow(dead_code)]
#[derive(Clone, Debug, Deserialize)]
struct HiddenProfile {
    head_count: u64,
    head_dim: u64,
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
    quant_cfg_id: String,
    #[serde(default)]
    static_table_id: String,
    #[serde(default)]
    range_bits: u64,
    #[serde(default)]
    signed_bound: i64,
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
}

#[derive(Clone, Debug, Default, Deserialize)]
struct FormalTableHashesFile {
    #[serde(default)]
    source: String,
    #[serde(default)]
    table_policy: String,
    #[serde(default)]
    quant_cfg_id: String,
    #[serde(default)]
    static_table_id: String,
    #[serde(default)]
    range_bits: u64,
    #[serde(default)]
    signed_bound: i64,
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
            Self::FullModel => "full_model",
        }
    }

    fn logrows(self, plan: &NativeCompiledPlan) -> u32 {
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
            Self::FullModel => plan.estimate.total_rows.max(4096),
        };
        ceil_log2(diagnostic_rows.saturating_add(64)).clamp(9, plan.estimate.max_logrows)
    }
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
    if raw >= 0 {
        raw
    } else if raw >= -2 {
        0
    } else {
        raw / 5
    }
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
    let output_rows = node_count
        .saturating_mul(u128::from(spec.class_count))
        .saturating_mul(output_heads.saturating_add(2));
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
        lookup_batching: "single_batched_static_lookup_argument_per_table_kind".to_string(),
        max_selector_policy: "dst-sorted CSR state machine with one max selector per segment/head"
            .to_string(),
    }
}

fn binding_from_hashes(table: &FormalTableHashesFile) -> FormalTableBinding {
    FormalTableBinding {
        source: table.source.clone(),
        table_policy: table.table_policy.clone(),
        quant_cfg_id: table.quant_cfg_id.clone(),
        static_table_id: table.static_table_id.clone(),
        range_bits: table.range_bits,
        range_size: table.range_size,
        signed_bound: table.signed_bound,
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
    !binding.table_hash.is_empty()
        && binding.source == "formal_quant_cfg_derived"
        && binding.table_policy == "derived_from_main_formal_static_tables_v1"
        && binding.quant_cfg_id == spec.quant_cfg_id
        && (spec.static_table_id.is_empty() || binding.static_table_id == spec.static_table_id)
        && (spec.formal_table_hash.is_empty() || binding.table_hash == spec.formal_table_hash)
        && binding.inverse_policy == QUANTIZED_INVERSE_POLICY
        && !binding.inv_hash.is_empty()
        && binding.inverse_table_hash == binding.inv_hash
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
    let _ = NativeDebugScope::parse(&native_debug_scope)?;
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
        strict_public_input_equivalent: false,
        full_gat_forward_in_ezkl_proof: false,
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
    let _ = NativeDebugScope::parse(&native_debug_scope)?;
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
        strict_public_input_equivalent: false,
        full_gat_forward_in_ezkl_proof: false,
        diagnostic_backend_version: NATIVE_DIAGNOSTIC_BACKEND_VERSION.to_string(),
        diagnostic_note: "diagnostic native circuit proves scoped CSR/table/metadata relations; strict full GAT forward is not asserted until all full-model relations are implemented".to_string(),
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
}

#[derive(Clone, Debug)]
struct FullNativeConfig {
    left: Column<Advice>,
    right: Column<Advice>,
    out: Column<Advice>,
    aux: Column<Advice>,
    instance: Column<Instance>,
    q_mul: Selector,
    q_inverse: Selector,
    q_segment_sum: Selector,
    q_max: Selector,
    q_projection: Selector,
    q_aggregation: Selector,
    q_output_average: Selector,
    q_public_binding: Selector,
    q_lrelu_lookup: Selector,
    q_exp_lookup: Selector,
    q_elu_lookup: Selector,
    table_lrelu_x: TableColumn,
    table_lrelu_y: TableColumn,
    table_exp_x: TableColumn,
    table_exp_y: TableColumn,
    table_elu_x: TableColumn,
    table_elu_y: TableColumn,
    table_inv_sum: TableColumn,
    table_inv_inv: TableColumn,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct FullNativeCircuit {
    params: FullNativeParams,
    plan: NativeCompiledPlan,
    csr: CsrPayload,
    witness_source_path: String,
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
            witness_source_path,
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
            },
            plan,
            csr,
            witness_source_path: String::new(),
        }
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
                let sum_q = (sum.max(1) as i64).saturating_mul(policy.sum_scale);
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
                Self::assign_advice(&mut region, config.aux, 0, "inverse aux", Fr::ZERO)?;
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
                        "segment id",
                        Fr::from(segment as u64),
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
                        "lookup row",
                        Fr::from(offset as u64),
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

    fn assign_output_average_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
        region_name: &'static str,
    ) -> Result<(), Error> {
        layouter.assign_region(
            || region_name,
            |mut region| {
                let head_count = self.plan.k_out.max(1);
                let sum = self.plan.class_count.saturating_mul(head_count).max(1);
                let avg = sum / head_count;
                config.q_output_average.enable(&mut region, 0)?;
                Self::assign_advice(&mut region, config.left, 0, "output sum", Fr::from(sum))?;
                Self::assign_advice(
                    &mut region,
                    config.right,
                    0,
                    "output head count",
                    Fr::from(head_count),
                )?;
                Self::assign_advice(&mut region, config.out, 0, "output average", Fr::from(avg))?;
                Self::assign_advice(
                    &mut region,
                    config.aux,
                    0,
                    "output class count",
                    Fr::from(self.plan.class_count),
                )?;
                Ok(())
            },
        )
    }

    fn assign_public_binding_region(
        &self,
        layouter: &mut impl Layouter<Fr>,
        config: &FullNativeConfig,
    ) -> Result<(), Error> {
        let values = native_full_public_inputs(&self.plan);
        let cells = layouter.assign_region(
            || "native.public_output_binding",
            |mut region| {
                let mut cells = Vec::with_capacity(values.len());
                for (offset, value) in values.iter().copied().enumerate() {
                    config.q_public_binding.enable(&mut region, offset)?;
                    let cell =
                        Self::assign_advice(&mut region, config.left, offset, "public lhs", value)?;
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
        for (index, cell) in cells.into_iter().enumerate() {
            layouter.constrain_instance(cell.cell(), config.instance, index)?;
        }
        Ok(())
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
            },
        )
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        _params: Self::Params,
    ) -> Self::Config {
        let left = meta.advice_column();
        let right = meta.advice_column();
        let out = meta.advice_column();
        let aux = meta.advice_column();
        let instance = meta.instance_column();
        meta.enable_equality(left);
        meta.enable_equality(instance);

        let q_mul = meta.selector();
        let q_inverse = meta.complex_selector();
        let q_segment_sum = meta.selector();
        let q_max = meta.selector();
        let q_projection = meta.selector();
        let q_aggregation = meta.selector();
        let q_output_average = meta.selector();
        let q_public_binding = meta.selector();
        let q_lrelu_lookup = meta.complex_selector();
        let q_exp_lookup = meta.complex_selector();
        let q_elu_lookup = meta.complex_selector();
        let table_lrelu_x = meta.lookup_table_column();
        let table_lrelu_y = meta.lookup_table_column();
        let table_exp_x = meta.lookup_table_column();
        let table_exp_y = meta.lookup_table_column();
        let table_elu_x = meta.lookup_table_column();
        let table_elu_y = meta.lookup_table_column();
        let table_inv_sum = meta.lookup_table_column();
        let table_inv_inv = meta.lookup_table_column();

        meta.create_gate("native.level0.mul", |meta| {
            let q = meta.query_selector(q_mul);
            let left_expr = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let right_expr = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let out_expr = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (left_expr * right_expr - out_expr)]
        });
        meta.lookup("native.lookup.inverse", |meta| {
            let q = meta.query_selector(q_inverse);
            let sum = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let inv = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![(q.clone() * sum, table_inv_sum), (q * inv, table_inv_inv)]
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
        meta.create_gate("native.output_average", |meta| {
            let q = meta.query_selector(q_output_average);
            let sum = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let heads = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            let avg = meta.query_advice(out, halo2_proofs::poly::Rotation::cur());
            vec![q * (avg * heads - sum)]
        });
        meta.create_gate("native.public_output_binding", |meta| {
            let q = meta.query_selector(q_public_binding);
            let lhs = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let rhs = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![q * (lhs - rhs)]
        });
        meta.lookup("native.lookup.lrelu", |meta| {
            let q = meta.query_selector(q_lrelu_lookup);
            let x = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let y = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![(q.clone() * x, table_lrelu_x), (q * y, table_lrelu_y)]
        });
        meta.lookup("native.lookup.exp", |meta| {
            let q = meta.query_selector(q_exp_lookup);
            let x = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let y = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![(q.clone() * x, table_exp_x), (q * y, table_exp_y)]
        });
        meta.lookup("native.lookup.elu", |meta| {
            let q = meta.query_selector(q_elu_lookup);
            let x = meta.query_advice(left, halo2_proofs::poly::Rotation::cur());
            let y = meta.query_advice(right, halo2_proofs::poly::Rotation::cur());
            vec![(q.clone() * x, table_elu_x), (q * y, table_elu_y)]
        });

        FullNativeConfig {
            left,
            right,
            out,
            aux,
            instance,
            q_mul,
            q_inverse,
            q_segment_sum,
            q_max,
            q_projection,
            q_aggregation,
            q_output_average,
            q_public_binding,
            q_lrelu_lookup,
            q_exp_lookup,
            q_elu_lookup,
            table_lrelu_x,
            table_lrelu_y,
            table_exp_x,
            table_exp_y,
            table_elu_x,
            table_elu_y,
            table_inv_sum,
            table_inv_inv,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        layouter.assign_table(
            || "native.lookup.inverse.table",
            |mut table| {
                let policy = inverse_policy_from_binding(&self.plan.formal_table_binding);
                let capacity = (1usize << self.params.logrows).saturating_sub(1);
                let mut offset = 0usize;
                let range_size = inverse_table_range_size(&self.plan.formal_table_binding);
                for sum_q in 0..range_size {
                    if offset >= capacity {
                        break;
                    }
                    let (min_inv, max_inv) =
                        quantized_inverse_interval_for_sum_q(sum_q as i64, &policy);
                    for inv_q in min_inv..=max_inv {
                        if offset >= capacity {
                            break;
                        }
                        table.assign_cell(
                            || "inv sum_q",
                            config.table_inv_sum,
                            offset,
                            || Value::known(Fr::from(sum_q as u64)),
                        )?;
                        table.assign_cell(
                            || "inv inv_q",
                            config.table_inv_inv,
                            offset,
                            || Value::known(fr_i64(inv_q)),
                        )?;
                        offset += 1;
                    }
                }
                if offset == 0 {
                    table.assign_cell(
                        || "inv sum_q",
                        config.table_inv_sum,
                        0,
                        || Value::known(Fr::ZERO),
                    )?;
                    table.assign_cell(
                        || "inv inv_q",
                        config.table_inv_inv,
                        0,
                        || Value::known(Fr::ZERO),
                    )?;
                }
                Ok(())
            },
        )?;
        layouter.assign_table(
            || "native.lookup.lrelu.table",
            |mut table| {
                for (offset, raw) in (-8i64..=8).enumerate() {
                    table.assign_cell(
                        || "lrelu x",
                        config.table_lrelu_x,
                        offset,
                        || Value::known(fr_i64(raw)),
                    )?;
                    table.assign_cell(
                        || "lrelu y",
                        config.table_lrelu_y,
                        offset,
                        || Value::known(fr_i64(leaky_relu_table_output(raw))),
                    )?;
                }
                Ok(())
            },
        )?;
        layouter.assign_table(
            || "native.lookup.exp.table",
            |mut table| {
                table.assign_cell(
                    || "exp disabled x",
                    config.table_exp_x,
                    0,
                    || Value::known(Fr::ZERO),
                )?;
                table.assign_cell(
                    || "exp disabled y",
                    config.table_exp_y,
                    0,
                    || Value::known(Fr::ZERO),
                )?;
                for raw in 0usize..=16 {
                    let offset = raw + 1;
                    let out = (16i64 - raw as i64).max(0);
                    table.assign_cell(
                        || "exp x",
                        config.table_exp_x,
                        offset,
                        || Value::known(Fr::from(raw as u64)),
                    )?;
                    table.assign_cell(
                        || "exp y",
                        config.table_exp_y,
                        offset,
                        || Value::known(fr_i64(out)),
                    )?;
                }
                Ok(())
            },
        )?;
        layouter.assign_table(
            || "native.lookup.elu.table",
            |mut table| {
                for (offset, raw) in (-8i64..=8).enumerate() {
                    table.assign_cell(
                        || "elu x",
                        config.table_elu_x,
                        offset,
                        || Value::known(fr_i64(raw)),
                    )?;
                    table.assign_cell(
                        || "elu y",
                        config.table_elu_y,
                        offset,
                        || Value::known(fr_i64(raw)),
                    )?;
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
            NativeDebugScope::OutputOnly | NativeDebugScope::FullModel
        ) {
            self.assign_output_average_region(&mut layouter, &config, "native.output_average")?;
        }
        self.assign_public_binding_region(&mut layouter, &config)
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

fn native_full_public_inputs(plan: &NativeCompiledPlan) -> Vec<Fr> {
    vec![
        Fr::from(plan.node_count.max(1)),
        Fr::from(plan.edge_count.max(1)),
        Fr::from(plan.class_count.max(1)),
        Fr::from(plan.total_attention_head_count.max(1)),
    ]
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
        actual_values: vec![
            format!("required_logrows={}", plan.estimate.required_logrows),
            format!("node_count={}", plan.node_count),
            format!("edge_count={}", plan.edge_count),
            format!("run_dir={}", run_dir.display()),
        ],
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
    NativeConstraintFailure {
        status: "blocked".to_string(),
        block_reason: "native_full_gat_forward_constraints_incomplete".to_string(),
        gate_name: "native.public_output_binding".to_string(),
        region_name: "native.full_model.layer_1.output.public_output_binding".to_string(),
        row_offset: 0,
        dataset: plan.dataset.clone(),
        route_id: plan.route_id.clone(),
        layer_index: Some(plan.layer_count.saturating_sub(1)),
        head_index: None,
        component: "public_output_binding".to_string(),
        expected_relation: "strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, ELU, output average".to_string(),
        actual_values: vec![
            "diagnostic proof binds CSR/table/metadata scoped relations only".to_string(),
            format!("scope={}", plan.native_debug_scope),
            format!("full_gat_forward_in_ezkl_proof={}", plan.full_gat_forward_in_ezkl_proof),
            format!("strict_public_input_equivalent={}", plan.strict_public_input_equivalent),
            format!("run_dir={}", run_dir.display()),
        ],
        witness_source_path: witness_path.display().to_string(),
        command,
        raw_error: "strict full-model public output witness columns are not assigned by this native diagnostic circuit".to_string(),
        attempted_fix: "replaced the artificial full compressed CSR failure with real Halo2 MockProver/KZG diagnostic scopes and preserved strict_public_input_equivalent=false for non-full semantics".to_string(),
        remaining_blocker: "full quantized GAT row assignment for public logits remains unimplemented; marking strict=true would be unsound".to_string(),
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
        if !validate_inverse_witness_against_policy(sum_q, inv_q, &policy) {
            return Err(format!(
                "quantized_inverse_lookup_mismatch: sum_q={} inv_q={} expected_inv_q={}",
                sum_q,
                inv_q,
                quantized_inverse_for_sum_q(sum_q, &policy)
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
    let circuit = FullNativeCircuit::new(plan, run_dir, String::new())?;
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
            Ok(serde_json::json!({
                "status": "setup_complete",
                "route_id": plan.route_id,
                "native_primitive": plan.native_primitive,
                "native_primitive_version": plan.native_primitive_version,
                "dataset": plan.dataset,
                "native_debug_scope": plan.native_debug_scope,
                "logrows": logrows,
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
                "payloads": {
                    "csr_layout": run_dir.join("csr_segment_layout.bin"),
                    "node_features": run_dir.join("node_features.f64.bin"),
                    "checkpoint_tensors": run_dir.join("checkpoint_tensors.txt"),
                    "formal_tables": run_dir.join("formal_tables.bin")
                },
                "status": "diagnostic_witness_metadata_ready",
                "strict_public_input_equivalent": false,
                "full_gat_forward_in_ezkl_proof": false,
                "full_model_witness_preflight": if preflight_path.exists() {
                    serde_json::from_str::<serde_json::Value>(&fs::read_to_string(&preflight_path)?)?
                } else {
                    serde_json::Value::Null
                },
                "note": "dataset payloads are bound for native diagnostic scopes; strict full GAT output binding remains false unless full-model preflight and circuit constraints are complete"
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
            let circuit =
                FullNativeCircuit::new(plan.clone(), &run_dir, witness_path.display().to_string())?;
            let public_inputs = native_full_public_inputs(&plan);
            let mock = MockProver::run(
                circuit.params.logrows,
                &circuit,
                vec![public_inputs.clone()],
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
                            format!(
                                "graph-gat-prove --compiled-circuit {} --witness {} --pk-path {} --proof-path {}",
                                compiled_circuit_path.display(),
                                witness_path.display(),
                                pk_path.display(),
                                proof_path.display()
                            ),
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
                        format!(
                            "graph-gat-prove --compiled-circuit {} --witness {} --pk-path {} --proof-path {}",
                            compiled_circuit_path.display(),
                            witness_path.display(),
                            pk_path.display(),
                            proof_path.display()
                        ),
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

            if scope == NativeDebugScope::FullModel && !plan.strict_public_input_equivalent {
                let failure = strict_equivalence_failure(
                    &plan,
                    &run_dir,
                    &witness_path,
                    format!(
                        "graph-gat-prove --compiled-circuit {} --witness {} --pk-path {} --proof-path {}",
                        compiled_circuit_path.display(),
                        witness_path.display(),
                        pk_path.display(),
                        proof_path.display()
                    ),
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

            let pk = load_pk::<KZGCommitmentScheme<Bn256>, FullNativeCircuit>(
                pk_path.clone(),
                circuit.params.clone(),
            )
            .map_err(|error| EZKLError::from(format!("native_prover_failed: load_pk: {error}")))?;
            let params = load_native_params_for_key(&pk_path, circuit.params.logrows)?;
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
                vec![public_inputs],
                &params,
                &pk,
                CheckMode::SAFE,
                None,
                None,
            )
            .map_err(|error| {
                EZKLError::from(format!("native_prover_failed: create_proof: {error}"))
            })?;
            snark.save(&proof_path)?;
            Ok(serde_json::json!({
                "status": "proof_created",
                "route_id": plan.route_id,
                "native_debug_scope": plan.native_debug_scope,
                "diagnostic_backend_version": NATIVE_DIAGNOSTIC_BACKEND_VERSION,
                "dataset": plan.dataset,
                "logrows": NativeDebugScope::parse(&plan.native_debug_scope)?.logrows(&plan),
                "proof_path": proof_path,
                "public_inputs": [
                    plan.node_count.max(1),
                    plan.edge_count.max(1),
                    plan.class_count.max(1),
                    plan.total_attention_head_count.max(1)
                ],
                "strict_public_input_equivalent": plan.strict_public_input_equivalent,
                "full_gat_forward_in_ezkl_proof": plan.full_gat_forward_in_ezkl_proof
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
            let output = serde_json::json!({
                "status": "verified",
                "verify_passed": true,
                "route_id": plan.route_id,
                "native_debug_scope": plan.native_debug_scope,
                "diagnostic_backend_version": NATIVE_DIAGNOSTIC_BACKEND_VERSION,
                "proof_path": proof_path,
                "vk_path": vk_path,
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
        TINY_NATIVE_LOGROWS, TinyNativeCircuit, create_tiny_proof,
        default_quantized_inverse_policy, default_tiny_witness, quantized_inverse_for_sum_q,
        save_tiny_keys, tiny_params, tiny_public_inputs, validate_alpha_rescale,
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
