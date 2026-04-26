use crate::EZKLError;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const ROUTE_ID: &str = "ezkl_gat_native_csr_segment_attention_full";
const PRIMITIVE: &str = "GraphSegmentAttention";
const PRIMITIVE_VERSION: &str = "csr_segment_attention_v1";
const DEFAULT_MAX_LOGROWS: u32 = 26;

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
    graph_hash: String,
    checkpoint_hash: String,
    metadata_hash: String,
    shape_hash: String,
    total_attention_head_count: u64,
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
    estimate: NativeEstimate,
    generated_unix_ms: u128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LookupSemantics {
    lrelu: String,
    elu: String,
    exp_map: String,
    range_checks: String,
    formal_status: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InverseSemantics {
    mode: String,
    constraint: String,
    witness_column: String,
    nonzero_argument: String,
    approximate_reciprocal_used: bool,
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
    estimate: NativeEstimate,
    csr_layout: String,
    graph_hash: String,
    checkpoint_hash: String,
    exact_inverse_gate: InverseSemantics,
    lookup_semantics: LookupSemantics,
}

fn now_unix_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or(0)
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

fn ceil_log2(value: u128) -> u32 {
    if value <= 1 {
        return 0;
    }
    u128::BITS - (value - 1).leading_zeros()
}

fn total_hidden_heads(spec: &GraphAttentionSpec) -> u64 {
    spec.hidden_profile
        .iter()
        .map(|profile| profile.head_count)
        .sum()
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
        projection_mul_add_rows = projection_mul_add_rows
            .saturating_add(node_count * head_count * prev_width * head_dim);
        aggregation_rows =
            aggregation_rows.saturating_add(edge_count * head_count * head_dim * 2);
        elu_lookup_rows =
            elu_lookup_rows.saturating_add(node_count * head_count * head_dim);
        prev_width = u128::from(hidden_concat_width(profile));
    }
    projection_mul_add_rows = projection_mul_add_rows
        .saturating_add(node_count * output_heads * prev_width * u128::from(spec.class_count));
    aggregation_rows =
        aggregation_rows.saturating_add(edge_count * output_heads * u128::from(spec.class_count) * 2);

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

fn lookup_semantics(spec: &GraphAttentionSpec) -> LookupSemantics {
    let formal_status = if spec.quant_cfg_id == "fp32_bundle_export" {
        "blocked_without_quant_cfg_bound_finite_lookup_tables"
    } else {
        "requires_quant_cfg_bound_lookup_tables"
    };
    LookupSemantics {
        lrelu: "native lookup table keyed by quant_cfg_id".to_string(),
        elu: "native lookup table keyed by quant_cfg_id".to_string(),
        exp_map: "native lookup table keyed by quant_cfg_id".to_string(),
        range_checks: "native signed fixed-point range checks".to_string(),
        formal_status: formal_status.to_string(),
    }
}

fn inverse_semantics() -> InverseSemantics {
    InverseSemantics {
        mode: "native_witness_inverse_constraint".to_string(),
        constraint: "for every dst segment and attention head: Sum_i * Inv_i = 1".to_string(),
        witness_column: "segment_inverse".to_string(),
        nonzero_argument: "self_loop plus positive ExpMap semantics makes every Sum_i nonzero".to_string(),
        approximate_reciprocal_used: false,
    }
}

fn validate_artifacts(spec: &GraphAttentionSpec, csr: &CsrLayoutManifest) -> Result<(), EZKLError> {
    if spec.route_id != ROUTE_ID {
        return Err(format!("unexpected route_id: {}", spec.route_id).into());
    }
    if spec.native_primitive != PRIMITIVE || spec.native_primitive_version != PRIMITIVE_VERSION {
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

pub(crate) fn graph_gat_gen_settings(
    spec_path: PathBuf,
    csr_layout_path: PathBuf,
    settings_path: PathBuf,
    max_logrows: Option<u32>,
) -> Result<String, EZKLError> {
    let (spec, _csr) = load_pair(&spec_path, &csr_layout_path)?;
    let max_logrows = max_logrows.unwrap_or(DEFAULT_MAX_LOGROWS);
    let estimate = estimate_rows(&spec, max_logrows);
    let block_reason = if estimate.within_public_srs {
        ""
    } else {
        "ezkl_native_csr_compile_resource_boundary"
    };
    let settings = NativeSettings {
        settings_kind: "graph_gat_native_csr_settings".to_string(),
        route_id: ROUTE_ID.to_string(),
        native_primitive: PRIMITIVE.to_string(),
        native_primitive_version: PRIMITIVE_VERSION.to_string(),
        model_type: "graph-segment-attention-native-csr".to_string(),
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
        lookup_semantics: lookup_semantics(&spec),
        inverse_semantics: inverse_semantics(),
        estimate,
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
) -> Result<String, EZKLError> {
    let (spec, _csr) = load_pair(&spec_path, &csr_layout_path)?;
    let max_logrows = max_logrows.unwrap_or(DEFAULT_MAX_LOGROWS);
    let estimate = estimate_rows(&spec, max_logrows);
    let lookup = lookup_semantics(&spec);
    let inverse = inverse_semantics();
    let mut status = "compiled".to_string();
    let mut block_reason = String::new();
    if !estimate.within_public_srs {
        status = "blocked".to_string();
        block_reason = "ezkl_native_csr_compile_resource_boundary".to_string();
    } else if lookup.formal_status == "blocked_without_quant_cfg_bound_finite_lookup_tables" {
        status = "blocked".to_string();
        block_reason = "ezkl_native_lookup_tables_missing_from_quant_cfg".to_string();
    }
    let plan = NativeCompiledPlan {
        compiled_kind: "graph_gat_native_csr_compiled_plan".to_string(),
        route_id: ROUTE_ID.to_string(),
        native_primitive: PRIMITIVE.to_string(),
        native_primitive_version: PRIMITIVE_VERSION.to_string(),
        status: status.clone(),
        block_reason: block_reason.clone(),
        failure_step: if block_reason.is_empty() {
            String::new()
        } else {
            "graph-gat-compile".to_string()
        },
        dataset: spec.dataset.clone(),
        estimate,
        csr_layout: "csr_dst_segment_ragged".to_string(),
        graph_hash: spec.graph_hash.clone(),
        checkpoint_hash: spec.checkpoint_hash.clone(),
        exact_inverse_gate: inverse,
        lookup_semantics: lookup,
    };
    write_json(&compiled_circuit_path, &plan)?;
    if !settings_path.exists() {
        graph_gat_gen_settings(
            spec_path,
            csr_layout_path,
            settings_path,
            Some(max_logrows),
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

fn load_compiled_plan(path: &Path, step: &str) -> Result<NativeCompiledPlan, EZKLError> {
    let plan = read_json::<NativeCompiledPlan>(path)?;
    if plan.status != "compiled" {
        return Err(format!(
            "{} refused because compiled plan status={} block_reason={}",
            step, plan.status, plan.block_reason
        )
        .into());
    }
    Ok(plan)
}

fn graph_gat_prover_backend_unavailable(step: &str) -> Result<String, EZKLError> {
    Err(format!(
        "{} reached compiled native CSR plan, but the production GraphSegmentAttention prover backend is not complete in this ezkl revision",
        step
    )
    .into())
}

pub(crate) fn graph_gat_setup(
    compiled_circuit_path: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
) -> Result<String, EZKLError> {
    let _ = (vk_path, pk_path);
    let _plan = load_compiled_plan(&compiled_circuit_path, "graph-gat-setup")?;
    graph_gat_prover_backend_unavailable("graph-gat-setup")
}

pub(crate) fn graph_gat_gen_witness(
    compiled_circuit_path: PathBuf,
    output_path: PathBuf,
) -> Result<String, EZKLError> {
    let _ = output_path;
    let _plan = load_compiled_plan(&compiled_circuit_path, "graph-gat-gen-witness")?;
    graph_gat_prover_backend_unavailable("graph-gat-gen-witness")
}

pub(crate) fn graph_gat_prove(
    compiled_circuit_path: PathBuf,
    proof_path: PathBuf,
) -> Result<String, EZKLError> {
    let _ = proof_path;
    let _plan = load_compiled_plan(&compiled_circuit_path, "graph-gat-prove")?;
    graph_gat_prover_backend_unavailable("graph-gat-prove")
}

pub(crate) fn graph_gat_verify(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
) -> Result<String, EZKLError> {
    let _ = (proof_path, settings_path, vk_path);
    Err("graph-gat-verify requires a proof produced by graph-gat-prove; none was generated".into())
}

#[cfg(test)]
mod tests {
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Selector},
    };
    use halo2curves::bn256::Fr;
    use halo2curves::ff::Field;

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
                        region.assign_advice(|| "sum", config.sum, offset, || Value::known(*sum))?;
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
}
