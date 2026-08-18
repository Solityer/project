//! Formal fixed-shape GAT circuit for the comparison baseline.
//!
//! The circuit is deliberately expressed as a small, auditable Halo2 trace
//! machine.  `FormalProgram` is generated only from the compiled architecture
//! and topology; witness values never select an operation or a wire.  Every
//! use of a logical value is connected to its canonical cell with a Halo2 copy
//! constraint.  This is the production GAT path -- there is no tiny,
//! diagnostic, sampled, compressed-equality, or host-only proof mode here.

use crate::circuit::modules::poseidon::spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH};
use crate::circuit::modules::poseidon::{PoseidonChip, PoseidonConfig};
use crate::circuit::modules::Module;
use crate::circuit::CheckMode;
use crate::fieldutils::integer_rep_to_felt;
use crate::pfsys::{
    create_keys, create_proof_circuit, load_pk, load_vk, save_params, save_pk, save_vk,
    verify_proof_circuit, Snark,
};
use crate::tensor::{Tensor, ValTensor, ValType};
use crate::EZKLError;
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner, Value};
use halo2_proofs::plonk::{
    Advice, Circuit, Column, ConstraintSystem, Error, Expression, Instance, Selector, TableColumn,
};
use halo2_proofs::poly::commitment::{Params, ParamsProver};
use halo2_proofs::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::strategy::SingleStrategy as KZGSingleStrategy;
use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};

const PLAN_KIND: &str = "ezkl_formal_fixed_gat_program_v1";
const SETTINGS_KIND: &str = "ezkl_formal_fixed_gat_settings_v1";
const WITNESS_KIND: &str = "ezkl_formal_gat_witness_v1";
const STATEMENT_KIND: &str = "ezkl_formal_gat_public_statement_v1";
const TABLE_GENERATOR_ID: &str = "main_gatzk_quantized_tables_v1";
const AUX_TABLE_GENERATOR_ID: &str = "ezkl_backend_aux_tables_v1";
const DEFAULT_MAX_LOGROWS: u32 = 28;
const PROGRAM_LANES: usize = 32;

type Var = usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct HiddenProfile {
    head_count: usize,
    head_dim: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct HeadSpec {
    #[serde(default)]
    layer_index: usize,
    #[serde(default)]
    local_head_index: usize,
    #[serde(default)]
    head_index: usize,
    seq_kernel: String,
    attn_src_kernel: String,
    attn_dst_kernel: String,
    #[serde(default)]
    output_bias: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GraphSpec {
    route_id: String,
    dataset: String,
    batching_rule: String,
    node_count: usize,
    edge_count: usize,
    graph_count: usize,
    feature_dim: usize,
    class_count: usize,
    #[serde(rename = "L")]
    layer_count: usize,
    #[serde(rename = "K_out")]
    k_out: usize,
    hidden_profile: Vec<HiddenProfile>,
    d_in_profile: Vec<usize>,
    hidden_heads: Vec<HeadSpec>,
    output_heads: Vec<HeadSpec>,
    model_arch_id: String,
    model_param_id: String,
    quant_cfg_id: String,
    #[serde(alias = "static_table_id")]
    static_tables_id: String,
    backend_aux_tables_id: String,
    source_quant_cfg_id: String,
    source_static_table_id: String,
    security_profile_id: String,
    graph_hash: String,
    checkpoint_hash: String,
    metadata_hash: String,
    shape_hash: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct CsrLayout {
    layout_kind: String,
    node_count: usize,
    edge_count: usize,
    graph_count: usize,
    max_in_degree: usize,
    src: Vec<usize>,
    dst: Vec<usize>,
    node_ptr: Vec<usize>,
    edge_ptr: Vec<usize>,
    #[serde(default)]
    absolute_ids: Vec<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct QuantConfig {
    rounding_mode: String,
    limb_bits: u32,
    feature_scale: i128,
    hidden_weight_scale: i128,
    hidden_attention_scale: i128,
    hidden_projection_scale: i128,
    delta_scale: i128,
    exp_scale: i128,
    hidden_recovery_num_scale: i128,
    hidden_recovery_den_scale: i128,
    elu_input_scale: i128,
    elu_output_scale: i128,
    agg_scale: i128,
    output_weight_scale: i128,
    output_attention_scale: i128,
    output_projection_scale: i128,
    output_pre_scale: i128,
    output_head_scale: i128,
    output_bias_scale: i128,
    output_scale: i128,
    source_quant_cfg_id: String,
    canonical_digest: String,
}

impl Default for QuantConfig {
    fn default() -> Self {
        Self {
            rounding_mode: "nearest_ties_away".to_owned(),
            limb_bits: 12,
            feature_scale: 1,
            hidden_weight_scale: 16,
            hidden_attention_scale: 16,
            hidden_projection_scale: 16,
            delta_scale: 256,
            exp_scale: 16,
            hidden_recovery_num_scale: 1,
            hidden_recovery_den_scale: 16,
            elu_input_scale: 16,
            elu_output_scale: 16,
            agg_scale: 16,
            output_weight_scale: 16,
            output_attention_scale: 1,
            output_projection_scale: 256,
            output_pre_scale: 16,
            output_head_scale: 16,
            output_bias_scale: 16,
            output_scale: 16,
            source_quant_cfg_id: String::new(),
            canonical_digest: String::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TableSpec {
    lrelu_min: i128,
    lrelu_max: i128,
    elu_min: i128,
    elu_max: i128,
    exp_max: i128,
    generator_id: String,
    profile_version: String,
    rounding_mode: String,
    clipping_rule: String,
    canonical_order: String,
    lrelu_scale: i128,
    elu_input_scale: i128,
    elu_output_scale: i128,
    exp_input_scale: i128,
    exp_output_scale: i128,
    source_static_table_id: String,
    rows_path: String,
    rows_digest: String,
    canonical_digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct AuxTableSpec {
    max_relu_min: i128,
    max_relu_max: i128,
    limb_max: i128,
    generator_id: String,
    profile_version: String,
    canonical_order: String,
    rows_path: String,
    rows_digest: String,
    canonical_digest: String,
}

#[derive(Clone, Debug, Default)]
struct CanonicalTables {
    rows: Vec<(u64, i128, i128)>,
}

impl CanonicalTables {
    fn lookup(&self, kind: LookupKind, x: i128) -> Result<i128, EZKLError> {
        self.rows
            .binary_search_by_key(&(kind as u64, x), |(row_kind, row_x, _)| {
                (*row_kind, *row_x)
            })
            .map(|index| self.rows[index].2)
            .map_err(|_| "formal lookup input escaped its frozen canonical table".into())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RoundDivBounds {
    a: i128,
    q: i128,
    r: i128,
    delta: i128,
    d_min: i128,
    d: i128,
    lhs_2a_plus_d: i128,
    mul_dq: i128,
    rhs_2dq_plus_r: i128,
    two_d_minus_one: i128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct BoundFamily {
    label: String,
    input_dim: usize,
    output_dim: usize,
    h: i128,
    w: i128,
    a_src: i128,
    a_dst: i128,
    projection: i128,
    e_src: i128,
    e_dst: i128,
    score: i128,
    z: i128,
    m: i128,
    delta_raw: i128,
    delta: i128,
    exp: i128,
    sum: i128,
    aggregate_num: i128,
    projection_mul: i128,
    projection_acc: i128,
    attention_mul: i128,
    attention_acc: i128,
    aggregate_mul: i128,
    aggregate_acc: i128,
    scalar_h: i128,
    scalar_projection: i128,
    scalar_e_src: i128,
    scalar_e_dst: i128,
    scalar_score: i128,
    scalar_z: i128,
    scalar_delta: i128,
    recovery: RoundDivBounds,
    #[serde(default)]
    alignment: Option<RoundDivBounds>,
    output_bound: i128,
    #[serde(default)]
    bias: i128,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct IntegerBoundLedger {
    ledger_kind: String,
    field_modulus: String,
    max_in_degree: usize,
    model_bounds_digest: String,
    families: Vec<BoundFamily>,
    output_head_bound: i128,
    output_sum_bound: i128,
    final_average: RoundDivBounds,
    no_wrap_verified: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SecurityProfile {
    profile_kind: String,
    ezkl_version: String,
    halo2_git_commit: String,
    curve: String,
    scalar_field: String,
    polynomial_commitment: String,
    multiopen: String,
    transcript: String,
    challenge: String,
    setup: String,
    native_security_level_claim: String,
    ceremony_descriptor_kind: String,
    ceremony_id: String,
    source_srs_sha256: String,
    source_srs_max_k: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ArtifactPaths {
    features_i64: String,
    model_i64: String,
    model_registration: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FormalPlan {
    kind: String,
    program_version: String,
    route_id: String,
    dataset: String,
    node_count: usize,
    edge_count: usize,
    graph_count: usize,
    feature_dim: usize,
    class_count: usize,
    layer_count: usize,
    k_out: usize,
    hidden_profile: Vec<HiddenProfile>,
    d_in_profile: Vec<usize>,
    hidden_heads: Vec<HeadSpec>,
    output_heads: Vec<HeadSpec>,
    src: Vec<usize>,
    dst: Vec<usize>,
    node_ptr: Vec<usize>,
    edge_ptr: Vec<usize>,
    absolute_ids: Vec<usize>,
    quant: QuantConfig,
    tables: TableSpec,
    aux_tables: AuxTableSpec,
    ledger: IntegerBoundLedger,
    model_parameter_bounds: BTreeMap<String, i128>,
    artifacts: ArtifactPaths,
    model_arch_id: String,
    model_param_id: String,
    quant_cfg_id: String,
    static_tables_id: String,
    backend_aux_tables_id: String,
    graph_hash: String,
    checkpoint_hash: String,
    metadata_hash: String,
    shape_hash: String,
    registered_model_hash: String,
    settings_id: String,
    compiled_circuit_id: String,
    security_profile_id: String,
    security_profile: SecurityProfile,
    logrows: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FormalSettings {
    kind: String,
    status: String,
    relation: String,
    route_id: String,
    dataset: String,
    logrows: u32,
    quant_cfg_id: String,
    static_tables_id: String,
    backend_aux_tables_id: String,
    security_profile_id: String,
    program_version: String,
    settings_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ModelRegistration {
    registration_kind: String,
    model_arch_id: String,
    model_param_id: String,
    quant_cfg_id: String,
    canonical_parameter_count: usize,
    canonical_parameter_digest: String,
    #[serde(default)]
    parameter_bounds: BTreeMap<String, i128>,
    model_bounds_digest: String,
    registered_model_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CircuitManifest {
    manifest_kind: String,
    compiled_circuit_id: String,
    settings_id: String,
    vk_id: String,
    pk_id: String,
    security_profile_id: String,
    quant_cfg_id: String,
    static_tables_id: String,
    backend_aux_tables_id: String,
    device_build: String,
    source_srs_sha256: String,
    source_srs_path: String,
    source_srs_k: u32,
    downsized_srs_k: u32,
    proving_srs_sha256: String,
    verification_srs_sha256: String,
    proving_srs_path: String,
    verification_srs_path: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PublicStatement {
    kind: String,
    dataset: String,
    model_arch_id: String,
    quant_cfg_id: String,
    static_tables_id: String,
    backend_aux_tables_id: String,
    security_profile_id: String,
    compiled_circuit_id: String,
    settings_id: String,
    vk_id: String,
    graph_hash: String,
    node_count: usize,
    edge_count: usize,
    graph_count: usize,
    feature_dim: usize,
    class_count: usize,
    t_h: Vec<i128>,
    indices: Vec<usize>,
    src: Vec<usize>,
    dst: Vec<usize>,
    node_ptr: Vec<usize>,
    edge_ptr: Vec<usize>,
    logits: Vec<i128>,
    model_hash: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FormalWitness {
    kind: String,
    compiled_circuit_id: String,
    values: Vec<String>,
    public_statement_path: String,
    logical_trace_digest: String,
}

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
struct ObservedFamily {
    input_dim: usize,
    output_dim: usize,
    h: i128,
    w: i128,
    a_src: i128,
    a_dst: i128,
    projection: i128,
    e_src: i128,
    e_dst: i128,
    score: i128,
    z: i128,
    m: i128,
    delta: i128,
    exp: i128,
    sum: i128,
    aggregate_num: i128,
    projection_mul: i128,
    projection_acc: i128,
    attention_mul: i128,
    attention_acc: i128,
    aggregate_mul: i128,
    aggregate_acc: i128,
    recovery_a: i128,
    recovery_d_min: i128,
    recovery_d: i128,
    recovery_q: i128,
    output_bound: i128,
    bias: i128,
    alignment_a: i128,
    alignment_q: i128,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
enum LookupKind {
    Limb = 1,
    LeakyRelu = 2,
    Elu = 3,
    ExpNeg = 4,
    MaxRelu = 5,
}

#[derive(Clone, Debug)]
enum Constraint {
    MulAdd { a: Var, b: Var, c: Var, out: Var },
    Boolean { value: Var },
    Constant { value: Var, constant: i128 },
    Lookup { kind: LookupKind, x: Var, y: Var },
}

#[derive(Clone, Debug, Default)]
struct FormalProgram {
    values: Vec<Option<Fr>>,
    integer_values: Vec<Option<i128>>,
    constraints: Vec<Constraint>,
    public_vars: Vec<Var>,
    model_vars: Vec<Var>,
    output_vars: Vec<Var>,
    constants: BTreeMap<i128, Var>,
    observed_families: BTreeMap<String, ObservedFamily>,
}

impl FormalProgram {
    fn witness(&mut self, value: Option<i128>) -> Var {
        let index = self.values.len();
        self.values.push(value.map(fr_i128));
        self.integer_values.push(value);
        index
    }

    fn field_witness(&mut self, value: Option<Fr>) -> Var {
        let index = self.values.len();
        self.values.push(value);
        self.integer_values.push(None);
        index
    }

    fn public(&mut self, value: Option<i128>) -> Var {
        let var = self.witness(value);
        self.public_vars.push(var);
        var
    }

    fn public_field(&mut self, value: Option<Fr>) -> Var {
        let var = self.field_witness(value);
        self.public_vars.push(var);
        var
    }

    fn constant(&mut self, value: i128) -> Var {
        if let Some(&var) = self.constants.get(&value) {
            return var;
        }
        let var = self.witness(Some(value));
        self.constraints.push(Constraint::Constant {
            value: var,
            constant: value,
        });
        self.constants.insert(value, var);
        var
    }

    fn mul_add(&mut self, a: Var, b: Var, c: Var) -> Result<Var, EZKLError> {
        let integer = match (
            self.integer_values[a],
            self.integer_values[b],
            self.integer_values[c],
        ) {
            (Some(a), Some(b), Some(c)) => Some(
                a.checked_mul(b)
                    .and_then(|value| value.checked_add(c))
                    .ok_or("canonical integer arithmetic overflow")?,
            ),
            _ => None,
        };
        let value = match (self.values[a], self.values[b], self.values[c]) {
            (Some(a), Some(b), Some(c)) => Some(a * b + c),
            _ => None,
        };
        let out = if let Some(integer) = integer {
            self.witness(Some(integer))
        } else {
            self.field_witness(value)
        };
        self.constraints.push(Constraint::MulAdd { a, b, c, out });
        Ok(out)
    }

    fn add(&mut self, a: Var, b: Var, one: Var) -> Result<Var, EZKLError> {
        self.mul_add(a, one, b)
    }

    fn mul(&mut self, a: Var, b: Var, zero: Var) -> Result<Var, EZKLError> {
        self.mul_add(a, b, zero)
    }

    fn assert_equal(&mut self, a: Var, b: Var, one: Var, zero: Var) {
        self.constraints.push(Constraint::MulAdd {
            a,
            b: one,
            c: zero,
            out: b,
        });
    }

    fn boolean(&mut self, value: Var) {
        self.constraints.push(Constraint::Boolean { value });
    }

    fn lookup(&mut self, kind: LookupKind, x: Var, y: Var) {
        self.constraints.push(Constraint::Lookup { kind, x, y });
    }
}

#[derive(Clone, Debug, Default)]
struct CircuitParams {
    #[allow(dead_code)]
    logrows: u32,
}

#[derive(Clone, Debug)]
struct FormalGatConfig {
    canonical: Vec<Column<Advice>>,
    op_a: Vec<Column<Advice>>,
    op_b: Vec<Column<Advice>>,
    op_c: Vec<Column<Advice>>,
    op_out: Vec<Column<Advice>>,
    lookup_kind: Vec<Column<Advice>>,
    fixed_constant: Vec<Column<halo2_proofs::plonk::Fixed>>,
    instance: Column<Instance>,
    q_mul_add: Vec<Selector>,
    q_boolean: Vec<Selector>,
    q_constant: Vec<Selector>,
    q_lookup: Vec<Selector>,
    table_kind: TableColumn,
    table_x: TableColumn,
    table_y: TableColumn,
    poseidon: PoseidonConfig<POSEIDON_WIDTH, POSEIDON_RATE>,
}

#[derive(Clone, Debug)]
struct FormalGatCircuit {
    plan: FormalPlan,
    program: FormalProgram,
    tables: CanonicalTables,
}

impl Circuit<Fr> for FormalGatCircuit {
    type Config = FormalGatConfig;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = CircuitParams;

    fn without_witnesses(&self) -> Self {
        let mut program = self.program.clone();
        program.values.fill(None);
        program.integer_values.fill(None);
        Self {
            plan: self.plan.clone(),
            program,
            tables: self.tables.clone(),
        }
    }

    fn params(&self) -> Self::Params {
        CircuitParams {
            logrows: self.plan.logrows,
        }
    }

    fn configure_with_params(
        meta: &mut ConstraintSystem<Fr>,
        _params: Self::Params,
    ) -> Self::Config {
        let canonical = (0..PROGRAM_LANES)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        let op_a = (0..PROGRAM_LANES)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        let op_b = (0..PROGRAM_LANES)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        let op_c = (0..PROGRAM_LANES)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        let op_out = (0..PROGRAM_LANES)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        let lookup_kind = (0..PROGRAM_LANES)
            .map(|_| meta.advice_column())
            .collect::<Vec<_>>();
        let fixed_constant = (0..PROGRAM_LANES)
            .map(|_| meta.fixed_column())
            .collect::<Vec<_>>();
        let instance = meta.instance_column();
        for columns in [&canonical, &op_a, &op_b, &op_c, &op_out] {
            for &column in columns {
                meta.enable_equality(column);
            }
        }
        meta.enable_equality(instance);
        let q_mul_add = (0..PROGRAM_LANES)
            .map(|_| meta.selector())
            .collect::<Vec<_>>();
        let q_boolean = (0..PROGRAM_LANES)
            .map(|_| meta.selector())
            .collect::<Vec<_>>();
        let q_constant = (0..PROGRAM_LANES)
            .map(|_| meta.selector())
            .collect::<Vec<_>>();
        let q_lookup = (0..PROGRAM_LANES)
            .map(|_| meta.complex_selector())
            .collect::<Vec<_>>();
        let table_kind = meta.lookup_table_column();
        let table_x = meta.lookup_table_column();
        let table_y = meta.lookup_table_column();

        for lane in 0..PROGRAM_LANES {
            let q_ma = q_mul_add[lane];
            let q_bool = q_boolean[lane];
            let q_const = q_constant[lane];
            let q_lu = q_lookup[lane];
            let a_col = op_a[lane];
            let b_col = op_b[lane];
            let c_col = op_c[lane];
            let out_col = op_out[lane];
            let kind_col = lookup_kind[lane];
            let fixed_col = fixed_constant[lane];
            meta.create_gate("formal_gat.mul_add_lane", move |meta| {
                let q = meta.query_selector(q_ma);
                let a = meta.query_advice(a_col, halo2_proofs::poly::Rotation::cur());
                let b = meta.query_advice(b_col, halo2_proofs::poly::Rotation::cur());
                let c = meta.query_advice(c_col, halo2_proofs::poly::Rotation::cur());
                let out = meta.query_advice(out_col, halo2_proofs::poly::Rotation::cur());
                vec![q * (a * b + c - out)]
            });
            meta.create_gate("formal_gat.boolean_lane", move |meta| {
                let q = meta.query_selector(q_bool);
                let value = meta.query_advice(a_col, halo2_proofs::poly::Rotation::cur());
                vec![q * value.clone() * (value - Expression::Constant(Fr::ONE))]
            });
            meta.create_gate("formal_gat.constant_lane", move |meta| {
                let q = meta.query_selector(q_const);
                let value = meta.query_advice(a_col, halo2_proofs::poly::Rotation::cur());
                let constant = meta.query_fixed(fixed_col, halo2_proofs::poly::Rotation::cur());
                vec![q * (value - constant)]
            });
            meta.lookup("formal_gat.exact_table_lane", move |meta| {
                let q = meta.query_selector(q_lu);
                let kind = meta.query_advice(kind_col, halo2_proofs::poly::Rotation::cur());
                let x = meta.query_advice(a_col, halo2_proofs::poly::Rotation::cur());
                let y = meta.query_advice(b_col, halo2_proofs::poly::Rotation::cur());
                vec![
                    (q.clone() * kind, table_kind),
                    (q.clone() * x, table_x),
                    (q * y, table_y),
                ]
            });
        }
        let poseidon = PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::
            configure_with_optional_instance(meta, Some(instance));
        Self::Config {
            canonical,
            op_a,
            op_b,
            op_c,
            op_out,
            lookup_kind,
            fixed_constant,
            instance,
            q_mul_add,
            q_boolean,
            q_constant,
            q_lookup,
            table_kind,
            table_x,
            table_y,
            poseidon,
        }
    }

    fn configure(meta: &mut ConstraintSystem<Fr>) -> Self::Config {
        Self::configure_with_params(meta, CircuitParams::default())
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<Fr>,
    ) -> Result<(), Error> {
        load_tables(&self.tables, &config, &mut layouter)?;
        let public_position = self
            .program
            .public_vars
            .iter()
            .enumerate()
            .map(|(position, var)| (*var, position))
            .collect::<HashMap<_, _>>();
        let canonical = layouter.assign_region(
            || "formal_gat.canonical_variables",
            |mut region| {
                let mut cells = Vec::with_capacity(self.program.values.len());
                for (index, value) in self.program.values.iter().enumerate() {
                    let lane = index % PROGRAM_LANES;
                    let row = index / PROGRAM_LANES;
                    let cell = if let Some(position) = public_position.get(&index) {
                        region.assign_advice_from_instance(
                            || "public canonical value",
                            config.instance,
                            *position,
                            config.canonical[lane],
                            row,
                        )
                    } else {
                        region.assign_advice(
                            || "private canonical value",
                            config.canonical[lane],
                            row,
                            || value.map(Value::known).unwrap_or(Value::unknown()),
                        )
                    }?;
                    cells.push(cell);
                }
                Ok(cells)
            },
        )?;

        layouter.assign_region(
            || "formal_gat.fixed_program",
            |mut region| {
                for (index, constraint) in self.program.constraints.iter().enumerate() {
                    let lane = index % PROGRAM_LANES;
                    let row = index / PROGRAM_LANES;
                    match *constraint {
                        Constraint::MulAdd { a, b, c, out } => {
                            config.q_mul_add[lane].enable(&mut region, row)?;
                            canonical[a].copy_advice(
                                || "a",
                                &mut region,
                                config.op_a[lane],
                                row,
                            )?;
                            canonical[b].copy_advice(
                                || "b",
                                &mut region,
                                config.op_b[lane],
                                row,
                            )?;
                            canonical[c].copy_advice(
                                || "c",
                                &mut region,
                                config.op_c[lane],
                                row,
                            )?;
                            canonical[out].copy_advice(
                                || "out",
                                &mut region,
                                config.op_out[lane],
                                row,
                            )?;
                        }
                        Constraint::Boolean { value } => {
                            config.q_boolean[lane].enable(&mut region, row)?;
                            canonical[value].copy_advice(
                                || "boolean",
                                &mut region,
                                config.op_a[lane],
                                row,
                            )?;
                        }
                        Constraint::Constant { value, constant } => {
                            config.q_constant[lane].enable(&mut region, row)?;
                            canonical[value].copy_advice(
                                || "constant value",
                                &mut region,
                                config.op_a[lane],
                                row,
                            )?;
                            region.assign_fixed(
                                || "program constant",
                                config.fixed_constant[lane],
                                row,
                                || Value::known(fr_i128(constant)),
                            )?;
                        }
                        Constraint::Lookup { kind, x, y } => {
                            config.q_lookup[lane].enable(&mut region, row)?;
                            canonical[x].copy_advice(
                                || "lookup x",
                                &mut region,
                                config.op_a[lane],
                                row,
                            )?;
                            canonical[y].copy_advice(
                                || "lookup y",
                                &mut region,
                                config.op_b[lane],
                                row,
                            )?;
                            region.assign_advice(
                                || "lookup kind",
                                config.lookup_kind[lane],
                                row,
                                || Value::known(Fr::from(kind as u64)),
                            )?;
                        }
                    }
                }
                Ok(())
            },
        )?;

        let mut model_tensor: Tensor<ValType<Fr>> = self
            .program
            .model_vars
            .iter()
            .map(|var| ValType::from(canonical[*var].clone()))
            .into();
        model_tensor
            .reshape(&[1, self.program.model_vars.len()])
            .map_err(|_| Error::Synthesis)?;
        let poseidon =
            PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::new(config.poseidon);
        let model_hash_position = self
            .program
            .public_vars
            .len()
            .checked_sub(1)
            .ok_or(Error::Synthesis)?;
        poseidon
            .layout(
                &mut layouter,
                &[ValTensor::from(model_tensor)],
                model_hash_position,
                &mut HashMap::new(),
            )
            .map_err(|_| Error::Synthesis)?;
        Ok(())
    }
}

fn fr_i128(value: i128) -> Fr {
    integer_rep_to_felt(value)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, EZKLError> {
    Ok(serde_json::from_slice(&fs::read(path)?)?)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), EZKLError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn sha256_bytes(bytes: &[u8]) -> String {
    sha256::digest(bytes)
}

fn sha256_file(path: &Path) -> Result<String, EZKLError> {
    Ok(sha256_bytes(&fs::read(path)?))
}

fn formal_domain_hash_fields(domain: &str, fields: &[(&str, &[u8])]) -> String {
    fn append_field(out: &mut Vec<u8>, value: &[u8]) {
        out.extend_from_slice(&(value.len() as u64).to_le_bytes());
        out.extend_from_slice(value);
    }
    let mut canonical = Vec::new();
    append_field(&mut canonical, b"GAT-ZKML/ezkl/canonical/v1");
    append_field(&mut canonical, domain.as_bytes());
    canonical.extend_from_slice(&(fields.len() as u64).to_le_bytes());
    for (role, bytes) in fields {
        append_field(&mut canonical, role.as_bytes());
        append_field(&mut canonical, bytes);
    }
    format!("sha256:{}", sha256_bytes(&canonical))
}

fn formal_domain_hash(domain: &str, role: &str, bytes: &[u8]) -> String {
    formal_domain_hash_fields(domain, &[(role, bytes)])
}

fn encode_u64_vector(values: &[usize]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(8 + values.len() * 8);
    bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
    for &value in values {
        bytes.extend_from_slice(&(value as u64).to_le_bytes());
    }
    bytes
}

fn canonical_graph_hash(
    node_count: usize,
    edge_count: usize,
    graph_count: usize,
    feature_dim: usize,
    class_count: usize,
    indices: &[usize],
    node_ptr_values: &[usize],
    edge_ptr_values: &[usize],
    src_values: &[usize],
    dst_values: &[usize],
    feature_values: &[i128],
) -> Result<String, EZKLError> {
    let mut dimensions = Vec::new();
    for value in [
        node_count,
        node_count,
        edge_count,
        graph_count,
        feature_dim,
        class_count,
    ] {
        dimensions.extend_from_slice(&(value as u64).to_le_bytes());
    }
    let absolute_ids = encode_u64_vector(indices);
    let node_ptr = encode_u64_vector(node_ptr_values);
    let edge_ptr = encode_u64_vector(edge_ptr_values);
    let src = encode_u64_vector(src_values);
    let dst = encode_u64_vector(dst_values);
    let mut features = Vec::with_capacity(8 + feature_values.len() * 8);
    features.extend_from_slice(&(feature_values.len() as u64).to_le_bytes());
    for &value in feature_values {
        let value = i64::try_from(value).map_err(|_| "public feature exceeds canonical i64")?;
        features.extend_from_slice(&value.to_le_bytes());
    }
    Ok(formal_domain_hash_fields(
        "public_graph",
        &[
            ("dimensions", &dimensions),
            ("absolute_ids", &absolute_ids),
            ("node_ptr", &node_ptr),
            ("edge_ptr", &edge_ptr),
            ("src", &src),
            ("dst", &dst),
            ("features_i64", &features),
        ],
    ))
}

fn public_graph_hash(statement: &PublicStatement) -> Result<String, EZKLError> {
    canonical_graph_hash(
        statement.node_count,
        statement.edge_count,
        statement.graph_count,
        statement.feature_dim,
        statement.class_count,
        &statement.indices,
        &statement.node_ptr,
        &statement.edge_ptr,
        &statement.src,
        &statement.dst,
        &statement.t_h,
    )
}

#[derive(Serialize)]
struct QuantCore<'a> {
    rounding_mode: &'a str,
    limb_bits: u32,
    feature_scale: i128,
    hidden_weight_scale: i128,
    hidden_attention_scale: i128,
    hidden_projection_scale: i128,
    delta_scale: i128,
    exp_scale: i128,
    hidden_recovery_num_scale: i128,
    hidden_recovery_den_scale: i128,
    elu_input_scale: i128,
    elu_output_scale: i128,
    agg_scale: i128,
    output_weight_scale: i128,
    output_attention_scale: i128,
    output_projection_scale: i128,
    output_pre_scale: i128,
    output_head_scale: i128,
    output_bias_scale: i128,
    output_scale: i128,
}

fn quant_profile_id(quant: &QuantConfig) -> Result<String, EZKLError> {
    let core = QuantCore {
        rounding_mode: &quant.rounding_mode,
        limb_bits: quant.limb_bits,
        feature_scale: quant.feature_scale,
        hidden_weight_scale: quant.hidden_weight_scale,
        hidden_attention_scale: quant.hidden_attention_scale,
        hidden_projection_scale: quant.hidden_projection_scale,
        delta_scale: quant.delta_scale,
        exp_scale: quant.exp_scale,
        hidden_recovery_num_scale: quant.hidden_recovery_num_scale,
        hidden_recovery_den_scale: quant.hidden_recovery_den_scale,
        elu_input_scale: quant.elu_input_scale,
        elu_output_scale: quant.elu_output_scale,
        agg_scale: quant.agg_scale,
        output_weight_scale: quant.output_weight_scale,
        output_attention_scale: quant.output_attention_scale,
        output_projection_scale: quant.output_projection_scale,
        output_pre_scale: quant.output_pre_scale,
        output_head_scale: quant.output_head_scale,
        output_bias_scale: quant.output_bias_scale,
        output_scale: quant.output_scale,
    };
    Ok(formal_domain_hash(
        "formal_quant_profile",
        "canonical_json",
        &serde_json::to_vec(&core)?,
    ))
}

fn load_security_profile(base: &Path) -> Result<(SecurityProfile, String), EZKLError> {
    let text = fs::read_to_string(base.join("security_profile.json"))?;
    let canonical = text.trim_end_matches(['\r', '\n']);
    let profile: SecurityProfile = serde_json::from_str(canonical)?;
    if serde_json::to_string(&profile)? != canonical
        || profile.profile_kind != "ezkl_native_security_profile_v1"
        || profile.ezkl_version != "local-pinned-source-8a190148c34021961596a057914a9ee8d724580a"
        || profile.halo2_git_commit != "01c88842679b4308e43ae5ed91c4183e861669bd"
        || profile.curve != "BN254"
        || profile.scalar_field != "BN254-Fr"
        || profile.polynomial_commitment != "KZG"
        || profile.multiopen != "SHPLONK"
        || profile.transcript != "Blake2b"
        || profile.challenge != "Challenge255"
        || !profile.setup.contains("trusted-ceremony")
        || profile.native_security_level_claim.is_empty()
        || profile.ceremony_descriptor_kind != "ezkl_trusted_ceremony_srs_v1"
        || profile.ceremony_id.is_empty()
        || !is_tagged_sha256(&profile.source_srs_sha256)
        || profile.source_srs_max_k != DEFAULT_MAX_LOGROWS
    {
        return Err(
            "security profile is non-canonical or disagrees with the pinned native backend".into(),
        );
    }
    let id = formal_domain_hash(
        "ezkl_security_profile",
        "canonical_json",
        canonical.as_bytes(),
    );
    Ok((profile, id))
}

fn validate_quant(quant: &QuantConfig) -> Result<(), EZKLError> {
    if quant.canonical_digest != quant_profile_id(quant)?
        || quant.rounding_mode != "nearest_ties_away"
        || (
            quant.feature_scale,
            quant.hidden_weight_scale,
            quant.hidden_attention_scale,
        ) != (1, 16, 16)
        || (
            quant.hidden_projection_scale,
            quant.delta_scale,
            quant.exp_scale,
        ) != (16, 256, 16)
        || (
            quant.hidden_recovery_num_scale,
            quant.hidden_recovery_den_scale,
        ) != (1, 16)
        || (
            quant.elu_input_scale,
            quant.elu_output_scale,
            quant.agg_scale,
        ) != (16, 16, 16)
        || (quant.output_weight_scale, quant.output_attention_scale) != (16, 1)
        || (quant.output_projection_scale, quant.output_pre_scale) != (256, 16)
        || (
            quant.output_head_scale,
            quant.output_bias_scale,
            quant.output_scale,
        ) != (16, 16, 16)
    {
        return Err(
            "formal quantization profile is not the frozen main-experiment scale chain".into(),
        );
    }
    Ok(())
}

fn canonical_digest<T: Serialize>(value: &T) -> Result<String, EZKLError> {
    Ok(sha256_bytes(&serde_json::to_vec(value)?))
}

fn is_tagged_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn bind_table_spec(tables: TableSpec) -> Result<TableSpec, EZKLError> {
    if tables.generator_id != TABLE_GENERATOR_ID {
        return Err("unregistered formal table generator".into());
    }
    let mut unhashed = tables.clone();
    let supplied = std::mem::take(&mut unhashed.canonical_digest);
    let expected = formal_domain_hash(
        "formal_static_table_profile",
        "canonical_json",
        &serde_json::to_vec(&unhashed)?,
    );
    if supplied != expected || !is_tagged_sha256(&tables.rows_digest) {
        return Err("formal table canonical digest is missing or malformed".into());
    }
    Ok(tables)
}

fn bind_aux_table_spec(tables: AuxTableSpec) -> Result<AuxTableSpec, EZKLError> {
    if tables.generator_id != AUX_TABLE_GENERATOR_ID {
        return Err("unregistered backend auxiliary table generator".into());
    }
    let mut unhashed = tables.clone();
    let supplied = std::mem::take(&mut unhashed.canonical_digest);
    let expected = formal_domain_hash(
        "ezkl_backend_aux_table_profile",
        "canonical_json",
        &serde_json::to_vec(&unhashed)?,
    );
    if supplied != expected || !is_tagged_sha256(&tables.rows_digest) {
        return Err("backend auxiliary table digest is missing or malformed".into());
    }
    Ok(tables)
}

fn validate_table_spec(tables: &TableSpec) -> Result<(), EZKLError> {
    if tables.generator_id != TABLE_GENERATOR_ID
        || tables.profile_version != "main_gatzk_table_profile_v1"
        || tables.rounding_mode != "nearest_ties_away"
        || tables.clipping_rule != "exp_nonnegative_clamp_zero"
        || tables.canonical_order != "kind_then_signed_input_ascending"
        || (
            tables.lrelu_scale,
            tables.elu_input_scale,
            tables.elu_output_scale,
            tables.exp_input_scale,
            tables.exp_output_scale,
        ) != (256, 16, 16, 256, 16)
        || tables.canonical_digest.is_empty()
    {
        return Err("formal table identity is missing or unregistered".into());
    }
    if tables.lrelu_min > 0
        || tables.lrelu_max < 0
        || tables.elu_min > 0
        || tables.elu_max < 0
        || tables.exp_max < 0
    {
        return Err("formal table domains do not cover their required origins".into());
    }
    Ok(())
}

fn validate_aux_table_spec(tables: &AuxTableSpec) -> Result<(), EZKLError> {
    if tables.generator_id != AUX_TABLE_GENERATOR_ID
        || tables.profile_version != "ezkl_backend_aux_table_profile_v1"
        || tables.canonical_order != "kind_then_signed_input_ascending"
        || tables.max_relu_min > 0
        || tables.max_relu_max < 0
        || tables.limb_max <= 0
    {
        return Err("invalid backend auxiliary table profile".into());
    }
    Ok(())
}

fn validate_main_table_domains(
    spec: &GraphSpec,
    quant: &QuantConfig,
    tables: &TableSpec,
    aux: &AuxTableSpec,
) -> Result<(), EZKLError> {
    let configured = 1i128
        .checked_shl(quant.limb_bits)
        .ok_or("range table overflow")?;
    let large = spec.batching_rule == "whole_graph_single"
        && (spec.edge_count >= (1usize << 18)
            || spec
                .node_count
                .checked_mul(spec.feature_dim)
                .ok_or("graph size overflow")?
                >= (1usize << 24));
    let range_size = if large {
        configured.max((1i128 << 23) - 2)
    } else {
        configured
    };
    let (lrelu_min, lrelu_max) = if large {
        (-(3i128 << 18), ((1i128 << 23) - 2) - (3i128 << 18) - 1)
    } else {
        (-configured, configured)
    };
    let elu_bound = if large {
        configured.max(1i128 << 20)
    } else {
        configured
    };
    if aux.limb_max != configured - 1
        || tables.lrelu_min != lrelu_min
        || tables.lrelu_max != lrelu_max
        || tables.elu_min != -elu_bound
        || tables.elu_max != elu_bound
        || tables.exp_max != range_size - 1
    {
        return Err("formal tables do not use the frozen main-experiment domain selection".into());
    }
    Ok(())
}

fn load_canonical_tables(
    base: &Path,
    spec: &TableSpec,
    aux: &AuxTableSpec,
) -> Result<CanonicalTables, EZKLError> {
    bind_table_spec(spec.clone())?;
    bind_aux_table_spec(aux.clone())?;
    validate_table_spec(spec)?;
    validate_aux_table_spec(aux)?;
    let main_payload = fs::read(base.join(&spec.rows_path))?;
    let aux_payload = fs::read(base.join(&aux.rows_path))?;
    let parse =
        |payload: &[u8], domain: &str, digest: &str| -> Result<Vec<(u64, i128, i128)>, EZKLError> {
            if payload.len() < 14
                || &payload[..14] != b"EZKLGATTABLE1\n"
                || (payload.len() - 14) % 24 != 0
            {
                return Err("invalid canonical table row payload".into());
            }
            let actual = formal_domain_hash(domain, "canonical_rows", payload);
            if actual != digest {
                return Err("canonical table row digest mismatch".into());
            }
            let mut rows = Vec::with_capacity((payload.len() - 14) / 24);
            for chunk in payload[14..].chunks_exact(24) {
                let kind = u64::from_le_bytes(chunk[..8].try_into().unwrap());
                let x = i64::from_le_bytes(chunk[8..16].try_into().unwrap()) as i128;
                let y = i64::from_le_bytes(chunk[16..24].try_into().unwrap()) as i128;
                rows.push((kind, x, y));
            }
            Ok(rows)
        };
    let mut rows = parse(&main_payload, "formal_static_table_rows", &spec.rows_digest)?;
    rows.extend(parse(
        &aux_payload,
        "ezkl_backend_aux_table_rows",
        &aux.rows_digest,
    )?);
    rows.sort_by_key(|(kind, x, _)| (*kind, *x));
    let mut expected = [0usize; 6];
    let to_usize = |value: i128| {
        usize::try_from(value).map_err(|_| EZKLError::from("table row count exceeds usize"))
    };
    expected[1] = to_usize(aux.limb_max + 1)?;
    expected[2] = to_usize(spec.lrelu_max - spec.lrelu_min + 1)?;
    expected[3] = to_usize(spec.elu_max - spec.elu_min + 1)?;
    expected[4] = to_usize(spec.exp_max + 1)?;
    expected[5] = to_usize(aux.max_relu_max - aux.max_relu_min + 1)?;
    let mut counts = [0usize; 6];
    for &(kind, x, y) in &rows {
        let index = usize::try_from(kind).map_err(|_| "invalid canonical table kind")?;
        if !(1..=5).contains(&index) {
            return Err("invalid canonical table kind".into());
        }
        let start = match index {
            1 => 0,
            2 => spec.lrelu_min,
            3 => spec.elu_min,
            4 => 0,
            5 => aux.max_relu_min,
            _ => unreachable!(),
        };
        if x != start + counts[index] as i128 {
            return Err("non-canonical table row order".into());
        }
        if index == 1 && y != x
            || index == 2 && y != deterministic_lrelu(x)
            || index == 3 && y != deterministic_elu(x)
            || index == 4 && y != deterministic_exp_neg(x)
            || index == 5 && y != x.max(0)
        {
            return Err(
                "canonical rows disagree with the frozen main-experiment table semantics".into(),
            );
        }
        counts[index] += 1;
    }
    if counts != expected {
        return Err("canonical table row domain mismatch".into());
    }
    Ok(CanonicalTables { rows })
}

fn validate_settings(
    settings: &FormalSettings,
    security_profile_id: &str,
) -> Result<(), EZKLError> {
    if settings.kind != SETTINGS_KIND || settings.settings_id.is_empty() {
        return Err("formal settings identity is missing".into());
    }
    let mut unhashed = settings.clone();
    let supplied = std::mem::take(&mut unhashed.settings_id);
    if canonical_digest(&unhashed)? != supplied {
        return Err("settings identity mismatch".into());
    }
    if settings.security_profile_id != security_profile_id || settings.program_version != PLAN_KIND
    {
        return Err("settings security/program identity mismatch".into());
    }
    Ok(())
}

fn validate_registration(
    plan: &FormalPlan,
    registration: &ModelRegistration,
) -> Result<(), EZKLError> {
    if registration.registration_kind != "ezkl_formal_model_registration_v1"
        || registration.model_arch_id != plan.model_arch_id
        || registration.model_param_id != plan.model_param_id
        || registration.quant_cfg_id != plan.quant_cfg_id
        || registration.canonical_parameter_count == 0
        || registration.canonical_parameter_digest.is_empty()
        || registration.registered_model_hash != plan.registered_model_hash
        || canonical_digest(&registration.parameter_bounds)? != registration.model_bounds_digest
        || registration.model_bounds_digest != plan.ledger.model_bounds_digest
    {
        return Err("canonical model registration identity mismatch".into());
    }
    Ok(())
}

fn fr_hex(value: Fr) -> String {
    hex::encode(value.to_repr().as_ref())
}

fn parse_fr_hex(value: &str) -> Result<Fr, EZKLError> {
    let bytes = hex::decode(value)
        .map_err(|error| EZKLError::from(format!("invalid field hex encoding: {error}")))?;
    let mut repr = <Fr as PrimeField>::Repr::default();
    if bytes.len() != repr.as_ref().len() {
        return Err("invalid field encoding length".into());
    }
    repr.as_mut().copy_from_slice(&bytes);
    Option::<Fr>::from(Fr::from_repr(repr)).ok_or_else(|| "non-canonical field encoding".into())
}

fn round_div_nearest_away(numerator: i128, denominator: i128) -> Result<i128, EZKLError> {
    if denominator <= 0 {
        return Err("SignedRoundDiv denominator must be positive".into());
    }
    let negative = numerator < 0;
    let magnitude = numerator
        .checked_abs()
        .ok_or("SignedRoundDiv magnitude overflow")?;
    let q = (magnitude.checked_mul(2).ok_or("SignedRoundDiv overflow")? + denominator)
        / denominator
            .checked_mul(2)
            .ok_or("SignedRoundDiv overflow")?;
    Ok(if negative { -q } else { q })
}

fn deterministic_lrelu(value: i128) -> i128 {
    if value >= 0 {
        value
    } else {
        -round_div_nearest_away(-value, 5).expect("positive divisor")
    }
}

fn round_f64_nearest_away(value: f64) -> i128 {
    if value >= 0.0 {
        (value + 0.5) as i128
    } else {
        (value - 0.5) as i128
    }
}

fn deterministic_elu(value: i128) -> i128 {
    if value >= 0 {
        value
    } else {
        round_f64_nearest_away(16.0 * ((value as f64 / 16.0).exp() - 1.0))
    }
}

fn deterministic_exp_neg(value: i128) -> i128 {
    round_f64_nearest_away(16.0 * (-(value as f64) / 256.0).exp()).max(0)
}

fn load_tables(
    tables: &CanonicalTables,
    config: &FormalGatConfig,
    layouter: &mut impl Layouter<Fr>,
) -> Result<(), Error> {
    let mut rows = Vec::with_capacity(tables.rows.len() + 1);
    rows.push((0u64, 0i128, 0i128));
    rows.extend(tables.rows.iter().copied());
    layouter.assign_table(
        || "formal_gat.canonical_exact_tables",
        |mut table| {
            for (row, (kind, x, y)) in rows.iter().copied().enumerate() {
                table.assign_cell(
                    || "kind",
                    config.table_kind,
                    row,
                    || Value::known(Fr::from(kind)),
                )?;
                table.assign_cell(|| "x", config.table_x, row, || Value::known(fr_i128(x)))?;
                table.assign_cell(|| "y", config.table_y, row, || Value::known(fr_i128(y)))?;
            }
            Ok(())
        },
    )
}

#[derive(Clone, Debug, Default)]
struct CanonicalInputs {
    features: Vec<i128>,
    tensors: BTreeMap<String, Vec<i128>>,
}

fn read_i64_matrix(path: &Path, rows: usize, columns: usize) -> Result<Vec<i128>, EZKLError> {
    let bytes = fs::read(path)?;
    if bytes.len() < 16 {
        return Err("canonical feature payload is truncated".into());
    }
    let payload_rows = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    let payload_columns = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
    if payload_rows != rows || payload_columns != columns {
        return Err("canonical feature payload shape mismatch".into());
    }
    let count = rows.checked_mul(columns).ok_or("feature shape overflow")?;
    if bytes.len() != 16 + count * 8 {
        return Err("canonical feature payload length mismatch".into());
    }
    Ok((0..count)
        .map(|index| {
            i64::from_le_bytes(bytes[16 + index * 8..24 + index * 8].try_into().unwrap()) as i128
        })
        .collect())
}

fn read_i64_tensors(path: &Path) -> Result<BTreeMap<String, Vec<i128>>, EZKLError> {
    let text = fs::read_to_string(path)?;
    let mut words = text.split_whitespace();
    let mut tensors = BTreeMap::new();
    while let Some(marker) = words.next() {
        if marker != "TENSOR" {
            return Err("invalid canonical model tensor marker".into());
        }
        let name = words
            .next()
            .ok_or("missing canonical tensor name")?
            .to_owned();
        let rank: usize = words
            .next()
            .ok_or("missing canonical tensor rank")?
            .parse()
            .map_err(|error| EZKLError::from(format!("invalid tensor rank: {error}")))?;
        let mut shape_product = 1usize;
        for _ in 0..rank {
            let dimension: usize = words
                .next()
                .ok_or("missing canonical tensor dimension")?
                .parse()
                .map_err(|error| EZKLError::from(format!("invalid tensor dimension: {error}")))?;
            shape_product = shape_product
                .checked_mul(dimension)
                .ok_or("tensor shape overflow")?;
        }
        let count: usize = words
            .next()
            .ok_or("missing canonical tensor length")?
            .parse()
            .map_err(|error| EZKLError::from(format!("invalid tensor length: {error}")))?;
        if count != shape_product {
            return Err("canonical tensor shape/length mismatch".into());
        }
        let mut values = Vec::with_capacity(count);
        for _ in 0..count {
            values.push(
                words
                    .next()
                    .ok_or("truncated canonical tensor")?
                    .parse::<i128>()
                    .map_err(|error| {
                        EZKLError::from(format!("invalid canonical tensor value: {error}"))
                    })?,
            );
        }
        if tensors.insert(name, values).is_some() {
            return Err("duplicate canonical tensor name".into());
        }
    }
    Ok(tensors)
}

fn load_inputs(plan: &FormalPlan, run_dir: &Path) -> Result<CanonicalInputs, EZKLError> {
    Ok(CanonicalInputs {
        features: read_i64_matrix(
            &run_dir.join(&plan.artifacts.features_i64),
            plan.node_count,
            plan.feature_dim,
        )?,
        tensors: read_i64_tensors(&run_dir.join(&plan.artifacts.model_i64))?,
    })
}

fn input_value(
    values: Option<&CanonicalInputs>,
    tensor: &str,
    index: usize,
) -> Result<Option<i128>, EZKLError> {
    match values {
        Some(values) => Ok(Some(
            *values
                .tensors
                .get(tensor)
                .ok_or_else(|| format!("canonical model tensor missing: {tensor}"))?
                .get(index)
                .ok_or_else(|| format!("canonical model tensor index out of range: {tensor}"))?,
        )),
        None => Ok(None),
    }
}

fn matrix(
    rows: usize,
    columns: usize,
    mut value: impl FnMut(usize) -> Result<Var, EZKLError>,
) -> Result<Vec<Vec<Var>>, EZKLError> {
    (0..rows)
        .map(|row| {
            (0..columns)
                .map(|column| value(row * columns + column))
                .collect()
        })
        .collect()
}

fn dot(
    program: &mut FormalProgram,
    left: &[Var],
    right: &[Var],
    zero: Var,
) -> Result<Var, EZKLError> {
    if left.len() != right.len() {
        return Err("dot-product shape mismatch".into());
    }
    let mut accumulator = zero;
    for (&left, &right) in left.iter().zip(right) {
        accumulator = program.mul_add(left, right, accumulator)?;
    }
    Ok(accumulator)
}

fn known_i128(program: &FormalProgram, var: Var) -> Result<i128, EZKLError> {
    program.integer_values[var].ok_or_else(|| "canonical observed trace value is unknown".into())
}

fn abs_i128(value: i128) -> Result<i128, EZKLError> {
    value
        .checked_abs()
        .ok_or_else(|| "observed trace absolute-value overflow".into())
}

fn observe_dot(
    program: &FormalProgram,
    left: &[Var],
    right: &[Var],
) -> Result<(i128, i128, i128), EZKLError> {
    let mut max_mul = 0;
    let mut max_acc = 0;
    let mut accumulator = 0i128;
    for (&left, &right) in left.iter().zip(right) {
        let product = known_i128(program, left)?
            .checked_mul(known_i128(program, right)?)
            .ok_or("observed dot product overflow")?;
        accumulator = accumulator
            .checked_add(product)
            .ok_or("observed dot accumulator overflow")?;
        max_mul = max_mul.max(abs_i128(product)?);
        max_acc = max_acc.max(abs_i128(accumulator)?);
    }
    Ok((max_mul, max_acc, abs_i128(accumulator)?))
}

fn lookup_apply(
    program: &mut FormalProgram,
    kind: LookupKind,
    input: Var,
    value: Option<i128>,
) -> Var {
    let output = program.witness(value);
    program.lookup(kind, input, output);
    output
}

fn frozen_lookup_value(
    tables: Option<&CanonicalTables>,
    kind: LookupKind,
    input: Option<i128>,
) -> Result<Option<i128>, EZKLError> {
    match input {
        None => Ok(None),
        Some(value) => Ok(Some(
            tables
                .ok_or("canonical frozen table rows are required for witness generation")?
                .lookup(kind, value)?,
        )),
    }
}

fn range_decompose(
    program: &mut FormalProgram,
    value: Var,
    bound: i128,
    limb_bits: u32,
    zero: Var,
) -> Result<(), EZKLError> {
    if bound < 0 || limb_bits == 0 || limb_bits >= 63 {
        return Err("invalid range-decomposition parameters".into());
    }
    let limb_count =
        ((128 - (bound as u128 + 1).leading_zeros() as usize).max(1) + limb_bits as usize - 1)
            / limb_bits as usize;
    let base = 1i128.checked_shl(limb_bits).ok_or("limb base overflow")?;
    let known_value = program.integer_values[value];
    if known_value.is_some_and(|value| value < 0 || value > bound) {
        return Err("integer-bound ledger violation while constructing witness".into());
    }
    let mut assign_limbs = |known: Option<i128>| -> Result<Var, EZKLError> {
        let mut reconstructed = zero;
        let mut coefficient = 1i128;
        for index in 0..limb_count {
            let limb_value = known.map(|value| (value >> (index as u32 * limb_bits)) & (base - 1));
            let limb = program.witness(limb_value);
            program.lookup(LookupKind::Limb, limb, limb);
            let coefficient_var = program.constant(coefficient);
            reconstructed = program.mul_add(limb, coefficient_var, reconstructed)?;
            coefficient = coefficient
                .checked_mul(base)
                .ok_or("limb coefficient overflow")?;
        }
        Ok(reconstructed)
    };
    let reconstructed = assign_limbs(known_value)?;
    let slack_value = known_value.map(|value| bound - value);
    let reconstructed_slack = assign_limbs(slack_value)?;
    drop(assign_limbs);
    let one = program.constant(1);
    program.assert_equal(reconstructed, value, one, zero);
    let value_plus_slack = program.add(value, reconstructed_slack, one)?;
    let bound_var = program.constant(bound);
    program.assert_equal(value_plus_slack, bound_var, one, zero);
    Ok(())
}

fn model_var(
    program: &mut FormalProgram,
    value: Option<i128>,
    _bound: i128,
    _limb_bits: u32,
    _zero: Var,
    _one: Var,
) -> Result<Var, EZKLError> {
    let var = program.witness(value);
    program.model_vars.push(var);
    // Protocol 0.5.1: canonical model-family bounds are recomputed and
    // authenticated during trusted setup.  Inference binds the same private
    // model through in-circuit Poseidon H(W), without repeating a range proof
    // for every parameter.
    Ok(var)
}

fn signed_round_div(
    program: &mut FormalProgram,
    x: Var,
    d: Var,
    bounds: &RoundDivBounds,
    limb_bits: u32,
    zero: Var,
    one: Var,
) -> Result<Var, EZKLError> {
    let x_value = program.integer_values[x];
    let d_value = program.integer_values[d];
    if d_value.is_some_and(|value| value <= 0) {
        return Err("SignedRoundDiv dynamic denominator is not positive".into());
    }
    let sigma_value = x_value.map(|value| i128::from(value < 0));
    let a_value = x_value
        .map(|value| value.checked_abs().ok_or("SignedRoundDiv abs overflow"))
        .transpose()?;
    let z_value = a_value.map(|value| i128::from(value == 0));
    let q_value = match (x_value, d_value) {
        (Some(x), Some(d)) => Some(round_div_nearest_away(x, d)?.abs()),
        _ => None,
    };
    let y_value = match (x_value, d_value) {
        (Some(x), Some(d)) => Some(round_div_nearest_away(x, d)?),
        _ => None,
    };
    let r_value = match (a_value, d_value, q_value) {
        (Some(a), Some(d), Some(q)) => Some(2 * a + d - 2 * d * q),
        _ => None,
    };
    let delta_value = match (d_value, r_value) {
        (Some(d), Some(r)) => Some(2 * d - 1 - r),
        _ => None,
    };
    let sigma = program.witness(sigma_value);
    let z = program.witness(z_value);
    let a = program.witness(a_value);
    let q = program.witness(q_value);
    let r = program.witness(r_value);
    let delta = program.witness(delta_value);
    let y = program.witness(y_value);
    let a_inverse = program.field_witness(match a_value {
        Some(0) => Some(Fr::ZERO),
        Some(value) => Option::<Fr>::from(fr_i128(value).invert()),
        None => None,
    });
    let d_inverse = program.field_witness(match d_value {
        Some(value) => Option::<Fr>::from(fr_i128(value).invert()),
        None => None,
    });
    program.boolean(sigma);
    program.boolean(z);
    let minus_two = program.constant(-2);
    let sign = program.mul_add(sigma, minus_two, one)?;
    let signed_a = program.mul(sign, a, zero)?;
    program.assert_equal(signed_a, x, one, zero);
    let signed_q = program.mul(sign, q, zero)?;
    program.assert_equal(signed_q, y, one, zero);
    let za = program.mul(z, a, zero)?;
    program.assert_equal(za, zero, one, zero);
    let a_times_inverse = program.mul(a, a_inverse, zero)?;
    let minus_one = program.constant(-1);
    let one_minus_z = program.mul_add(z, minus_one, one)?;
    program.assert_equal(a_times_inverse, one_minus_z, one, zero);
    let zsigma = program.mul(z, sigma, zero)?;
    program.assert_equal(zsigma, zero, one, zero);
    let zainv = program.mul(z, a_inverse, zero)?;
    program.assert_equal(zainv, zero, one, zero);
    let ddinv = program.mul(d, d_inverse, zero)?;
    program.assert_equal(ddinv, one, one, zero);
    let two = program.constant(2);
    let two_a = program.mul(a, two, zero)?;
    let lhs = program.add(two_a, d, one)?;
    let dq = program.mul(d, q, zero)?;
    let two_dq = program.mul(dq, two, zero)?;
    let rhs = program.add(two_dq, r, one)?;
    program.assert_equal(lhs, rhs, one, zero);
    let r_plus_delta = program.add(r, delta, one)?;
    let two_d = program.mul(d, two, zero)?;
    let rhs_remainder = program.add(two_d, minus_one, one)?;
    program.assert_equal(r_plus_delta, rhs_remainder, one, zero);
    let d_minus_one = program.add(d, minus_one, one)?;
    range_decompose(program, a, bounds.a, limb_bits, zero)?;
    range_decompose(program, q, bounds.q, limb_bits, zero)?;
    range_decompose(program, r, bounds.r, limb_bits, zero)?;
    range_decompose(program, delta, bounds.delta, limb_bits, zero)?;
    range_decompose(program, d_minus_one, bounds.d - 1, limb_bits, zero)?;
    let d_min = program.constant(bounds.d_min);
    let neg_d_min = program.mul(d_min, minus_one, zero)?;
    let d_above_min = program.add(d, neg_d_min, one)?;
    range_decompose(
        program,
        d_above_min,
        bounds.d - bounds.d_min,
        limb_bits,
        zero,
    )?;
    Ok(y)
}

fn matrix_projection(
    program: &mut FormalProgram,
    input: &[Vec<Var>],
    weights: &[Vec<Var>],
    zero: Var,
) -> Result<Vec<Vec<Var>>, EZKLError> {
    if input.is_empty() || weights.is_empty() || input[0].len() != weights.len() {
        return Err("projection matrix shape mismatch".into());
    }
    let output_width = weights[0].len();
    (0..input.len())
        .map(|row| {
            (0..output_width)
                .map(|column| {
                    let right = weights
                        .iter()
                        .map(|weights| weights[column])
                        .collect::<Vec<_>>();
                    dot(program, &input[row], &right, zero)
                })
                .collect()
        })
        .collect()
}

fn head_forward(
    program: &mut FormalProgram,
    plan: &FormalPlan,
    label: &str,
    input: &[Vec<Var>],
    projection: &[Vec<Var>],
    a_src: &[Var],
    a_dst: &[Var],
    hidden: bool,
    recovery_bounds: &RoundDivBounds,
    tables: Option<&CanonicalTables>,
    zero: Var,
    one: Var,
) -> Result<Vec<Vec<Var>>, EZKLError> {
    let observing = input
        .iter()
        .flatten()
        .chain(projection.iter().flatten())
        .chain(a_src.iter())
        .chain(a_dst.iter())
        .all(|var| program.integer_values[*var].is_some());
    let mut observed = ObservedFamily {
        input_dim: input.first().map(Vec::len).unwrap_or(0),
        output_dim: projection.first().map(Vec::len).unwrap_or(0),
        recovery_d_min: i128::MAX,
        ..Default::default()
    };
    if observing {
        for &var in input.iter().flatten() {
            observed.h = observed.h.max(abs_i128(known_i128(program, var)?)?);
        }
        for &var in projection.iter().flatten() {
            observed.w = observed.w.max(abs_i128(known_i128(program, var)?)?);
        }
        for &var in a_src {
            observed.a_src = observed.a_src.max(abs_i128(known_i128(program, var)?)?);
        }
        for &var in a_dst {
            observed.a_dst = observed.a_dst.max(abs_i128(known_i128(program, var)?)?);
        }
        for row in input {
            for column in 0..observed.output_dim {
                let right = projection
                    .iter()
                    .map(|weights| weights[column])
                    .collect::<Vec<_>>();
                let (mul, acc, output) = observe_dot(program, row, &right)?;
                observed.projection_mul = observed.projection_mul.max(mul);
                observed.projection_acc = observed.projection_acc.max(acc);
                observed.projection = observed.projection.max(output);
            }
        }
    }
    let h_prime = matrix_projection(program, input, projection, zero)?;
    let e_src = h_prime
        .iter()
        .map(|row| dot(program, row, a_src, zero))
        .collect::<Result<Vec<_>, _>>()?;
    let e_dst = h_prime
        .iter()
        .map(|row| dot(program, row, a_dst, zero))
        .collect::<Result<Vec<_>, _>>()?;
    if observing {
        for row in &h_prime {
            let (src_mul, src_acc, src_out) = observe_dot(program, row, a_src)?;
            let (dst_mul, dst_acc, dst_out) = observe_dot(program, row, a_dst)?;
            observed.attention_mul = observed.attention_mul.max(src_mul).max(dst_mul);
            observed.attention_acc = observed.attention_acc.max(src_acc).max(dst_acc);
            observed.e_src = observed.e_src.max(src_out);
            observed.e_dst = observed.e_dst.max(dst_out);
        }
    }
    let mut z = Vec::with_capacity(plan.edge_count);
    for edge in 0..plan.edge_count {
        let score = program.add(e_src[plan.src[edge]], e_dst[plan.dst[edge]], one)?;
        z.push(lookup_apply(
            program,
            LookupKind::LeakyRelu,
            score,
            frozen_lookup_value(tables, LookupKind::LeakyRelu, program.integer_values[score])?,
        ));
        if observing {
            observed.score = observed.score.max(abs_i128(known_i128(program, score)?)?);
            observed.z = observed
                .z
                .max(abs_i128(known_i128(program, *z.last().unwrap())?)?);
        }
    }
    let mut groups = vec![Vec::new(); plan.node_count];
    for edge in 0..plan.edge_count {
        groups[plan.dst[edge]].push(edge);
    }
    let mut max = Vec::with_capacity(plan.node_count);
    for edges in &groups {
        if edges.is_empty() {
            return Err("topology contains a node without an incoming edge".into());
        }
        let max_value = if edges
            .iter()
            .all(|edge| program.integer_values[z[*edge]].is_some())
        {
            Some(
                edges
                    .iter()
                    .map(|edge| program.integer_values[z[*edge]].unwrap())
                    .max()
                    .unwrap(),
            )
        } else {
            None
        };
        let max_var = program.witness(max_value);
        let mut y_sum = zero;
        for edge in edges {
            let minus_one = program.constant(-1);
            let m_minus_one = program.add(max_var, minus_one, one)?;
            let neg_m_minus_one = program.mul(m_minus_one, minus_one, zero)?;
            let w = program.add(z[*edge], neg_m_minus_one, one)?;
            let y = lookup_apply(
                program,
                LookupKind::MaxRelu,
                w,
                frozen_lookup_value(tables, LookupKind::MaxRelu, program.integer_values[w])?,
            );
            program.boolean(y);
            y_sum = program.add(y_sum, y, one)?;
        }
        let minus_one = program.constant(-1);
        let neg_sum = program.mul(y_sum, minus_one, zero)?;
        let one_minus_sum = program.add(one, neg_sum, one)?;
        let max_zero = lookup_apply(
            program,
            LookupKind::MaxRelu,
            one_minus_sum,
            frozen_lookup_value(
                tables,
                LookupKind::MaxRelu,
                program.integer_values[one_minus_sum],
            )?,
        );
        program.assert_equal(max_zero, zero, one, zero);
        max.push(max_var);
        if observing {
            observed.m = observed.m.max(abs_i128(known_i128(program, max_var)?)?);
        }
    }
    let mut u = Vec::with_capacity(plan.edge_count);
    for edge in 0..plan.edge_count {
        let minus_one = program.constant(-1);
        let neg_z = program.mul(z[edge], minus_one, zero)?;
        let delta = program.add(max[plan.dst[edge]], neg_z, one)?;
        u.push(lookup_apply(
            program,
            LookupKind::ExpNeg,
            delta,
            frozen_lookup_value(tables, LookupKind::ExpNeg, program.integer_values[delta])?,
        ));
        if observing {
            observed.delta = observed.delta.max(known_i128(program, delta)?);
            observed.exp = observed.exp.max(known_i128(program, *u.last().unwrap())?);
        }
    }
    let mut sum = vec![zero; plan.node_count];
    for edge in 0..plan.edge_count {
        let dst = plan.dst[edge];
        sum[dst] = program.add(sum[dst], u[edge], one)?;
        if observing {
            observed.sum = observed.sum.max(known_i128(program, sum[dst])?);
        }
    }
    let inverse = sum
        .iter()
        .map(|sum| {
            program.field_witness(
                program.values[*sum].and_then(|value| Option::<Fr>::from(value.invert())),
            )
        })
        .collect::<Vec<_>>();
    for node in 0..plan.node_count {
        let product = program.mul(sum[node], inverse[node], zero)?;
        program.assert_equal(product, one, one, zero);
    }
    let alpha = (0..plan.edge_count)
        .map(|edge| program.mul(u[edge], inverse[plan.dst[edge]], zero))
        .collect::<Result<Vec<_>, _>>()?;
    let width = h_prime[0].len();
    let mut field_aggregate = vec![vec![zero; width]; plan.node_count];
    let mut integer_numerator = vec![vec![zero; width]; plan.node_count];
    for edge in 0..plan.edge_count {
        let src = plan.src[edge];
        let dst = plan.dst[edge];
        for column in 0..width {
            let term = program.mul(alpha[edge], h_prime[src][column], zero)?;
            field_aggregate[dst][column] = program.add(field_aggregate[dst][column], term, one)?;
            let integer_term = program.mul(u[edge], h_prime[src][column], zero)?;
            integer_numerator[dst][column] =
                program.add(integer_numerator[dst][column], integer_term, one)?;
            if observing {
                observed.aggregate_mul = observed
                    .aggregate_mul
                    .max(abs_i128(known_i128(program, integer_term)?)?);
                observed.aggregate_acc = observed.aggregate_acc.max(abs_i128(known_i128(
                    program,
                    integer_numerator[dst][column],
                )?)?);
                observed.aggregate_num = observed.aggregate_acc;
            }
        }
    }
    let numerator_scale = program.constant(if hidden {
        plan.quant.hidden_recovery_num_scale
    } else {
        plan.quant.output_pre_scale
    });
    let projection_scale = program.constant(if hidden {
        plan.quant.hidden_recovery_den_scale
    } else {
        plan.quant.output_projection_scale
    });
    let mut output = vec![vec![zero; width]; plan.node_count];
    for node in 0..plan.node_count {
        for column in 0..width {
            let consistency = program.mul(sum[node], field_aggregate[node][column], zero)?;
            program.assert_equal(consistency, integer_numerator[node][column], one, zero);
            let dividend = program.mul(numerator_scale, integer_numerator[node][column], zero)?;
            let denominator = program.mul(projection_scale, sum[node], zero)?;
            let recovered = signed_round_div(
                program,
                dividend,
                denominator,
                recovery_bounds,
                plan.quant.limb_bits,
                zero,
                one,
            )?;
            if observing {
                observed.recovery_a = observed
                    .recovery_a
                    .max(abs_i128(known_i128(program, dividend)?)?);
                let actual_d = known_i128(program, denominator)?;
                observed.recovery_d_min = observed.recovery_d_min.min(actual_d);
                observed.recovery_d = observed.recovery_d.max(actual_d);
                observed.recovery_q = observed
                    .recovery_q
                    .max(abs_i128(known_i128(program, recovered)?)?);
            }
            output[node][column] = if hidden {
                lookup_apply(
                    program,
                    LookupKind::Elu,
                    recovered,
                    frozen_lookup_value(
                        tables,
                        LookupKind::Elu,
                        program.integer_values[recovered],
                    )?,
                )
            } else {
                recovered
            };
            if observing {
                observed.output_bound = observed
                    .output_bound
                    .max(abs_i128(known_i128(program, output[node][column])?)?);
            }
        }
    }
    if observing {
        program.observed_families.insert(label.to_owned(), observed);
    }
    Ok(output)
}

fn family_bounds<'a>(plan: &'a FormalPlan, label: &str) -> Result<&'a BoundFamily, EZKLError> {
    plan.ledger
        .families
        .iter()
        .find(|family| family.label == label)
        .ok_or_else(|| format!("missing integer-bound family: {label}").into())
}

fn build_program(
    plan: &FormalPlan,
    inputs: Option<&CanonicalInputs>,
    tables: Option<&CanonicalTables>,
) -> Result<FormalProgram, EZKLError> {
    validate_plan(plan)?;
    let mut program = FormalProgram::default();
    let zero = program.constant(0);
    let one = program.constant(1);
    let mut hidden = matrix(plan.node_count, plan.feature_dim, |index| {
        let value = inputs.map(|inputs| inputs.features[index]);
        Ok(program.public(value))
    })?;

    let mut hidden_specs = plan.hidden_heads.clone();
    hidden_specs.sort_by_key(|head| (head.layer_index, head.local_head_index));
    for layer in 0..plan.hidden_profile.len() {
        let profile = &plan.hidden_profile[layer];
        let input_width = plan.d_in_profile[layer];
        if hidden.first().map(Vec::len) != Some(input_width) {
            return Err("hidden layer input width mismatch".into());
        }
        let heads = hidden_specs
            .iter()
            .filter(|head| head.layer_index == layer)
            .collect::<Vec<_>>();
        if heads.len() != profile.head_count {
            return Err("hidden head registry mismatch".into());
        }
        let mut outputs = Vec::with_capacity(profile.head_count);
        for head in heads {
            let label = format!("hidden.layer_{layer}.head_{}", head.local_head_index);
            let bounds = family_bounds(plan, &label)?;
            let weights = matrix(input_width, profile.head_dim, |index| {
                let value = input_value(inputs, &head.seq_kernel, index)?;
                model_var(
                    &mut program,
                    value,
                    bounds.w,
                    plan.quant.limb_bits,
                    zero,
                    one,
                )
            })?;
            let a_src = (0..profile.head_dim)
                .map(|index| {
                    model_var(
                        &mut program,
                        input_value(inputs, &head.attn_src_kernel, index)?,
                        bounds.a_src,
                        plan.quant.limb_bits,
                        zero,
                        one,
                    )
                })
                .collect::<Result<Vec<_>, EZKLError>>()?;
            let a_dst = (0..profile.head_dim)
                .map(|index| {
                    model_var(
                        &mut program,
                        input_value(inputs, &head.attn_dst_kernel, index)?,
                        bounds.a_dst,
                        plan.quant.limb_bits,
                        zero,
                        one,
                    )
                })
                .collect::<Result<Vec<_>, EZKLError>>()?;
            outputs.push(head_forward(
                &mut program,
                plan,
                &label,
                &hidden,
                &weights,
                &a_src,
                &a_dst,
                true,
                &bounds.recovery,
                tables,
                zero,
                one,
            )?);
        }
        hidden = (0..plan.node_count)
            .map(|node| {
                outputs
                    .iter()
                    .flat_map(|head| head[node].iter().copied())
                    .collect::<Vec<_>>()
            })
            .collect();
    }

    let output_width = hidden.first().map(Vec::len).ok_or("empty hidden output")?;
    let mut output_specs = plan.output_heads.clone();
    output_specs.sort_by_key(|head| head.head_index);
    if output_specs.len() != plan.k_out {
        return Err("output head registry mismatch".into());
    }
    let mut output_heads = Vec::with_capacity(plan.k_out);
    for head in &output_specs {
        let label = format!("output.head_{}", head.head_index);
        let bounds = family_bounds(plan, &label)?;
        let weights = matrix(output_width, plan.class_count, |index| {
            model_var(
                &mut program,
                input_value(inputs, &head.seq_kernel, index)?,
                bounds.w,
                plan.quant.limb_bits,
                zero,
                one,
            )
        })?;
        let a_src = (0..plan.class_count)
            .map(|index| {
                model_var(
                    &mut program,
                    input_value(inputs, &head.attn_src_kernel, index)?,
                    bounds.a_src,
                    plan.quant.limb_bits,
                    zero,
                    one,
                )
            })
            .collect::<Result<Vec<_>, EZKLError>>()?;
        let a_dst = (0..plan.class_count)
            .map(|index| {
                model_var(
                    &mut program,
                    input_value(inputs, &head.attn_dst_kernel, index)?,
                    bounds.a_dst,
                    plan.quant.limb_bits,
                    zero,
                    one,
                )
            })
            .collect::<Result<Vec<_>, EZKLError>>()?;
        let recovered = head_forward(
            &mut program,
            plan,
            &label,
            &hidden,
            &weights,
            &a_src,
            &a_dst,
            false,
            &bounds.recovery,
            tables,
            zero,
            one,
        )?;
        let output_head_scale = program.constant(plan.quant.output_head_scale);
        let output_pre_scale = program.constant(plan.quant.output_pre_scale);
        let mut with_bias = vec![vec![zero; plan.class_count]; plan.node_count];
        let mut observed_bias = 0i128;
        let mut observed_alignment_a = 0i128;
        let mut observed_alignment_q = 0i128;
        let mut observed_output_bound = 0i128;
        for class in 0..plan.class_count {
            let bias = model_var(
                &mut program,
                input_value(inputs, &head.output_bias, class)?,
                bounds.bias,
                plan.quant.limb_bits,
                zero,
                one,
            )?;
            if inputs.is_some() {
                observed_bias = observed_bias.max(abs_i128(known_i128(&program, bias)?)?);
            }
            for node in 0..plan.node_count {
                let dividend = program.mul(output_head_scale, recovered[node][class], zero)?;
                let aligned = signed_round_div(
                    &mut program,
                    dividend,
                    output_pre_scale,
                    bounds
                        .alignment
                        .as_ref()
                        .ok_or("output family is missing alignment division bounds")?,
                    plan.quant.limb_bits,
                    zero,
                    one,
                )?;
                with_bias[node][class] = program.add(aligned, bias, one)?;
                if inputs.is_some() {
                    observed_alignment_a =
                        observed_alignment_a.max(abs_i128(known_i128(&program, dividend)?)?);
                    observed_alignment_q =
                        observed_alignment_q.max(abs_i128(known_i128(&program, aligned)?)?);
                    observed_output_bound = observed_output_bound
                        .max(abs_i128(known_i128(&program, with_bias[node][class])?)?);
                }
            }
        }
        if let Some(observed) = program.observed_families.get_mut(&label) {
            observed.bias = observed_bias;
            observed.alignment_a = observed_alignment_a;
            observed.alignment_q = observed_alignment_q;
            observed.output_bound = observed_output_bound;
        }
        output_heads.push(with_bias);
    }
    let head_count = program.constant(plan.k_out as i128);
    let mut logits = Vec::with_capacity(plan.node_count * plan.class_count);
    for node in 0..plan.node_count {
        for class in 0..plan.class_count {
            let mut sum = zero;
            for head in &output_heads {
                sum = program.add(sum, head[node][class], one)?;
            }
            let logit = signed_round_div(
                &mut program,
                sum,
                head_count,
                &plan.ledger.final_average,
                plan.quant.limb_bits,
                zero,
                one,
            )?;
            program.public_vars.push(logit);
            program.output_vars.push(logit);
            logits.push(logit);
        }
    }
    let model_hash = if program
        .model_vars
        .iter()
        .all(|var| program.values[*var].is_some())
    {
        let message = program
            .model_vars
            .iter()
            .map(|var| program.values[*var].unwrap())
            .collect::<Vec<_>>();
        Some(PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::run(message)?[0][0])
    } else {
        None
    };
    let model_hash_var = program.public_field(model_hash);
    if let Some(hash) = model_hash {
        if !plan.registered_model_hash.is_empty() && fr_hex(hash) != plan.registered_model_hash {
            return Err(
                "canonical private model does not match registered in-circuit Poseidon hash".into(),
            );
        }
    }
    let _ = model_hash_var;
    Ok(program)
}

fn validate_topology(plan: &FormalPlan) -> Result<(), EZKLError> {
    if plan.node_count == 0 || plan.edge_count == 0 || plan.graph_count == 0 {
        return Err("empty formal GAT instance".into());
    }
    if plan.src.len() != plan.edge_count || plan.dst.len() != plan.edge_count {
        return Err("topology edge length mismatch".into());
    }
    if plan.absolute_ids != (0..plan.node_count).collect::<Vec<_>>() {
        return Err("formal full-dataset route requires canonical I=[0,N) node identities".into());
    }
    if plan.node_ptr.len() != plan.graph_count + 1 || plan.edge_ptr.len() != plan.graph_count + 1 {
        return Err("batch pointer length mismatch".into());
    }
    if plan.node_ptr.first() != Some(&0)
        || plan.node_ptr.last() != Some(&plan.node_count)
        || plan.edge_ptr.first() != Some(&0)
        || plan.edge_ptr.last() != Some(&plan.edge_count)
    {
        return Err("batch pointer endpoints are invalid".into());
    }
    if plan.node_ptr.windows(2).any(|pair| pair[0] >= pair[1])
        || plan.edge_ptr.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return Err("batch pointers are not strictly increasing".into());
    }
    let mut graph_for_node = vec![0usize; plan.node_count];
    for graph in 0..plan.graph_count {
        for node in plan.node_ptr[graph]..plan.node_ptr[graph + 1] {
            graph_for_node[node] = graph;
        }
        let mut previous_dst = None;
        for edge in plan.edge_ptr[graph]..plan.edge_ptr[graph + 1] {
            let src = plan.src[edge];
            let dst = plan.dst[edge];
            if src >= plan.node_count
                || dst >= plan.node_count
                || graph_for_node[src] != graph
                || graph_for_node[dst] != graph
            {
                return Err("cross-graph or out-of-range edge".into());
            }
            if previous_dst.is_some_and(|previous| previous > dst) {
                return Err(
                    "edges are not stable nondecreasing by destination in graph block".into(),
                );
            }
            previous_dst = Some(dst);
        }
    }
    let mut degree = vec![0usize; plan.node_count];
    for dst in &plan.dst {
        degree[*dst] += 1;
    }
    if degree.iter().any(|degree| *degree == 0) {
        return Err("self-loop/in-degree invariant failed".into());
    }
    Ok(())
}

fn bound_add(left: i128, right: i128, label: &str) -> Result<i128, EZKLError> {
    left.checked_add(right)
        .ok_or_else(|| format!("integer-bound overflow in {label}").into())
}

fn bound_mul(left: i128, right: i128, label: &str) -> Result<i128, EZKLError> {
    left.checked_mul(right)
        .ok_or_else(|| format!("integer-bound overflow in {label}").into())
}

fn nearest_quotient_bound(a: i128, d_min: i128) -> Result<i128, EZKLError> {
    if a < 0 || d_min <= 0 {
        return Err("invalid SignedRoundDiv quotient-bound inputs".into());
    }
    let numerator = bound_add(
        bound_mul(2, a, "round quotient 2a")?,
        d_min,
        "round quotient",
    )?;
    Ok(numerator / bound_mul(2, d_min, "round quotient 2d")?)
}

fn validate_round_div_bounds(
    bounds: &RoundDivBounds,
    derived_a: i128,
    derived_d_min: i128,
    derived_d: i128,
    label: &str,
) -> Result<(), EZKLError> {
    if Fr::NUM_BITS < 254 {
        return Err("unexpected backend scalar field width".into());
    }
    if derived_a < 0 || derived_d_min <= 0 || derived_d < derived_d_min {
        return Err(format!("invalid derived division bounds: {label}").into());
    }
    let derived_q = nearest_quotient_bound(derived_a, derived_d_min)?;
    let derived_remainder = bound_add(bound_mul(2, derived_d, label)?, -1, label)?;
    if bounds.a < derived_a
        || bounds.q < derived_q
        || bounds.r < derived_remainder
        || bounds.delta < derived_remainder
        || bounds.d_min > derived_d_min
        || bounds.d < derived_d
        || bounds.d_min <= 0
        || bounds.d < bounds.d_min
    {
        return Err(format!("undersized SignedRoundDiv certificate: {label}").into());
    }
    if bounds.q < nearest_quotient_bound(bounds.a, bounds.d_min)? {
        return Err(format!(
            "division quotient certificate is not closed under supplied bounds: {label}"
        )
        .into());
    }
    let lhs = bound_add(bound_mul(2, bounds.a, label)?, bounds.d, label)?;
    let mul_dq = bound_mul(bounds.d, bounds.q, label)?;
    let rhs = bound_add(bound_mul(2, mul_dq, label)?, bounds.r, label)?;
    let two_d_minus_one = bound_add(bound_mul(2, bounds.d, label)?, -1, label)?;
    if bounds.lhs_2a_plus_d < lhs
        || bounds.mul_dq < mul_dq
        || bounds.rhs_2dq_plus_r < rhs
        || bounds.two_d_minus_one < two_d_minus_one
    {
        return Err(format!("undersized SignedRoundDiv intermediate certificate: {label}").into());
    }
    // Every non-negative checked i128 value is < 2^127, while BN254 Fr's
    // signed-canonical half interval is > 2^252. Checked arithmetic above is
    // therefore an explicit strict field-fit proof, not a boolean assertion.
    Ok(())
}

fn validate_round_div_exact(
    bounds: &RoundDivBounds,
    a: i128,
    d_min: i128,
    d: i128,
    label: &str,
) -> Result<(), EZKLError> {
    validate_round_div_bounds(bounds, a, d_min, d, label)?;
    let q = nearest_quotient_bound(a, d_min)?;
    let r = bound_add(bound_mul(2, d, label)?, -1, label)?;
    let lhs = bound_add(bound_mul(2, a, label)?, d, label)?;
    let mul_dq = bound_mul(d, q, label)?;
    let rhs = bound_add(bound_mul(2, mul_dq, label)?, r, label)?;
    if bounds.a != a
        || bounds.q != q
        || bounds.r != r
        || bounds.delta != r
        || bounds.d_min != d_min
        || bounds.d != d
        || bounds.lhs_2a_plus_d != lhs
        || bounds.mul_dq != mul_dq
        || bounds.rhs_2dq_plus_r != rhs
        || bounds.two_d_minus_one != r
    {
        return Err(format!("non-canonical conservative SignedRoundDiv ledger: {label}").into());
    }
    Ok(())
}

fn validate_plan(plan: &FormalPlan) -> Result<(), EZKLError> {
    if plan.kind != PLAN_KIND || plan.program_version != PLAN_KIND {
        return Err("not a formal fixed GAT program".into());
    }
    if [
        &plan.graph_hash,
        &plan.checkpoint_hash,
        &plan.metadata_hash,
        &plan.shape_hash,
        &plan.quant_cfg_id,
        &plan.static_tables_id,
        &plan.backend_aux_tables_id,
        &plan.security_profile_id,
    ]
    .iter()
    .any(|value| !is_tagged_sha256(value))
    {
        return Err("formal public identity is not a domain-separated SHA-256 digest".into());
    }
    if plan.layer_count < 2
        || plan.layer_count != plan.hidden_profile.len() + 1
        || plan.d_in_profile.len() != plan.hidden_profile.len()
        || plan.k_out == 0
        || plan.class_count == 0
    {
        return Err("invalid formal GAT architecture".into());
    }
    validate_quant(&plan.quant)?;
    if plan.quant_cfg_id != plan.quant.canonical_digest
        || plan.static_tables_id != plan.tables.canonical_digest
        || plan.backend_aux_tables_id != plan.aux_tables.canonical_digest
    {
        return Err("compiled plan quant/table identity mismatch".into());
    }
    if plan.quant.limb_bits == 0
        || plan.quant.limb_bits >= 63
        || plan.aux_tables.limb_max
            != (1i128
                .checked_shl(plan.quant.limb_bits)
                .ok_or("limb table width overflow")?
                - 1)
        || [
            plan.quant.feature_scale,
            plan.quant.hidden_weight_scale,
            plan.quant.hidden_attention_scale,
            plan.quant.hidden_projection_scale,
            plan.quant.delta_scale,
            plan.quant.exp_scale,
            plan.quant.hidden_recovery_num_scale,
            plan.quant.hidden_recovery_den_scale,
            plan.quant.elu_input_scale,
            plan.quant.elu_output_scale,
            plan.quant.agg_scale,
            plan.quant.output_weight_scale,
            plan.quant.output_attention_scale,
            plan.quant.output_projection_scale,
            plan.quant.output_pre_scale,
            plan.quant.output_head_scale,
            plan.quant.output_bias_scale,
        ]
        .iter()
        .any(|scale| *scale <= 0)
    {
        return Err("invalid formal integer quantization chain".into());
    }
    if plan.ledger.ledger_kind != "ezkl_integer_bound_ledger_v2"
        || plan.ledger.field_modulus != "bn254_fr"
        || !plan.ledger.no_wrap_verified
        || plan.ledger.max_in_degree == 0
    {
        return Err("integer bound ledger is absent or not field-fit verified".into());
    }
    if plan.security_profile_id
        != formal_domain_hash(
            "ezkl_security_profile",
            "canonical_json",
            &serde_json::to_vec(&plan.security_profile)?,
        )
    {
        return Err("unexpected ezkl/Halo2 security profile".into());
    }
    validate_table_spec(&plan.tables)?;
    validate_aux_table_spec(&plan.aux_tables)?;
    if !plan.compiled_circuit_id.is_empty() {
        let mut unhashed = plan.clone();
        let supplied = std::mem::take(&mut unhashed.compiled_circuit_id);
        if canonical_digest(&unhashed)? != supplied {
            return Err("compiled circuit identity mismatch".into());
        }
    }
    for name in [
        "feature",
        "weight",
        "hidden_attention",
        "output_attention",
        "output_bias",
    ] {
        let bound = plan
            .model_parameter_bounds
            .get(name)
            .copied()
            .ok_or_else(|| EZKLError::from(format!("missing registered bound: {name}")))?;
        if bound < 0 || bound > i64::MAX as i128 {
            return Err("registered model/data bound is outside the no-wrap profile".into());
        }
    }
    if plan.ledger.families.len() != plan.hidden_heads.len() + plan.output_heads.len() {
        return Err("integer ledger does not cover every attention head".into());
    }
    validate_topology(plan)?;
    let mut degree = vec![0usize; plan.node_count];
    for &dst in &plan.dst {
        degree[dst] += 1;
    }
    if degree.into_iter().max() != Some(plan.ledger.max_in_degree) {
        return Err("integer ledger maximum degree does not match fixed topology".into());
    }
    let registered_weight = plan.model_parameter_bounds["weight"];
    let registered_hidden_attention = plan.model_parameter_bounds["hidden_attention"];
    let registered_output_attention = plan.model_parameter_bounds["output_attention"];
    let registered_output_bias = plan.model_parameter_bounds["output_bias"];
    let mut seen = std::collections::BTreeSet::new();
    let mut scalar_hidden_h = plan.model_parameter_bounds["feature"];
    let mut hidden_specs = plan.hidden_heads.clone();
    hidden_specs.sort_by_key(|head| (head.layer_index, head.local_head_index));
    for layer in 0..plan.hidden_profile.len() {
        let profile = &plan.hidden_profile[layer];
        for head in hidden_specs.iter().filter(|head| head.layer_index == layer) {
            let label = format!("hidden.layer_{layer}.head_{}", head.local_head_index);
            let family = family_bounds(plan, &label)?;
            if !seen.insert(label.clone())
                || family.input_dim != plan.d_in_profile[layer]
                || family.output_dim != profile.head_dim
                || family.h > i64::MAX as i128
                || family.w > registered_weight
                || family.a_src > registered_hidden_attention
                || family.a_dst > registered_hidden_attention
                || [family.h, family.w, family.a_src, family.a_dst]
                    .iter()
                    .any(|bound| *bound < 0)
            {
                return Err(format!("invalid hidden family registration: {label}").into());
            }
            // Protocol 0.4.5 mandatory scalar family gate.  The tighter
            // canonical-trace certificate below is additional evidence and
            // must never replace this conservative B_H/B_W derivation.
            let scalar_projection = bound_mul(
                family.input_dim as i128,
                bound_mul(scalar_hidden_h, family.w, &label)?,
                &label,
            )?;
            let scalar_e_src = bound_mul(
                family.output_dim as i128,
                bound_mul(scalar_projection, family.a_src, &label)?,
                &label,
            )?;
            let scalar_e_dst = bound_mul(
                family.output_dim as i128,
                bound_mul(scalar_projection, family.a_dst, &label)?,
                &label,
            )?;
            let scalar_score = bound_add(scalar_e_src, scalar_e_dst, &label)?;
            let scalar_z = deterministic_lrelu(-scalar_score)
                .checked_abs()
                .ok_or("scalar LReLU bound overflow")?
                .max(deterministic_lrelu(scalar_score).abs());
            let scalar_delta = bound_mul(2, scalar_z, &label)?;
            let scalar_sum = bound_mul(plan.ledger.max_in_degree as i128, 16, &label)?;
            let scalar_projection_mul = bound_mul(scalar_hidden_h, family.w, &label)?;
            let scalar_attention_mul = bound_mul(scalar_projection, family.a_src, &label)?
                .max(bound_mul(scalar_projection, family.a_dst, &label)?);
            let scalar_aggregate_mul = bound_mul(16, scalar_projection, &label)?;
            let scalar_aggregate = bound_mul(scalar_sum, scalar_projection, &label)?;
            if family.scalar_h != scalar_hidden_h
                || family.scalar_projection != scalar_projection
                || family.scalar_e_src != scalar_e_src
                || family.scalar_e_dst != scalar_e_dst
                || family.scalar_score != scalar_score
                || family.scalar_z != scalar_z
                || family.scalar_delta != scalar_delta
                || plan.tables.lrelu_min > -scalar_score
                || plan.tables.lrelu_max < scalar_score
                || plan.tables.exp_max < scalar_delta
            {
                return Err(
                    format!("protocol 0.4.5 scalar family/table gate failed: {label}").into(),
                );
            }
            if family.h != scalar_hidden_h
                || family.projection_mul != scalar_projection_mul
                || family.projection_acc != scalar_projection
                || family.projection != scalar_projection
                || family.attention_mul != scalar_attention_mul
                || family.attention_acc != scalar_e_src.max(scalar_e_dst)
                || family.e_src != scalar_e_src
                || family.e_dst != scalar_e_dst
                || family.score != scalar_score
                || family.z != scalar_z
                || family.m != scalar_z
                || family.delta_raw != scalar_delta
                || family.delta != scalar_delta
                || family.exp != 16
                || family.sum != scalar_sum
                || family.aggregate_mul != scalar_aggregate_mul
                || family.aggregate_acc != scalar_aggregate
                || family.aggregate_num != scalar_aggregate
                || plan.tables.lrelu_min > -family.score
                || plan.tables.lrelu_max < family.score
                || plan.tables.exp_max < family.delta
                || plan.aux_tables.max_relu_min > -(family.delta + 1)
                || plan.aux_tables.max_relu_max < family.delta + 1
            {
                return Err(format!("undersized hidden family/table certificate: {label}").into());
            }
            let recovery_a = bound_mul(
                plan.quant.hidden_recovery_num_scale,
                family.aggregate_num,
                &label,
            )?;
            let recovery_d = bound_mul(plan.quant.hidden_recovery_den_scale, family.sum, &label)?;
            validate_round_div_exact(
                &family.recovery,
                recovery_a,
                256,
                recovery_d,
                &format!("{label}.recovery"),
            )?;
            if family.alignment.is_some()
                || family.bias != 0
                || family.output_bound != family.recovery.q.max(16)
                || plan.tables.elu_min > -family.recovery.q
                || plan.tables.elu_max < family.recovery.q
            {
                return Err(format!("invalid hidden recovery/ELU certificate: {label}").into());
            }
        }
        let mut next_scalar_h = 0i128;
        for head in hidden_specs.iter().filter(|head| head.layer_index == layer) {
            let family = family_bounds(
                plan,
                &format!("hidden.layer_{layer}.head_{}", head.local_head_index),
            )?;
            next_scalar_h = next_scalar_h.max(family.output_bound);
        }
        scalar_hidden_h = next_scalar_h;
    }
    let output_input_dim = plan
        .hidden_profile
        .last()
        .map(|profile| profile.head_count * profile.head_dim)
        .ok_or("missing hidden profile")?;
    let mut maximum_output_head = 0i128;
    let mut output_specs = plan.output_heads.clone();
    output_specs.sort_by_key(|head| head.head_index);
    for head in output_specs {
        let label = format!("output.head_{}", head.head_index);
        let family = family_bounds(plan, &label)?;
        if !seen.insert(label.clone())
            || family.input_dim != output_input_dim
            || family.output_dim != plan.class_count
            || family.h > i64::MAX as i128
            || family.w > registered_weight
            || family.a_src > registered_output_attention
            || family.a_dst > registered_output_attention
            || family.bias > registered_output_bias
        {
            return Err(format!("invalid output family registration: {label}").into());
        }
        let scalar_projection = bound_mul(
            family.input_dim as i128,
            bound_mul(scalar_hidden_h, family.w, &label)?,
            &label,
        )?;
        let scalar_e_src = bound_mul(
            family.output_dim as i128,
            bound_mul(scalar_projection, family.a_src, &label)?,
            &label,
        )?;
        let scalar_e_dst = bound_mul(
            family.output_dim as i128,
            bound_mul(scalar_projection, family.a_dst, &label)?,
            &label,
        )?;
        let scalar_score = bound_add(scalar_e_src, scalar_e_dst, &label)?;
        let scalar_z = deterministic_lrelu(-scalar_score)
            .checked_abs()
            .ok_or("scalar LReLU bound overflow")?
            .max(deterministic_lrelu(scalar_score).abs());
        let scalar_delta = bound_mul(2, scalar_z, &label)?;
        let scalar_sum = bound_mul(plan.ledger.max_in_degree as i128, 16, &label)?;
        let scalar_projection_mul = bound_mul(scalar_hidden_h, family.w, &label)?;
        let scalar_attention_mul = bound_mul(scalar_projection, family.a_src, &label)?
            .max(bound_mul(scalar_projection, family.a_dst, &label)?);
        let scalar_aggregate_mul = bound_mul(16, scalar_projection, &label)?;
        let scalar_aggregate = bound_mul(scalar_sum, scalar_projection, &label)?;
        if family.scalar_h != scalar_hidden_h
            || family.scalar_projection != scalar_projection
            || family.scalar_e_src != scalar_e_src
            || family.scalar_e_dst != scalar_e_dst
            || family.scalar_score != scalar_score
            || family.scalar_z != scalar_z
            || family.scalar_delta != scalar_delta
            || plan.tables.lrelu_min > -scalar_score
            || plan.tables.lrelu_max < scalar_score
            || plan.tables.exp_max < scalar_delta
        {
            return Err(format!("protocol 0.4.5 scalar family/table gate failed: {label}").into());
        }
        if family.h != scalar_hidden_h
            || family.projection_mul != scalar_projection_mul
            || family.projection_acc != scalar_projection
            || family.projection != scalar_projection
            || family.attention_mul != scalar_attention_mul
            || family.attention_acc != scalar_e_src.max(scalar_e_dst)
            || family.e_src != scalar_e_src
            || family.e_dst != scalar_e_dst
            || family.score != scalar_score
            || family.z != scalar_z
            || family.m != scalar_z
            || family.delta_raw != scalar_delta
            || family.delta != scalar_delta
            || family.exp != 16
            || family.sum != scalar_sum
            || family.aggregate_mul != scalar_aggregate_mul
            || family.aggregate_acc != scalar_aggregate
            || family.aggregate_num != scalar_aggregate
            || plan.tables.lrelu_min > -family.score
            || plan.tables.lrelu_max < family.score
            || plan.tables.exp_max < family.delta
            || plan.aux_tables.max_relu_min > -(family.delta + 1)
            || plan.aux_tables.max_relu_max < family.delta + 1
        {
            return Err(format!("undersized output family/table certificate: {label}").into());
        }
        let recovery_a = bound_mul(plan.quant.output_pre_scale, family.aggregate_num, &label)?;
        let recovery_d = bound_mul(plan.quant.output_projection_scale, family.sum, &label)?;
        validate_round_div_exact(
            &family.recovery,
            recovery_a,
            4096,
            recovery_d,
            &format!("{label}.recovery"),
        )?;
        let alignment = family
            .alignment
            .as_ref()
            .ok_or_else(|| EZKLError::from(format!("missing output alignment bounds: {label}")))?;
        let alignment_a = bound_mul(plan.quant.output_head_scale, family.recovery.q, &label)?;
        validate_round_div_exact(
            alignment,
            alignment_a,
            plan.quant.output_pre_scale,
            plan.quant.output_pre_scale,
            &format!("{label}.alignment"),
        )?;
        let output_bound = bound_add(alignment.q, family.bias, &label)?;
        if family.output_bound != output_bound {
            return Err(format!("undersized output head bound: {label}").into());
        }
        maximum_output_head = maximum_output_head.max(family.output_bound);
    }
    if seen.len() != plan.ledger.families.len()
        || plan.ledger.output_head_bound != maximum_output_head
    {
        return Err(
            "ledger contains missing/extra families or an undersized output head bound".into(),
        );
    }
    let output_sum = bound_mul(
        plan.k_out as i128,
        plan.ledger.output_head_bound,
        "final average",
    )?;
    if plan.ledger.output_sum_bound != output_sum {
        return Err("undersized final output sum bound".into());
    }
    validate_round_div_exact(
        &plan.ledger.final_average,
        plan.ledger.output_sum_bound,
        plan.k_out as i128,
        plan.k_out as i128,
        "final_average",
    )?;
    let required_aux_magnitude = plan.ledger.max_in_degree.saturating_sub(1) as i128;
    if plan.aux_tables.max_relu_min > -required_aux_magnitude
        || plan.aux_tables.max_relu_max < required_aux_magnitude
    {
        return Err("MaxReLU auxiliary table does not cover 1-sum(y)".into());
    }
    Ok(())
}

fn read_u64(bytes: &mut &[u8]) -> Result<u64, EZKLError> {
    if bytes.len() < 8 {
        return Err("truncated CSR payload".into());
    }
    let value = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    *bytes = &bytes[8..];
    Ok(value)
}

fn read_usize_vec(bytes: &mut &[u8]) -> Result<Vec<usize>, EZKLError> {
    let length = read_u64(bytes)? as usize;
    (0..length)
        .map(|_| usize::try_from(read_u64(bytes)?).map_err(|_| "CSR index exceeds usize".into()))
        .collect()
}

fn read_csr_payload(path: &Path) -> Result<CsrLayout, EZKLError> {
    let payload = fs::read(path)?;
    if payload.len() < 12 || &payload[..12] != b"EZKLGATCSR2\n" {
        return Err("invalid canonical CSR payload magic".into());
    }
    let mut bytes = &payload[12..];
    let src = read_usize_vec(&mut bytes)?;
    let dst = read_usize_vec(&mut bytes)?;
    let segment_ptr = read_usize_vec(&mut bytes)?;
    let edge_ptr = read_usize_vec(&mut bytes)?;
    let node_ptr = read_usize_vec(&mut bytes)?;
    let _segment_graph_id = read_usize_vec(&mut bytes)?;
    let absolute_ids = read_usize_vec(&mut bytes)?;
    let max_in_degree = read_u64(&mut bytes)? as usize;
    if !bytes.is_empty() {
        return Err("unexpected trailing CSR payload bytes".into());
    }
    Ok(CsrLayout {
        layout_kind: "csr_dst_segment_ragged".to_owned(),
        node_count: segment_ptr.len().saturating_sub(1),
        edge_count: src.len(),
        graph_count: node_ptr.len().saturating_sub(1),
        max_in_degree,
        src,
        dst,
        node_ptr,
        edge_ptr,
        absolute_ids,
    })
}

fn canonical_model_values(
    spec: &GraphSpec,
    tensors: &BTreeMap<String, Vec<i128>>,
) -> Result<Vec<i128>, EZKLError> {
    let mut values = Vec::new();
    let mut hidden = spec.hidden_heads.clone();
    hidden.sort_by_key(|head| (head.layer_index, head.local_head_index));
    for head in hidden {
        for name in [
            &head.seq_kernel,
            &head.attn_src_kernel,
            &head.attn_dst_kernel,
        ] {
            values.extend(
                tensors
                    .get(name)
                    .ok_or_else(|| format!("canonical tensor missing: {name}"))?,
            );
        }
    }
    let mut output = spec.output_heads.clone();
    output.sort_by_key(|head| head.head_index);
    for head in output {
        for name in [
            &head.seq_kernel,
            &head.attn_src_kernel,
            &head.attn_dst_kernel,
            &head.output_bias,
        ] {
            values.extend(
                tensors
                    .get(name)
                    .ok_or_else(|| format!("canonical tensor missing: {name}"))?,
            );
        }
    }
    Ok(values)
}

fn absolute_bound(values: &[i128]) -> Result<i128, EZKLError> {
    values.iter().try_fold(0i128, |bound, value| {
        Ok(bound.max(
            value
                .checked_abs()
                .ok_or("canonical parameter absolute-value overflow")?,
        ))
    })
}

fn derive_registered_bounds(
    spec: &GraphSpec,
    tensors: &BTreeMap<String, Vec<i128>>,
    features: &[i128],
) -> Result<BTreeMap<String, i128>, EZKLError> {
    let mut result = BTreeMap::new();
    result.insert("feature".to_owned(), absolute_bound(features)?);
    let mut weight = 0i128;
    let mut hidden_attention = 0i128;
    let mut output_attention = 0i128;
    let mut output_bias = 0i128;
    for head in &spec.hidden_heads {
        weight = weight.max(absolute_bound(
            tensors
                .get(&head.seq_kernel)
                .ok_or("missing hidden weight")?,
        )?);
        hidden_attention = hidden_attention.max(absolute_bound(
            tensors
                .get(&head.attn_src_kernel)
                .ok_or("missing hidden source attention")?,
        )?);
        hidden_attention = hidden_attention.max(absolute_bound(
            tensors
                .get(&head.attn_dst_kernel)
                .ok_or("missing hidden destination attention")?,
        )?);
    }
    for head in &spec.output_heads {
        weight = weight.max(absolute_bound(
            tensors
                .get(&head.seq_kernel)
                .ok_or("missing output weight")?,
        )?);
        output_attention = output_attention.max(absolute_bound(
            tensors
                .get(&head.attn_src_kernel)
                .ok_or("missing output source attention")?,
        )?);
        output_attention = output_attention.max(absolute_bound(
            tensors
                .get(&head.attn_dst_kernel)
                .ok_or("missing output destination attention")?,
        )?);
        output_bias = output_bias.max(absolute_bound(
            tensors
                .get(&head.output_bias)
                .ok_or("missing output bias")?,
        )?);
    }
    result.insert("weight".to_owned(), weight);
    result.insert("hidden_attention".to_owned(), hidden_attention);
    result.insert("output_attention".to_owned(), output_attention);
    result.insert("output_bias".to_owned(), output_bias);
    Ok(result)
}

fn estimate_logrows(spec: &GraphSpec, tables: &TableSpec, aux: &AuxTableSpec) -> u32 {
    let n = spec.node_count as u128;
    let e = spec.edge_count as u128;
    let mut arithmetic = 0u128;
    for (layer, profile) in spec.hidden_profile.iter().enumerate() {
        let input = spec.d_in_profile[layer] as u128;
        let width = profile.head_dim as u128;
        let heads = profile.head_count as u128;
        arithmetic +=
            heads * (n * width * input + n * width * 2 + e * width * 4 + e * 40 + n * width * 80);
    }
    let output_input = spec
        .hidden_profile
        .last()
        .map(|profile| profile.head_count * profile.head_dim)
        .unwrap_or(spec.feature_dim) as u128;
    arithmetic += spec.k_out as u128
        * (n * spec.class_count as u128 * output_input
            + n * spec.class_count as u128 * 170
            + e * spec.class_count as u128 * 4
            + e * 40);
    arithmetic += n * spec.class_count as u128 * (40 + spec.k_out as u128 * 4);
    // The fixed trace machine assigns canonical variables and independent
    // constraints into PROGRAM_LANES parallel column lanes.  The factor three
    // conservatively covers canonical cells, operation rows, range auxiliaries
    // and Poseidon overhead before packing.
    let uncompressed = arithmetic
        .saturating_mul(3)
        .saturating_add(n * spec.feature_dim as u128);
    let variables_and_constraints =
        uncompressed.saturating_add(PROGRAM_LANES as u128 - 1) / PROGRAM_LANES as u128;
    let table_rows = (aux.limb_max + 1).max(0) as u128
        + (tables.lrelu_max - tables.lrelu_min + 1).max(0) as u128
        + (tables.elu_max - tables.elu_min + 1).max(0) as u128
        + (tables.exp_max + 1).max(0) as u128
        + (aux.max_relu_max - aux.max_relu_min + 1).max(0) as u128;
    let rows = variables_and_constraints.max(table_rows).max(1024);
    (128 - (rows - 1).leading_zeros()).max(10)
}

fn plan_dir(path: &Path) -> PathBuf {
    path.parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf()
}

pub(crate) fn graph_gat_gen_settings(
    spec_path: PathBuf,
    csr_layout_path: PathBuf,
    settings_path: PathBuf,
    max_logrows: Option<u32>,
) -> Result<String, EZKLError> {
    let spec: GraphSpec = read_json(&spec_path)?;
    let csr = read_csr_payload(&csr_layout_path.with_file_name("csr_segment_layout.bin"))?;
    if (spec.node_count, spec.edge_count, spec.graph_count)
        != (csr.node_count, csr.edge_count, csr.graph_count)
    {
        return Err("graph spec and canonical CSR payload disagree".into());
    }
    let base = spec_path.parent().unwrap_or_else(|| Path::new("."));
    let quant: QuantConfig = read_json(&base.join("formal_quant_config.json"))?;
    validate_quant(&quant)?;
    let tables = bind_table_spec(read_json(&base.join("formal_table_spec.json"))?)?;
    let aux_tables = bind_aux_table_spec(read_json(&base.join("backend_aux_table_spec.json"))?)?;
    validate_main_table_domains(&spec, &quant, &tables, &aux_tables)?;
    let _canonical_tables = load_canonical_tables(base, &tables, &aux_tables)?;
    let (_, security_profile_id) = load_security_profile(base)?;
    if spec.quant_cfg_id != quant.canonical_digest
        || spec.static_tables_id != tables.canonical_digest
        || spec.backend_aux_tables_id != aux_tables.canonical_digest
        || spec.source_quant_cfg_id != quant.source_quant_cfg_id
        || spec.source_static_table_id != tables.source_static_table_id
        || spec.security_profile_id != security_profile_id
    {
        return Err("graph spec does not bind the canonical quant/table/security profiles".into());
    }
    let logrows = estimate_logrows(&spec, &tables, &aux_tables);
    let limit = max_logrows.unwrap_or(DEFAULT_MAX_LOGROWS);
    if logrows > limit {
        return Err(format!(
            "formal fixed GAT circuit requires logrows={logrows}, configured maximum={limit}"
        )
        .into());
    }
    let mut settings = FormalSettings {
        kind: SETTINGS_KIND.to_owned(),
        status: "ready_for_formal_execution".to_owned(),
        relation: "R_GAT-ezkl".to_owned(),
        route_id: spec.route_id,
        dataset: spec.dataset,
        logrows,
        quant_cfg_id: spec.quant_cfg_id,
        static_tables_id: spec.static_tables_id,
        backend_aux_tables_id: spec.backend_aux_tables_id,
        security_profile_id,
        program_version: PLAN_KIND.to_owned(),
        settings_id: String::new(),
    };
    settings.settings_id = canonical_digest(&settings)?;
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
    if !settings_path.exists() {
        graph_gat_gen_settings(
            spec_path.clone(),
            csr_layout_path.clone(),
            settings_path.clone(),
            max_logrows,
        )?;
    }
    let spec: GraphSpec = read_json(&spec_path)?;
    let csr = read_csr_payload(&csr_layout_path.with_file_name("csr_segment_layout.bin"))?;
    if csr.absolute_ids.len() != csr.node_count {
        return Err("canonical CSR payload is missing absolute node identities".into());
    }
    let base = spec_path.parent().unwrap_or_else(|| Path::new("."));
    let quant: QuantConfig = read_json(&base.join("formal_quant_config.json"))?;
    validate_quant(&quant)?;
    let tables = bind_table_spec(read_json(&base.join("formal_table_spec.json"))?)?;
    let aux_tables = bind_aux_table_spec(read_json(&base.join("backend_aux_table_spec.json"))?)?;
    validate_main_table_domains(&spec, &quant, &tables, &aux_tables)?;
    let _canonical_tables = load_canonical_tables(base, &tables, &aux_tables)?;
    let (security_profile, security_profile_id) = load_security_profile(base)?;
    let settings: FormalSettings = read_json(&settings_path)?;
    validate_settings(&settings, &security_profile_id)?;
    let expected_logrows = estimate_logrows(&spec, &tables, &aux_tables);
    if settings.status != "ready_for_formal_execution"
        || settings.relation != "R_GAT-ezkl"
        || settings.route_id != spec.route_id
        || settings.dataset != spec.dataset
        || settings.quant_cfg_id != quant.canonical_digest
        || settings.static_tables_id != tables.canonical_digest
        || settings.backend_aux_tables_id != aux_tables.canonical_digest
        || settings.security_profile_id != security_profile_id
        || settings.logrows != expected_logrows
        || expected_logrows > max_logrows.unwrap_or(DEFAULT_MAX_LOGROWS)
        || spec.quant_cfg_id != quant.canonical_digest
        || spec.static_tables_id != tables.canonical_digest
        || spec.backend_aux_tables_id != aux_tables.canonical_digest
        || spec.source_quant_cfg_id != quant.source_quant_cfg_id
        || spec.source_static_table_id != tables.source_static_table_id
        || spec.security_profile_id != security_profile_id
    {
        return Err("settings/spec/quant/table/security cross-binding mismatch".into());
    }
    let mut ledger: IntegerBoundLedger = read_json(&base.join("integer_bound_ledger.json"))?;
    let mut registration: ModelRegistration = read_json(&base.join("model_registration.json"))?;
    let artifacts = ArtifactPaths {
        features_i64: "node_features.i64.bin".to_owned(),
        model_i64: "model_parameters.i64.txt".to_owned(),
        model_registration: "model_registration.json".to_owned(),
    };
    let tensors = read_i64_tensors(&base.join(&artifacts.model_i64))?;
    let features = read_i64_matrix(
        &base.join(&artifacts.features_i64),
        spec.node_count,
        spec.feature_dim,
    )?;
    if canonical_graph_hash(
        spec.node_count,
        spec.edge_count,
        spec.graph_count,
        spec.feature_dim,
        spec.class_count,
        &csr.absolute_ids,
        &csr.node_ptr,
        &csr.edge_ptr,
        &csr.src,
        &csr.dst,
        &features,
    )? != spec.graph_hash
    {
        return Err("graph spec hash does not match canonical public features/topology".into());
    }
    let derived_bounds = derive_registered_bounds(&spec, &tensors, &features)?;
    if registration.parameter_bounds != derived_bounds
        || registration.model_arch_id != spec.model_arch_id
        || registration.model_param_id != spec.model_param_id
        || registration.quant_cfg_id != quant.canonical_digest
    {
        return Err("trusted setup rejected canonical model registration/bounds mismatch".into());
    }
    for head in &spec.hidden_heads {
        let label = format!(
            "hidden.layer_{}.head_{}",
            head.layer_index, head.local_head_index
        );
        let family = ledger
            .families
            .iter()
            .find(|family| family.label == label)
            .ok_or("ledger is missing a canonical hidden model family")?;
        if family.w
            != absolute_bound(
                tensors
                    .get(&head.seq_kernel)
                    .ok_or("missing hidden weight")?,
            )?
            || family.a_src
                != absolute_bound(
                    tensors
                        .get(&head.attn_src_kernel)
                        .ok_or("missing hidden attention")?,
                )?
            || family.a_dst
                != absolute_bound(
                    tensors
                        .get(&head.attn_dst_kernel)
                        .ok_or("missing hidden attention")?,
                )?
        {
            return Err("hidden family certificate is not bound to canonical model values".into());
        }
    }
    for head in &spec.output_heads {
        let label = format!("output.head_{}", head.head_index);
        let family = ledger
            .families
            .iter()
            .find(|family| family.label == label)
            .ok_or("ledger is missing a canonical output model family")?;
        if family.w
            != absolute_bound(
                tensors
                    .get(&head.seq_kernel)
                    .ok_or("missing output weight")?,
            )?
            || family.a_src
                != absolute_bound(
                    tensors
                        .get(&head.attn_src_kernel)
                        .ok_or("missing output attention")?,
                )?
            || family.a_dst
                != absolute_bound(
                    tensors
                        .get(&head.attn_dst_kernel)
                        .ok_or("missing output attention")?,
                )?
            || family.bias
                != absolute_bound(
                    tensors
                        .get(&head.output_bias)
                        .ok_or("missing output bias")?,
                )?
        {
            return Err("output family certificate is not bound to canonical model values".into());
        }
    }
    let model_values = canonical_model_values(&spec, &tensors)?;
    let model_fields = model_values
        .iter()
        .map(|value| fr_i128(*value))
        .collect::<Vec<_>>();
    let model_hash =
        PoseidonChip::<PoseidonSpec, POSEIDON_WIDTH, POSEIDON_RATE>::run(model_fields)?[0][0];
    let model_hash_hex = fr_hex(model_hash);
    let parameter_digest = sha256_bytes(
        &model_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    if registration.canonical_parameter_count != 0
        && registration.canonical_parameter_count != model_values.len()
    {
        return Err("model registration canonical parameter identity mismatch".into());
    }
    if !registration.canonical_parameter_digest.is_empty()
        && registration.canonical_parameter_digest != parameter_digest
    {
        return Err("model registration canonical parameter digest mismatch".into());
    }
    registration.canonical_parameter_count = model_values.len();
    registration.canonical_parameter_digest = parameter_digest;
    registration.model_bounds_digest = canonical_digest(&registration.parameter_bounds)?;
    if !ledger.model_bounds_digest.is_empty()
        && ledger.model_bounds_digest != registration.model_bounds_digest
    {
        return Err("integer ledger/model registration bounds digest mismatch".into());
    }
    ledger.model_bounds_digest = registration.model_bounds_digest.clone();
    write_json(&base.join("integer_bound_ledger.json"), &ledger)?;
    if !registration.registered_model_hash.is_empty()
        && registration.registered_model_hash != model_hash_hex
    {
        return Err("model registration Poseidon hash mismatch".into());
    }
    registration.registered_model_hash = model_hash_hex.clone();
    write_json(&base.join("model_registration.json"), &registration)?;
    let mut plan = FormalPlan {
        kind: PLAN_KIND.to_owned(),
        program_version: PLAN_KIND.to_owned(),
        route_id: spec.route_id,
        dataset: spec.dataset,
        node_count: spec.node_count,
        edge_count: spec.edge_count,
        graph_count: spec.graph_count,
        feature_dim: spec.feature_dim,
        class_count: spec.class_count,
        layer_count: spec.layer_count,
        k_out: spec.k_out,
        hidden_profile: spec.hidden_profile,
        d_in_profile: spec.d_in_profile,
        hidden_heads: spec.hidden_heads,
        output_heads: spec.output_heads,
        src: csr.src,
        dst: csr.dst,
        node_ptr: csr.node_ptr,
        edge_ptr: csr.edge_ptr,
        absolute_ids: csr.absolute_ids,
        quant,
        tables,
        aux_tables,
        ledger,
        model_parameter_bounds: registration.parameter_bounds.clone(),
        artifacts,
        model_arch_id: spec.model_arch_id,
        model_param_id: spec.model_param_id,
        quant_cfg_id: spec.quant_cfg_id,
        static_tables_id: spec.static_tables_id,
        backend_aux_tables_id: spec.backend_aux_tables_id,
        graph_hash: spec.graph_hash,
        checkpoint_hash: spec.checkpoint_hash,
        metadata_hash: spec.metadata_hash,
        shape_hash: spec.shape_hash,
        registered_model_hash: model_hash_hex,
        settings_id: settings.settings_id,
        compiled_circuit_id: String::new(),
        security_profile_id,
        security_profile,
        logrows: settings.logrows,
    };
    validate_plan(&plan)?;
    plan.compiled_circuit_id = canonical_digest(&plan)?;
    write_json(&compiled_circuit_path, &plan)?;
    Ok(serde_json::to_string(&plan)?)
}

fn native_srs_path(key_path: &Path) -> PathBuf {
    key_path.with_extension("srs")
}

fn validate_resolved_srs_files(
    plan: &FormalPlan,
    manifest: &CircuitManifest,
) -> Result<(), EZKLError> {
    let source_path = Path::new(&manifest.source_srs_path);
    let source_digest = format!("sha256:{}", sha256_file(source_path)?);
    if source_digest != plan.security_profile.source_srs_sha256
        || manifest.source_srs_sha256 != plan.security_profile.source_srs_sha256
    {
        return Err("source SRS bytes differ from the resolved security profile".into());
    }
    let mut params = crate::pfsys::srs::load_srs_prover::<KZGCommitmentScheme<Bn256>>(
        source_path.to_path_buf(),
    )?;
    if params.k() != plan.security_profile.source_srs_max_k
        || manifest.source_srs_k != plan.security_profile.source_srs_max_k
        || plan.logrows > params.k()
    {
        return Err("source SRS k differs from the resolved security profile".into());
    }
    params.downsize(plan.logrows);
    let mut expected_bytes = Vec::new();
    params.write(&mut expected_bytes)?;
    let expected_digest = format!("sha256:{}", sha256_bytes(&expected_bytes));
    let proving_path = Path::new(&manifest.proving_srs_path);
    let verification_path = Path::new(&manifest.verification_srs_path);
    if manifest.downsized_srs_k != plan.logrows
        || manifest.proving_srs_sha256 != expected_digest
        || manifest.verification_srs_sha256 != expected_digest
        || format!("sha256:{}", sha256_file(proving_path)?) != expected_digest
        || format!("sha256:{}", sha256_file(verification_path)?) != expected_digest
        || fs::read(proving_path)? != expected_bytes
        || fs::read(verification_path)? != expected_bytes
    {
        return Err("downsized PK/VK SRS bytes are not the deterministic source-SRS prefix".into());
    }
    Ok(())
}

fn load_params_for_key(key_path: &Path) -> Result<ParamsKZG<Bn256>, EZKLError> {
    crate::pfsys::srs::load_srs_prover::<KZGCommitmentScheme<Bn256>>(native_srs_path(key_path))
        .map_err(EZKLError::from)
}

fn configure_device(device: &str) -> Result<&'static str, EZKLError> {
    match device {
        "cpu" => {
            #[cfg(feature = "gpu-accelerated")]
            return Err(
                "CPU route requires the CPU build; GPU-enabled binaries cannot silently fall back"
                    .into(),
            );
            #[cfg(not(feature = "gpu-accelerated"))]
            Ok("halo2_cpu")
        }
        "cuda" => {
            #[cfg(feature = "gpu-accelerated")]
            {
                halo2_proofs::icicle::try_load_and_set_backend_device("CUDA");
                Ok("halo2_icicle_cuda_device_0")
            }
            #[cfg(not(feature = "gpu-accelerated"))]
            Err("CUDA route requires a binary built with the gpu-accelerated feature".into())
        }
        _ => Err("device must be cpu or cuda".into()),
    }
}

pub(crate) fn graph_gat_setup(
    compiled_circuit_path: PathBuf,
    vk_path: PathBuf,
    pk_path: PathBuf,
    srs_path: PathBuf,
    device: String,
) -> Result<String, EZKLError> {
    let device_marker = configure_device(&device)?;
    let plan: FormalPlan = read_json(&compiled_circuit_path)?;
    validate_plan(&plan)?;
    let tables = load_canonical_tables(
        &plan_dir(&compiled_circuit_path),
        &plan.tables,
        &plan.aux_tables,
    )?;
    let program = build_program(&plan, None, None)?;
    let circuit = FormalGatCircuit {
        plan: plan.clone(),
        program,
        tables,
    };
    if !srs_path.is_file() {
        return Err("formal setup requires an existing trusted KZG SRS".into());
    }
    let source_srs_path = fs::canonicalize(&srs_path)?;
    let source_srs_sha256 = format!("sha256:{}", sha256_file(&source_srs_path)?);
    let mut params = crate::pfsys::srs::load_srs_prover::<KZGCommitmentScheme<Bn256>>(srs_path)?;
    let source_srs_k = params.k();
    if source_srs_sha256 != plan.security_profile.source_srs_sha256
        || source_srs_k != plan.security_profile.source_srs_max_k
        || source_srs_k < plan.logrows
    {
        return Err("trusted KZG SRS differs from the exact resolved security profile".into());
    }
    params.downsize(plan.logrows);
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, FormalGatCircuit>(&circuit, &params, true)?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_params::<KZGCommitmentScheme<Bn256>>(&native_srs_path(&pk_path), &params)?;
    fs::copy(native_srs_path(&pk_path), native_srs_path(&vk_path))?;
    let proving_srs_sha256 = format!("sha256:{}", sha256_file(&native_srs_path(&pk_path))?);
    let verification_srs_sha256 = format!("sha256:{}", sha256_file(&native_srs_path(&vk_path))?);
    let proving_srs_path = fs::canonicalize(native_srs_path(&pk_path))?;
    let verification_srs_path = fs::canonicalize(native_srs_path(&vk_path))?;
    let run_dir = plan_dir(&compiled_circuit_path);
    let manifest = CircuitManifest {
        manifest_kind: "ezkl_formal_gat_circuit_manifest_v1".to_owned(),
        compiled_circuit_id: plan.compiled_circuit_id.clone(),
        settings_id: plan.settings_id.clone(),
        vk_id: sha256_file(&vk_path)?,
        pk_id: sha256_file(&pk_path)?,
        security_profile_id: plan.security_profile_id.clone(),
        quant_cfg_id: plan.quant_cfg_id.clone(),
        static_tables_id: plan.static_tables_id.clone(),
        backend_aux_tables_id: plan.backend_aux_tables_id.clone(),
        device_build: device_marker.to_owned(),
        source_srs_sha256,
        source_srs_path: source_srs_path.display().to_string(),
        source_srs_k,
        downsized_srs_k: plan.logrows,
        proving_srs_sha256,
        verification_srs_sha256,
        proving_srs_path: proving_srs_path.display().to_string(),
        verification_srs_path: verification_srs_path.display().to_string(),
    };
    validate_resolved_srs_files(&plan, &manifest)?;
    write_json(&run_dir.join("circuit_manifest.json"), &manifest)?;
    Ok(serde_json::to_string(&manifest)?)
}

fn program_instances(program: &FormalProgram) -> Result<Vec<Fr>, EZKLError> {
    program
        .public_vars
        .iter()
        .map(|var| program.values[*var].ok_or_else(|| "public instance value is unknown".into()))
        .collect()
}

fn statement_instances(
    plan: &FormalPlan,
    statement: &PublicStatement,
) -> Result<Vec<Fr>, EZKLError> {
    validate_statement(plan, statement)?;
    let n_total = statement.t_h.len() / plan.feature_dim;
    let mut values = Vec::with_capacity(
        plan.node_count * plan.feature_dim + plan.node_count * plan.class_count + 1,
    );
    for &index in &statement.indices {
        if index >= n_total {
            return Err("public absolute node index out of range".into());
        }
        values.extend(
            statement.t_h[index * plan.feature_dim..(index + 1) * plan.feature_dim]
                .iter()
                .map(|value| fr_i128(*value)),
        );
    }
    values.extend(statement.logits.iter().map(|value| fr_i128(*value)));
    values.push(parse_fr_hex(&statement.model_hash)?);
    Ok(values)
}

fn validate_statement(plan: &FormalPlan, statement: &PublicStatement) -> Result<(), EZKLError> {
    if statement.kind != STATEMENT_KIND
        || statement.dataset != plan.dataset
        || statement.model_arch_id != plan.model_arch_id
        || statement.quant_cfg_id != plan.quant_cfg_id
        || statement.static_tables_id != plan.static_tables_id
        || statement.backend_aux_tables_id != plan.backend_aux_tables_id
        || statement.security_profile_id != plan.security_profile_id
        || statement.compiled_circuit_id != plan.compiled_circuit_id
        || statement.settings_id != plan.settings_id
        || statement.graph_hash != plan.graph_hash
    {
        return Err("public statement metadata/circuit binding mismatch".into());
    }
    if (
        statement.node_count,
        statement.edge_count,
        statement.graph_count,
        statement.feature_dim,
        statement.class_count,
    ) != (
        plan.node_count,
        plan.edge_count,
        plan.graph_count,
        plan.feature_dim,
        plan.class_count,
    ) {
        return Err("public statement shape mismatch".into());
    }
    if statement.src != plan.src
        || statement.dst != plan.dst
        || statement.node_ptr != plan.node_ptr
        || statement.edge_ptr != plan.edge_ptr
    {
        return Err("public topology does not match compiled circuit wiring".into());
    }
    if statement.indices != plan.absolute_ids
        || statement.logits.len() != plan.node_count * plan.class_count
        || statement.t_h.len() != plan.node_count * plan.feature_dim
    {
        return Err("public statement tensor length mismatch".into());
    }
    if public_graph_hash(statement)? != plan.graph_hash {
        return Err(
            "public canonical features/topology hash does not match the compiled graph identity"
                .into(),
        );
    }
    let mut unique = statement.indices.clone();
    unique.sort_unstable();
    unique.dedup();
    if unique.len() != statement.indices.len() {
        return Err("public node indices contain duplicates".into());
    }
    if statement.model_hash != plan.registered_model_hash {
        return Err("public model hash does not match registered circuit model".into());
    }
    let public_bound = plan
        .model_parameter_bounds
        .get("feature")
        .copied()
        .unwrap_or(-1);
    if public_bound < 0
        || statement.t_h.iter().any(|value| {
            value
                .checked_abs()
                .is_none_or(|magnitude| magnitude > public_bound)
        })
    {
        return Err("public feature range exceeds integer bound ledger".into());
    }
    Ok(())
}

pub(crate) fn graph_gat_gen_witness(
    compiled_circuit_path: PathBuf,
    output_path: PathBuf,
    public_statement_path: PathBuf,
) -> Result<String, EZKLError> {
    let plan: FormalPlan = read_json(&compiled_circuit_path)?;
    validate_plan(&plan)?;
    let run_dir = plan_dir(&compiled_circuit_path);
    let manifest: CircuitManifest = read_json(&run_dir.join("circuit_manifest.json"))?;
    if manifest.compiled_circuit_id != plan.compiled_circuit_id
        || manifest.settings_id != plan.settings_id
    {
        return Err("circuit manifest does not match compiled circuit".into());
    }
    let inputs = load_inputs(&plan, &run_dir)?;
    let tables = load_canonical_tables(&run_dir, &plan.tables, &plan.aux_tables)?;
    let program = build_program(&plan, Some(&inputs), Some(&tables))?;
    let logits = program
        .output_vars
        .iter()
        .map(|var| program.integer_values[*var].ok_or("final logit is not a canonical integer"))
        .collect::<Result<Vec<_>, _>>()?;
    let statement = PublicStatement {
        kind: STATEMENT_KIND.to_owned(),
        dataset: plan.dataset.clone(),
        model_arch_id: plan.model_arch_id.clone(),
        quant_cfg_id: plan.quant_cfg_id.clone(),
        static_tables_id: plan.static_tables_id.clone(),
        backend_aux_tables_id: plan.backend_aux_tables_id.clone(),
        security_profile_id: plan.security_profile_id.clone(),
        compiled_circuit_id: plan.compiled_circuit_id.clone(),
        settings_id: plan.settings_id.clone(),
        vk_id: manifest.vk_id,
        graph_hash: plan.graph_hash.clone(),
        node_count: plan.node_count,
        edge_count: plan.edge_count,
        graph_count: plan.graph_count,
        feature_dim: plan.feature_dim,
        class_count: plan.class_count,
        t_h: inputs.features.clone(),
        indices: plan.absolute_ids.clone(),
        src: plan.src.clone(),
        dst: plan.dst.clone(),
        node_ptr: plan.node_ptr.clone(),
        edge_ptr: plan.edge_ptr.clone(),
        logits,
        model_hash: plan.registered_model_hash.clone(),
    };
    validate_statement(&plan, &statement)?;
    let expected = statement_instances(&plan, &statement)?;
    if program_instances(&program)? != expected {
        return Err("generated public statement does not match circuit public instance".into());
    }
    write_json(&public_statement_path, &statement)?;
    let values = program
        .values
        .iter()
        .map(|value| {
            value
                .map(fr_hex)
                .ok_or_else(|| EZKLError::from("formal trace contains an unknown witness value"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let witness = FormalWitness {
        kind: WITNESS_KIND.to_owned(),
        compiled_circuit_id: plan.compiled_circuit_id,
        logical_trace_digest: sha256_bytes(values.join("\n").as_bytes()),
        values,
        public_statement_path: public_statement_path.display().to_string(),
    };
    write_json(&output_path, &witness)?;
    Ok(serde_json::to_string(&serde_json::json!({
        "status": "formal_witness_created",
        "witness_path": output_path,
        "public_statement_path": public_statement_path,
        "logical_trace_digest": witness.logical_trace_digest,
    }))?)
}

fn program_with_serialized_witness(
    plan: &FormalPlan,
    witness: &FormalWitness,
) -> Result<FormalProgram, EZKLError> {
    if witness.kind != WITNESS_KIND || witness.compiled_circuit_id != plan.compiled_circuit_id {
        return Err("formal witness/circuit identity mismatch".into());
    }
    if witness.logical_trace_digest != sha256_bytes(witness.values.join("\n").as_bytes()) {
        return Err("formal witness trace digest mismatch".into());
    }
    let mut program = build_program(plan, None, None)?;
    if witness.values.len() != program.values.len() {
        return Err("formal witness variable count mismatch".into());
    }
    program.values = witness
        .values
        .iter()
        .map(|value| parse_fr_hex(value).map(Some))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(program)
}

pub(crate) fn graph_gat_prove(
    compiled_circuit_path: PathBuf,
    witness_path: PathBuf,
    public_statement_path: PathBuf,
    pk_path: PathBuf,
    proof_path: PathBuf,
    device: String,
) -> Result<String, EZKLError> {
    let device_marker = configure_device(&device)?;
    let plan: FormalPlan = read_json(&compiled_circuit_path)?;
    validate_plan(&plan)?;
    let circuit_manifest: CircuitManifest =
        read_json(&plan_dir(&compiled_circuit_path).join("circuit_manifest.json"))?;
    validate_resolved_srs_files(&plan, &circuit_manifest)?;
    if circuit_manifest.manifest_kind != "ezkl_formal_gat_circuit_manifest_v1"
        || circuit_manifest.compiled_circuit_id != plan.compiled_circuit_id
        || circuit_manifest.security_profile_id != plan.security_profile_id
        || circuit_manifest.quant_cfg_id != plan.quant_cfg_id
        || circuit_manifest.static_tables_id != plan.static_tables_id
        || circuit_manifest.backend_aux_tables_id != plan.backend_aux_tables_id
        || circuit_manifest.pk_id != sha256_file(&pk_path)?
        || Path::new(&circuit_manifest.proving_srs_path)
            != fs::canonicalize(native_srs_path(&pk_path))?
        || circuit_manifest.device_build != device_marker
    {
        return Err("proving key/circuit/device identity mismatch".into());
    }
    let witness: FormalWitness = read_json(&witness_path)?;
    let statement: PublicStatement = read_json(&public_statement_path)?;
    let program = program_with_serialized_witness(&plan, &witness)?;
    let public = program_instances(&program)?;
    if public != statement_instances(&plan, &statement)? {
        return Err("witness public values differ from independent public statement".into());
    }
    let circuit = FormalGatCircuit {
        plan: plan.clone(),
        program,
        tables: load_canonical_tables(
            &plan_dir(&compiled_circuit_path),
            &plan.tables,
            &plan.aux_tables,
        )?,
    };
    let pk =
        load_pk::<KZGCommitmentScheme<Bn256>, FormalGatCircuit>(pk_path.clone(), circuit.params())?;
    let params = load_params_for_key(&pk_path)?;
    if params.k() != plan.logrows {
        return Err("proving SRS k does not match compiled circuit".into());
    }
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
        vec![public],
        &params,
        &pk,
        CheckMode::SAFE,
        None,
        None,
    )?;
    snark.save(&proof_path)?;
    Ok(serde_json::to_string(&serde_json::json!({
        "status": "proof_created", "proof_path": proof_path,
        "compiled_circuit_id": plan.compiled_circuit_id,
        "public_statement_path": public_statement_path,
        "device_requested": device, "device_runtime_marker": device_marker,
        "gpu_backend_selected": if device == "cuda" { "halo2_icicle" } else { "none" },
        "gpu_kernel_execution": if device == "cuda" { "pending_formal_experiment_validation" } else { "not_applicable" },
        "no_fallback": true,
    }))?)
}

pub(crate) fn graph_gat_verify(
    compiled_circuit_path: PathBuf,
    proof_path: PathBuf,
    public_statement_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    output_path: PathBuf,
) -> Result<String, EZKLError> {
    let plan: FormalPlan = read_json(&compiled_circuit_path)?;
    validate_plan(&plan)?;
    let settings: FormalSettings = read_json(&settings_path)?;
    let (_, security_profile_id) = load_security_profile(&plan_dir(&compiled_circuit_path))?;
    validate_settings(&settings, &security_profile_id)?;
    let statement: PublicStatement = read_json(&public_statement_path)?;
    let run_dir = plan_dir(&compiled_circuit_path);
    let circuit_manifest: CircuitManifest = read_json(&run_dir.join("circuit_manifest.json"))?;
    validate_resolved_srs_files(&plan, &circuit_manifest)?;
    let registration: ModelRegistration =
        read_json(&run_dir.join(&plan.artifacts.model_registration))?;
    validate_registration(&plan, &registration)?;
    if circuit_manifest.manifest_kind != "ezkl_formal_gat_circuit_manifest_v1"
        || settings.settings_id != plan.settings_id
        || settings.status != "ready_for_formal_execution"
        || settings.relation != "R_GAT-ezkl"
        || settings.route_id != plan.route_id
        || settings.dataset != plan.dataset
        || settings.quant_cfg_id != plan.quant_cfg_id
        || settings.static_tables_id != plan.static_tables_id
        || settings.backend_aux_tables_id != plan.backend_aux_tables_id
        || settings.logrows != plan.logrows
        || settings.security_profile_id != plan.security_profile_id
        || circuit_manifest.compiled_circuit_id != plan.compiled_circuit_id
        || circuit_manifest.settings_id != plan.settings_id
        || circuit_manifest.vk_id != sha256_file(&vk_path)?
        || Path::new(&circuit_manifest.verification_srs_path)
            != fs::canonicalize(native_srs_path(&vk_path))?
        || statement.vk_id != circuit_manifest.vk_id
        || circuit_manifest.security_profile_id != plan.security_profile_id
        || circuit_manifest.quant_cfg_id != plan.quant_cfg_id
        || circuit_manifest.static_tables_id != plan.static_tables_id
        || circuit_manifest.backend_aux_tables_id != plan.backend_aux_tables_id
        || !matches!(
            circuit_manifest.device_build.as_str(),
            "halo2_cpu" | "halo2_icicle_cuda_device_0"
        )
        || registration.registered_model_hash != statement.model_hash
    {
        return Err("settings/vk/circuit/table/ledger/model registration binding mismatch".into());
    }
    let expected_instances = statement_instances(&plan, &statement)?;
    let mut proof = Snark::<Fr, G1Affine>::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
    if proof.instances != vec![expected_instances.clone()] {
        return Err(
            "proof-carried instances differ from verifier-reconstructed public statement".into(),
        );
    }
    // The native verifier is always invoked with the independently reconstructed
    // instances, never with an untrusted serialization carried by the proof.
    proof.instances = vec![expected_instances];
    let circuit = FormalGatCircuit {
        plan: plan.clone(),
        program: build_program(&plan, None, None)?,
        tables: load_canonical_tables(&run_dir, &plan.tables, &plan.aux_tables)?,
    };
    let vk =
        load_vk::<KZGCommitmentScheme<Bn256>, FormalGatCircuit>(vk_path.clone(), circuit.params())?;
    let params = load_params_for_key(&vk_path)?;
    if params.k() != plan.logrows {
        return Err("verification SRS k does not match compiled circuit".into());
    }
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
        1 << plan.logrows,
    )?;
    let output = serde_json::json!({
        "status": "verified", "verify_passed": true,
        "native_halo2_verification": true,
        "independent_public_statement_reconstruction": true,
        "proof_instances_trusted": false,
        "compiled_circuit_id": plan.compiled_circuit_id,
        "settings_id": plan.settings_id,
        "vk_id": circuit_manifest.vk_id,
        "static_tables_id": plan.static_tables_id,
        "backend_aux_tables_id": plan.backend_aux_tables_id,
        "integer_bound_ledger_digest": plan.ledger.model_bounds_digest,
        "model_hash": statement.model_hash,
        "prover_device_build": circuit_manifest.device_build,
        "verification_runtime_marker": "halo2_kzg_shplonk_independent_statement_ok",
        "full_gat_forward_in_ezkl_proof": true,
        "strict_public_input_equivalent": true,
    });
    write_json(&output_path, &output)?;
    Ok(output.to_string())
}
