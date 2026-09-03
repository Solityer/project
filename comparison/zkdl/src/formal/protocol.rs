use crate::algebra::{Point, Scalar};
use crate::compiler::SUPPORTED_ROUNDING_MODE;
use crate::hyrax::{commit, commit_with_backend, CommitmentParameters, VectorCommitment};
use crate::prover_backend::{GpuTelemetry, ProverBackend};
use crate::registry::{RelationRegistry, Visibility, WitnessPackage};
use crate::sumcheck::{prove_operation, verify_operation, OperationProof};
use crate::transcript::Transcript;
use anyhow::{bail, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

static ATOMIC_WRITE_COUNTER: AtomicU64 = AtomicU64::new(0);

pub const PROOF_FORMAT: &str = "zkDL-GAT-proof-envelope-v3";

/// The native formal CLI is compiled from exactly one baseline.  Keeping the
/// root in the binary (rather than accepting an arbitrary caller supplied
/// root) prevents a run from silently consuming another baseline's resources
/// or publishing artifacts outside zkDL.
pub const BASELINE_ROOT: &str = env!("CARGO_MANIFEST_DIR");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CliPathRoot {
    Artifacts,
    RunsTmp,
}

impl CliPathRoot {
    fn name(self) -> &'static str {
        match self {
            Self::Artifacts => "artifacts",
            // The public runner's single-run output directory is `runs/`.
            // Keep the internal variant name for source compatibility, but
            // authenticate the actual baseline-local production root.
            Self::RunsTmp => "runs",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicStatement {
    pub protocol_id: String,
    pub adapter_schema: String,
    pub adapter_version: u32,
    pub protocol_sha256: String,
    pub canonical_relation_digest: String,
    pub proof_task: String,
    pub privacy_policy: String,
    pub input_privacy: String,
    pub output_mode: String,
    pub rounding_mode: String,
    pub dataset: String,
    pub input_preprocess_id: String,
    pub data_identity: String,
    pub pp_id: String,
    pub registry: RelationRegistry,
    pub node_count: usize,
    pub edge_count: usize,
    pub graph_count: usize,
    pub feature_dim: usize,
    pub class_count: usize,
    pub public_features: Vec<i64>,
    pub node_ptr: Vec<usize>,
    pub edge_ptr: Vec<usize>,
    pub src: Vec<usize>,
    pub dst: Vec<usize>,
    pub graph_features_digest: String,
    pub topology_digest: String,
    pub checkpoint_manifest_id: String,
    pub model_shape_id: String,
    pub commitment_set_id: String,
    pub quant_cfg_id: String,
    pub static_tables_id: String,
    pub static_table_rows_id: String,
    pub static_table_profile: serde_json::Value,
    pub integer_bound_ledger_id: String,
    pub integer_bound_ledger: serde_json::Value,
    pub model_arch_id: String,
    pub model_param_id: String,
    pub public_logits: Vec<i64>,
    pub resource_manifest_id: String,
    pub config_sha256: String,
    pub source_input_id: String,
    pub statement_id: String,
}
impl PublicStatement {
    pub fn canonical_digest(&self) -> Result<[u8; 32]> {
        let mut x = self.clone();
        x.statement_id.clear();
        let bytes = bincode::serialize(&x)?;
        let mut h = Sha256::new();
        h.update(b"zkDL-GAT-public-statement-v1\0");
        h.update(bytes);
        Ok(h.finalize().into())
    }
    pub fn seal(mut self) -> Result<Self> {
        self.registry.validate()?;
        self.statement_id = hex::encode(self.canonical_digest()?);
        self.validate()?;
        Ok(self)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.protocol_id == "GAT-ZKML-zkDL"
                && self.proof_task == "node_classification_full_gat",
            "statement task mismatch"
        );
        ensure!(
            self.adapter_schema == crate::compiler::CANONICAL_ADAPTER_SCHEMA
                && self.adapter_version == crate::compiler::CANONICAL_ADAPTER_VERSION
                && self.protocol_sha256 == crate::compiler::protocol_digest()
                && self.canonical_relation_digest.starts_with("sha256:")
                && self.canonical_relation_digest.len() == 71,
            "statement canonical adapter identity mismatch"
        );
        ensure!(
            self.privacy_policy == "model_and_trace_hidden"
                && self.input_privacy == "public_graph_public_features"
                && self.output_mode == "public_logits",
            "statement privacy/output mismatch"
        );
        ensure!(
            self.rounding_mode == SUPPORTED_ROUNDING_MODE,
            "statement rounding mode is unsupported or was silently substituted"
        );
        ensure!(
            !self.dataset.is_empty()
                && !self.pp_id.is_empty()
                && ((self.dataset == "cora"
                    || self.dataset == "citeseer"
                    || self.dataset == "pubmed"
                    || self.dataset == "ppi")
                    && self.input_preprocess_id == "row_normalize"
                    || self.dataset == "ogbn-arxiv"
                        && self.input_preprocess_id == "ogbn_arxiv_raw_float32"),
            "dataset/PP/input preprocessing identity missing or non-canonical"
        );
        ensure!(
            self.quant_cfg_id == "gat_l3_scale_chain",
            "statement quantization identity is not the canonical scale chain"
        );
        ensure!(
            canonical_sha256_id(&self.data_identity)
                && canonical_sha256_id(&self.source_input_id),
            "statement source/data identity is malformed"
        );
        for (label, value) in [
            ("checkpoint manifest", &self.checkpoint_manifest_id),
            ("model shape", &self.model_shape_id),
            ("commitment set", &self.commitment_set_id),
            ("static table rows", &self.static_table_rows_id),
            ("static tables", &self.static_tables_id),
            ("integer bounds", &self.integer_bound_ledger_id),
            ("resource manifest", &self.resource_manifest_id),
            ("formal config", &self.config_sha256),
        ] {
            ensure!(
                value
                    .strip_prefix("sha256:")
                    .map(|x| x.len() == 64 && x.bytes().all(|b| b.is_ascii_hexdigit()))
                    == Some(true),
                "{} digest is not canonical sha256",
                label
            );
        }
        ensure!(
            self.node_count > 0
                && self.edge_count >= self.node_count
                && self.graph_count > 0
                && self.feature_dim > 0
                && self.class_count > 0,
            "invalid public instance dimensions"
        );
        ensure!(
            self.public_features.len()
                == self
                    .node_count
                    .checked_mul(self.feature_dim)
                    .ok_or_else(|| anyhow::anyhow!("feature dimension overflow"))?,
            "public feature shape mismatch"
        );
        ensure!(
            self.public_logits.len()
                == self
                    .node_count
                    .checked_mul(self.class_count)
                    .ok_or_else(|| anyhow::anyhow!("logit dimension overflow"))?,
            "public logits shape mismatch"
        );
        ensure!(
            self.src.len() == self.edge_count && self.dst.len() == self.edge_count,
            "public edge payload shape mismatch"
        );
        ensure!(
            self.node_ptr.len() == self.graph_count + 1
                && self.edge_ptr.len() == self.graph_count + 1
                && self.node_ptr[0] == 0
                && self.edge_ptr[0] == 0
                && *self.node_ptr.last().unwrap() == self.node_count
                && *self.edge_ptr.last().unwrap() == self.edge_count,
            "public graph pointers malformed"
        );
        ensure!(
            self.node_ptr.windows(2).all(|x| x[0] < x[1])
                && self.edge_ptr.windows(2).all(|x| x[0] < x[1]),
            "public graph pointers must be strictly increasing"
        );
        let mut indegree = vec![0usize; self.node_count];
        let mut self_loop = vec![false; self.node_count];
        for graph in 0..self.graph_count {
            let nlo = self.node_ptr[graph];
            let nhi = self.node_ptr[graph + 1];
            let elo = self.edge_ptr[graph];
            let ehi = self.edge_ptr[graph + 1];
            let mut previous_dst = None;
            for edge in elo..ehi {
                let src = self.src[edge];
                let dst = self.dst[edge];
                ensure!(
                    src >= nlo && src < nhi && dst >= nlo && dst < nhi,
                    "cross-graph edge or endpoint out of range"
                );
                if let Some(previous) = previous_dst {
                    ensure!(
                        previous <= dst,
                        "edges are not in canonical graph/destination order"
                    );
                }
                previous_dst = Some(dst);
                indegree[dst] += 1;
                if src == dst {
                    self_loop[dst] = true;
                }
            }
        }
        ensure!(
            indegree.iter().all(|x| *x > 0) && self_loop.iter().all(|x| *x),
            "canonical topology must include every self-loop and nonempty in-neighborhood"
        );
        ensure!(
            self.graph_features_digest
                == feature_digest(self.node_count, self.feature_dim, &self.public_features),
            "public feature digest mismatch"
        );
        ensure!(
            self.topology_digest
                == topology_digest(
                    self.node_count,
                    self.edge_count,
                    self.graph_count,
                    &self.node_ptr,
                    &self.edge_ptr,
                    &self.src,
                    &self.dst
                ),
            "public topology digest mismatch"
        );
        ensure!(
            scalar_payload_matches_i64(
                self.registry.public_tensors.get("public.features"),
                &self.public_features
            ),
            "public feature registry payload mismatch"
        );
        ensure!(
            scalar_payload_matches_usize(
                self.registry.public_tensors.get("public.edge_src"),
                &self.src
            ),
            "public src registry payload mismatch"
        );
        ensure!(
            scalar_payload_matches_usize(
                self.registry.public_tensors.get("public.edge_dst"),
                &self.dst
            ),
            "public dst registry payload mismatch"
        );
        ensure!(
            scalar_payload_matches_usize(
                self.registry.public_tensors.get("public.node_ptr"),
                &self.node_ptr
            ),
            "public node_ptr registry payload mismatch"
        );
        ensure!(
            scalar_payload_matches_usize(
                self.registry.public_tensors.get("public.edge_ptr"),
                &self.edge_ptr
            ),
            "public edge_ptr registry payload mismatch"
        );
        ensure!(
            scalar_payload_matches_i64(
                self.registry.public_tensors.get("public.logits"),
                &self.public_logits
            ),
            "public logits registry payload mismatch"
        );
        ensure!(
            self.statement_id == hex::encode(self.canonical_digest()?),
            "statement digest mismatch"
        );
        self.registry.validate()?;
        self.registry.validate_canonical_gat()?;
        ensure!(
            self.static_table_rows_id == self.registry.static_table_digest()?,
            "relation table rows do not match authenticated static-table digest"
        );
        ensure!(
            self.commitment_set_id == self.registry.registered_family_set_digest(),
            "registered commitment family set mismatch"
        );
        let table_profile = self
            .static_table_profile
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("static table profile must be an object"))?;
        ensure!(
            table_profile
                .get("static_tables_id")
                .and_then(serde_json::Value::as_str)
                == Some(self.static_tables_id.as_str()),
            "static table profile ID mismatch"
        );
        ensure!(
            json_digest_without(
                &self.static_table_profile,
                "static_tables_id",
                b"zkDL-GAT-static-table-profile-v1\0"
            )? == self.static_tables_id,
            "static table profile digest recomputation failed"
        );
        validate_static_tables(&self.registry, &self.static_table_profile)?;
        ensure!(
            table_profile
                .get("rows_digest")
                .and_then(serde_json::Value::as_str)
                .map(|value| value.starts_with("sha256:") && value.len() == 71)
                == Some(true),
            "static table row digest missing"
        );
        let ledger = self
            .integer_bound_ledger
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("integer bound ledger must be an object"))?;
        ensure!(
            ledger
                .get("integer_bound_ledger_id")
                .and_then(serde_json::Value::as_str)
                == Some(self.integer_bound_ledger_id.as_str()),
            "integer bound ledger ID mismatch"
        );
        ensure!(
            json_digest_without(&self.integer_bound_ledger, "integer_bound_ledger_id", b"")?
                == self.integer_bound_ledger_id,
            "integer bound ledger digest recomputation failed"
        );
        ensure!(
            ledger
                .get("bit_width")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0)
                >= 2,
            "integer bound ledger bit width missing"
        );
        validate_integer_bound_ledger(&self.integer_bound_ledger, &self.static_table_profile)?;
        validate_ledger_statement_binding(self)?;
        validate_public_feature_bound(self)?;
        self.registry.validate_canonical_gat()?;
        Ok(())
    }
}

fn scalar_payload_matches_i64(actual: Option<&Vec<Scalar>>, expected: &[i64]) -> bool {
    actual
        .map(|values| {
            values.len() == expected.len()
                && values
                    .iter()
                    .zip(expected)
                    .all(|(left, right)| *left == Scalar::from_i64(*right))
        })
        .unwrap_or(false)
}

fn json_digest_without(value: &serde_json::Value, field: &str, domain: &[u8]) -> Result<String> {
    let mut copy = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("digest source must be an object"))?
        .clone();
    copy.remove(field);
    let bytes = serde_json::to_vec(&serde_json::Value::Object(copy))?;
    let mut h = Sha256::new();
    h.update(domain);
    h.update(bytes);
    Ok(format!("sha256:{}", hex::encode(h.finalize())))
}

fn profile_object<'a>(
    profile: &'a serde_json::Value,
    name: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    profile
        .get(name)
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("static table profile entry {name} is missing"))
}

fn profile_i64(profile: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<i64> {
    profile
        .get(key)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| anyhow::anyhow!("static table profile integer {key} is missing"))
}

fn profile_str<'a>(
    profile: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<&'a str> {
    profile
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("static table profile string {key} is missing"))
}

fn table_output_lrelu(value: i64) -> i64 {
    if value >= 0 {
        value
    } else {
        let a = -(value as i128);
        -((2 * a + 5) / 10) as i64
    }
}

fn round_ties_away(value: f64) -> Result<i64> {
    ensure!(value.is_finite(), "static table output is not finite");
    let rounded = if value >= 0.0 {
        (value + 0.5).floor()
    } else {
        (value - 0.5).ceil()
    };
    ensure!(
        rounded >= i64::MIN as f64 && rounded <= i64::MAX as f64,
        "static table output exceeds signed integer range"
    );
    Ok(rounded as i64)
}

fn table_output_elu(value: i64) -> Result<i64> {
    if value >= 0 {
        Ok(value)
    } else {
        round_ties_away(16.0 * ((value as f64 / 16.0).exp() - 1.0))
    }
}

fn table_output_exp(value: i64) -> Result<i64> {
    Ok(1.max(round_ties_away(16.0 * (-(value as f64) / 256.0).exp())?))
}

fn expected_table(
    min: i64,
    max: i64,
    output: impl Fn(i64) -> Result<i64>,
) -> Result<(Vec<Scalar>, Vec<Scalar>, usize)> {
    ensure!(max >= min, "static table domain is inverted");
    let raw_len = (max as i128 - min as i128 + 1) as usize;
    ensure!(
        raw_len > 0 && raw_len <= (1usize << 27),
        "static table domain is too large for the native full public MLE vector"
    );
    let padded_len = raw_len.max(2).next_power_of_two();
    let mut values = Vec::with_capacity(padded_len);
    for index in 0..raw_len {
        values.push(Scalar::from_i64(output(min + index as i64)?));
    }
    values.resize(padded_len, Scalar::zero());
    let mut selector = vec![Scalar::one(); raw_len];
    selector.resize(padded_len, Scalar::zero());
    Ok((values, selector, padded_len.trailing_zeros() as usize))
}

fn validate_static_tables(registry: &RelationRegistry, profile: &serde_json::Value) -> Result<()> {
    let profile_obj = profile
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("static table profile must be an object"))?;
    ensure!(
        profile_str(profile_obj, "schema")? == "zkDL_gat_static_tables_v1"
            && profile_str(profile_obj, "generator_id")? == "main_gatzk_quantized_tables_v1"
            && profile_str(profile_obj, "rounding_mode")? == "nearest_ties_away"
            && profile_str(profile_obj, "canonical_order")? == "kind_then_signed_input_ascending",
        "static table generator/rounding profile mismatch"
    );
    let lrelu = profile_object(profile, "lrelu")?;
    let elu = profile_object(profile, "elu")?;
    let exp = profile_object(profile, "exp")?;
    ensure!(
        profile_i64(lrelu, "input_scale")? == 256
            && profile_i64(lrelu, "output_scale")? == 256
            && profile_i64(elu, "input_scale")? == 16
            && profile_i64(elu, "output_scale")? == 16
            && profile_i64(exp, "input_scale")? == 256
            && profile_i64(exp, "output_scale")? == 16,
        "static table quantization scales are not the canonical zkDL scales"
    );
    let lmin = profile_i64(lrelu, "min")?;
    let lmax = profile_i64(lrelu, "max")?;
    let emin = profile_i64(elu, "min")?;
    let emax = profile_i64(elu, "max")?;
    let xmin = profile_i64(exp, "min")?;
    let xmax = profile_i64(exp, "max")?;
    ensure!(
        lmin <= 0 && lmax >= 0 && emin <= 0 && emax >= 0 && xmin == 0 && xmax >= 0,
        "static table domains do not cover the canonical zero point"
    );
    for (entry, min, max) in [(lrelu, lmin, lmax), (elu, emin, emax), (exp, xmin, xmax)] {
        let rows = (max as i128 - min as i128 + 1) as i64;
        ensure!(
            rows > 0 && profile_i64(entry, "rows")? == rows,
            "static table row count mismatch"
        );
    }
    let (l_table, l_selector, l_bits) =
        expected_table(lmin, lmax, |value| Ok(table_output_lrelu(value)))?;
    let (e_table, e_selector, e_bits) = expected_table(emin, emax, table_output_elu)?;
    let (x_table, x_selector, x_bits) = expected_table(xmin, xmax, table_output_exp)?;
    let expected = [
        (&l_table, l_bits),
        (&l_selector, l_bits),
        (&e_table, e_bits),
        (&e_selector, e_bits),
        (&x_table, x_bits),
        (&x_selector, x_bits),
    ];
    let tables = registry.public_tables();
    for (values, bits) in &tables {
        ensure!(
            expected.iter().any(|(candidate, width)| {
                *width == bits.len() && candidate.as_slice() == values.as_slice()
            }),
            "relation contains a public lookup table outside the authenticated static profile"
        );
    }
    for (candidate, width) in expected {
        ensure!(
            tables
                .iter()
                .any(|(values, bits)| bits.len() == width
                    && values.as_slice() == candidate.as_slice()),
            "authenticated static table row set is missing from the relation"
        );
    }
    let mut rows_digest = Sha256::new();
    rows_digest.update(b"zkDL-GAT-static-table-rows-v1\0");
    for (kind, min, max, output) in [
        ("lrelu", lmin, lmax, 0u8),
        ("elu", emin, emax, 1u8),
        ("exp", xmin, xmax, 2u8),
    ] {
        rows_digest.update(kind.as_bytes());
        rows_digest.update([0]);
        for value in min..=max {
            let output = match output {
                0 => table_output_lrelu(value),
                1 => table_output_elu(value)?,
                _ => table_output_exp(value)?,
            };
            rows_digest.update(value.to_be_bytes());
            rows_digest.update(output.to_be_bytes());
        }
    }
    let expected_rows_digest = format!("sha256:{}", hex::encode(rows_digest.finalize()));
    ensure!(
        profile_str(profile_obj, "rows_digest")? == expected_rows_digest,
        "static table rows digest does not match the canonical generator"
    );
    Ok(())
}

fn ledger_u128(value: &serde_json::Value, label: &str) -> Result<u128> {
    if let Some(number) = value.as_u64() {
        return Ok(number as u128);
    }
    if let Some(text) = value.as_str() {
        return text
            .parse::<u128>()
            .with_context(|| format!("invalid integer ledger value {label}"));
    }
    Err(anyhow::anyhow!(
        "integer ledger value {label} is not a nonnegative integer"
    ))
}

fn ledger_i64(value: &serde_json::Value, label: &str) -> Result<i64> {
    value
        .as_i64()
        .ok_or_else(|| anyhow::anyhow!("integer ledger signed value {label} is missing"))
}

fn ledger_domain(value: &serde_json::Value, label: &str) -> Result<(i64, i64)> {
    let values = value
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("integer ledger table domain {label} is not an array"))?;
    ensure!(
        values.len() == 2,
        "integer ledger table domain {label} shape mismatch"
    );
    Ok((
        ledger_i64(&values[0], label)?,
        ledger_i64(&values[1], label)?,
    ))
}

fn validate_integer_bound_ledger(
    ledger: &serde_json::Value,
    profile: &serde_json::Value,
) -> Result<()> {
    const FIELD_HALF_DEC: &str =
        "26217937587563095239723870254092982918845276250263818911301829349969290592256";
    let object = ledger
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("integer bound ledger must be an object"))?;
    ensure!(
        object.get("schema").and_then(serde_json::Value::as_str)
            == Some("zkDL_gat_integer_bound_ledger_v3"),
        "integer bound ledger schema mismatch"
    );
    ensure!(
        object.get("field").and_then(serde_json::Value::as_str) == Some("BLS12-381-Fr")
            && object
                .get("field_fit")
                .and_then(serde_json::Value::as_str)
                .map(|text| !text.is_empty())
                == Some(true),
        "integer bound ledger field certificate missing"
    );
    ensure!(
        object
            .get("rounding_mode")
            .and_then(serde_json::Value::as_str)
            == Some(SUPPORTED_ROUNDING_MODE),
        "integer bound ledger rounding mode mismatch"
    );
    ensure!(
        object
            .get("field_half_modulus")
            .and_then(serde_json::Value::as_str)
            == Some(FIELD_HALF_DEC),
        "integer bound ledger field half-modulus mismatch"
    );
    ensure!(
        object
            .get("failures")
            .and_then(serde_json::Value::as_array)
            .map(Vec::is_empty)
            == Some(true),
        "integer bound ledger contains preflight failures"
    );
    let bit_width = object
        .get("bit_width")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("integer bound ledger bit width missing"))?;
    ensure!(
        (2..254).contains(&bit_width),
        "integer bound ledger bit width cannot fit the signed field"
    );
    let profile_lrelu = profile_object(profile, "lrelu")?;
    let profile_elu = profile_object(profile, "elu")?;
    let profile_exp = profile_object(profile, "exp")?;
    let domains = object
        .get("table_domains")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("integer ledger table domains missing"))?;
    for (name, profile_entry) in [
        ("lrelu", profile_lrelu),
        ("elu", profile_elu),
        ("exp", profile_exp),
    ] {
        let ledger_range = ledger_domain(
            domains
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("integer ledger domain {name} missing"))?,
            name,
        )?;
        let profile_range = (
            profile_i64(profile_entry, "min")?,
            profile_i64(profile_entry, "max")?,
        );
        ensure!(
            ledger_range == profile_range,
            "integer ledger/profile domain mismatch for {name}"
        );
    }
    let half = FIELD_HALF_DEC.parse::<u128>()?;
    let mut max_bound = 0u128;
    let heads = object
        .get("heads")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("integer bound ledger heads missing"))?;
    ensure!(
        !heads.is_empty(),
        "integer bound ledger has no family certificates"
    );
    for (index, head) in heads.iter().enumerate() {
        let record = head
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("integer ledger head {index} is not an object"))?;
        ensure!(
            record
                .get("kind")
                .and_then(serde_json::Value::as_str)
                .map(|kind| !kind.is_empty())
                == Some(true),
            "integer ledger head kind missing"
        );
        let mut values = std::collections::BTreeMap::new();
        for (key, value) in record {
            if key.starts_with("B_") {
                let parsed = ledger_u128(value, key)?;
                ensure!(
                    parsed < half,
                    "integer ledger bound {key} exceeds field half-modulus"
                );
                max_bound = max_bound.max(parsed);
                values.insert(key.as_str(), parsed);
            }
        }
        for key in [
            "B_a",
            "B_d",
            "B_q",
            "B_r",
            "B_delta",
            "B_2a_plus_d",
            "B_2dq_plus_r",
            "B_r_plus_delta",
        ] {
            ensure!(
                values.contains_key(key),
                "integer ledger head {index} missing {key}"
            );
        }
        let ba = values["B_a"];
        let bd = values["B_d"];
        let bq = values["B_q"];
        let br = values["B_r"];
        let bdelta = values["B_delta"];
        ensure!(bd > 0, "integer ledger denominator bound must be positive");
        let expected_r = 2u128
            .checked_mul(ba)
            .and_then(|value| value.checked_add(bd))
            .and_then(|value| value.checked_add(2u128.checked_mul(bd)?.checked_mul(bq)?))
            .ok_or_else(|| anyhow::anyhow!("integer ledger composite bound overflow"))?;
        let expected_delta = 2u128
            .checked_mul(bd)
            .and_then(|value| value.checked_sub(1))
            .and_then(|value| value.checked_add(br))
            .ok_or_else(|| anyhow::anyhow!("integer ledger delta bound overflow"))?;
        ensure!(
            br == expected_r && bdelta == expected_delta,
            "integer ledger r/delta relation mismatch"
        );
        ensure!(
            values["B_2a_plus_d"]
                == 2u128
                    .checked_mul(ba)
                    .and_then(|value| value.checked_add(bd))
                    .ok_or_else(|| anyhow::anyhow!("integer ledger 2a+d overflow"))?
                && values["B_2dq_plus_r"]
                    == 2u128
                        .checked_mul(bd)
                        .and_then(|value| value.checked_mul(bq))
                        .and_then(|value| value.checked_add(br))
                        .ok_or_else(|| anyhow::anyhow!("integer ledger 2dq+r overflow"))?
                && values["B_r_plus_delta"]
                    == br
                        .checked_add(bdelta)
                        .ok_or_else(|| anyhow::anyhow!("integer ledger r+delta overflow"))?,
            "integer ledger composite equality mismatch"
        );
        if let Some(score) = values.get("B_S") {
            let range = ledger_domain(domains.get("lrelu").unwrap(), "lrelu")?;
            ensure!(
                -range.0 as u128 <= *score && range.1 as u128 >= *score,
                "LReLU ledger domain does not cover B_S"
            );
        }
        if let Some(delta) = values.get("B_delta_raw") {
            let range = ledger_domain(domains.get("exp").unwrap(), "exp")?;
            ensure!(
                range.0 >= 0 && range.1 as u128 >= *delta,
                "ExpMap ledger domain does not cover B_delta"
            );
        }
        if record.get("kind").and_then(serde_json::Value::as_str) == Some("hidden") {
            if let Some(q) = values.get("B_q_div") {
                let range = ledger_domain(domains.get("elu").unwrap(), "elu")?;
                ensure!(
                    -range.0 as u128 <= *q && range.1 as u128 >= *q,
                    "ELU ledger domain does not cover B_q"
                );
            }
        }
    }
    ensure!(
        max_bound < half,
        "integer ledger maximum bound exceeds field half-modulus"
    );
    let required_bits = if max_bound == 0 {
        0
    } else {
        128 - max_bound.leading_zeros() as u64
    };
    ensure!(
        bit_width >= required_bits,
        "integer ledger bit width does not cover certified bounds"
    );
    Ok(())
}

fn scalar_payload_matches_usize(actual: Option<&Vec<Scalar>>, expected: &[usize]) -> bool {
    actual
        .map(|values| {
            values.len() == expected.len()
                && values.iter().zip(expected).all(|(left, right)| {
                    i64::try_from(*right)
                        .map(|value| *left == Scalar::from_i64(value))
                        .unwrap_or(false)
                })
        })
        .unwrap_or(false)
}

fn ledger_string<'a>(
    object: &'a serde_json::Map<String, serde_json::Value>,
    key: &str,
) -> Result<&'a str> {
    object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow::anyhow!("integer ledger identity field {key} is missing"))
}

fn ledger_usize(object: &serde_json::Map<String, serde_json::Value>, key: &str) -> Result<usize> {
    let value = object
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("integer ledger identity field {key} is missing"))?;
    usize::try_from(value)
        .map_err(|_| anyhow::anyhow!("integer ledger field {key} overflows usize"))
}

fn validate_ledger_statement_binding(statement: &PublicStatement) -> Result<()> {
    let object = statement
        .integer_bound_ledger
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("integer bound ledger must be an object"))?;
    ensure!(
        ledger_string(object, "dataset")? == statement.dataset,
        "integer ledger dataset identity mismatch"
    );
    ensure!(
        ledger_string(object, "input_preprocess_id")? == statement.input_preprocess_id,
        "integer ledger input preprocessing identity mismatch"
    );
    ensure!(
        ledger_string(object, "quant_cfg_id")? == statement.quant_cfg_id,
        "integer ledger quantization identity mismatch"
    );
    for (key, expected) in [
        ("node_count", statement.node_count),
        ("edge_count", statement.edge_count),
        ("graph_count", statement.graph_count),
        ("feature_dim", statement.feature_dim),
        ("class_count", statement.class_count),
    ] {
        ensure!(
            ledger_usize(object, key)? == expected,
            "integer ledger instance identity mismatch for {key}"
        );
    }
    for (key, expected) in [
        (
            "resource_manifest_id",
            statement.resource_manifest_id.as_str(),
        ),
        ("config_sha256", statement.config_sha256.as_str()),
        (
            "checkpoint_manifest_id",
            statement.checkpoint_manifest_id.as_str(),
        ),
        ("model_shape_id", statement.model_shape_id.as_str()),
        ("static_tables_id", statement.static_tables_id.as_str()),
        (
            "static_table_rows_id",
            statement.static_table_rows_id.as_str(),
        ),
        ("model_arch_id", statement.model_arch_id.as_str()),
        ("model_param_id", statement.model_param_id.as_str()),
        ("data_identity", statement.data_identity.as_str()),
    ] {
        ensure!(
            ledger_string(object, key)? == expected,
            "integer ledger authenticated identity mismatch for {key}"
        );
    }
    ensure!(
        ledger_string(object, "rounding_mode")? == statement.rounding_mode,
        "integer ledger rounding mode mismatch"
    );
    let scales = object
        .get("scales")
        .ok_or_else(|| anyhow::anyhow!("integer ledger quantization scales are missing"))?;
    let scale_bytes = serde_json::to_vec(scales)?;
    let mut h = Sha256::new();
    h.update(scale_bytes);
    let computed = format!("sha256:{}", hex::encode(h.finalize()));
    ensure!(
        object
            .get("quantization_scales_digest")
            .and_then(serde_json::Value::as_str)
            == Some(computed.as_str()),
        "integer ledger quantization scale digest mismatch"
    );
    Ok(())
}

fn ledger_head_for_input<'a>(
    ledger: &'a serde_json::Value,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    let heads = ledger
        .get("heads")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("integer ledger heads missing"))?;
    heads
        .iter()
        .find(|head| {
            let object = match head.as_object() {
                Some(value) => value,
                None => return false,
            };
            let kind = object.get("kind").and_then(serde_json::Value::as_str);
            let layer = object.get("layer").and_then(serde_json::Value::as_u64);
            let head = object.get("head").and_then(serde_json::Value::as_u64);
            (kind == Some("hidden") && layer == Some(0) && head == Some(0))
                || (kind == Some("output") && layer.is_none() && head == Some(0))
        })
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("integer ledger has no canonical input-bound family"))
}

fn validate_public_feature_bound(statement: &PublicStatement) -> Result<()> {
    let head = ledger_head_for_input(&statement.integer_bound_ledger)?;
    let bound = ledger_u128(
        head.get("B_H")
            .ok_or_else(|| anyhow::anyhow!("canonical input-bound family is missing B_H"))?,
        "B_H",
    )?;
    ensure!(
        bound < (i64::MAX as u128) + 1,
        "public feature bound exceeds signed input range"
    );
    for (index, value) in statement.public_features.iter().enumerate() {
        let magnitude = (*value as i128).unsigned_abs();
        ensure!(
            magnitude <= bound,
            "public feature {} exceeds authenticated B_H input bound",
            index
        );
    }
    Ok(())
}

fn feature_digest(nodes: usize, features: usize, values: &[i64]) -> String {
    let mut h = Sha256::new();
    h.update(b"zkDL-GAT-public-features-v1\0");
    h.update((nodes as u64).to_be_bytes());
    h.update((features as u64).to_be_bytes());
    for value in values {
        h.update(value.to_be_bytes());
    }
    format!("sha256:{}", hex::encode(h.finalize()))
}
fn topology_digest(
    nodes: usize,
    edges: usize,
    graphs: usize,
    node_ptr: &[usize],
    edge_ptr: &[usize],
    src: &[usize],
    dst: &[usize],
) -> String {
    let mut h = Sha256::new();
    h.update(b"zkDL-GAT-public-topology-v1\0");
    for value in [nodes, edges, graphs] {
        h.update((value as u64).to_be_bytes());
    }
    for values in [node_ptr, edge_ptr, src, dst] {
        h.update((values.len() as u64).to_be_bytes());
        for value in values {
            h.update((*value as u64).to_be_bytes());
        }
    }
    format!("sha256:{}", hex::encode(h.finalize()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum ModelRole {
    Weight(usize),
    AttentionSrc,
    AttentionDst,
    Bias,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ParsedModelFamily {
    kind: String,
    layer: Option<usize>,
    head: usize,
    role: ModelRole,
}

fn parse_decimal_suffix(value: &str, prefix: &str, label: &str) -> Result<usize> {
    let text = value
        .strip_prefix(prefix)
        .ok_or_else(|| anyhow::anyhow!("model family {label} prefix mismatch"))?;
    ensure!(!text.is_empty(), "model family {label} index is empty");
    text.parse::<usize>()
        .with_context(|| format!("model family {label} index is not decimal"))
}

fn parse_model_family_id(family_id: &str) -> Result<ParsedModelFamily> {
    let (kind, layer, head, role_text) = if let Some(rest) = family_id.strip_prefix("hidden.layer_")
    {
        let (layer_text, rest) = rest.split_once(".head_").ok_or_else(|| {
            anyhow::anyhow!("hidden model family layer/head mapping is not canonical")
        })?;
        let (head_text, role) = rest
            .split_once('.')
            .ok_or_else(|| anyhow::anyhow!("hidden model family role mapping is not canonical"))?;
        ensure!(
            !layer_text.is_empty() && !head_text.is_empty(),
            "hidden model family index is empty"
        );
        (
            "hidden".to_string(),
            Some(
                layer_text
                    .parse::<usize>()
                    .context("hidden layer index is not decimal")?,
            ),
            head_text
                .parse::<usize>()
                .context("hidden head index is not decimal")?,
            role,
        )
    } else if let Some(rest) = family_id.strip_prefix("output.head_") {
        let (head_text, role) = rest
            .split_once('.')
            .ok_or_else(|| anyhow::anyhow!("output model family role mapping is not canonical"))?;
        ensure!(!head_text.is_empty(), "output model family index is empty");
        (
            "output".to_string(),
            None,
            head_text
                .parse::<usize>()
                .context("output head index is not decimal")?,
            role,
        )
    } else {
        bail!("registered-model family is outside the canonical model namespace: {family_id}");
    };
    let role = if role_text == "a_src" {
        ModelRole::AttentionSrc
    } else if role_text == "a_dst" {
        ModelRole::AttentionDst
    } else if role_text == "bias" {
        ensure!(kind == "output", "hidden model family cannot carry a bias");
        ModelRole::Bias
    } else if role_text.starts_with("Wcol_") {
        ModelRole::Weight(parse_decimal_suffix(role_text, "Wcol_", family_id)?)
    } else {
        bail!("registered-model family role is not canonical: {family_id}");
    };
    Ok(ParsedModelFamily {
        kind,
        layer,
        head,
        role,
    })
}

fn validate_registered_model_family_layout(statement: &PublicStatement) -> Result<()> {
    let mut groups: BTreeMap<(String, Option<usize>, usize), BTreeSet<ModelRole>> = BTreeMap::new();
    for family in &statement.registry.tensor_families {
        if family.visibility == Visibility::RegisteredModel {
            let parsed = parse_model_family_id(&family.family_id)?;
            let role_set = groups
                .entry((parsed.kind, parsed.layer, parsed.head))
                .or_default();
            ensure!(
                role_set.insert(parsed.role),
                "duplicate canonical model role"
            );
        }
    }
    ensure!(
        !groups.is_empty(),
        "canonical registry has no registered model families"
    );
    let mut hidden_layers: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut output_heads = BTreeSet::new();
    for (key, _) in &groups {
        match (key.0.as_str(), key.1, key.2) {
            ("hidden", Some(layer), head) => {
                hidden_layers.entry(layer).or_default().insert(head);
            }
            ("output", None, head) => {
                output_heads.insert(head);
            }
            _ => bail!("canonical model family namespace mapping is malformed"),
        }
    }
    if let Some(max_layer) = hidden_layers.keys().next_back().copied() {
        ensure!(
            hidden_layers.keys().copied().eq(0..=max_layer),
            "canonical hidden model layers are not contiguous from layer 0"
        );
        for (layer, heads) in hidden_layers {
            let max_head = heads
                .iter()
                .next_back()
                .copied()
                .ok_or_else(|| anyhow::anyhow!("canonical hidden layer has no heads"))?;
            ensure!(
                heads.iter().copied().eq(0..=max_head),
                "canonical hidden heads are not contiguous in layer {layer}"
            );
        }
    }
    if let Some(max_head) = output_heads.iter().next_back().copied() {
        ensure!(
            output_heads.iter().copied().eq(0..=max_head),
            "canonical output heads are not contiguous from head 0"
        );
    }
    for ((kind, _, _), roles) in groups {
        ensure!(
            roles.contains(&ModelRole::AttentionSrc) && roles.contains(&ModelRole::AttentionDst),
            "canonical {kind} model family is missing an attention vector"
        );
        let weight_indices = roles
            .iter()
            .filter_map(|role| match role {
                ModelRole::Weight(index) => Some(*index),
                _ => None,
            })
            .collect::<Vec<_>>();
        ensure!(
            !weight_indices.is_empty(),
            "canonical model family has no weight columns"
        );
        ensure!(
            weight_indices
                .iter()
                .enumerate()
                .all(|(position, index)| *index == position),
            "canonical model weight columns are not contiguous from Wcol_0"
        );
        if kind == "output" {
            ensure!(
                roles.contains(&ModelRole::Bias),
                "canonical output family is missing bias"
            );
        } else {
            ensure!(
                !roles.contains(&ModelRole::Bias),
                "canonical hidden family has an output bias"
            );
        }
    }
    Ok(())
}

fn ledger_head_for_model<'a>(
    ledger: &'a serde_json::Value,
    parsed: &ParsedModelFamily,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    let heads = ledger
        .get("heads")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("integer ledger heads missing"))?;
    let matches = heads
        .iter()
        .filter_map(serde_json::Value::as_object)
        .filter(|record| {
            let kind = record.get("kind").and_then(serde_json::Value::as_str);
            let head = record.get("head").and_then(serde_json::Value::as_u64);
            let layer = record.get("layer").and_then(serde_json::Value::as_u64);
            kind == Some(parsed.kind.as_str())
                && head == Some(parsed.head as u64)
                && layer == parsed.layer.map(|value| value as u64)
        })
        .collect::<Vec<_>>();
    ensure!(
        matches.len() == 1,
        "integer ledger has an ambiguous/missing canonical model family mapping"
    );
    Ok(matches[0])
}

fn model_bound_key(role: ModelRole) -> &'static str {
    match role {
        ModelRole::Weight(_) => "B_W",
        ModelRole::AttentionSrc => "B_a_src",
        ModelRole::AttentionDst => "B_a_dst",
        ModelRole::Bias => "B_b",
    }
}

const FIELD_MODULUS_BE: [u8; 32] = [
    0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48, 0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1, 0xd8, 0x05,
    0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01,
];
const FIELD_HALF_BE: [u8; 32] = [
    0x39, 0xf6, 0xd3, 0xa9, 0x94, 0xce, 0xbe, 0xa4, 0x19, 0x9c, 0xec, 0x04, 0x04, 0xd0, 0xec, 0x02,
    0xa9, 0xde, 0xd2, 0x01, 0x7f, 0xff, 0x2d, 0xff, 0x7f, 0xff, 0xff, 0xff, 0x80, 0x00, 0x00, 0x00,
];

fn cmp_be(left: &[u8; 32], right: &[u8; 32]) -> std::cmp::Ordering {
    left.iter().cmp(right.iter())
}

fn sub_be(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut borrow = 0i16;
    for index in (0..32).rev() {
        let value = left[index] as i16 - right[index] as i16 - borrow;
        if value < 0 {
            out[index] = (value + 256) as u8;
            borrow = 1;
        } else {
            out[index] = value as u8;
            borrow = 0;
        }
    }
    debug_assert_eq!(borrow, 0);
    out
}

fn scalar_abs_le(value: Scalar, bound: u128) -> bool {
    let magnitude = if cmp_be(&value.0, &FIELD_HALF_BE) == std::cmp::Ordering::Greater {
        sub_be(&FIELD_MODULUS_BE, &value.0)
    } else {
        value.0
    };
    if magnitude[..16].iter().any(|byte| *byte != 0) {
        return false;
    }
    u128::from_be_bytes(magnitude[16..].try_into().expect("16-byte scalar suffix")) <= bound
}

fn validate_registered_model_parameter_bounds(
    statement: &PublicStatement,
    witness: &WitnessPackage,
) -> Result<()> {
    validate_registered_model_family_layout(statement)?;
    for family in &statement.registry.tensor_families {
        if family.visibility != Visibility::RegisteredModel {
            continue;
        }
        let parsed = parse_model_family_id(&family.family_id)?;
        let record = ledger_head_for_model(&statement.integer_bound_ledger, &parsed)?;
        let key = model_bound_key(parsed.role);
        let bound = ledger_u128(
            record
                .get(key)
                .ok_or_else(|| anyhow::anyhow!("integer ledger family bound {key} is missing"))?,
            key,
        )?;
        let values = witness
            .tensors
            .get(&family.family_id)
            .ok_or_else(|| anyhow::anyhow!("missing registered-model witness family"))?;
        ensure!(
            values.len() == family.logical_len,
            "registered-model witness family shape mismatch"
        );
        ensure!(
            values.iter().all(|value| scalar_abs_le(*value, bound)),
            "registered-model family {} exceeds authenticated {}",
            family.family_id,
            key
        );
    }
    Ok(())
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub format: String,
    pub pp_id: String,
    pub registry_id: String,
    pub model_arch_id: String,
    pub model_param_id: String,
    pub input_preprocess_id: String,
    pub data_identity: String,
    pub source_input_id: String,
    pub checkpoint_manifest_id: String,
    pub model_shape_id: String,
    pub quant_cfg_id: String,
    pub static_tables_id: String,
    pub static_table_rows_id: String,
    pub certified_bounds_id: String,
    pub registration_transaction_id: String,
    pub registration_binding_digest: String,
    pub commitment_set_id: String,
    pub commitment_points_id: String,
    pub commitments: BTreeMap<String, VectorCommitment>,
    pub registry_object_id: String,
}
impl ModelRegistry {
    fn commitment_points_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"zkDL-GAT-model-commitment-points-v1\0");
        for (family_id, commitment) in &self.commitments {
            h.update((family_id.len() as u64).to_be_bytes());
            h.update(family_id.as_bytes());
            h.update((commitment.logical_len as u64).to_be_bytes());
            h.update(commitment.point.0);
        }
        format!("sha256:{}", hex::encode(h.finalize()))
    }
    fn digest(&self) -> Result<[u8; 32]> {
        let mut x = self.clone();
        x.registry_object_id.clear();
        let b = bincode::serialize(&x)?;
        let mut h = Sha256::new();
        h.update(b"zkDL-GAT-model-registry-v1\0");
        h.update(b);
        Ok(h.finalize().into())
    }
    fn seal(mut self) -> Result<Self> {
        self.registry_object_id = hex::encode(self.digest()?);
        Ok(self)
    }
    pub fn validate(&self, pp: &CommitmentParameters, s: &PublicStatement) -> Result<()> {
        ensure!(
            self.format == "zkDL-GAT-registered-model-v1"
                && self.pp_id == pp.parameters_id
                && self.registry_id == s.registry.registry_id,
            "model registry context mismatch"
        );
        validate_registered_model_family_layout(s)?;
        ensure!(
            self.model_arch_id == s.model_arch_id && self.model_param_id == s.model_param_id,
            "model identity mismatch"
        );
        ensure!(
            self.input_preprocess_id == s.input_preprocess_id
                && self.data_identity == s.data_identity
                && self.source_input_id == s.source_input_id,
            "registered model source/data identity mismatch"
        );
        ensure!(
            self.checkpoint_manifest_id == s.checkpoint_manifest_id
                && self.model_shape_id == s.model_shape_id
                && self.quant_cfg_id == s.quant_cfg_id
                && self.static_tables_id == s.static_tables_id
                && self.static_table_rows_id == s.static_table_rows_id,
            "registered model metadata does not match the public statement"
        );
        ensure!(
            self.certified_bounds_id == s.integer_bound_ledger_id,
            "registered model is not bound to the statement's certified integer-bound ledger"
        );
        ensure!(
            canonical_sha256_id(&self.registration_transaction_id)
                && canonical_sha256_id(&self.registration_binding_digest),
            "registered model transaction/binding identity is malformed"
        );
        ensure!(
            self.commitment_set_id == s.commitment_set_id,
            "registered model family-set digest mismatch"
        );
        ensure!(
            self.commitment_points_id == self.commitment_points_digest(),
            "registered model commitment-point digest mismatch"
        );
        ensure!(
            self.registry_object_id == hex::encode(self.digest()?),
            "model registry digest mismatch"
        );
        ensure!(
            self.commitments.len()
                == s.registry
                    .tensor_families
                    .iter()
                    .filter(|tensor| tensor.visibility == Visibility::RegisteredModel)
                    .count(),
            "registered model commitment set mismatch"
        );
        for t in &s.registry.tensor_families {
            if t.visibility == Visibility::RegisteredModel {
                let c = self
                    .commitments
                    .get(&t.family_id)
                    .ok_or_else(|| anyhow::anyhow!("missing registered model family"))?;
                ensure!(
                    c.logical_len == t.logical_len,
                    "model commitment shape mismatch"
                );
                ensure!(
                    !c.point.eq_checked(Point::identity())?,
                    "registered model commitment must be non-identity"
                );
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelOpeningSecret {
    pub registry_object_id: String,
    pub registration_transaction_id: String,
    pub registration_binding_digest: String,
    pub blinders: BTreeMap<String, Scalar>,
}

fn canonical_sha256_id(value: &str) -> bool {
    value
        .strip_prefix("sha256:")
        .map(|text| text.len() == 64 && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        == Some(true)
}

fn fresh_registration_transaction_id() -> Result<String> {
    let mut nonce = [0u8; 32];
    getrandom::getrandom(&mut nonce)
        .map_err(|error| anyhow::anyhow!("OS randomness failed: {error}"))?;
    let mut hash = Sha256::new();
    hash.update(b"zkDL-GAT-registration-transaction-v1\0");
    hash.update(nonce);
    Ok(format!("sha256:{}", hex::encode(hash.finalize())))
}

/// Bind the public registry and private opening secret without putting any
/// secret bytes in the public object.  The transaction and object IDs are
/// cleared while hashing so the binding remains stable when the final object
/// digest is sealed after the binding is attached.
fn compute_registration_binding_digest(
    registry: &ModelRegistry,
    secret: &ModelOpeningSecret,
) -> Result<String> {
    let mut public = registry.clone();
    public.registry_object_id.clear();
    public.registration_binding_digest.clear();
    let mut private = secret.clone();
    private.registry_object_id.clear();
    private.registration_transaction_id.clear();
    private.registration_binding_digest.clear();
    let mut hash = Sha256::new();
    hash.update(b"zkDL-GAT-registration-binding-v1\0");
    hash.update(bincode::serialize(&public)?);
    hash.update(bincode::serialize(&private)?);
    Ok(format!("sha256:{}", hex::encode(hash.finalize())))
}

/// Schnorr proof that a proof-local registered-model commitment is a
/// Pedersen re-randomization of the authenticated registry commitment.
/// The delta itself is never serialized; only this proof and the session
/// commitment are sent to the verifier.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionRerandomizationProof {
    pub nonce_commitment: Point,
    pub response: Scalar,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofEnvelope {
    pub format: String,
    pub statement_id: String,
    pub model_registry_id: String,
    pub device: String,
    pub fresh_nonce: [u8; 32],
    pub model_session_commitments: BTreeMap<String, VectorCommitment>,
    pub model_rerandomization: BTreeMap<String, SessionRerandomizationProof>,
    pub dynamic_commitments: BTreeMap<String, VectorCommitment>,
    pub operations: Vec<OperationProof>,
    pub gpu_telemetry: Option<GpuTelemetry>,
    pub commitment_time_ms: f64,
    pub prove_time_ms: f64,
    pub transcript_digest: [u8; 32],
}

fn absorb_gpu_telemetry(
    transcript: &mut Transcript,
    device: &str,
    telemetry: &Option<GpuTelemetry>,
) -> Result<()> {
    transcript.label("formal-prover-device-telemetry-v1");
    transcript.label(device);
    match telemetry {
        Some(value) => transcript.bytes(b"gpu-telemetry", &bincode::serialize(value)?),
        None => transcript.bytes(b"gpu-telemetry", b"cpu-no-gpu-telemetry"),
    }
    Ok(())
}

fn validate_gpu_telemetry(device: &str, telemetry: &Option<GpuTelemetry>) -> Result<()> {
    if device == "cpu" {
        ensure!(telemetry.is_none(), "CPU proof carries GPU telemetry");
        return Ok(());
    }
    let value = telemetry
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("GPU proof is missing native kernel telemetry"))?;
    ensure!(
        value.provider_id == "zkDL-native-CUDA-BLS12-381-v1"
            && value.probe_kernel_invocations == 1
            && value.cross_backend_kat_passed,
        "GPU provider identity/probe/KAT telemetry rejected"
    );
    ensure!(
        value.kat_vectors_digest.len() == 64
            && value
                .kat_vectors_digest
                .bytes()
                .all(|x| x.is_ascii_hexdigit()),
        "GPU cross-backend KAT digest rejected"
    );
    ensure!(
        value.msm_kernel_invocations > 0
            && value.field_inner_product_invocations > 0
            && value.scalar_fold_invocations > 0
            && value.point_fold_invocations > 0
            && value.sumcheck_kernel_invocations > 0,
        "GPU proof does not attest full commitment/IPA/Sumcheck kernel coverage"
    );
    ensure!(
        !value.device_name.is_empty() && !value.compute_capability.is_empty(),
        "GPU device identity missing"
    );
    Ok(())
}

fn begin_transcript(
    s: &PublicStatement,
    m: &ModelRegistry,
    nonce: [u8; 32],
    device: &str,
    session_commitments: &BTreeMap<String, VectorCommitment>,
    model_sessions: &BTreeMap<String, VectorCommitment>,
) -> Transcript {
    let mut t = Transcript::new(PROOF_FORMAT);
    t.bytes(b"statement-id", s.statement_id.as_bytes());
    t.bytes(b"adapter-schema", s.adapter_schema.as_bytes());
    t.u64(b"adapter-version", s.adapter_version as u64);
    t.bytes(b"protocol-sha256", s.protocol_sha256.as_bytes());
    t.bytes(
        b"canonical-relation-digest",
        s.canonical_relation_digest.as_bytes(),
    );
    t.label("zkDL-rounding-mode-v1");
    t.bytes(b"rounding-mode", s.rounding_mode.as_bytes());
    t.bytes(b"model-registry-id", m.registry_object_id.as_bytes());
    t.bytes(b"proof-nonce", &nonce);
    let public_logits = s
        .registry
        .public_tensors
        .get("public.logits")
        .expect("validated canonical public logits family");
    t.bytes(
        b"public-logits-v1",
        &bincode::serialize(public_logits).expect("canonical scalar serialization"),
    );
    t.label(device);
    t.label("zkDL-model-session-commitments-v1");
    for family in &s.registry.tensor_families {
        if family.visibility != Visibility::Public {
            let c = session_commitments
                .get(&family.family_id)
                .expect("validated commitment map");
            t.label(&family.family_id);
            t.u64(b"tensor-len", c.logical_len as u64);
            t.point(b"tensor-commitment", c.point);
            if family.visibility == Visibility::RegisteredModel {
                let registry = m
                    .commitments
                    .get(&family.family_id)
                    .expect("validated registered model map");
                let session = model_sessions
                    .get(&family.family_id)
                    .expect("validated model session map");
                t.point(b"registry-model-commitment", registry.point);
                t.point(b"session-model-commitment", session.point);
            }
        }
    }
    t
}

fn session_rerandomization_challenge(
    transcript: &mut Transcript,
    family_id: &str,
    registry: &VectorCommitment,
    session: &VectorCommitment,
    nonce_commitment: Point,
) -> Scalar {
    transcript.label("zkDL-model-session-rerandomization-v1");
    transcript.label(family_id);
    transcript.u64(b"logical-len", registry.logical_len as u64);
    transcript.point(b"registry-commitment", registry.point);
    transcript.point(b"session-commitment", session.point);
    transcript.point(b"rerandomization-nonce", nonce_commitment);
    transcript.challenge("rerandomization-schnorr-challenge")
}

fn random_nonzero_scalar() -> Result<Scalar> {
    loop {
        let value = Scalar::random()?;
        if value != Scalar::zero() {
            return Ok(value);
        }
    }
}

fn prove_session_rerandomization(
    pp: &CommitmentParameters,
    family_id: &str,
    registry: &VectorCommitment,
    session: &VectorCommitment,
    delta: Scalar,
    transcript: &mut Transcript,
) -> Result<SessionRerandomizationProof> {
    ensure!(
        delta != Scalar::zero(),
        "model session delta must be fresh and nonzero"
    );
    ensure!(
        registry.logical_len == session.logical_len,
        "model session shape mismatch"
    );
    let generator = pp.blinder_generator(registry.logical_len)?;
    let nonce = random_nonzero_scalar()?;
    let nonce_commitment = generator.mul(nonce)?;
    ensure!(
        !nonce_commitment.eq_checked(Point::identity())?,
        "model session Schnorr nonce commitment must be non-identity"
    );
    ensure!(
        !session.point.eq_checked(registry.point)?,
        "model session commitment must differ from registry commitment"
    );
    let challenge = session_rerandomization_challenge(
        transcript,
        family_id,
        registry,
        session,
        nonce_commitment,
    );
    Ok(SessionRerandomizationProof {
        nonce_commitment,
        response: nonce + challenge * delta,
    })
}

fn verify_session_rerandomization(
    pp: &CommitmentParameters,
    family_id: &str,
    registry: &VectorCommitment,
    session: &VectorCommitment,
    proof: &SessionRerandomizationProof,
    transcript: &mut Transcript,
) -> Result<()> {
    ensure!(
        registry.logical_len == session.logical_len,
        "model session shape mismatch"
    );
    ensure!(
        !session.point.eq_checked(registry.point)?,
        "model session commitment must differ from registry commitment"
    );
    ensure!(
        !proof.nonce_commitment.eq_checked(Point::identity())?,
        "model session Schnorr nonce commitment must be non-identity"
    );
    let generator = pp.blinder_generator(registry.logical_len)?;
    let challenge = session_rerandomization_challenge(
        transcript,
        family_id,
        registry,
        session,
        proof.nonce_commitment,
    );
    // The registry point is subtracted explicitly below.  Keeping the
    // operation in the point API avoids accepting malformed compressed input.
    let expected = proof.nonce_commitment.add(
        session
            .point
            .add(registry.point.mul(-Scalar::one())?)?
            .mul(challenge)?,
    )?;
    ensure!(
        generator.mul(proof.response)?.eq_checked(expected)?,
        "registered-model session re-randomization proof rejected"
    );
    Ok(())
}

pub fn register_model(
    pp: &CommitmentParameters,
    s: &PublicStatement,
    w: &WitnessPackage,
    certified_bounds_id: String,
) -> Result<(ModelRegistry, ModelOpeningSecret)> {
    pp.validate()?;
    s.validate()?;
    w.validate(&s.registry)?;
    validate_registered_model_parameter_bounds(s, w)?;
    // WitnessPackage::validate performs the private source-input digest check.
    ensure!(
        certified_bounds_id == s.integer_bound_ledger_id,
        "certified bounds ID must equal the statement integer-bound ledger ID"
    );
    let registration_transaction_id = fresh_registration_transaction_id()?;
    let mut commitments = BTreeMap::new();
    let mut blinders = BTreeMap::new();
    for family in &s.registry.tensor_families {
        if family.visibility == Visibility::RegisteredModel {
            let r = random_nonzero_scalar()?;
            let (c, _) = commit(pp, w.tensors[&family.family_id].clone(), r)?;
            commitments.insert(family.family_id.clone(), c);
            blinders.insert(family.family_id.clone(), r);
        }
    }
    let registry = ModelRegistry {
        format: "zkDL-GAT-registered-model-v1".into(),
        pp_id: pp.parameters_id.clone(),
        registry_id: s.registry.registry_id.clone(),
        model_arch_id: s.model_arch_id.clone(),
        model_param_id: s.model_param_id.clone(),
        input_preprocess_id: s.input_preprocess_id.clone(),
        data_identity: s.data_identity.clone(),
        source_input_id: s.source_input_id.clone(),
        checkpoint_manifest_id: s.checkpoint_manifest_id.clone(),
        model_shape_id: s.model_shape_id.clone(),
        quant_cfg_id: s.quant_cfg_id.clone(),
        static_tables_id: s.static_tables_id.clone(),
        static_table_rows_id: s.static_table_rows_id.clone(),
        certified_bounds_id,
        registration_transaction_id: registration_transaction_id.clone(),
        registration_binding_digest: String::new(),
        commitment_set_id: s.registry.registered_family_set_digest(),
        commitment_points_id: String::new(),
        commitments,
        registry_object_id: String::new(),
    };
    let mut registry = registry;
    registry.commitment_points_id = registry.commitment_points_digest();
    let mut secret = ModelOpeningSecret {
        registry_object_id: String::new(),
        registration_transaction_id,
        registration_binding_digest: String::new(),
        blinders,
    };
    registry.registration_binding_digest = compute_registration_binding_digest(&registry, &secret)?;
    let registry = registry.seal()?;
    secret.registry_object_id = registry.registry_object_id.clone();
    secret.registration_binding_digest = registry.registration_binding_digest.clone();
    registry.validate(pp, s)?;
    Ok((registry, secret))
}

pub fn prove(
    pp: &CommitmentParameters,
    s: &PublicStatement,
    m: &ModelRegistry,
    secret: &ModelOpeningSecret,
    w: &WitnessPackage,
    device: &str,
) -> Result<ProofEnvelope> {
    pp.validate()?;
    s.validate()?;
    m.validate(pp, s)?;
    w.validate(&s.registry)?;
    // WitnessPackage::validate performs the private source-input digest check.
    ensure!(
        secret.registry_object_id == m.registry_object_id,
        "model opening secret mismatch"
    );
    ensure!(
        secret.registration_transaction_id == m.registration_transaction_id
            && secret.registration_binding_digest == m.registration_binding_digest,
        "model opening secret registration transaction mismatch"
    );
    ensure!(
        compute_registration_binding_digest(m, secret)? == m.registration_binding_digest,
        "model opening secret registration binding mismatch"
    );
    let expected_model_families: std::collections::BTreeSet<_> = s
        .registry
        .tensor_families
        .iter()
        .filter(|family| family.visibility == Visibility::RegisteredModel)
        .map(|family| family.family_id.as_str())
        .collect();
    ensure!(
        secret
            .blinders
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == expected_model_families,
        "model opening secret family set mismatch"
    );
    ensure!(device == "cpu" || device == "gpu", "device must be cpu|gpu");
    let backend = ProverBackend::open(device)?;
    let mut all = BTreeMap::new();
    let mut openings = BTreeMap::new();
    let mut dynamic = BTreeMap::new();
    let mut model_sessions = BTreeMap::new();
    let mut model_deltas = BTreeMap::new();
    let commitment_start = Instant::now();
    for family in &s.registry.tensor_families {
        match family.visibility {
            Visibility::Public => {}
            Visibility::RegisteredModel => {
                let r = *secret
                    .blinders
                    .get(&family.family_id)
                    .ok_or_else(|| anyhow::anyhow!("missing model blinder"))?;
                let delta = random_nonzero_scalar()?;
                let registry = m
                    .commitments
                    .get(&family.family_id)
                    .ok_or_else(|| anyhow::anyhow!("missing registered model commitment"))?;
                let session = pp.rerandomize_commitment(registry, delta)?;
                ensure!(
                    !session.point.eq_checked(registry.point)?,
                    "model session commitment did not change under fresh delta"
                );
                let session_blinder = r + delta;
                let (c, cw) = commit_with_backend(
                    pp,
                    w.tensors[&family.family_id].clone(),
                    session_blinder,
                    &backend,
                )?;
                ensure!(
                    c.logical_len == session.logical_len && c.point.eq_checked(session.point)?,
                    "model witness does not open proof-local session commitment"
                );
                model_deltas.insert(family.family_id.clone(), delta);
                model_sessions.insert(family.family_id.clone(), session.clone());
                all.insert(family.family_id.clone(), session);
                openings.insert(family.family_id.clone(), cw);
            }
            Visibility::DynamicPrivate => {
                let (c, cw) = commit_with_backend(
                    pp,
                    w.tensors[&family.family_id].clone(),
                    random_nonzero_scalar()?,
                    &backend,
                )?;
                ensure!(
                    !c.point.eq_checked(Point::identity())?,
                    "dynamic commitment must be non-identity"
                );
                dynamic.insert(family.family_id.clone(), c.clone());
                all.insert(family.family_id.clone(), c);
                openings.insert(family.family_id.clone(), cw);
            }
        }
    }
    let commitment_time_ms = commitment_start.elapsed().as_secs_f64() * 1000.0;
    let prove_start = Instant::now();
    let nonce = loop {
        let mut candidate = [0u8; 32];
        getrandom::getrandom(&mut candidate)
            .map_err(|e| anyhow::anyhow!("OS randomness failed: {e}"))?;
        if candidate.iter().any(|byte| *byte != 0) {
            break candidate;
        }
    };
    let mut transcript = begin_transcript(s, m, nonce, device, &all, &model_sessions);
    let mut model_rerandomization = BTreeMap::new();
    for family in &s.registry.tensor_families {
        if family.visibility == Visibility::RegisteredModel {
            let registry = m
                .commitments
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing registered model commitment"))?;
            let session = model_sessions
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing model session commitment"))?;
            let delta = *model_deltas
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing model session delta"))?;
            let proof = prove_session_rerandomization(
                pp,
                &family.family_id,
                registry,
                session,
                delta,
                &mut transcript,
            )?;
            model_rerandomization.insert(family.family_id.clone(), proof);
        }
    }
    let mut operations = Vec::with_capacity(s.registry.operation_families.len());
    for op in &s.registry.operation_families {
        operations.push(prove_operation(
            pp,
            &s.registry,
            op,
            w,
            &all,
            &openings,
            &mut transcript,
            &backend,
        )?);
    }
    let gpu_telemetry = backend.finish_gpu()?;
    validate_gpu_telemetry(device, &gpu_telemetry)?;
    absorb_gpu_telemetry(&mut transcript, device, &gpu_telemetry)?;
    let transcript_digest = transcript.digest();
    let prove_time_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
    Ok(ProofEnvelope {
        format: PROOF_FORMAT.into(),
        statement_id: s.statement_id.clone(),
        model_registry_id: m.registry_object_id.clone(),
        device: device.into(),
        fresh_nonce: nonce,
        model_session_commitments: model_sessions,
        model_rerandomization,
        dynamic_commitments: dynamic,
        operations,
        gpu_telemetry,
        commitment_time_ms,
        prove_time_ms,
        transcript_digest,
    })
}

pub fn verify(
    pp: &CommitmentParameters,
    s: &PublicStatement,
    m: &ModelRegistry,
    p: &ProofEnvelope,
    expected_device: Option<&str>,
) -> Result<()> {
    pp.validate()?;
    s.validate()?;
    m.validate(pp, s)?;
    ensure!(
        p.format == PROOF_FORMAT
            && p.statement_id == s.statement_id
            && p.model_registry_id == m.registry_object_id,
        "proof envelope context mismatch"
    );
    ensure!(
        p.fresh_nonce.iter().any(|byte| *byte != 0),
        "proof fresh nonce must be nonzero"
    );
    if let Some(d) = expected_device {
        ensure!(p.device == d, "proof device mismatch");
    }
    ensure!(
        p.device == "cpu" || p.device == "gpu",
        "invalid proof device"
    );
    ensure!(
        p.commitment_time_ms.is_finite()
            && p.commitment_time_ms >= 0.0
            && p.prove_time_ms.is_finite()
            && p.prove_time_ms >= 0.0,
        "proof timing fields are malformed"
    );
    validate_gpu_telemetry(&p.device, &p.gpu_telemetry)?;
    let registered_count = s
        .registry
        .tensor_families
        .iter()
        .filter(|x| x.visibility == Visibility::RegisteredModel)
        .count();
    ensure!(
        p.model_session_commitments.len() == registered_count
            && p.model_rerandomization.len() == registered_count,
        "registered-model session proof count mismatch"
    );
    let mut all = BTreeMap::new();
    let mut model_sessions = BTreeMap::new();
    for family in &s.registry.tensor_families {
        if family.visibility == Visibility::RegisteredModel {
            let registry = m
                .commitments
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing registered model commitment"))?;
            let session = p
                .model_session_commitments
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing model session commitment"))?;
            ensure!(
                session.logical_len == registry.logical_len
                    && session.logical_len == family.logical_len,
                "model session commitment shape mismatch"
            );
            all.insert(family.family_id.clone(), session.clone());
            model_sessions.insert(family.family_id.clone(), session.clone());
        }
    }
    for family in &s.registry.tensor_families {
        if family.visibility == Visibility::DynamicPrivate {
            let c = p
                .dynamic_commitments
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing dynamic commitment"))?;
            ensure!(
                c.logical_len == family.logical_len,
                "dynamic commitment shape mismatch"
            );
            ensure!(
                !c.point.eq_checked(Point::identity())?,
                "dynamic commitment must be non-identity"
            );
            all.insert(family.family_id.clone(), c.clone());
        }
    }
    ensure!(
        p.dynamic_commitments.len()
            == s.registry
                .tensor_families
                .iter()
                .filter(|x| x.visibility == Visibility::DynamicPrivate)
                .count(),
        "unexpected dynamic commitment"
    );
    let expected_registered: std::collections::BTreeSet<_> = s
        .registry
        .tensor_families
        .iter()
        .filter(|x| x.visibility == Visibility::RegisteredModel)
        .map(|x| x.family_id.as_str())
        .collect();
    ensure!(
        p.model_session_commitments
            .keys()
            .map(String::as_str)
            .collect::<std::collections::BTreeSet<_>>()
            == expected_registered
            && p.model_rerandomization
                .keys()
                .map(String::as_str)
                .collect::<std::collections::BTreeSet<_>>()
                == expected_registered,
        "unexpected registered-model session commitment"
    );
    ensure!(
        p.operations.len() == s.registry.operation_families.len(),
        "operation proof count mismatch"
    );
    let mut transcript = begin_transcript(s, m, p.fresh_nonce, &p.device, &all, &model_sessions);
    for family in &s.registry.tensor_families {
        if family.visibility == Visibility::RegisteredModel {
            let registry = m
                .commitments
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing registered model commitment"))?;
            let session = model_sessions
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing model session commitment"))?;
            let proof = p
                .model_rerandomization
                .get(&family.family_id)
                .ok_or_else(|| anyhow::anyhow!("missing model re-randomization proof"))?;
            verify_session_rerandomization(
                pp,
                &family.family_id,
                registry,
                session,
                proof,
                &mut transcript,
            )?;
        }
    }
    for (op, proof) in s.registry.operation_families.iter().zip(&p.operations) {
        verify_operation(pp, &s.registry, op, &all, proof, &mut transcript)?;
    }
    absorb_gpu_telemetry(&mut transcript, &p.device, &p.gpu_telemetry)?;
    ensure!(
        p.transcript_digest == transcript.digest(),
        "proof transcript digest mismatch"
    );
    Ok(())
}

pub fn read_bytes(path: &Path) -> Result<Vec<u8>> {
    let path = validate_secure_path(path, true)?;
    fs::read(&path).with_context(|| format!("read {}", path.display()))
}

pub fn read_bin<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    let bytes = read_bytes(path)?;
    Ok(bincode::deserialize(&bytes).with_context(|| format!("parse {}", path.display()))?)
}

fn lexical_absolute(path: &Path) -> Result<PathBuf> {
    ensure!(!path.as_os_str().is_empty(), "path must not be empty");
    let raw = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut absolute = PathBuf::new();
    for component in raw.components() {
        match component {
            Component::RootDir => absolute.push(Path::new(std::path::MAIN_SEPARATOR_STR)),
            Component::Prefix(prefix) => absolute.push(prefix.as_os_str()),
            Component::CurDir => {}
            Component::Normal(value) => absolute.push(value),
            Component::ParentDir => bail!("path contains a lexical parent component"),
        }
    }
    ensure!(
        absolute.is_absolute(),
        "path did not resolve to an absolute path"
    );
    Ok(absolute)
}

fn metadata_is_single_regular_file(metadata: &fs::Metadata, label: &str) -> Result<()> {
    ensure!(
        !metadata.file_type().is_symlink() && metadata.is_file(),
        "{label} must be a regular file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        ensure!(metadata.nlink() == 1, "{label} must not be hard-linked");
    }
    Ok(())
}

/// Check a path without following symlinks in any existing component.  The
/// final component may be absent only when the caller is preparing an output;
/// all existing final components must still be regular, single-link files.
fn validate_secure_path(path: &Path, require_final: bool) -> Result<PathBuf> {
    let absolute = lexical_absolute(path)?;
    let parent = absolute
        .parent()
        .ok_or_else(|| anyhow::anyhow!("path has no parent"))?;
    let mut current = if absolute.has_root() {
        Path::new(std::path::MAIN_SEPARATOR_STR).to_path_buf()
    } else {
        PathBuf::new()
    };
    for component in parent.components() {
        match component {
            Component::RootDir => {}
            Component::Prefix(prefix) => current.push(prefix.as_os_str()),
            Component::CurDir => {}
            Component::Normal(value) => {
                current.push(value);
                match fs::symlink_metadata(&current) {
                    Ok(metadata) => ensure!(
                        !metadata.file_type().is_symlink() && metadata.is_dir(),
                        "path parent is not a real directory: {}",
                        current.display()
                    ),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                    Err(error) => return Err(error.into()),
                }
            }
            Component::ParentDir => bail!("path contains a lexical parent component"),
        }
    }
    match fs::symlink_metadata(&absolute) {
        Ok(metadata) => metadata_is_single_regular_file(&metadata, "path")?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && !require_final => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            bail!("path does not exist: {}", absolute.display())
        }
        Err(error) => return Err(error.into()),
    }
    Ok(absolute)
}

fn baseline_root_path() -> Result<PathBuf> {
    let configured = lexical_absolute(Path::new(BASELINE_ROOT))?;
    let metadata = fs::symlink_metadata(&configured)?;
    ensure!(
        !metadata.file_type().is_symlink() && metadata.is_dir(),
        "compiled baseline root is not a real directory"
    );
    Ok(fs::canonicalize(configured)?)
}

/// Validate the explicit native CLI project root.  It must be the exact
/// baseline directory captured at compile time, including after symlink
/// resolution; sibling baselines and aliases are rejected.
pub fn validate_project_root(path: &Path) -> Result<PathBuf> {
    let expected = baseline_root_path()?;
    let supplied = lexical_absolute(path)?;
    let supplied_metadata = fs::symlink_metadata(&supplied)?;
    ensure!(
        !supplied_metadata.file_type().is_symlink() && supplied_metadata.is_dir(),
        "project-root must be a real directory"
    );
    ensure!(
        fs::canonicalize(&supplied)? == expected,
        "project-root must equal the compiled comparison/zkdl baseline"
    );
    Ok(expected)
}

/// Validate a native CLI argument against one of the baseline production roots.
/// This is intentionally lexical: a symlink that happens to point back inside
/// the baseline is still rejected, as are hard-linked files and special files.
pub fn validate_cli_path(path: &Path, root: CliPathRoot, output: bool) -> Result<PathBuf> {
    let baseline = baseline_root_path()?;
    let absolute = lexical_absolute(path)?;
    let relative = absolute
        .strip_prefix(&baseline)
        .map_err(|_| anyhow::anyhow!("CLI path escapes the zkDL baseline: {}", path.display()))?;
    let mut components = relative.components();
    let first = components
        .next()
        .and_then(|component| match component {
            Component::Normal(value) => Some(value),
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("CLI path must be below a canonical production root"))?;
    ensure!(
        first == std::ffi::OsStr::new(root.name()),
        "CLI path must be below {}/",
        root.name()
    );
    let checked = validate_secure_path(&absolute, !output)?;
    Ok(checked)
}

/// Return a stable baseline-relative spelling for authenticated manifest
/// fields.  Absolute paths are never serialized into formal manifests.
pub fn relative_baseline_path(path: &Path) -> Result<String> {
    let baseline = baseline_root_path()?;
    let absolute = lexical_absolute(path)?;
    let relative = absolute
        .strip_prefix(&baseline)
        .map_err(|_| anyhow::anyhow!("path escapes the zkDL baseline: {}", path.display()))?;
    ensure!(
        !relative.as_os_str().is_empty(),
        "baseline root itself is not a file path"
    );
    Ok(relative.to_string_lossy().replace('\\', "/"))
}

fn validate_output_path(path: &Path) -> Result<()> {
    validate_secure_path(path, false).map(|_| ())
}

/// Create missing output parents one component at a time, checking every
/// existing component before descending.  `create_dir_all` may otherwise
/// follow a symlink in a missing suffix and create artifacts outside the
/// caller's path root.
fn ensure_regular_parent(path: &Path) -> Result<()> {
    validate_output_path(path)?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("output path has no parent"))?;
    let mut current = if parent.is_absolute() {
        Path::new(std::path::MAIN_SEPARATOR_STR).to_path_buf()
    } else {
        std::env::current_dir()?
    };
    for component in parent.components() {
        match component {
            std::path::Component::RootDir | std::path::Component::CurDir => {}
            std::path::Component::Normal(value) => {
                current.push(value);
                match fs::symlink_metadata(&current) {
                    Ok(metadata) => ensure!(
                        !metadata.file_type().is_symlink() && metadata.is_dir(),
                        "output path parent is not a regular directory"
                    ),
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        fs::create_dir(&current)?;
                        let metadata = fs::symlink_metadata(&current)?;
                        ensure!(
                            !metadata.file_type().is_symlink() && metadata.is_dir(),
                            "new output path parent is not a regular directory"
                        );
                    }
                    Err(error) => return Err(error.into()),
                }
            }
            std::path::Component::Prefix(_) | std::path::Component::ParentDir => {
                bail!("output path contains an unsupported component")
            }
        }
    }
    validate_output_path(path)
}

/// Publish a regular-file artifact atomically.  The destination is never
/// followed when it already exists, and private artifacts receive restrictive
/// permissions before becoming visible to another process.
pub fn write_bytes_atomic(path: &Path, bytes: &[u8], private: bool) -> Result<()> {
    validate_output_path(path)?;
    ensure_regular_parent(path)?;
    let parent = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("output path has no parent"))?;
    let name = path
        .file_name()
        .ok_or_else(|| anyhow::anyhow!("output path has no file name"))?
        .to_string_lossy();
    let counter = ATOMIC_WRITE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(".{name}.tmp.{}.{}", std::process::id(), counter));
    let result = (|| -> Result<()> {
        let mut options = fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(if private { 0o600 } else { 0o644 });
        }
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(
                path,
                fs::Permissions::from_mode(if private { 0o600 } else { 0o644 }),
            )?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub fn write_bin<T: Serialize>(path: &Path, value: &T, private: bool) -> Result<()> {
    let bytes = bincode::serialize(value)?;
    write_bytes_atomic(path, &bytes, private)
}

fn ensure_destination_absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => bail!(
            "registration destination already exists: {}",
            path.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

/// Publish the registered-model public object and its private opening secret
/// with a cross-directory two-phase transaction.  The two serialized objects
/// carry the same transaction and binding digest, so a loader can reject a
/// crash-left half or a pair assembled from different registrations.
pub fn write_model_registration_atomic(
    registry_path: &Path,
    registry: &ModelRegistry,
    secret_path: &Path,
    secret: &ModelOpeningSecret,
) -> Result<()> {
    ensure!(
        registry_path != secret_path,
        "registered model public and private destinations must differ"
    );
    for path in [registry_path, secret_path] {
        validate_output_path(path)?;
        ensure_destination_absent(path)?;
    }
    ensure!(
        registry.registration_transaction_id == secret.registration_transaction_id
            && registry.registration_binding_digest == secret.registration_binding_digest,
        "registration transaction metadata mismatch"
    );
    ensure!(
        canonical_sha256_id(&registry.registration_transaction_id)
            && canonical_sha256_id(&registry.registration_binding_digest),
        "registration transaction metadata is malformed"
    );
    ensure!(
        compute_registration_binding_digest(registry, secret)?
            == registry.registration_binding_digest,
        "registration transaction binding digest mismatch"
    );
    let registry_bytes = bincode::serialize(registry)?;
    let secret_bytes = bincode::serialize(secret)?;
    let registry_parent = registry_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("registered model path has no parent"))?;
    let secret_parent = secret_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("model secret path has no parent"))?;
    ensure_regular_parent(registry_path)?;
    ensure_regular_parent(secret_path)?;
    validate_output_path(registry_path)?;
    validate_output_path(secret_path)?;
    ensure_destination_absent(registry_path)?;
    ensure_destination_absent(secret_path)?;
    let token = registry
        .registration_transaction_id
        .strip_prefix("sha256:")
        .unwrap_or(&registry.registration_transaction_id);
    let registry_tmp = registry_parent.join(format!(
        ".{}.registration.{}.tmp",
        registry_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("registry path has no file name"))?
            .to_string_lossy(),
        token
    ));
    let secret_tmp = secret_parent.join(format!(
        ".{}.registration.{}.tmp",
        secret_path
            .file_name()
            .ok_or_else(|| anyhow::anyhow!("secret path has no file name"))?
            .to_string_lossy(),
        token
    ));
    let mut registry_published = false;
    let mut secret_published = false;
    let result = (|| -> Result<()> {
        ensure_destination_absent(&registry_tmp)?;
        ensure_destination_absent(&secret_tmp)?;
        write_bytes_atomic(&registry_tmp, &registry_bytes, false)?;
        write_bytes_atomic(&secret_tmp, &secret_bytes, true)?;
        ensure_destination_absent(registry_path)?;
        ensure_destination_absent(secret_path)?;
        fs::rename(&registry_tmp, registry_path)?;
        registry_published = true;
        fs::rename(&secret_tmp, secret_path)?;
        secret_published = true;
        let published_registry: ModelRegistry = read_bin(registry_path)?;
        let published_secret: ModelOpeningSecret = read_bin(secret_path)?;
        ensure!(
            published_registry.registration_transaction_id
                == published_secret.registration_transaction_id
                && published_registry.registration_binding_digest
                    == published_secret.registration_binding_digest
                && published_secret.registry_object_id == published_registry.registry_object_id,
            "published registration pair failed transaction identity check"
        );
        ensure!(
            compute_registration_binding_digest(&published_registry, &published_secret)?
                == published_registry.registration_binding_digest,
            "published registration pair failed binding digest check"
        );
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&registry_tmp);
        let _ = fs::remove_file(&secret_tmp);
        if registry_published {
            let _ = fs::remove_file(registry_path);
        }
        if secret_published {
            let _ = fs::remove_file(secret_path);
        }
    }
    result
}
