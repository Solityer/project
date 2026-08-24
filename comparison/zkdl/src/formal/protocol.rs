use crate::algebra::{Point, Scalar};
use crate::hyrax::{commit, commit_with_backend, CommitmentParameters, VectorCommitment};
use crate::prover_backend::{GpuTelemetry, ProverBackend};
use crate::registry::{RelationRegistry, Visibility, WitnessPackage};
use crate::sumcheck::{prove_operation, verify_operation, OperationProof};
use crate::transcript::Transcript;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const PROOF_FORMAT: &str = "zkDL-GAT-proof-envelope-v3";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicStatement {
    pub protocol_id: String,
    pub proof_task: String,
    pub privacy_policy: String,
    pub input_privacy: String,
    pub output_mode: String,
    pub dataset: String,
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
            self.privacy_policy == "model_and_trace_hidden"
                && self.input_privacy == "public_graph_public_features"
                && self.output_mode == "public_logits",
            "statement privacy/output mismatch"
        );
        ensure!(
            !self.dataset.is_empty() && !self.pp_id.is_empty(),
            "dataset/PP identity missing"
        );
        for (label, value) in [
            ("checkpoint manifest", &self.checkpoint_manifest_id),
            ("model shape", &self.model_shape_id),
            ("commitment set", &self.commitment_set_id),
            ("static table rows", &self.static_table_rows_id),
            ("quantization", &self.quant_cfg_id),
            ("static tables", &self.static_tables_id),
            ("integer bounds", &self.integer_bound_ledger_id),
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
            self.statement_id == hex::encode(self.canonical_digest()?),
            "statement digest mismatch"
        );
        self.registry.validate()?;
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
        raw_len > 0 && raw_len <= (1usize << 26),
        "static table domain is too large"
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelRegistry {
    pub format: String,
    pub pp_id: String,
    pub registry_id: String,
    pub model_arch_id: String,
    pub model_param_id: String,
    pub checkpoint_manifest_id: String,
    pub model_shape_id: String,
    pub quant_cfg_id: String,
    pub static_tables_id: String,
    pub static_table_rows_id: String,
    pub certified_bounds_id: String,
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
        ensure!(
            self.model_arch_id == s.model_arch_id && self.model_param_id == s.model_param_id,
            "model identity mismatch"
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
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelOpeningSecret {
    pub registry_object_id: String,
    pub blinders: BTreeMap<String, Scalar>,
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
    t.bytes(b"model-registry-id", m.registry_object_id.as_bytes());
    t.bytes(b"proof-nonce", &nonce);
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
    let nonce = loop {
        let value = Scalar::random()?;
        if value != Scalar::zero() {
            break value;
        }
    };
    let nonce_commitment = generator.mul(nonce)?;
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
    // WitnessPackage::validate performs the private source-input digest check.
    ensure!(
        certified_bounds_id == s.integer_bound_ledger_id,
        "certified bounds ID must equal the statement integer-bound ledger ID"
    );
    let mut commitments = BTreeMap::new();
    let mut blinders = BTreeMap::new();
    for family in &s.registry.tensor_families {
        if family.visibility == Visibility::RegisteredModel {
            let r = Scalar::random()?;
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
        checkpoint_manifest_id: s.checkpoint_manifest_id.clone(),
        model_shape_id: s.model_shape_id.clone(),
        quant_cfg_id: s.quant_cfg_id.clone(),
        static_tables_id: s.static_tables_id.clone(),
        static_table_rows_id: s.static_table_rows_id.clone(),
        certified_bounds_id,
        commitment_set_id: s.registry.registered_family_set_digest(),
        commitment_points_id: String::new(),
        commitments,
        registry_object_id: String::new(),
    };
    let mut registry = registry;
    registry.commitment_points_id = registry.commitment_points_digest();
    let registry = registry.seal()?;
    let secret = ModelOpeningSecret {
        registry_object_id: registry.registry_object_id.clone(),
        blinders,
    };
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
    for family in &s.registry.tensor_families {
        match family.visibility {
            Visibility::Public => {}
            Visibility::RegisteredModel => {
                let r = *secret
                    .blinders
                    .get(&family.family_id)
                    .ok_or_else(|| anyhow::anyhow!("missing model blinder"))?;
                let delta = loop {
                    let value = Scalar::random()?;
                    if value != Scalar::zero() {
                        break value;
                    }
                };
                let registry = m
                    .commitments
                    .get(&family.family_id)
                    .ok_or_else(|| anyhow::anyhow!("missing registered model commitment"))?;
                let session = pp.rerandomize_commitment(registry, delta)?;
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
                    Scalar::random()?,
                    &backend,
                )?;
                dynamic.insert(family.family_id.clone(), c.clone());
                all.insert(family.family_id.clone(), c);
                openings.insert(family.family_id.clone(), cw);
            }
        }
    }
    let mut nonce = [0u8; 32];
    getrandom::getrandom(&mut nonce).map_err(|e| anyhow::anyhow!("OS randomness failed: {e}"))?;
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
    if let Some(d) = expected_device {
        ensure!(p.device == d, "proof device mismatch");
    }
    ensure!(
        p.device == "cpu" || p.device == "gpu",
        "invalid proof device"
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

pub fn read_bin<T: for<'a> Deserialize<'a>>(path: &Path) -> Result<T> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    Ok(bincode::deserialize(&bytes).with_context(|| format!("parse {}", path.display()))?)
}
pub fn write_bin<T: Serialize>(path: &Path, value: &T, private: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let bytes = bincode::serialize(value)?;
    #[cfg(unix)]
    if private {
        use std::io::Write;
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(path)?;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
        return Ok(());
    }
    fs::write(path, &bytes)?;
    Ok(())
}
