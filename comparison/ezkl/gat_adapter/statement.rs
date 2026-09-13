//! Strict verifier-facing statement schema for the native GAT circuit.
//!
//! A proof file is not its statement.  The runner loads this external
//! statement, authenticates it against the resource/artifact manifest, and
//! reconstructs the exact public-instance vector before invoking ezkl's native
//! verifier.  In particular, no witness field is accepted here.

use crate::gat_adapter::manifest::BoundCircuitRef;
use crate::gat_adapter::schema::{sha256_digest, DigestIdentity, ModelBinding};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

/// The complete public ezkl/GAT boundary.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublicEzkl {
    /// Public topology identity.
    pub topology: DigestIdentity,
    /// Public static-table identity.
    pub static_tables: DigestIdentity,
    /// Public quantization identity.
    pub quant: DigestIdentity,
    /// Public integer-ledger identity.
    pub ledger: DigestIdentity,
    /// Native circuit/settings/VK binding.
    pub circuit: BoundCircuitRef,
    /// Native Poseidon model-parameter closure H(W), represented in the
    /// canonical field encoding emitted by the native module.
    pub model_hash: String,
    /// Canonical model binding (audit digest and hash domain).
    pub model: ModelBinding,
    /// Public feature table, row-major fixed-point integers.
    pub feature_table: Vec<i128>,
    /// Public selected node ids.
    pub node_ids: Vec<usize>,
    /// Public source indices in canonical edge order.
    pub src: Vec<usize>,
    /// Public destination indices in canonical edge order.
    pub dst: Vec<usize>,
    /// Public node boundaries.
    pub node_ptr: Vec<usize>,
    /// Public edge boundaries.
    pub edge_ptr: Vec<usize>,
    /// Public output logits, row-major fixed-point integers.
    pub y: Vec<i128>,
}

/// A statement with all identity fields needed by the formal validator.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PublicStatement {
    /// Statement schema version.
    pub schema_version: u32,
    /// Relation identifier.
    pub relation_id: String,
    /// Full relation scope; only complete GAT is accepted.
    pub proof_scope: String,
    /// Model and trace privacy boundary.
    pub privacy_policy: String,
    /// Statement is supplied independently of the proof artifact.
    pub public_statement_mode: String,
    /// Public logits are part of the native instance.
    pub output_mode: String,
    /// Dataset name.
    pub dataset: String,
    /// Model architecture identity.
    pub model_arch_id: String,
    /// Quantization configuration id.
    pub quant_cfg_id: String,
    /// Static table id.
    pub static_table_id: String,
    /// Integer-bound ledger id.
    pub degree_bound_id: String,
    /// ezkl security profile id.
    pub ezkl_security_profile_id: String,
    /// The public relation boundary.
    pub public_ezkl: PublicEzkl,
    /// Canonical output digest.
    pub output_digest: String,
    /// SHA-256 of the external canonical public-output JSON file.
    pub public_output_sha256: String,
    /// Digest of the canonical statement bytes excluding this digest.
    pub statement_digest: String,
}

impl PublicEzkl {
    /// Validate shape/identity fields that can be checked without a witness.
    pub fn validate(&self) -> Result<(), String> {
        if self.circuit.native.compiled_circuit_id.is_empty()
            || self.circuit.native.settings_id.is_empty()
            || self.circuit.native.vk_id.is_empty()
        {
            return Err("public circuit identity is incomplete".into());
        }
        if self.model_hash.is_empty() {
            return Err("native model H(W) is missing".into());
        }
        self.topology.validate("statement.topology")?;
        self.static_tables.validate("statement.static_tables")?;
        self.quant.validate("statement.quant")?;
        self.ledger.validate("statement.ledger")?;
        self.circuit
            .validate()
            .map_err(|e| format!("statement circuit: {}", e))?;
        self.model
            .validate()
            .map_err(|e| format!("statement model: {}", e))?;
        if self.feature_table.is_empty() || self.node_ids.is_empty() || self.y.is_empty() {
            return Err("statement public feature/node/output arrays must not be empty".into());
        }
        if self.src.len() != self.dst.len() || self.node_ptr.len() < 2 || self.edge_ptr.len() < 2 {
            return Err("statement graph arrays have inconsistent lengths".into());
        }
        if self.node_ptr[0] != 0
            || self.edge_ptr[0] != 0
            || self.node_ptr.last().copied() != Some(self.node_ids.len())
            || self.edge_ptr.last().copied() != Some(self.src.len())
        {
            return Err("statement graph pointers do not cover the public graph".into());
        }
        if self
            .node_ptr
            .windows(2)
            .any(|window| window[0] >= window[1])
            || self
                .edge_ptr
                .windows(2)
                .any(|window| window[0] >= window[1])
        {
            return Err("statement graph pointers must be strictly increasing".into());
        }
        let mut node_ids = BTreeSet::new();
        if self.node_ids.iter().any(|id| !node_ids.insert(*id)) {
            return Err("statement node_ids contain duplicates".into());
        }
        for graph_id in 0..self.node_ptr.len() - 1 {
            let node_start = self.node_ptr[graph_id];
            let node_end = self.node_ptr[graph_id + 1];
            let edge_start = self.edge_ptr[graph_id];
            let edge_end = self.edge_ptr[graph_id + 1];
            for edge in edge_start..edge_end {
                let src = self.src[edge];
                let dst = self.dst[edge];
                if src < node_start || src >= node_end || dst < node_start || dst >= node_end {
                    return Err("statement edge crosses a graph batch boundary".into());
                }
            }
        }
        Ok(())
    }
}

impl PublicStatement {
    /// Validate a statement.  Native public instances are deliberately not a
    /// statement field: they are rebuilt from this external boundary through
    /// `GraphCircuit::prepare_public_inputs` at witness and verify time.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1 || self.relation_id != "R_GAT-ezkl" {
            return Err("unsupported GAT statement relation/schema".into());
        }
        if self.proof_scope != "full_gat"
            || self.privacy_policy != "model_and_trace_hidden"
            || self.public_statement_mode != "external"
            || self.output_mode != "public_logits"
        {
            return Err("statement proof/privacy/output scope is not canonical".into());
        }
        for (name, value) in [
            ("dataset", &self.dataset),
            ("model_arch_id", &self.model_arch_id),
            ("quant_cfg_id", &self.quant_cfg_id),
            ("static_table_id", &self.static_table_id),
            ("degree_bound_id", &self.degree_bound_id),
            ("ezkl_security_profile_id", &self.ezkl_security_profile_id),
        ] {
            if value.is_empty() {
                return Err(format!("statement {} must not be empty", name));
            }
        }
        self.public_ezkl.validate()?;
        if !is_native_field_encoding(&self.public_ezkl.model_hash) {
            return Err("native model H(W) is not a canonical Halo2 field encoding".into());
        }
        if self.public_ezkl.model.hash != self.public_ezkl.model_hash {
            return Err("model H(W) disagrees with model binding hash".into());
        }
        if self.public_ezkl.circuit.native.ezkl_security_profile_id != self.ezkl_security_profile_id
        {
            return Err("statement security profile mismatch".into());
        }
        if self.public_ezkl.quant.id != self.quant_cfg_id
            || self.public_ezkl.static_tables.id != self.static_table_id
            || self.public_ezkl.ledger.id != self.degree_bound_id
        {
            return Err("statement registry ids do not match public boundary".into());
        }
        valid_digest("output_digest", &self.output_digest)?;
        valid_digest("public_output_sha256", &self.public_output_sha256)?;
        valid_digest("statement_digest", &self.statement_digest)?;
        if self.output_digest != digest_i128(&self.public_ezkl.y)? {
            return Err("output_digest does not match public Y".into());
        }
        if self.statement_digest != self.digest_without_self()? {
            return Err("statement_digest does not match canonical statement".into());
        }
        Ok(())
    }

    /// Digest of canonical statement fields excluding `statement_digest`.
    pub fn digest_without_self(&self) -> Result<String, String> {
        let mut clone = self.clone();
        clone.statement_digest.clear();
        let bytes =
            serde_json::to_vec(&clone).map_err(|e| format!("serialize statement: {}", e))?;
        sha256_digest(&bytes)
    }

    /// Save after validating strict statement and public-instance bindings.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {}", parent.display(), e))?;
        }
        let bytes =
            serde_json::to_vec_pretty(self).map_err(|e| format!("serialize statement: {}", e))?;
        fs::write(path, bytes).map_err(|e| format!("write {}: {}", path.display(), e))
    }

    /// Load and fully validate an external statement.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
        let raw: Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse statement {}: {}", path.display(), e))?;
        validate_json_shape(&raw)?;
        let value: Self = serde_json::from_value(raw)
            .map_err(|e| format!("parse statement {}: {}", path.display(), e))?;
        value.validate()?;
        Ok(value)
    }
}

fn validate_json_shape(value: &Value) -> Result<(), String> {
    ensure_object_keys(
        value,
        &[
            "schema_version",
            "relation_id",
            "proof_scope",
            "privacy_policy",
            "public_statement_mode",
            "output_mode",
            "dataset",
            "model_arch_id",
            "quant_cfg_id",
            "static_table_id",
            "degree_bound_id",
            "ezkl_security_profile_id",
            "public_ezkl",
            "output_digest",
            "public_output_sha256",
            "statement_digest",
        ],
        "statement",
    )?;
    let public = value
        .get("public_ezkl")
        .ok_or_else(|| "statement public_ezkl is missing".to_string())?;
    ensure_object_keys(
        public,
        &[
            "topology",
            "static_tables",
            "quant",
            "ledger",
            "circuit",
            "model_hash",
            "model",
            "feature_table",
            "node_ids",
            "src",
            "dst",
            "node_ptr",
            "edge_ptr",
            "y",
        ],
        "statement.public_ezkl",
    )?;
    for key in ["topology", "static_tables", "quant", "ledger"] {
        ensure_object_keys(
            public
                .get(key)
                .ok_or_else(|| format!("statement.public_ezkl.{} is missing", key))?,
            &["id", "digest"],
            &format!("statement.public_ezkl.{}", key),
        )?;
    }
    ensure_object_keys(
        public
            .get("circuit")
            .ok_or_else(|| "statement public_ezkl.circuit is missing".to_string())?,
        &[
            "relation_id",
            "compiled_circuit_id",
            "compiled_circuit_digest",
            "settings_id",
            "settings_digest",
            "vk_id",
            "vk_digest",
            "model_graph_id",
            "model_graph_digest",
            "ezkl_security_profile_id",
        ],
        "statement.public_ezkl.circuit",
    )?;
    ensure_object_keys(
        public
            .get("model")
            .ok_or_else(|| "statement public_ezkl.model is missing".to_string())?,
        &["hash", "hash_domain", "canonical_model_digest"],
        "statement.public_ezkl.model",
    )?;
    Ok(())
}

fn ensure_object_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{} must be a JSON object", label))?;
    let actual = object.keys().cloned().collect::<BTreeSet<_>>();
    let wanted = expected
        .iter()
        .map(|key| (*key).to_string())
        .collect::<BTreeSet<_>>();
    if actual != wanted {
        return Err(format!(
            "{} schema mismatch; missing={:?}, extra={:?}",
            label,
            wanted.difference(&actual).collect::<Vec<_>>(),
            actual.difference(&wanted).collect::<Vec<_>>()
        ));
    }
    Ok(())
}

fn valid_digest(label: &str, digest: &str) -> Result<(), String> {
    if digest.len() != 64 || digest.bytes().any(|b| !b.is_ascii_hexdigit()) {
        return Err(format!("{} must be a 32-byte hexadecimal digest", label));
    }
    if digest.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err(format!("{} must use lowercase hexadecimal", label));
    }
    Ok(())
}

fn is_native_field_encoding(value: &str) -> bool {
    value.len() == 66
        && value.starts_with("0x")
        && value[2..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn digest_i128(values: &[i128]) -> Result<String, String> {
    let mut bytes = b"ezkl-gat-public-y-v1".to_vec();
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_digest(&bytes)
}

/// Public helpers for statement tests and verifier adapters.
pub fn output_digest(values: &[i128]) -> Result<String, String> {
    digest_i128(values)
}
