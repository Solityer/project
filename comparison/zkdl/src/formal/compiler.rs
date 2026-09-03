use crate::algebra::Scalar;
use crate::protocol::PublicStatement;
use crate::registry::*;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

pub const CANONICAL_ADAPTER_SCHEMA: &str = "zkDL_gat_canonical_adapter_v2";
pub const CANONICAL_ADAPTER_VERSION: u32 = 2;
pub const SUPPORTED_ROUNDING_MODE: &str = "nearest_ties_away";
pub const CANONICAL_RELATION_SCHEMA: &str = "zkDL_gat_canonical_relation_schema_v1";

pub(crate) fn protocol_digest() -> String {
    let mut h = Sha256::new();
    h.update(include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/GAT-ZKML-zkDL.md"
    )));
    format!("sha256:{}", hex::encode(h.finalize()))
}

fn canonical_json_bytes<T: Serialize>(value: &T) -> Result<Vec<u8>> {
    // serde_json::Map uses its deterministic BTreeMap representation in this
    // crate.  The adapter uses the same sorted-key/no-whitespace encoding.
    Ok(serde_json::to_vec(value)?)
}

fn relation_digest(input: &RelationCompilerInput) -> Result<String> {
    let mut value = serde_json::to_value(input)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("canonical relation input is not an object"))?;
    // The digest authenticates every other field, including source_input_id;
    // only the digest field itself is excluded to avoid a circular hash.
    object.remove("canonical_relation_digest");
    let mut h = Sha256::new();
    h.update(b"zkDL-GAT-canonical-relation-v2\0");
    h.update(canonical_json_bytes(&value)?);
    Ok(format!("sha256:{}", hex::encode(h.finalize())))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FieldValue {
    Signed(i64),
    CanonicalHex(String),
}
impl FieldValue {
    fn field(&self) -> Result<Scalar> {
        match self {
            Self::Signed(v) => Ok(Scalar::from_i64(*v)),
            Self::CanonicalHex(v) => Scalar::from_be_hex(v),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IntegerViewIndex {
    Explicit(Vec<u32>),
    Affine {
        offset: u32,
        stride: u32,
    },
    PaddedAffine {
        offset: u32,
        stride: u32,
        logical_rows: u32,
        padding_index: u32,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IntegerColumnSource {
    TensorView {
        family_id: String,
        canonical_index_by_row: IntegerViewIndex,
    },
    Public {
        values: Vec<FieldValue>,
    },
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerOperationColumn {
    pub column_id: String,
    pub source: IntegerColumnSource,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum IntegerExpr {
    Const(FieldValue),
    Column(u32),
    Add(Box<IntegerExpr>, Box<IntegerExpr>),
    Sub(Box<IntegerExpr>, Box<IntegerExpr>),
    Mul(Box<IntegerExpr>, Box<IntegerExpr>),
    PublicTableLookup {
        values: Vec<FieldValue>,
        bit_columns: Vec<u32>,
    },
}
impl IntegerExpr {
    fn field(&self) -> Result<Expr> {
        Ok(match self {
            Self::Const(v) => Expr::Const(v.field()?),
            Self::Column(i) => Expr::Column(*i),
            Self::Add(a, b) => Expr::Add(Box::new(a.field()?), Box::new(b.field()?)),
            Self::Sub(a, b) => Expr::Sub(Box::new(a.field()?), Box::new(b.field()?)),
            Self::Mul(a, b) => Expr::Mul(Box::new(a.field()?), Box::new(b.field()?)),
            Self::PublicTableLookup {
                values,
                bit_columns,
            } => Expr::PublicTableLookup {
                values: values
                    .iter()
                    .map(FieldValue::field)
                    .collect::<Result<Vec<_>>>()?,
                bit_columns: bit_columns.clone(),
            },
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntegerOperationFamily {
    pub family_id: String,
    pub operation_kind: String,
    pub rows: usize,
    pub real_rows: usize,
    pub degree_bound: usize,
    pub columns: Vec<IntegerOperationColumn>,
    pub residuals: Vec<IntegerExpr>,
    pub requires_line8: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelationCompilerInput {
    pub schema: String,
    pub adapter_schema: String,
    pub adapter_version: u32,
    pub protocol_sha256: String,
    pub rounding_mode: String,
    pub canonical_relation_digest: String,
    pub dataset: String,
    pub input_preprocess_id: String,
    pub data_identity: String,
    pub pp_id: String,
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
    pub canonical_tensor_digest: String,
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
    pub tensor_families: Vec<TensorFamilySpec>,
    pub operation_families: Vec<IntegerOperationFamily>,
    pub tensors: BTreeMap<String, Vec<FieldValue>>,
    pub resource_manifest_id: String,
    pub config_sha256: String,
    pub source_input_id: String,
    /// Adapter-generated, explicit canonical schema.  It is checked field by
    /// field in Rust; a digest alone is deliberately not sufficient.
    pub canonical_schema: serde_json::Value,
}

fn source_digest(input: &RelationCompilerInput) -> Result<String> {
    let mut value = serde_json::to_value(input)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("canonical relation input is not an object"))?;
    object.remove("source_input_id");
    object.remove("canonical_relation_digest");
    let mut h = Sha256::new();
    h.update(b"zkDL-GAT-canonical-adapter-input-v2\0");
    h.update(canonical_json_bytes(&value)?);
    Ok(format!("sha256:{}", hex::encode(h.finalize())))
}

pub fn compile(input: RelationCompilerInput) -> Result<(PublicStatement, WitnessPackage)> {
    ensure!(
        input.schema == CANONICAL_ADAPTER_SCHEMA,
        "only the baseline-local canonical GAT adapter schema is accepted"
    );
    ensure!(
        input.adapter_schema == CANONICAL_ADAPTER_SCHEMA
            && input.adapter_version == CANONICAL_ADAPTER_VERSION,
        "canonical adapter schema/version authentication failed"
    );
    ensure!(
        input.protocol_sha256 == protocol_digest(),
        "canonical adapter protocol digest does not match the immutable zkDL protocol"
    );
    ensure!(
        input.rounding_mode == SUPPORTED_ROUNDING_MODE,
        "unsupported or silently substituted rounding mode"
    );
    ensure!(
        input.quant_cfg_id == "gat_l3_scale_chain",
        "canonical adapter quantization identity mismatch"
    );
    ensure!(
        (matches!(input.dataset.as_str(), "cora" | "citeseer" | "pubmed" | "ppi")
            && input.input_preprocess_id == "row_normalize")
            || (input.dataset == "ogbn-arxiv"
                && input.input_preprocess_id == "ogbn_arxiv_raw_float32"),
        "canonical adapter input preprocessing identity mismatch"
    );
    ensure!(
        input.data_identity.starts_with("sha256:")
            && input.data_identity.len() == 71,
        "canonical adapter data identity is malformed"
    );
    ensure!(
        input.source_input_id == source_digest(&input)?,
        "canonical adapter source digest mismatch"
    );
    ensure!(
        input.canonical_relation_digest == relation_digest(&input)?,
        "canonical adapter relation digest mismatch"
    );
    ensure!(
        input
            .canonical_schema
            .get("schema")
            .and_then(serde_json::Value::as_str)
            == Some(CANONICAL_RELATION_SCHEMA),
        "canonical relation schema descriptor is missing"
    );
    let architecture = input
        .canonical_schema
        .get("architecture")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("canonical architecture profile is missing"))?;
    for (key, expected) in [
        ("model_arch_id", input.model_arch_id.as_str()),
        ("model_param_id", input.model_param_id.as_str()),
    ] {
        ensure!(
            architecture.get(key).and_then(serde_json::Value::as_str) == Some(expected),
            "canonical architecture identity mismatch: {key}"
        );
    }
    for (key, expected) in [
        ("node_count", input.node_count),
        ("edge_count", input.edge_count),
        ("graph_count", input.graph_count),
        ("feature_dim", input.feature_dim),
        ("class_count", input.class_count),
    ] {
        ensure!(
            architecture.get(key).and_then(serde_json::Value::as_u64) == Some(expected as u64),
            "canonical architecture dimension mismatch: {key}"
        );
    }
    for value in [
        &input.checkpoint_manifest_id,
        &input.canonical_tensor_digest,
        &input.model_shape_id,
        &input.commitment_set_id,
        &input.static_table_rows_id,
        &input.resource_manifest_id,
        &input.config_sha256,
    ] {
        ensure!(
            value
                .strip_prefix("sha256:")
                .map(|x| x.len() == 64 && x.bytes().all(|b| b.is_ascii_hexdigit()))
                == Some(true),
            "canonical adapter metadata digest malformed"
        );
    }
    let operations = input
        .operation_families
        .iter()
        .map(|o| -> Result<OperationFamilySpec> {
            Ok(OperationFamilySpec {
                family_id: o.family_id.clone(),
                operation_kind: o.operation_kind.clone(),
                rows: o.rows,
                real_rows: o.real_rows,
                degree_bound: o.degree_bound,
                columns: o
                    .columns
                    .iter()
                    .map(|c| -> Result<OperationColumn> {
                        Ok(OperationColumn {
                            column_id: c.column_id.clone(),
                            source: match &c.source {
                                IntegerColumnSource::TensorView {
                                    family_id,
                                    canonical_index_by_row,
                                } => ColumnSource::TensorView {
                                    family_id: family_id.clone(),
                                    canonical_index_by_row: match canonical_index_by_row {
                                        IntegerViewIndex::Explicit(v) => {
                                            ViewIndex::Explicit(v.clone())
                                        }
                                        IntegerViewIndex::Affine { offset, stride } => {
                                            ViewIndex::Affine {
                                                offset: *offset,
                                                stride: *stride,
                                            }
                                        }
                                        IntegerViewIndex::PaddedAffine {
                                            offset,
                                            stride,
                                            logical_rows,
                                            padding_index,
                                        } => ViewIndex::PaddedAffine {
                                            offset: *offset,
                                            stride: *stride,
                                            logical_rows: *logical_rows,
                                            padding_index: *padding_index,
                                        },
                                    },
                                },
                                IntegerColumnSource::Public { values } => ColumnSource::Public {
                                    values: values
                                        .iter()
                                        .map(FieldValue::field)
                                        .collect::<Result<Vec<_>>>()?,
                                },
                            },
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                residuals: o
                    .residuals
                    .iter()
                    .map(IntegerExpr::field)
                    .collect::<Result<Vec<_>>>()?,
                requires_line8: o.requires_line8,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let all_tensors = input
        .tensors
        .iter()
        .map(|(k, v)| {
            Ok((
                k.clone(),
                v.iter()
                    .map(FieldValue::field)
                    .collect::<Result<Vec<_>>>()?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    ensure!(
        all_tensors.len() == input.tensor_families.len(),
        "canonical adapter tensor set mismatch"
    );
    let mut public_tensors = BTreeMap::new();
    let mut private_tensors = BTreeMap::new();
    for spec in &input.tensor_families {
        let values = all_tensors
            .get(&spec.family_id)
            .ok_or_else(|| anyhow::anyhow!("missing compiler tensor {}", spec.family_id))?
            .clone();
        if spec.visibility == Visibility::Public {
            public_tensors.insert(spec.family_id.clone(), values);
        } else {
            private_tensors.insert(spec.family_id.clone(), values);
        }
    }
    let registry = RelationRegistry {
        schema: "zkDL_gat_fac_registry_v1".into(),
        tensor_families: input.tensor_families.clone(),
        public_tensors,
        operation_families: operations,
        mandatory_operation_kinds: MANDATORY_GAT_OPERATIONS
            .iter()
            .map(|x| x.to_string())
            .collect(),
        registry_id: String::new(),
        canonical_schema: input.canonical_schema.clone(),
    }
    .seal()?;
    registry.validate_canonical_gat()?;
    let witness = WitnessPackage {
        registry_id: registry.registry_id.clone(),
        canonical_tensor_digest: input.canonical_tensor_digest.clone(),
        tensors: private_tensors,
        witness_id: input.source_input_id.clone(),
    };
    witness.validate(&registry)?;
    ensure!(
        witness.computed_canonical_tensor_digest(&registry)? == input.canonical_tensor_digest,
        "canonical tensor digest does not match relation input"
    );
    let statement = PublicStatement {
        protocol_id: "GAT-ZKML-zkDL".into(),
        adapter_schema: input.adapter_schema,
        adapter_version: input.adapter_version,
        protocol_sha256: input.protocol_sha256,
        canonical_relation_digest: input.canonical_relation_digest,
        proof_task: "node_classification_full_gat".into(),
        privacy_policy: "model_and_trace_hidden".into(),
        input_privacy: "public_graph_public_features".into(),
        output_mode: "public_logits".into(),
        dataset: input.dataset,
        input_preprocess_id: input.input_preprocess_id,
        data_identity: input.data_identity,
        pp_id: input.pp_id,
        node_count: input.node_count,
        edge_count: input.edge_count,
        graph_count: input.graph_count,
        feature_dim: input.feature_dim,
        class_count: input.class_count,
        public_features: input.public_features,
        node_ptr: input.node_ptr,
        edge_ptr: input.edge_ptr,
        src: input.src,
        dst: input.dst,
        registry,
        graph_features_digest: input.graph_features_digest,
        topology_digest: input.topology_digest,
        checkpoint_manifest_id: input.checkpoint_manifest_id,
        model_shape_id: input.model_shape_id,
        commitment_set_id: input.commitment_set_id,
        quant_cfg_id: input.quant_cfg_id,
        static_tables_id: input.static_tables_id,
        static_table_rows_id: input.static_table_rows_id,
        static_table_profile: input.static_table_profile,
        rounding_mode: input.rounding_mode,
        integer_bound_ledger_id: input.integer_bound_ledger_id,
        integer_bound_ledger: input.integer_bound_ledger,
        model_arch_id: input.model_arch_id,
        model_param_id: input.model_param_id,
        public_logits: input.public_logits,
        resource_manifest_id: input.resource_manifest_id,
        config_sha256: input.config_sha256,
        source_input_id: input.source_input_id,
        statement_id: String::new(),
    }
    .seal()?;
    Ok((statement, witness))
}
