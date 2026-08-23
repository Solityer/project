use crate::algebra::Scalar;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Visibility {
    Public,
    RegisteredModel,
    DynamicPrivate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ViewIndex {
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
impl ViewIndex {
    pub fn at(&self, row: usize) -> Result<usize> {
        Ok(match self {
            Self::Explicit(v) => *v
                .get(row)
                .ok_or_else(|| anyhow::anyhow!("view row out of range"))?
                as usize,
            Self::Affine { offset, stride } => (*offset as usize)
                .checked_add(
                    row.checked_mul(*stride as usize)
                        .ok_or_else(|| anyhow::anyhow!("view index overflow"))?,
                )
                .ok_or_else(|| anyhow::anyhow!("view index overflow"))?,
            Self::PaddedAffine {
                offset,
                stride,
                logical_rows,
                padding_index,
            } => {
                if row < *logical_rows as usize {
                    (*offset as usize)
                        .checked_add(
                            row.checked_mul(*stride as usize)
                                .ok_or_else(|| anyhow::anyhow!("view index overflow"))?,
                        )
                        .ok_or_else(|| anyhow::anyhow!("view index overflow"))?
                } else {
                    *padding_index as usize
                }
            }
        })
    }
    pub fn validate(&self, rows: usize, logical_len: usize) -> Result<()> {
        if let Self::Explicit(v) = self {
            ensure!(v.len() == rows, "view mapping length mismatch");
        }
        for row in 0..rows {
            ensure!(self.at(row)? < logical_len, "view mapping out of range");
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorFamilySpec {
    pub family_id: String,
    pub logical_len: usize,
    pub padded_len: usize,
    pub logical_shape: Vec<usize>,
    pub mle_order: Vec<String>,
    pub instance_order: Vec<String>,
    pub visibility: Visibility,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ColumnSource {
    TensorView {
        family_id: String,
        canonical_index_by_row: ViewIndex,
    },
    Public {
        values: Vec<Scalar>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationColumn {
    pub column_id: String,
    pub source: ColumnSource,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Expr {
    Const(Scalar),
    Column(u32),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    PublicTableLookup {
        values: Vec<Scalar>,
        bit_columns: Vec<u32>,
    },
}

impl Expr {
    pub fn eval(&self, columns: &[Scalar]) -> Result<Scalar> {
        Ok(match self {
            Self::Const(v) => *v,
            Self::Column(i) => *columns
                .get(*i as usize)
                .ok_or_else(|| anyhow::anyhow!("expression column out of range"))?,
            Self::Add(a, b) => a.eval(columns)? + b.eval(columns)?,
            Self::Sub(a, b) => a.eval(columns)? - b.eval(columns)?,
            Self::Mul(a, b) => a.eval(columns)? * b.eval(columns)?,
            // Adapter bit columns are canonical little-endian (bit_0 first).  The
            // split-half MLE fold consumes the most-significant selector first, so
            // the columns must be traversed in reverse order.
            Self::PublicTableLookup {
                values,
                bit_columns,
            } => {
                let mut layer = values.clone();
                for bit in bit_columns.iter().rev() {
                    let b = *columns
                        .get(*bit as usize)
                        .ok_or_else(|| anyhow::anyhow!("table bit column out of range"))?;
                    ensure!(
                        layer.len() >= 2 && layer.len().is_power_of_two(),
                        "public table is not power-of-two padded"
                    );
                    let half = layer.len() / 2;
                    layer = (0..half)
                        .map(|i| layer[i] * (Scalar::one() - b) + layer[i + half] * b)
                        .collect();
                }
                ensure!(layer.len() == 1, "public table bit dimension mismatch");
                layer[0]
            }
        })
    }
    pub fn degree(&self) -> usize {
        match self {
            Self::Const(_) => 0,
            Self::Column(_) => 1,
            Self::Add(a, b) | Self::Sub(a, b) => a.degree().max(b.degree()),
            Self::Mul(a, b) => a.degree() + b.degree(),
            Self::PublicTableLookup { bit_columns, .. } => bit_columns.len(),
        }
    }
    pub fn validate(&self, column_count: usize) -> Result<()> {
        match self {
            Self::Column(i) => ensure!(
                (*i as usize) < column_count,
                "expression column out of range"
            ),
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.validate(column_count)?;
                b.validate(column_count)?
            }
            Self::PublicTableLookup {
                values,
                bit_columns,
            } => {
                ensure!(
                    values.len() >= 2 && values.len().is_power_of_two(),
                    "public table must be power-of-two padded"
                );
                ensure!(
                    values.len().trailing_zeros() as usize == bit_columns.len(),
                    "public table bit dimension mismatch"
                );
                ensure!(
                    bit_columns.iter().all(|i| (*i as usize) < column_count),
                    "table bit column out of range"
                );
            }
            _ => {}
        };
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationFamilySpec {
    pub family_id: String,
    pub operation_kind: String,
    pub rows: usize,
    pub degree_bound: usize,
    pub columns: Vec<OperationColumn>,
    pub residuals: Vec<Expr>,
    pub requires_line8: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RelationRegistry {
    pub schema: String,
    pub tensor_families: Vec<TensorFamilySpec>,
    pub public_tensors: BTreeMap<String, Vec<Scalar>>,
    pub operation_families: Vec<OperationFamilySpec>,
    pub mandatory_operation_kinds: Vec<String>,
    pub registry_id: String,
}

impl RelationRegistry {
    pub fn canonical_digest(&self) -> Result<[u8; 32]> {
        let mut copy = self.clone();
        copy.registry_id.clear();
        let bytes = bincode::serialize(&copy)?;
        let mut h = Sha256::new();
        h.update(b"zkDL-GAT-registry-v1\0");
        h.update(bytes);
        Ok(h.finalize().into())
    }
    pub fn seal(mut self) -> Result<Self> {
        self.registry_id = hex::encode(self.canonical_digest()?);
        self.validate()?;
        Ok(self)
    }
    pub fn tensor(&self, id: &str) -> Result<&TensorFamilySpec> {
        self.tensor_families
            .iter()
            .find(|x| x.family_id == id)
            .ok_or_else(|| anyhow::anyhow!("unknown tensor family {id}"))
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == "zkDL_gat_fac_registry_v1",
            "registry schema mismatch"
        );
        ensure!(
            self.registry_id == hex::encode(self.canonical_digest()?),
            "registry digest mismatch"
        );
        let mut ids = BTreeSet::new();
        for t in &self.tensor_families {
            ensure!(ids.insert(&t.family_id), "duplicate tensor family");
            ensure!(
                t.logical_len > 0
                    && t.padded_len >= t.logical_len
                    && t.padded_len.is_power_of_two(),
                "invalid tensor shape"
            );
            if t.visibility == Visibility::Public {
                ensure!(
                    self.public_tensors.get(&t.family_id).map(Vec::len) == Some(t.logical_len),
                    "public tensor payload missing/shape mismatch"
                );
            } else {
                ensure!(
                    !self.public_tensors.contains_key(&t.family_id),
                    "private tensor leaked into public registry"
                );
            }
        }
        ensure!(
            self.public_tensors.len()
                == self
                    .tensor_families
                    .iter()
                    .filter(|t| t.visibility == Visibility::Public)
                    .count(),
            "unexpected public tensor payload"
        );
        let mut ops = BTreeSet::new();
        let mut kinds = BTreeSet::new();
        for op in &self.operation_families {
            ensure!(ops.insert(&op.family_id), "duplicate operation family");
            ensure!(
                op.rows >= 2 && op.rows.is_power_of_two(),
                "operation rows must be power-of-two >=2"
            );
            ensure!(!op.residuals.is_empty(), "operation has no residuals");
            kinds.insert(op.operation_kind.clone());
            let mut cols = BTreeSet::new();
            for c in &op.columns {
                ensure!(cols.insert(&c.column_id), "duplicate operation column");
                match &c.source {
                    ColumnSource::Public { values } => ensure!(
                        values.len() == op.rows,
                        "public operation column length mismatch"
                    ),
                    ColumnSource::TensorView {
                        family_id,
                        canonical_index_by_row,
                    } => {
                        let t = self.tensor(family_id)?;
                        canonical_index_by_row.validate(op.rows, t.logical_len)?;
                    }
                }
            }
            for e in &op.residuals {
                e.validate(op.columns.len())?;
                ensure!(
                    e.degree() + 1 <= op.degree_bound,
                    "operation degree bound too small"
                );
            }
            let private_claims = op
                .columns
                .iter()
                .filter(|c| match &c.source {
                    ColumnSource::TensorView { family_id, .. } => self
                        .tensor(family_id)
                        .map(|t| t.visibility != Visibility::Public)
                        .unwrap_or(false),
                    _ => false,
                })
                .count();
            ensure!(
                op.requires_line8 || private_claims <= 1,
                "Line8 cannot be omitted with multiple private terminal claims"
            );
        }
        for required in &self.mandatory_operation_kinds {
            ensure!(
                kinds.contains(required),
                "mandatory operation kind missing: {required}"
            );
        }
        let expected = MANDATORY_GAT_OPERATIONS
            .iter()
            .map(|value| value.to_string())
            .collect::<BTreeSet<_>>();
        ensure!(
            self.mandatory_operation_kinds.len() == expected.len()
                && self
                    .mandatory_operation_kinds
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>()
                    == expected,
            "mandatory operation registry differs from the fixed GAT relation"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WitnessPackage {
    pub registry_id: String,
    pub tensors: BTreeMap<String, Vec<Scalar>>,
    pub witness_id: String,
}

impl WitnessPackage {
    pub fn validate(&self, registry: &RelationRegistry) -> Result<()> {
        ensure!(
            self.registry_id == registry.registry_id,
            "witness registry mismatch"
        );
        for t in &registry.tensor_families {
            if t.visibility != Visibility::Public {
                let v = self
                    .tensors
                    .get(&t.family_id)
                    .ok_or_else(|| anyhow::anyhow!("missing witness tensor {}", t.family_id))?;
                ensure!(v.len() == t.logical_len, "witness tensor length mismatch");
            }
        }
        Ok(())
    }
}

pub const MANDATORY_GAT_OPERATIONS: &[&str] = &[
    "projection_matmul",
    "attention_dot",
    "public_sparse_route",
    "score_add",
    "lrelu_indexed_public_table",
    "exact_group_max",
    "exp_indexed_public_table",
    "group_sum",
    "inverse_alpha",
    "u_message_hadamard",
    "alpha_message_hadamard",
    "field_aggregate",
    "integer_numerator",
    "dual_aggregate_consistency",
    "signed_round_div",
    "elu_indexed_public_table",
    "hidden_concat",
    "output_bias",
    "head_sum_final_average",
    "public_logits_terminal",
];

#[cfg(test)]
mod tests {
    use super::Expr;
    use crate::algebra::Scalar;

    #[test]
    fn public_table_lookup_uses_little_endian_bit_columns() {
        let expression = Expr::PublicTableLookup {
            values: vec![10, 20, 30, 40]
                .into_iter()
                .map(Scalar::from_i64)
                .collect(),
            bit_columns: vec![0, 1],
        };
        assert_eq!(
            expression.eval(&[Scalar::one(), Scalar::zero()]).unwrap(),
            Scalar::from_i64(20)
        );
        assert_eq!(
            expression.eval(&[Scalar::zero(), Scalar::one()]).unwrap(),
            Scalar::from_i64(30)
        );
    }
}
