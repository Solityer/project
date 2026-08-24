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

    fn collect_public_tables(&self, out: &mut Vec<(Vec<Scalar>, Vec<u32>)>) {
        match self {
            Self::PublicTableLookup {
                values,
                bit_columns,
            } => out.push((values.clone(), bit_columns.clone())),
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.collect_public_tables(out);
                b.collect_public_tables(out);
            }
            Self::Const(_) | Self::Column(_) => {}
        }
    }

    fn validate_table_bits(
        &self,
        columns: &[OperationColumn],
        registry: &RelationRegistry,
    ) -> Result<()> {
        match self {
            Self::PublicTableLookup { bit_columns, .. } => {
                let mut previous = None;
                for (position, bit) in bit_columns.iter().enumerate() {
                    let column = columns
                        .get(*bit as usize)
                        .ok_or_else(|| anyhow::anyhow!("table bit column out of range"))?;
                    let (family_id, _) = match &column.source {
                        ColumnSource::TensorView {
                            family_id,
                            canonical_index_by_row,
                        } => (family_id, canonical_index_by_row),
                        ColumnSource::Public { .. } => {
                            return Err(anyhow::anyhow!("public table bit column cannot be public"))
                        }
                    };
                    ensure!(
                        registry.tensor(family_id)?.visibility != Visibility::Public,
                        "public table bit column cannot expose a public tensor"
                    );
                    let suffix = column
                        .column_id
                        .rsplit_once("_bit_")
                        .and_then(|(_, value)| value.parse::<usize>().ok())
                        .ok_or_else(|| {
                            anyhow::anyhow!("table bit column is not a canonical bit family")
                        })?;
                    ensure!(
                        suffix == position,
                        "table bit columns are not little-endian and contiguous"
                    );
                    ensure!(
                        previous.map(|value| *bit > value).unwrap_or(true),
                        "table bit columns are duplicated or unsorted"
                    );
                    previous = Some(*bit);
                }
            }
            Self::Add(a, b) | Self::Sub(a, b) | Self::Mul(a, b) => {
                a.validate_table_bits(columns, registry)?;
                b.validate_table_bits(columns, registry)?;
            }
            Self::Const(_) | Self::Column(_) => {}
        }
        Ok(())
    }

    fn is_column(&self, index: u32) -> bool {
        matches!(self, Self::Column(value) if *value == index)
    }

    fn has_padding_selectors(&self, selector: u32, valid: u32) -> bool {
        match self {
            Self::Mul(left, right) if left.is_column(selector) => match right.as_ref() {
                Self::Mul(inner_left, _) if inner_left.is_column(valid) => true,
                _ => false,
            },
            _ => false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OperationFamilySpec {
    pub family_id: String,
    pub operation_kind: String,
    pub rows: usize,
    pub real_rows: usize,
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

    /// Digest the exact public lookup rows carried by the relation plan.  The
    /// preflight report supplies the domain/rounding policy, while this digest
    /// is recomputed from the serialized relation itself so a caller cannot
    /// silently swap table rows after preflight.
    pub fn static_table_digest(&self) -> Result<String> {
        let mut tables = self.public_tables();
        tables.sort_by(|a, b| {
            let key = |table: &(Vec<Scalar>, Vec<u32>)| {
                let mut out = Vec::new();
                out.extend_from_slice(&(table.0.len() as u64).to_be_bytes());
                for value in &table.0 {
                    out.extend_from_slice(&value.0);
                }
                out.extend_from_slice(&(table.1.len() as u64).to_be_bytes());
                for bit in &table.1 {
                    out.extend_from_slice(&bit.to_be_bytes());
                }
                out
            };
            key(a).cmp(&key(b))
        });
        tables.dedup_by(|a, b| a == b);
        let mut h = Sha256::new();
        h.update(b"zkDL-GAT-registry-static-table-rows-v1\0");
        for table in tables {
            h.update((table.0.len() as u64).to_be_bytes());
            for value in table.0 {
                h.update(value.0);
            }
            h.update((table.1.len() as u64).to_be_bytes());
            for bit in table.1 {
                h.update(bit.to_be_bytes());
            }
        }
        Ok(format!("sha256:{}", hex::encode(h.finalize())))
    }

    pub fn public_tables(&self) -> Vec<(Vec<Scalar>, Vec<u32>)> {
        let mut tables = Vec::new();
        for op in &self.operation_families {
            for residual in &op.residuals {
                residual.collect_public_tables(&mut tables);
            }
        }
        tables
    }

    /// Stable digest of the registered-model family set.  The authenticated
    /// commitment points themselves are covered by ModelRegistry's object
    /// digest; this digest binds that object to the statement's expected
    /// family/shape set before registration.
    pub fn registered_family_set_digest(&self) -> String {
        let mut h = Sha256::new();
        h.update(b"zkDL-GAT-registered-family-set-v1\0");
        for tensor in &self.tensor_families {
            if tensor.visibility == Visibility::RegisteredModel {
                h.update((tensor.family_id.len() as u64).to_be_bytes());
                h.update(tensor.family_id.as_bytes());
                h.update((tensor.logical_len as u64).to_be_bytes());
                h.update((tensor.padded_len as u64).to_be_bytes());
                h.update((tensor.logical_shape.len() as u64).to_be_bytes());
                for value in &tensor.logical_shape {
                    h.update((*value as u64).to_be_bytes());
                }
                h.update((tensor.mle_order.len() as u64).to_be_bytes());
                for value in &tensor.mle_order {
                    h.update((value.len() as u64).to_be_bytes());
                    h.update(value.as_bytes());
                }
            }
        }
        format!("sha256:{}", hex::encode(h.finalize()))
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
                op.rows >= 4
                    && op.rows.is_power_of_two()
                    && op.real_rows > 0
                    && op.real_rows <= op.rows / 2,
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
            ensure!(
                op.columns.len() >= 2,
                "operation is missing q/valid selector columns"
            );
            let valid_index = (op.columns.len() - 2) as u32;
            let selector_index = (op.columns.len() - 1) as u32;
            for (index, name) in [(valid_index, "valid"), (selector_index, "domain_selector")] {
                let column = &op.columns[index as usize];
                ensure!(
                    column.column_id == name,
                    "operation selector column order mismatch"
                );
                let values = match &column.source {
                    ColumnSource::Public { values } => values,
                    ColumnSource::TensorView { .. } => {
                        return Err(anyhow::anyhow!("operation selector must be public"))
                    }
                };
                ensure!(
                    values.iter().enumerate().all(|(row, value)| *value
                        == if row < op.real_rows {
                            Scalar::one()
                        } else {
                            Scalar::zero()
                        }),
                    "operation selector does not define the canonical real/padding halves"
                );
            }
            for e in &op.residuals {
                e.validate(op.columns.len())?;
                e.validate_table_bits(&op.columns, self)?;
                ensure!(
                    e.has_padding_selectors(selector_index, valid_index),
                    "operation residual is not masked by the canonical domain selector"
                );
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
    /// Private source binding; this is intentionally not copied into the
    /// public statement because it covers dynamic trace tensors.
    pub canonical_tensor_digest: String,
    pub tensors: BTreeMap<String, Vec<Scalar>>,
    pub witness_id: String,
}

impl WitnessPackage {
    pub fn computed_canonical_tensor_digest(&self, registry: &RelationRegistry) -> Result<String> {
        ensure!(
            self.registry_id == registry.registry_id,
            "witness registry mismatch"
        );
        let mut h = Sha256::new();
        h.update(b"zkDL-GAT-canonical-tensor-v1\0");
        for spec in &registry.tensor_families {
            h.update((spec.family_id.len() as u64).to_be_bytes());
            h.update(spec.family_id.as_bytes());
            let values = if spec.visibility == Visibility::Public {
                registry.public_tensors.get(&spec.family_id)
            } else {
                self.tensors.get(&spec.family_id)
            }
            .ok_or_else(|| anyhow::anyhow!("missing canonical tensor {}", spec.family_id))?;
            ensure!(
                values.len() == spec.logical_len,
                "canonical tensor shape mismatch"
            );
            h.update((values.len() as u64).to_be_bytes());
            for value in values {
                h.update(value.0);
            }
        }
        Ok(format!("sha256:{}", hex::encode(h.finalize())))
    }

    pub fn validate(&self, registry: &RelationRegistry) -> Result<()> {
        ensure!(
            self.registry_id == registry.registry_id,
            "witness registry mismatch"
        );
        let expected_private: BTreeSet<_> = registry
            .tensor_families
            .iter()
            .filter(|tensor| tensor.visibility != Visibility::Public)
            .map(|tensor| tensor.family_id.as_str())
            .collect();
        ensure!(
            self.tensors
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>()
                == expected_private,
            "witness tensor set does not match the private registry families"
        );
        ensure!(
            self.canonical_tensor_digest == self.computed_canonical_tensor_digest(registry)?,
            "private canonical tensor digest mismatch"
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
