use crate::algebra::Scalar;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const CANONICAL_RELATION_SCHEMA: &str = "zkDL_gat_canonical_relation_schema_v1";
const CANONICAL_RELATION_SCHEMA_VERSION: u64 = 1;
const CANONICAL_TENSOR_MLE_ORDER: &str = "row_major_lsb_first";
const CANONICAL_TENSOR_INSTANCE_ORDER: &str = "canonical_program_order";

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
    /// The adapter's explicit canonical schema descriptor.  Validation below
    /// compares it against the deserialized relation object field-by-field;
    /// callers cannot make an arbitrary operation list acceptable merely by
    /// supplying a matching digest.
    #[serde(default)]
    pub canonical_schema: serde_json::Value,
}

fn is_canonical_terminal_view(view: &ViewIndex, logical_len: usize) -> bool {
    match view {
        ViewIndex::Affine {
            offset: 0,
            stride: 1,
        } => logical_len.is_power_of_two(),
        ViewIndex::PaddedAffine {
            offset: 0,
            stride: 1,
            logical_rows,
            padding_index: 0,
        } => *logical_rows as usize == logical_len,
        _ => false,
    }
}

fn decimal_component(value: &str, prefix: &str) -> bool {
    let Some(number) = value.strip_prefix(prefix) else {
        return false;
    };
    !number.is_empty() && number.bytes().all(|byte| byte.is_ascii_digit())
}

fn canonical_head_namespace(family_id: &str) -> Option<&str> {
    if let Some(rest) = family_id.strip_prefix("hidden.layer_") {
        let (layer, rest) = rest.split_once(".head_")?;
        let (head, suffix) = rest.split_once('.')?;
        if decimal_component(layer, "") && decimal_component(head, "") {
            return Some(suffix);
        }
        return None;
    }
    if let Some(rest) = family_id.strip_prefix("output.head_") {
        let (head, suffix) = rest.split_once('.')?;
        if decimal_component(head, "") {
            return Some(suffix);
        }
    }
    None
}

fn canonical_registered_parameter_family(family_id: &str) -> bool {
    let Some(suffix) = canonical_head_namespace(family_id) else {
        return false;
    };
    suffix == "a_src"
        || suffix == "a_dst"
        || suffix == "bias"
        || suffix
            .strip_prefix("Wcol_")
            .map(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
            .unwrap_or(false)
}

fn canonical_public_auxiliary_family(family_id: &str) -> bool {
    family_id.starts_with("output.head_")
        && canonical_head_namespace(family_id) == Some("alignment_d")
}

fn canonical_dynamic_head_suffix(suffix: &str) -> bool {
    const EXACT: &[&str] = &[
        "projection",
        "E_src",
        "E_dst",
        "route_E_src",
        "route_E_dst",
        "score",
        "z",
        "max",
        "delta",
        "max_selector",
        "max_selector_acc",
        "u",
        "u_acc",
        "sum",
        "inv",
        "alpha",
        "route_projection",
        "u_message",
        "alpha_message",
        "field_aggregate",
        "integer_numerator",
        "num_acc",
        "field_acc",
        "recovery_x",
        "recovery_d",
        "pre",
        "aggregate",
        "alignment_x",
        "alignment_d",
        "aligned",
        "head_output",
    ];
    if EXACT.contains(&suffix) {
        return true;
    }
    let indexed = |value: &str, prefix: &str| {
        value
            .strip_prefix(prefix)
            .map(|index| !index.is_empty() && index.bytes().all(|byte| byte.is_ascii_digit()))
            .unwrap_or(false)
    };
    if indexed(suffix, "Pcol_") || indexed(suffix, "delta_bit_") {
        return true;
    }
    for prefix in ["lrelu", "exp", "elu"] {
        if suffix == format!("{prefix}.idx") || indexed(suffix, &format!("{prefix}.bit_")) {
            return true;
        }
    }
    for prefix in ["recovery_div", "alignment_div"] {
        let Some(role) = suffix.strip_prefix(&format!("{prefix}.")) else {
            continue;
        };
        if ["sigma", "z", "a", "d", "a_inv", "d_inv", "q", "r", "delta"].contains(&role) {
            return true;
        }
        for bit_prefix in ["a_bit_", "d_bit_", "q_bit_", "r_bit_", "delta_bit_"] {
            if indexed(role, bit_prefix) {
                return true;
            }
        }
    }
    false
}

fn canonical_dynamic_family(family_id: &str) -> bool {
    if family_id == "output.head_sum" || family_id == "output.final_logits" {
        return true;
    }
    if let Some(rest) = family_id.strip_prefix("hidden.layer_") {
        if rest.ends_with(".concat") {
            let layer = rest.trim_end_matches(".concat");
            return decimal_component(layer, "");
        }
    }
    canonical_head_namespace(family_id)
        .map(canonical_dynamic_head_suffix)
        .unwrap_or(false)
}

fn canonical_operation_id_matches_kind(operation: &OperationFamilySpec) -> bool {
    let id = operation.family_id.as_str();
    match operation.operation_kind.as_str() {
        "projection_matmul" => id.contains(".projection.col_"),
        "attention_dot" => id.ends_with(".attention_src") || id.ends_with(".attention_dst"),
        "public_sparse_route" => {
            id.ends_with(".route_src")
                || id.ends_with(".route_dst")
                || id.ends_with(".route_projection_relation")
        }
        "score_add" => id.ends_with(".score_add"),
        "lrelu_indexed_public_table" => id.ends_with(".lrelu"),
        "exact_group_max" => id.ends_with(".max_relation"),
        "exp_indexed_public_table" => id.ends_with(".exp"),
        "group_sum" => id.ends_with(".group_sum"),
        "inverse_alpha" => id.ends_with(".inverse") || id.ends_with(".alpha"),
        "u_message_hadamard" => id.ends_with(".u_message_relation"),
        "alpha_message_hadamard" => id.ends_with(".alpha_message_relation"),
        "field_aggregate" => id.ends_with(".field_aggregate_relation"),
        "integer_numerator" => id.ends_with(".integer_aggregate"),
        "dual_aggregate_consistency" => id.ends_with(".dual_consistency"),
        "signed_round_div" => {
            id.ends_with(".recovery_div")
                || id.ends_with(".alignment_div")
                || id.ends_with(".final_average")
        }
        "elu_indexed_public_table" => id.ends_with(".elu"),
        "hidden_concat" => {
            id.contains(".concat.head_") && id.ends_with(".col_0")
                || id.contains(".concat.head_") && id.contains(".col_")
        }
        "output_bias" => id.ends_with(".bias_add"),
        "head_sum_final_average" => id == "output.head_sum" || id == "output.final_average",
        "public_logits_terminal" => id == "output.public_logits",
        _ => false,
    }
}

#[derive(Clone, Debug)]
struct ExpectedTensorFamily {
    family_id: String,
    logical_len: usize,
    logical_shape: Vec<usize>,
    visibility: Visibility,
}

#[derive(Clone, Debug)]
struct CanonicalLayerProfile {
    layer_index: usize,
    input_dim: usize,
    head_count: usize,
    head_dim: usize,
}

#[derive(Clone, Debug)]
struct CanonicalArchitectureProfile {
    layer_count: usize,
    hidden_profile: Vec<(usize, usize)>,
    d_in_profile: Vec<usize>,
    k_out: usize,
    node_count: usize,
    edge_count: usize,
    graph_count: usize,
    feature_dim: usize,
    class_count: usize,
    output_head_count: usize,
    bit_width: usize,
    lookup_bits: BTreeMap<String, usize>,
    hidden_layers: Vec<CanonicalLayerProfile>,
}

fn ensure_exact_object_keys<'a>(
    value: &'a serde_json::Value,
    expected: &[&str],
    label: &str,
) -> Result<&'a serde_json::Map<String, serde_json::Value>> {
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("{label} must be an object"))?;
    let actual = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let expected = expected.iter().copied().collect::<BTreeSet<_>>();
    ensure!(actual == expected, "{label} has unknown or missing fields");
    Ok(object)
}

fn canonical_required_string(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    label: &str,
) -> Result<String> {
    let value = object
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("{label}.{key} must be a string"))?;
    ensure!(!value.is_empty(), "{label}.{key} must not be empty");
    Ok(value.to_string())
}

fn canonical_required_usize(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    label: &str,
) -> Result<usize> {
    let value = object
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("{label}.{key} must be a non-negative integer"))?;
    usize::try_from(value).map_err(|_| anyhow::anyhow!("{label}.{key} overflows usize"))
}

fn canonical_required_usize_array(
    object: &serde_json::Map<String, serde_json::Value>,
    key: &str,
    label: &str,
) -> Result<Vec<usize>> {
    let value = object
        .get(key)
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("{label}.{key} must be an integer array"))?;
    value
        .iter()
        .map(|item| {
            let number = item
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("{label}.{key} contains a non-negative integer"))?;
            usize::try_from(number).map_err(|_| anyhow::anyhow!("{label}.{key} overflows usize"))
        })
        .collect()
}

fn canonical_architecture_profile(
    schema: &serde_json::Value,
    n: usize,
    e: usize,
    g: usize,
    d: usize,
    c: usize,
) -> Result<CanonicalArchitectureProfile> {
    let schema_object = ensure_exact_object_keys(
        schema,
        &[
            "architecture",
            "fac_plan",
            "operation_families",
            "operation_family_order",
            "schema",
            "terminal_plan",
            "tensor_families",
            "tensor_family_order",
            "version",
        ],
        "canonical relation schema",
    )?;
    ensure!(
        schema_object
            .get("schema")
            .and_then(serde_json::Value::as_str)
            == Some(CANONICAL_RELATION_SCHEMA),
        "canonical relation schema descriptor mismatch"
    );
    ensure!(
        schema_object
            .get("version")
            .and_then(serde_json::Value::as_u64)
            == Some(CANONICAL_RELATION_SCHEMA_VERSION),
        "canonical relation schema descriptor version mismatch"
    );
    let architecture = schema_object
        .get("architecture")
        .ok_or_else(|| anyhow::anyhow!("canonical architecture profile is missing"))?;
    let architecture = ensure_exact_object_keys(
        architecture,
        &[
            "L",
            "bit_width",
            "class_count",
            "d_in_profile",
            "edge_count",
            "feature_dim",
            "graph_count",
            "hidden_head_specs",
            "hidden_layers",
            "hidden_profile",
            "K_out",
            "lookup_bits",
            "model_arch_id",
            "model_param_id",
            "node_count",
            "output_head_count",
            "output_head_specs",
        ],
        "canonical architecture profile",
    )?;
    canonical_required_string(architecture, "model_arch_id", "architecture")?;
    canonical_required_string(architecture, "model_param_id", "architecture")?;
    let profile = CanonicalArchitectureProfile {
        layer_count: canonical_required_usize(architecture, "L", "architecture")?,
        hidden_profile: {
            let values = architecture
                .get("hidden_profile")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| anyhow::anyhow!("architecture.hidden_profile must be an array"))?;
            values
                .iter()
                .map(|value| {
                    let object = ensure_exact_object_keys(
                        value,
                        &["head_count", "head_dim"],
                        "architecture.hidden_profile entry",
                    )?;
                    Ok((
                        canonical_required_usize(object, "head_count", "hidden profile")?,
                        canonical_required_usize(object, "head_dim", "hidden profile")?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?
        },
        d_in_profile: canonical_required_usize_array(architecture, "d_in_profile", "architecture")?,
        k_out: canonical_required_usize(architecture, "K_out", "architecture")?,
        node_count: canonical_required_usize(architecture, "node_count", "architecture")?,
        edge_count: canonical_required_usize(architecture, "edge_count", "architecture")?,
        graph_count: canonical_required_usize(architecture, "graph_count", "architecture")?,
        feature_dim: canonical_required_usize(architecture, "feature_dim", "architecture")?,
        class_count: canonical_required_usize(architecture, "class_count", "architecture")?,
        output_head_count: canonical_required_usize(
            architecture,
            "output_head_count",
            "architecture",
        )?,
        bit_width: canonical_required_usize(architecture, "bit_width", "architecture")?,
        lookup_bits: {
            let value = architecture
                .get("lookup_bits")
                .ok_or_else(|| anyhow::anyhow!("architecture.lookup_bits is missing"))?;
            let object = ensure_exact_object_keys(
                value,
                &["elu", "exp", "lrelu"],
                "architecture.lookup_bits",
            )?;
            let mut result = BTreeMap::new();
            for key in ["lrelu", "exp", "elu"] {
                let bits = canonical_required_usize(object, key, "architecture.lookup_bits")?;
                ensure!(
                    (1..32).contains(&bits),
                    "architecture lookup-bit width is invalid"
                );
                result.insert(key.to_string(), bits);
            }
            result
        },
        hidden_layers: Vec::new(),
    };
    ensure!(
        profile.node_count == n
            && profile.edge_count == e
            && profile.graph_count == g
            && profile.feature_dim == d
            && profile.class_count == c,
        "canonical architecture dimensions do not match public tensor shapes"
    );
    ensure!(
        profile.output_head_count > 0,
        "canonical architecture has no output heads"
    );
    ensure!(
        profile.layer_count == 3
            && profile.k_out == 2
            && profile.hidden_profile == vec![(8, 8), (8, 8)],
        "canonical architecture is not the current L3 two-hidden-layer/two-output GAT"
    );
    ensure!(
        (2..254).contains(&profile.bit_width),
        "canonical integer bit width is invalid"
    );

    let layers = architecture
        .get("hidden_layers")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("architecture.hidden_layers must be an array"))?;
    ensure!(
        !layers.is_empty(),
        "canonical architecture has no hidden layers"
    );
    ensure!(
        profile.layer_count == layers.len() + 1
            && profile.hidden_profile.len() == layers.len()
            && profile.d_in_profile.len() == layers.len(),
        "canonical architecture L/profile/input-profile lengths are inconsistent"
    );
    let mut previous_dim = d;
    let mut parsed_layers = Vec::with_capacity(layers.len());
    for (position, value) in layers.iter().enumerate() {
        let object = ensure_exact_object_keys(
            value,
            &["head_count", "head_dim", "input_dim", "layer_index"],
            &format!("architecture.hidden_layers[{position}"),
        )?;
        let layer_index = canonical_required_usize(object, "layer_index", "hidden layer")?;
        let input_dim = canonical_required_usize(object, "input_dim", "hidden layer")?;
        let head_count = canonical_required_usize(object, "head_count", "hidden layer")?;
        let head_dim = canonical_required_usize(object, "head_dim", "hidden layer")?;
        ensure!(
            layer_index == position && input_dim == previous_dim && head_count > 0 && head_dim > 0,
            "canonical hidden layer index/dimension mapping is invalid"
        );
        previous_dim = head_count
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow::anyhow!("canonical hidden layer width overflows usize"))?;
        parsed_layers.push(CanonicalLayerProfile {
            layer_index,
            input_dim,
            head_count,
            head_dim,
        });
    }
    ensure!(
        profile
            .d_in_profile
            .iter()
            .copied()
            .eq(parsed_layers.iter().map(|layer| layer.input_dim))
            && profile.hidden_profile.iter().copied().eq(parsed_layers
                .iter()
                .map(|layer| (layer.head_count, layer.head_dim)),),
        "canonical architecture profile does not match hidden layer metadata"
    );
    ensure!(
        profile.k_out == profile.output_head_count,
        "canonical K_out does not match output head count"
    );

    let hidden_specs = architecture
        .get("hidden_head_specs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("architecture.hidden_head_specs must be an array"))?;
    let expected_hidden_count: usize = parsed_layers.iter().map(|layer| layer.head_count).sum();
    ensure!(
        hidden_specs.len() == expected_hidden_count,
        "canonical hidden head specification count mismatch"
    );
    let mut cursor = 0usize;
    let mut global_indices = BTreeSet::new();
    for layer in &parsed_layers {
        for local_head_index in 0..layer.head_count {
            let value = hidden_specs
                .get(cursor)
                .ok_or_else(|| anyhow::anyhow!("canonical hidden head specification is missing"))?;
            let object = ensure_exact_object_keys(
                value,
                &[
                    "attn_dst_bias",
                    "attn_dst_kernel",
                    "attn_src_bias",
                    "attn_src_kernel",
                    "global_head_index",
                    "layer_index",
                    "local_head_index",
                    "output_bias",
                    "seq_kernel",
                ],
                &format!("architecture.hidden_head_specs[{cursor}]"),
            )?;
            ensure!(
                canonical_required_usize(object, "layer_index", "hidden head")?
                    == layer.layer_index
                    && canonical_required_usize(object, "local_head_index", "hidden head")?
                        == local_head_index,
                "canonical hidden head layer/local index mapping is invalid"
            );
            if let Some(global_index) = object.get("global_head_index") {
                if !global_index.is_null() {
                    let global_index =
                        canonical_required_usize(object, "global_head_index", "hidden head")?;
                    ensure!(
                        global_index < expected_hidden_count,
                        "canonical hidden head global index is out of range"
                    );
                    ensure!(
                        global_indices.insert(global_index),
                        "canonical hidden head global index collides"
                    );
                }
            }
            for key in [
                "seq_kernel",
                "attn_dst_kernel",
                "attn_dst_bias",
                "attn_src_kernel",
                "attn_src_bias",
                "output_bias",
            ] {
                canonical_required_string(object, key, "hidden head")?;
            }
            cursor += 1;
        }
    }
    let output_specs = architecture
        .get("output_head_specs")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("architecture.output_head_specs must be an array"))?;
    ensure!(
        output_specs.len() == profile.output_head_count,
        "canonical output head specification count mismatch"
    );
    for (head_index, value) in output_specs.iter().enumerate() {
        let object = ensure_exact_object_keys(
            value,
            &[
                "attn_dst_bias",
                "attn_dst_kernel",
                "attn_src_bias",
                "attn_src_kernel",
                "head_index",
                "output_bias",
                "seq_kernel",
            ],
            &format!("architecture.output_head_specs[{head_index}]"),
        )?;
        ensure!(
            canonical_required_usize(object, "head_index", "output head")? == head_index,
            "canonical output head order/index mapping is invalid"
        );
        for key in [
            "seq_kernel",
            "attn_dst_kernel",
            "attn_dst_bias",
            "attn_src_kernel",
            "attn_src_bias",
            "output_bias",
        ] {
            canonical_required_string(object, key, "output head")?;
        }
    }

    Ok(CanonicalArchitectureProfile {
        hidden_layers: parsed_layers,
        ..profile
    })
}

fn expected_tensor(
    out: &mut Vec<ExpectedTensorFamily>,
    family_id: impl Into<String>,
    logical_len: usize,
    logical_shape: Vec<usize>,
    visibility: Visibility,
) {
    out.push(ExpectedTensorFamily {
        family_id: family_id.into(),
        logical_len,
        logical_shape,
        visibility,
    });
}

fn append_division_aux_layout(
    out: &mut Vec<ExpectedTensorFamily>,
    prefix: &str,
    label: &str,
    logical_len: usize,
    bit_width: usize,
) {
    for suffix in ["sigma", "z", "a", "d", "a_inv", "d_inv", "q", "r", "delta"] {
        expected_tensor(
            out,
            format!("{prefix}.{label}.{suffix}"),
            logical_len,
            vec![logical_len],
            Visibility::DynamicPrivate,
        );
    }
    for suffix in ["a", "d", "q", "r", "delta"] {
        for bit in 0..bit_width {
            expected_tensor(
                out,
                format!("{prefix}.{label}.{suffix}_bit_{bit}"),
                logical_len,
                vec![logical_len],
                Visibility::DynamicPrivate,
            );
        }
    }
}

fn append_head_tensor_layout(
    out: &mut Vec<ExpectedTensorFamily>,
    prefix: &str,
    n: usize,
    e: usize,
    input_dim: usize,
    output_dim: usize,
    bit_width: usize,
    lookup_bits: (usize, usize, usize),
    hidden: bool,
) {
    let nd = n * output_dim;
    expected_tensor(
        out,
        format!("{prefix}.projection"),
        n * output_dim,
        vec![n, output_dim],
        Visibility::DynamicPrivate,
    );
    for col in 0..output_dim {
        expected_tensor(
            out,
            format!("{prefix}.Wcol_{col}"),
            input_dim,
            vec![input_dim],
            Visibility::RegisteredModel,
        );
        expected_tensor(
            out,
            format!("{prefix}.Pcol_{col}"),
            n,
            vec![n],
            Visibility::DynamicPrivate,
        );
    }
    for (suffix, len, shape) in [
        ("E_src", n, vec![n]),
        ("E_dst", n, vec![n]),
        ("a_src", output_dim, vec![output_dim]),
        ("a_dst", output_dim, vec![output_dim]),
        ("route_E_src", e, vec![e]),
        ("route_E_dst", e, vec![e]),
        ("score", e, vec![e]),
        ("z", e, vec![e]),
    ] {
        expected_tensor(
            out,
            format!("{prefix}.{suffix}"),
            len,
            shape,
            Visibility::DynamicPrivate,
        );
    }
    for (name, bits) in [("lrelu", lookup_bits.0)] {
        expected_tensor(
            out,
            format!("{prefix}.{name}.idx"),
            e,
            vec![e],
            Visibility::DynamicPrivate,
        );
        for bit in 0..bits {
            expected_tensor(
                out,
                format!("{prefix}.{name}.bit_{bit}"),
                e,
                vec![e],
                Visibility::DynamicPrivate,
            );
        }
    }
    for (suffix, len, shape) in [
        ("max", n, vec![n]),
        ("delta", e, vec![e]),
        ("max_selector", e, vec![e]),
        ("max_selector_acc", e, vec![e]),
    ] {
        expected_tensor(
            out,
            format!("{prefix}.{suffix}"),
            len,
            shape,
            Visibility::DynamicPrivate,
        );
    }
    for bit in 0..bit_width {
        expected_tensor(
            out,
            format!("{prefix}.delta_bit_{bit}"),
            e,
            vec![e],
            Visibility::DynamicPrivate,
        );
    }
    for (suffix, len, shape) in [("u", e, vec![e])] {
        expected_tensor(
            out,
            format!("{prefix}.{suffix}"),
            len,
            shape,
            Visibility::DynamicPrivate,
        );
    }
    for (name, bits) in [("exp", lookup_bits.1)] {
        expected_tensor(
            out,
            format!("{prefix}.{name}.idx"),
            e,
            vec![e],
            Visibility::DynamicPrivate,
        );
        for bit in 0..bits {
            expected_tensor(
                out,
                format!("{prefix}.{name}.bit_{bit}"),
                e,
                vec![e],
                Visibility::DynamicPrivate,
            );
        }
    }
    for (suffix, len, shape) in [
        ("u_acc", e, vec![e]),
        ("sum", n, vec![n]),
        ("inv", n, vec![n]),
        ("alpha", e, vec![e]),
        ("route_projection", e * output_dim, vec![e, output_dim]),
        ("u_message", e * output_dim, vec![e, output_dim]),
        ("alpha_message", e * output_dim, vec![e, output_dim]),
        ("field_aggregate", n * output_dim, vec![n, output_dim]),
        ("integer_numerator", n * output_dim, vec![n, output_dim]),
        ("num_acc", e * output_dim, vec![e, output_dim]),
        ("field_acc", e * output_dim, vec![e, output_dim]),
        ("recovery_x", n * output_dim, vec![n * output_dim]),
        ("recovery_d", n * output_dim, vec![n * output_dim]),
        ("pre", n * output_dim, vec![n, output_dim]),
    ] {
        expected_tensor(
            out,
            format!("{prefix}.{suffix}"),
            len,
            shape,
            Visibility::DynamicPrivate,
        );
    }
    append_division_aux_layout(out, prefix, "recovery_div", nd, bit_width);
    if hidden {
        for (name, bits) in [("elu", lookup_bits.2)] {
            expected_tensor(
                out,
                format!("{prefix}.{name}.idx"),
                nd,
                vec![nd],
                Visibility::DynamicPrivate,
            );
            for bit in 0..bits {
                expected_tensor(
                    out,
                    format!("{prefix}.{name}.bit_{bit}"),
                    nd,
                    vec![nd],
                    Visibility::DynamicPrivate,
                );
            }
        }
        expected_tensor(
            out,
            format!("{prefix}.aggregate"),
            nd,
            vec![n, output_dim],
            Visibility::DynamicPrivate,
        );
    } else {
        expected_tensor(
            out,
            format!("{prefix}.alignment_x"),
            nd,
            vec![nd],
            Visibility::DynamicPrivate,
        );
        expected_tensor(
            out,
            format!("{prefix}.alignment_d"),
            nd,
            vec![nd],
            Visibility::Public,
        );
        expected_tensor(
            out,
            format!("{prefix}.aligned"),
            nd,
            vec![n, output_dim],
            Visibility::DynamicPrivate,
        );
        append_division_aux_layout(out, prefix, "alignment_div", nd, bit_width);
        expected_tensor(
            out,
            format!("{prefix}.bias"),
            output_dim,
            vec![output_dim],
            Visibility::RegisteredModel,
        );
        expected_tensor(
            out,
            format!("{prefix}.head_output"),
            nd,
            vec![n, output_dim],
            Visibility::DynamicPrivate,
        );
    }
}

fn canonical_expected_tensor_layout(
    registry: &RelationRegistry,
    schema: &serde_json::Value,
) -> Result<Vec<ExpectedTensorFamily>> {
    let features = registry.tensor("public.features")?;
    let edge_src = registry.tensor("public.edge_src")?;
    let node_ptr = registry.tensor("public.node_ptr")?;
    let logits = registry.tensor("public.logits")?;
    let n = features.logical_shape.first().copied().unwrap_or(0);
    let d = features.logical_shape.get(1).copied().unwrap_or(0);
    let e = edge_src.logical_len;
    let g = node_ptr.logical_len.checked_sub(1).unwrap_or(0);
    let c = logits.logical_shape.get(1).copied().unwrap_or(0);
    ensure!(
        n > 0 && d > 0 && e > 0 && g > 0 && c > 0,
        "canonical public shape profile is incomplete"
    );
    ensure!(
        features.logical_len == n * d,
        "canonical feature shape is inconsistent"
    );
    let profile = canonical_architecture_profile(schema, n, e, g, d, c)?;
    let bit_width = profile.bit_width;
    let lookup_bits = (
        *profile
            .lookup_bits
            .get("lrelu")
            .ok_or_else(|| anyhow::anyhow!("canonical lrelu lookup width is missing"))?,
        *profile
            .lookup_bits
            .get("exp")
            .ok_or_else(|| anyhow::anyhow!("canonical exp lookup width is missing"))?,
        *profile
            .lookup_bits
            .get("elu")
            .ok_or_else(|| anyhow::anyhow!("canonical elu lookup width is missing"))?,
    );

    let mut expected = Vec::new();
    expected_tensor(
        &mut expected,
        "public.features",
        n * d,
        vec![n, d],
        Visibility::Public,
    );
    expected_tensor(
        &mut expected,
        "public.edge_src",
        e,
        vec![e],
        Visibility::Public,
    );
    expected_tensor(
        &mut expected,
        "public.edge_dst",
        e,
        vec![e],
        Visibility::Public,
    );
    expected_tensor(
        &mut expected,
        "public.node_ptr",
        g + 1,
        vec![g + 1],
        Visibility::Public,
    );
    expected_tensor(
        &mut expected,
        "public.edge_ptr",
        g + 1,
        vec![g + 1],
        Visibility::Public,
    );
    let mut previous_dim = d;
    for layer in &profile.hidden_layers {
        ensure!(
            layer.input_dim == previous_dim,
            "hidden layer input dimension does not match previous concat"
        );
        for head in 0..layer.head_count {
            append_head_tensor_layout(
                &mut expected,
                &format!("hidden.layer_{}.head_{head}", layer.layer_index),
                n,
                e,
                layer.input_dim,
                layer.head_dim,
                bit_width,
                lookup_bits,
                true,
            );
        }
        previous_dim = layer
            .head_count
            .checked_mul(layer.head_dim)
            .ok_or_else(|| anyhow::anyhow!("hidden concat width overflows usize"))?;
        expected_tensor(
            &mut expected,
            format!("hidden.layer_{}.concat", layer.layer_index),
            n * previous_dim,
            vec![n, previous_dim],
            Visibility::DynamicPrivate,
        );
    }
    for head in 0..profile.output_head_count {
        append_head_tensor_layout(
            &mut expected,
            &format!("output.head_{head}"),
            n,
            e,
            previous_dim,
            c,
            bit_width,
            lookup_bits,
            false,
        );
    }
    let nd = n * c;
    expected_tensor(
        &mut expected,
        "output.head_sum",
        nd,
        vec![n, c],
        Visibility::DynamicPrivate,
    );
    expected_tensor(
        &mut expected,
        "output.final_logits",
        nd,
        vec![n, c],
        Visibility::DynamicPrivate,
    );
    expected_tensor(
        &mut expected,
        "public.logits",
        nd,
        vec![n, c],
        Visibility::Public,
    );
    expected_tensor(
        &mut expected,
        "public.k_out",
        nd,
        vec![nd],
        Visibility::Public,
    );
    Ok(expected)
}

fn canonical_expected_operation_ids(
    registry: &RelationRegistry,
    schema: &serde_json::Value,
) -> Result<Vec<(String, String, usize, usize)>> {
    let n = registry.tensor("public.features")?.logical_shape[0];
    let e = registry.tensor("public.edge_src")?.logical_len;
    let c = registry.tensor("public.logits")?.logical_shape[1];
    let profile = canonical_architecture_profile(
        schema,
        n,
        e,
        registry.tensor("public.node_ptr")?.logical_len - 1,
        registry.tensor("public.features")?.logical_shape[1],
        c,
    )?;
    let mut out = Vec::new();
    let mut add = |id: String, kind: &str, logical: usize| {
        let base = logical.max(2).next_power_of_two();
        out.push((id, kind.to_string(), logical, base * 2));
    };
    for layer in &profile.hidden_layers {
        for head in 0..layer.head_count {
            let prefix = format!("hidden.layer_{}.head_{head}", layer.layer_index);
            for col in 0..layer.head_dim {
                add(
                    format!("{prefix}.projection.col_{col}"),
                    "projection_matmul",
                    n,
                );
            }
            add(format!("{prefix}.attention_src"), "attention_dot", n);
            add(format!("{prefix}.attention_dst"), "attention_dot", n);
            add(format!("{prefix}.route_src"), "public_sparse_route", e);
            add(format!("{prefix}.route_dst"), "public_sparse_route", e);
            add(format!("{prefix}.score_add"), "score_add", e);
            add(format!("{prefix}.lrelu"), "lrelu_indexed_public_table", e);
            add(format!("{prefix}.max_relation"), "exact_group_max", e);
            add(format!("{prefix}.exp"), "exp_indexed_public_table", e);
            add(format!("{prefix}.group_sum"), "group_sum", e);
            add(format!("{prefix}.inverse"), "inverse_alpha", n);
            add(format!("{prefix}.alpha"), "inverse_alpha", e);
            add(
                format!("{prefix}.route_projection_relation"),
                "public_sparse_route",
                e * layer.head_dim,
            );
            add(
                format!("{prefix}.u_message_relation"),
                "u_message_hadamard",
                e * layer.head_dim,
            );
            add(
                format!("{prefix}.alpha_message_relation"),
                "alpha_message_hadamard",
                e * layer.head_dim,
            );
            add(
                format!("{prefix}.integer_aggregate"),
                "integer_numerator",
                e * layer.head_dim,
            );
            add(
                format!("{prefix}.field_aggregate_relation"),
                "field_aggregate",
                e * layer.head_dim,
            );
            add(
                format!("{prefix}.dual_consistency"),
                "dual_aggregate_consistency",
                n * layer.head_dim,
            );
            add(
                format!("{prefix}.recovery_div"),
                "signed_round_div",
                n * layer.head_dim,
            );
            add(
                format!("{prefix}.elu"),
                "elu_indexed_public_table",
                n * layer.head_dim,
            );
        }
        for head_index in 0..layer.head_count {
            for col in 0..layer.head_dim {
                add(
                    format!(
                        "hidden.layer_{}.concat.head_{head_index}.col_{col}",
                        layer.layer_index
                    ),
                    "hidden_concat",
                    n,
                );
            }
        }
    }
    for head in 0..profile.output_head_count {
        let prefix = format!("output.head_{head}");
        for col in 0..c {
            add(
                format!("{prefix}.projection.col_{col}"),
                "projection_matmul",
                n,
            );
        }
        add(format!("{prefix}.attention_src"), "attention_dot", n);
        add(format!("{prefix}.attention_dst"), "attention_dot", n);
        add(format!("{prefix}.route_src"), "public_sparse_route", e);
        add(format!("{prefix}.route_dst"), "public_sparse_route", e);
        add(format!("{prefix}.score_add"), "score_add", e);
        add(format!("{prefix}.lrelu"), "lrelu_indexed_public_table", e);
        add(format!("{prefix}.max_relation"), "exact_group_max", e);
        add(format!("{prefix}.exp"), "exp_indexed_public_table", e);
        add(format!("{prefix}.group_sum"), "group_sum", e);
        add(format!("{prefix}.inverse"), "inverse_alpha", n);
        add(format!("{prefix}.alpha"), "inverse_alpha", e);
        add(
            format!("{prefix}.route_projection_relation"),
            "public_sparse_route",
            e * c,
        );
        add(
            format!("{prefix}.u_message_relation"),
            "u_message_hadamard",
            e * c,
        );
        add(
            format!("{prefix}.alpha_message_relation"),
            "alpha_message_hadamard",
            e * c,
        );
        add(
            format!("{prefix}.integer_aggregate"),
            "integer_numerator",
            e * c,
        );
        add(
            format!("{prefix}.field_aggregate_relation"),
            "field_aggregate",
            e * c,
        );
        add(
            format!("{prefix}.dual_consistency"),
            "dual_aggregate_consistency",
            n * c,
        );
        add(format!("{prefix}.recovery_div"), "signed_round_div", n * c);
        add(format!("{prefix}.alignment_div"), "signed_round_div", n * c);
        add(format!("{prefix}.bias_add"), "output_bias", n * c);
    }
    let nd = n * c;
    add("output.head_sum".into(), "head_sum_final_average", nd);
    add("output.final_average".into(), "head_sum_final_average", nd);
    add("output.public_logits".into(), "public_logits_terminal", nd);
    ensure!(
        profile.output_head_count > 0,
        "canonical output head count is zero"
    );
    Ok(out)
}

fn canonical_fac_plan(registry: &RelationRegistry) -> Result<serde_json::Value> {
    let mut operations = Vec::with_capacity(registry.operation_families.len());
    for op in &registry.operation_families {
        let mut groups: BTreeMap<String, Vec<(usize, &ViewIndex)>> = BTreeMap::new();
        for (column_index, column) in op.columns.iter().enumerate() {
            if let ColumnSource::TensorView {
                family_id,
                canonical_index_by_row,
            } = &column.source
            {
                if registry.tensor(family_id)?.visibility != Visibility::Public {
                    groups
                        .entry(family_id.clone())
                        .or_default()
                        .push((column_index, canonical_index_by_row));
                }
            }
        }
        let mut claims = Vec::with_capacity(groups.len());
        for (family_id, columns) in groups {
            let spec = registry.tensor(&family_id)?;
            let non_identity = columns.iter().any(|(_, view)| {
                !matches!(view, ViewIndex::PaddedAffine { offset: 0, stride: 1, logical_rows, padding_index: 0 } if *logical_rows as usize == op.real_rows && spec.logical_len == op.real_rows)
            });
            claims.push(serde_json::json!({
                "family_id": family_id,
                "column_indices": columns.iter().map(|(index, _)| *index).collect::<Vec<_>>(),
                "column_ids": columns.iter().map(|(index, _)| op.columns[*index].column_id.clone()).collect::<Vec<_>>(),
                "view_count": columns.len(),
                "requires_reindex": non_identity || columns.len() > 1,
                "requires_line8": true,
            }));
        }
        operations.push(serde_json::json!({
            "operation_id": op.family_id,
            "terminal_claims": claims,
            "terminal_target": "canonical_tensor_family",
        }));
    }
    Ok(serde_json::json!({
        "schema": "zkDL_gat_fac_reindex_plan_v1",
        "operations": operations,
    }))
}

fn validate_canonical_schema_descriptor(registry: &RelationRegistry) -> Result<()> {
    let schema = &registry.canonical_schema;
    let _ = canonical_architecture_profile(
        schema,
        registry.tensor("public.features")?.logical_shape[0],
        registry.tensor("public.edge_src")?.logical_len,
        registry
            .tensor("public.node_ptr")?
            .logical_len
            .checked_sub(1)
            .ok_or_else(|| anyhow::anyhow!("canonical public node pointer family is empty"))?,
        registry.tensor("public.features")?.logical_shape[1],
        registry.tensor("public.logits")?.logical_shape[1],
    )?;
    ensure!(
        schema.get("schema").and_then(serde_json::Value::as_str)
            == Some("zkDL_gat_canonical_relation_schema_v1"),
        "canonical relation schema descriptor mismatch"
    );
    ensure!(
        schema.get("version").and_then(serde_json::Value::as_u64) == Some(1),
        "canonical relation schema descriptor version mismatch"
    );
    let expected_tensors = canonical_expected_tensor_layout(registry, schema)?;
    let expected_tensor_ids = expected_tensors
        .iter()
        .map(|tensor| tensor.family_id.clone())
        .collect::<Vec<_>>();
    let actual_tensor_ids = registry
        .tensor_families
        .iter()
        .map(|tensor| tensor.family_id.clone())
        .collect::<Vec<_>>();
    ensure!(
        actual_tensor_ids == expected_tensor_ids,
        "canonical tensor family IDs/order/unknown entries mismatch"
    );
    ensure!(
        schema.get("tensor_family_order") == Some(&serde_json::json!(expected_tensor_ids)),
        "canonical tensor family order descriptor mismatch"
    );
    for (actual, expected) in registry.tensor_families.iter().zip(expected_tensors) {
        ensure!(
            actual.logical_len == expected.logical_len
                && actual.logical_shape == expected.logical_shape
                && actual.visibility == expected.visibility
                && actual.padded_len == expected.logical_len.max(2).next_power_of_two()
                && actual.mle_order.len() == 1
                && actual.mle_order[0] == CANONICAL_TENSOR_MLE_ORDER
                && actual.instance_order.len() == 1
                && actual.instance_order[0] == CANONICAL_TENSOR_INSTANCE_ORDER,
            "canonical tensor family shape/visibility/padding/MLE order mismatch: {}",
            actual.family_id
        );
    }
    ensure!(
        schema.get("tensor_families") == Some(&serde_json::to_value(&registry.tensor_families)?),
        "canonical tensor family descriptor does not match the registry"
    );

    let expected_operations = canonical_expected_operation_ids(registry, schema)?;
    ensure!(
        registry.operation_families.len() == expected_operations.len(),
        "canonical operation family count mismatch"
    );
    for (operation, (id, kind, real_rows, rows)) in
        registry.operation_families.iter().zip(expected_operations)
    {
        ensure!(
            operation.family_id == id
                && operation.operation_kind == kind
                && operation.real_rows == real_rows
                && operation.rows == rows,
            "canonical operation ID/order/kind/domain mismatch: {}",
            operation.family_id
        );
    }
    ensure!(
        schema.get("operation_family_order")
            == Some(&serde_json::json!(registry
                .operation_families
                .iter()
                .map(|op| op.family_id.clone())
                .collect::<Vec<_>>())),
        "canonical operation family order descriptor mismatch"
    );
    ensure!(
        schema.get("operation_families")
            == Some(&serde_json::to_value(&registry.operation_families)?),
        "canonical operation columns/residuals/view descriptor does not match the registry"
    );
    ensure!(
        schema.get("fac_plan") == Some(&canonical_fac_plan(registry)?),
        "canonical FAC re-index/compression plan mismatch"
    );
    let terminal_value = schema
        .get("terminal_plan")
        .ok_or_else(|| anyhow::anyhow!("canonical terminal plan is missing"))?;
    let terminal = ensure_exact_object_keys(
        terminal_value,
        &["public_logits_family", "public_logits_operation", "target"],
        "canonical terminal plan",
    )?;
    ensure!(
        terminal
            .get("public_logits_operation")
            .and_then(serde_json::Value::as_str)
            == Some("output.public_logits")
            && terminal
                .get("public_logits_family")
                .and_then(serde_json::Value::as_str)
                == Some("public.logits")
            && terminal.get("target").and_then(serde_json::Value::as_str)
                == Some("verifier_local_public_y"),
        "canonical terminal binding plan mismatch"
    );
    Ok(())
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

    /// Enforce the baseline's fixed public boundary and terminal relation.
    ///
    /// `RelationRegistry::validate` checks generic FAC well-formedness.  That
    /// is deliberately not sufficient for the production adapter: a caller
    /// must not be able to rename `public.logits`, introduce an alias family,
    /// or replace the public-logits terminal with an arbitrary equality.  The
    /// canonical adapter emits the exact shapes and views checked here.
    pub fn validate_canonical_gat(&self) -> Result<()> {
        validate_canonical_schema_descriptor(self)?;
        const PUBLIC_FAMILIES: &[&str] = &[
            "public.features",
            "public.edge_src",
            "public.edge_dst",
            "public.node_ptr",
            "public.edge_ptr",
            "public.logits",
            "public.k_out",
        ];

        for family in &self.tensor_families {
            if family.family_id.starts_with("public.") {
                ensure!(
                    PUBLIC_FAMILIES.contains(&family.family_id.as_str()),
                    "non-canonical public tensor family alias: {}",
                    family.family_id
                );
                ensure!(
                    family.visibility == Visibility::Public,
                    "canonical public tensor is not public: {}",
                    family.family_id
                );
            }
            if family.visibility == Visibility::Public && !family.family_id.starts_with("public.") {
                ensure!(
                    canonical_public_auxiliary_family(&family.family_id),
                    "non-canonical auxiliary public tensor family alias: {}",
                    family.family_id
                );
            }
            if family.visibility == Visibility::DynamicPrivate {
                ensure!(
                    canonical_dynamic_family(&family.family_id),
                    "non-canonical dynamic tensor family alias: {}",
                    family.family_id
                );
            }
            if family.visibility == Visibility::RegisteredModel {
                ensure!(
                    canonical_registered_parameter_family(&family.family_id),
                    "non-canonical registered model family alias: {}",
                    family.family_id
                );
            }
        }
        for family_id in PUBLIC_FAMILIES {
            let matches = self
                .tensor_families
                .iter()
                .filter(|family| family.family_id == *family_id)
                .collect::<Vec<_>>();
            ensure!(
                matches.len() == 1,
                "canonical public tensor family must occur exactly once: {}",
                family_id
            );
            ensure!(
                matches[0].visibility == Visibility::Public,
                "canonical public tensor family visibility mismatch: {}",
                family_id
            );
        }
        let logits = self.tensor("public.logits")?.logical_len;
        ensure!(
            self.public_tensors.get("public.logits").map(Vec::len) == Some(logits),
            "canonical public logits payload shape mismatch"
        );

        let terminals = self
            .operation_families
            .iter()
            .filter(|op| op.operation_kind == "public_logits_terminal")
            .collect::<Vec<_>>();
        ensure!(
            terminals.len() == 1,
            "canonical public-logits terminal must occur exactly once"
        );
        let op = terminals[0];
        ensure!(
            op.family_id == "output.public_logits",
            "canonical public-logits terminal family ID mismatch"
        );
        let base = logits.max(2).next_power_of_two();
        ensure!(
            op.real_rows == logits && op.rows == base * 2,
            "canonical public-logits terminal domain shape mismatch"
        );
        ensure!(
            op.columns.len() == 4
                && op.columns[0].column_id == "private_final"
                && op.columns[1].column_id == "public_y"
                && op.columns[2].column_id == "valid"
                && op.columns[3].column_id == "domain_selector",
            "canonical public-logits terminal column order mismatch"
        );
        match &op.columns[0].source {
            ColumnSource::TensorView {
                family_id,
                canonical_index_by_row,
            } => {
                ensure!(
                    family_id == "output.final_logits",
                    "public-logits terminal private source is not output.final_logits"
                );
                ensure!(
                    self.tensor(family_id)?.visibility == Visibility::DynamicPrivate,
                    "public-logits terminal private source visibility mismatch"
                );
                ensure!(
                    self.tensor(family_id)?.logical_len == logits,
                    "public-logits terminal private source shape mismatch"
                );
                ensure!(
                    is_canonical_terminal_view(canonical_index_by_row, logits),
                    "public-logits private source mapping is not canonical"
                );
            }
            _ => bail!("public-logits terminal private source must be a tensor view"),
        }
        match &op.columns[1].source {
            ColumnSource::TensorView {
                family_id,
                canonical_index_by_row,
            } => {
                ensure!(
                    family_id == "public.logits",
                    "public-logits terminal public source is not public.logits"
                );
                ensure!(
                    is_canonical_terminal_view(canonical_index_by_row, logits),
                    "public-logits public source mapping is not canonical"
                );
            }
            _ => bail!("public-logits terminal public source must be a tensor view"),
        }
        for (index, expected) in [(2usize, "valid"), (3usize, "domain_selector")] {
            let values = match &op.columns[index].source {
                ColumnSource::Public { values } => values,
                _ => bail!("canonical terminal selector must be public"),
            };
            ensure!(
                values.len() == op.rows
                    && values.iter().enumerate().all(|(row, value)| {
                        *value
                            == if row < op.real_rows {
                                Scalar::one()
                            } else {
                                Scalar::zero()
                            }
                    }),
                "canonical terminal {} selector payload mismatch",
                expected
            );
        }
        for operation in &self.operation_families {
            ensure!(
                operation.family_id.starts_with("hidden.layer_")
                    || operation.family_id.starts_with("output."),
                "non-canonical operation family alias: {}",
                operation.family_id
            );
            ensure!(
                canonical_operation_id_matches_kind(operation),
                "operation family ID does not match its canonical operation kind: {} ({})",
                operation.family_id,
                operation.operation_kind
            );
            for column in &operation.columns {
                if let ColumnSource::TensorView { family_id, .. } = &column.source {
                    if canonical_public_auxiliary_family(family_id) {
                        ensure!(
                            operation.operation_kind == "signed_round_div"
                                && operation.family_id.ends_with(".alignment_div"),
                            "public alignment denominator is used outside its canonical relation"
                        );
                    }
                }
            }
        }
        ensure!(
            op.residuals.len() == 1,
            "canonical terminal residual count mismatch"
        );
        let selector = 3u32;
        let valid = 2u32;
        let residual = &op.residuals[0];
        let canonical = match residual {
            Expr::Mul(a, b) if a.is_column(selector) => match b.as_ref() {
                Expr::Mul(a, b) if a.is_column(valid) => match b.as_ref() {
                    Expr::Sub(a, b) => a.is_column(0) && b.is_column(1),
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        };
        ensure!(
            canonical,
            "public-logits terminal residual is not canonical Y binding"
        );
        ensure!(
            op.degree_bound == 4,
            "canonical public-logits terminal degree bound mismatch"
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
