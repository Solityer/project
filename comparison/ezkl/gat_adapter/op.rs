//! Native ezkl `Op` implementation for the canonical integer GAT relation.
//!
//! The operation is deliberately one graph node so the adapter can preserve
//! the official `GraphCircuit`/`SupportedOp`/settings path.  Internally it is
//! composed from ezkl's existing einsum, polynomial, shuffle/max, exact
//! dynamic lookup and equality layouts. Public graph routing uses fixed
//! topology wiring after the statement inputs are bound to the circuit graph.
//! No witness is accepted without a matching circuit relation.

use std::collections::HashMap;

use halo2_proofs::circuit::Value;
use halo2curves::bn256::Fr as Fp;
use halo2curves::ff::{Field, PrimeField};

use crate::circuit::ops::region::RegionCtx;
use crate::circuit::ops::{base::BaseOp, layouts, Op};
use crate::circuit::{BaseConfig, CircuitError};
use crate::fieldutils::{integer_rep_to_felt, IntegerRep};
use crate::tensor::{Tensor, ValTensor, ValType};

use super::relation::RoundingMode;
use super::schema::{scale_exponent_to_multiplier, BoundFamily, CanonicalGat, ExactTable};

pub(super) const MAX_GAT_EDGE_CHUNK_CELLS: usize = 1 << 18;

/// Source-derived bound for adapter-owned `ValType` payloads around one
/// `aggregate_head` call. This does not include Halo2's region-assignment,
/// lookup-table, permutation, or proof-system workspaces; callers combine
/// those separately with the native resource census.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GatAggregateTensorLiveBound {
    /// Largest channel count among the relation's hidden and output heads.
    pub channels: usize,
    /// Adapter input tensors borrowed by `aggregate_head`: source, destination,
    /// projected nodes and the two attention vectors.
    pub input_cells: usize,
    /// Score/max-prefix stage, including the score and repeated maximum that
    /// remain live while the segmented-prefix recurrence is assembled.
    pub score_prefix_cells: usize,
    /// Chunked numerator/field aggregation stage, including edge endpoints,
    /// carries, and bounded chunk-local route/product/prefix tensors.
    pub chunk_aggregation_cells: usize,
    /// SignedRoundDiv stage's dividend-sized witness tensors and node vectors.
    pub signed_round_cells: usize,
    /// Largest phase-local component, before caller-owned aggregate inputs.
    pub phase_peak_cells: usize,
    /// Aggregate inputs plus the largest phase-local component.
    pub peak_cells: usize,
    /// Conservative peak adapter-owned `ValType` payload bytes.
    pub peak_bytes: u64,
}

/// Bound the largest source-visible adapter tensor payload in any GAT head.
/// The cell formulas follow the explicit `aggregate_head` lifetimes: 12E for
/// score/max-repeat plus segmented-prefix working tensors and an assignment
/// conversion allowance, and a separate chunk
/// phase with persistent alpha/exp, two N*C endpoint buffers, four bounded
/// edge-chunk channel tensors, the scalar edge-chunk tensor, and carry cells.
/// Exact element size comes from the live native `ValType<Fp>` representation.
pub(crate) fn gat_aggregate_tensor_live_bound(
    spec: &CanonicalGat,
) -> Result<GatAggregateTensorLiveBound, String> {
    fn add(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
        lhs.checked_add(rhs)
            .ok_or_else(|| format!("GAT tensor live-bound {} overflow", label))
    }
    fn mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
        lhs.checked_mul(rhs)
            .ok_or_else(|| format!("GAT tensor live-bound {} overflow", label))
    }

    let graph = &spec.graph;
    if graph.n == 0 || graph.e == 0 {
        return Err("GAT tensor live bound requires nonempty nodes and edges".into());
    }
    let channels = spec
        .model
        .hidden_layers
        .iter()
        .map(|layer| layer.d_head)
        .chain(spec.model.output_heads.iter().map(|_| spec.model.classes))
        .max()
        .ok_or_else(|| "GAT tensor live bound requires at least one attention head".to_string())?;
    if channels == 0 || channels > MAX_GAT_EDGE_CHUNK_CELLS {
        return Err(format!(
            "GAT tensor live bound has unsupported channel count {}",
            channels
        ));
    }

    let node_channels = mul(graph.n, channels, "node-channel cells")?;
    let attention_cells = mul(channels, 2, "attention-vector cells")?;
    let topology_edge_cells = mul(graph.e, 2, "source/destination edge cells")?;
    let input_cells = add(
        add(topology_edge_cells, node_channels, "aggregate input cells")?,
        attention_cells,
        "aggregate input cells",
    )?;

    let score_prefix_cells = add(
        mul(graph.e, 12, "score-prefix edge cells")?,
        mul(graph.n, 4, "score-prefix node cells")?,
        "score-prefix cells",
    )?;

    let chunk_rows = (MAX_GAT_EDGE_CHUNK_CELLS / channels).max(1).min(graph.e);
    let chunk_channel_cells = mul(chunk_rows, channels, "chunk channel cells")?;
    let chunk_aggregation_cells = add(
        add(
            add(
                mul(graph.e, 2, "persistent alpha/exp cells")?,
                mul(node_channels, 2, "endpoint capacity cells")?,
                "chunk aggregation cells",
            )?,
            mul(chunk_channel_cells, 4, "chunk channel scratch cells")?,
            "chunk aggregation cells",
        )?,
        add(chunk_rows, mul(channels, 2, "chunk carry cells")?, "chunk row/carry cells")?,
        "chunk aggregation cells",
    )?;

    // signed_round_div_layout materializes q, q_floor, rem, abs, sigma, zero,
    // and rounded_up at dividend shape, then keeps the assigned witnesses,
    // signed reconstruction intermediates and equality outputs live through
    // the rounding identities. Sixteen N*C slots conservatively cover those
    // named dividend-shaped groups; node slots cover divisor/inverse and
    // endpoint/assignment intermediates.
    let signed_round_cells = add(
        mul(node_channels, 16, "SignedRoundDiv channel cells")?,
        mul(graph.n, 8, "SignedRoundDiv node cells")?,
        "SignedRoundDiv cells",
    )?;
    let phase_peak_cells = score_prefix_cells
        .max(chunk_aggregation_cells)
        .max(signed_round_cells);
    let peak_cells = add(input_cells, phase_peak_cells, "peak cells")?;
    let bytes = (peak_cells as u128)
        .checked_mul(std::mem::size_of::<ValType<Fp>>() as u128)
        .ok_or_else(|| "GAT tensor live-bound byte size overflow".to_string())?;
    let peak_bytes = u64::try_from(bytes)
        .map_err(|_| "GAT tensor live-bound byte size exceeds u64".to_string())?;

    Ok(GatAggregateTensorLiveBound {
        channels,
        input_cells,
        score_prefix_cells,
        chunk_aggregation_cells,
        signed_round_cells,
        phase_peak_cells,
        peak_cells,
        peak_bytes,
    })
}

/// Conservative adapter-owned `ValType` payload envelope over one complete
/// GAT layout. It combines the per-head aggregate bound with the source-visible
/// model/input tensors retained around each head, projection input/output
/// copies, completed heads, head interleaves, output-head accumulation and the
/// final signed average. Native Halo2 layout scratch and proof-system buffers
/// remain separate resource terms.
pub(crate) fn gat_transient_witness_bytes_upper(spec: &CanonicalGat) -> Result<u64, String> {
    fn add(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
        lhs.checked_add(rhs)
            .ok_or_else(|| format!("GAT relation tensor bound {} overflow", label))
    }
    fn mul(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
        lhs.checked_mul(rhs)
            .ok_or_else(|| format!("GAT relation tensor bound {} overflow", label))
    }

    let graph = &spec.graph;
    let aggregate = gat_aggregate_tensor_live_bound(spec)?;
    let params = spec.model.flattened_params_len()?;
    let feature_cells = mul(graph.n_total, spec.feature_dim, "input features")?;
    let pointer_len = add(graph.g_batch, 1, "pointer length")?;
    let topology_input_cells = add(
        add(graph.n, mul(graph.e, 2, "edge topology")?, "topology")?,
        mul(pointer_len, 2, "graph pointers")?,
        "topology",
    )?;
    // The flat model input remains owned by the graph while weight_tensors
    // holds its shape-specific copies. Public topology inputs likewise remain
    // live beside GatOp's bound source/destination/node/pointer tensors.
    let outer_cells = add(
        add(
            mul(params, 2, "flat and split model parameters")?,
            feature_cells,
            "outer model inputs",
        )?,
        mul(topology_input_cells, 2, "public and bound topology")?,
        "outer model inputs",
    )?;

    let mut phase_peak_cells = 0usize;
    let feature_gather_cells = add(
        feature_cells,
        mul(graph.n, spec.feature_dim, "gathered input features")?,
        "feature gather phase",
    )?;
    phase_peak_cells = phase_peak_cells.max(feature_gather_cells);
    phase_peak_cells = phase_peak_cells.max(graph.n.max(graph.e).max(graph.g_batch + 1));
    let mut current_dim = spec.feature_dim;
    for layer in &spec.model.hidden_layers {
        let mut completed_head_channels = 0usize;
        for _head in &layer.heads {
            let channels = layer.d_head;
            let input_rows = mul(graph.n, layer.d_in, "hidden layer input")?;
            let completed_outputs = mul(graph.n, completed_head_channels, "completed hidden heads")?;
            let projection_weights = mul(layer.d_in, channels, "hidden projection weights")?;
            let projected_nodes = mul(graph.n, channels, "hidden projection output")?;
            let attention_weights = mul(channels, 2, "hidden attention weights")?;
            // The native projection consumes the input and weight tensors and
            // materializes the projected output while its assigned einsum
            // operands remain live. Count one additional input/weight/output
            // working set beyond those retained head tensors.
            let projection_workspace = add(
                add(input_rows, projection_weights, "hidden projection workspace")?,
                projected_nodes,
                "hidden projection workspace",
            )?;
            let head_context = add(
                add(
                    add(input_rows, completed_outputs, "hidden head context")?,
                    projection_weights,
                    "hidden head context",
                )?,
                add(
                    add(projected_nodes, attention_weights, "hidden head context")?,
                    projection_workspace,
                    "hidden projection workspace",
                )?,
                "hidden head context",
            )?;
            phase_peak_cells = phase_peak_cells.max(add(
                head_context,
                aggregate.phase_peak_cells,
                "hidden aggregate phase",
            )?);
            completed_head_channels = add(
                completed_head_channels,
                channels,
                "completed hidden head channels",
            )?;
        }
        // During head interleave, the previous layer input, all completed
        // heads, and the new concatenated tensor coexist.
        let previous_layer = mul(graph.n, layer.d_in, "interleave input")?;
        let all_head_outputs = mul(graph.n, completed_head_channels, "interleave heads")?;
        phase_peak_cells = phase_peak_cells.max(add(
            previous_layer,
            mul(all_head_outputs, 2, "interleave input/output heads")?,
            "hidden interleave phase",
        )?);
        current_dim = completed_head_channels;
    }

    let output_head_count = spec.model.output_heads.len();
    for (head_idx, _head) in spec.model.output_heads.iter().enumerate() {
        let channels = spec.model.classes;
        let input_rows = mul(graph.n, current_dim, "output layer input")?;
        let completed_outputs = mul(
            mul(graph.n, channels, "completed output heads")?,
            head_idx,
            "completed output heads",
        )?;
        let projection_weights = mul(current_dim, channels, "output projection weights")?;
        let projected_nodes = mul(graph.n, channels, "output projection output")?;
        let attention_and_bias = mul(channels, 3, "output attention and bias")?;
        let projection_workspace = add(
            add(input_rows, projection_weights, "output projection workspace")?,
            projected_nodes,
            "output projection workspace",
        )?;
        let head_context = add(
            add(
                add(input_rows, completed_outputs, "output head context")?,
                projection_weights,
                "output head context",
            )?,
            add(
                add(projected_nodes, attention_and_bias, "output head context")?,
                projection_workspace,
                "output projection workspace",
            )?,
            "output head context",
        )?;
        let output_signed_round = mul(
            mul(graph.n, channels, "output signed-round shape")?,
            16,
            "output signed-round workspace",
        )?;
        let output_bias_and_alignment = mul(
            mul(graph.n, channels, "output bias/alignment shape")?,
            4,
            "output bias/alignment workspace",
        )?;
        phase_peak_cells = phase_peak_cells.max(add(
            add(head_context, aggregate.phase_peak_cells, "output aggregate phase")?,
            add(
                output_signed_round,
                output_bias_and_alignment,
                "output head post-aggregation phase",
            )?,
            "output head phase",
        )?);
    }

    let output_cells = mul(
        mul(graph.n, spec.model.classes, "output head shape")?,
        output_head_count,
        "all output heads",
    )?;
    let output_accumulator_phase = add(
        add(
            mul(graph.n, current_dim, "final output hidden input")?,
            output_cells,
            "final output head values",
        )?,
        mul(
            mul(graph.n, spec.model.classes, "output accumulator shape")?,
            19,
            "output accumulator and final signed average",
        )?,
        "final output phase",
    )?;
    phase_peak_cells = phase_peak_cells.max(output_accumulator_phase);

    let peak_cells = add(outer_cells, phase_peak_cells, "whole relation peak")?;
    let bytes = (peak_cells as u128)
        .checked_mul(std::mem::size_of::<ValType<Fp>>() as u128)
        .ok_or_else(|| "GAT relation tensor bound byte size overflow".to_string())?;
    u64::try_from(bytes)
        .map_err(|_| "GAT relation tensor bound byte size exceeds u64".to_string())
}

fn checked_shape_len(shape: &[usize]) -> Option<usize> {
    shape.iter().try_fold(1usize, |len, dimension| len.checked_mul(*dimension))
}

fn is_exact_single_shape(shapes: &[Vec<usize>], expected: &[usize]) -> bool {
    shapes.len() == 1 && shapes[0].as_slice() == expected
}

/// Input positions used by [`GatOp`].
pub const FEATURE_INPUT: usize = 0;
/// Public node-id input position.
pub const NODE_ID_INPUT: usize = 1;
/// Public source-edge input position.
pub const SRC_INPUT: usize = 2;
/// Public destination-edge input position.
pub const DST_INPUT: usize = 3;
/// Public node-prefix input position.
pub const NODE_PTR_INPUT: usize = 4;
/// Public edge-prefix input position.
pub const EDGE_PTR_INPUT: usize = 5;
/// First private model tensor input position.
pub const FIRST_WEIGHT_INPUT: usize = 6;

/// A native graph operation carrying the complete canonical GAT contract.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GatOp {
    /// Canonical topology, model shapes, table identities and range ledger.
    pub spec: CanonicalGat,
    /// Exact signed-division rounding mode.
    pub rounding: RoundingMode,
}

impl GatOp {
    /// Construct and validate a canonical operation.
    pub fn new(spec: CanonicalGat, rounding: RoundingMode) -> Result<Self, String> {
        spec.validate_for_compile()?;
        Ok(Self { spec, rounding })
    }

    fn bound_family(&self, layer: usize, head: usize, is_output: bool,) -> Result<&BoundFamily, CircuitError> {
        self.spec
            .ledger
            .families
            .iter()
            .find(|family| { family.layer == layer && family.head == head && family.is_output == is_output})
            .ok_or(CircuitError::UnsupportedOp)
    }

    /// Remove private model values from a serialized compiled graph while
    /// retaining every shape, scale, bound and public H(W) binding needed to
    /// lay out the runtime witness.  The normal in-memory operation is never
    /// redacted; this is called only on the clone made by `GraphCircuit::save`.
    pub fn redact_private_params(&mut self) {
        for layer in &mut self.spec.model.hidden_layers {
            for head in &mut layer.heads {
                for value in &mut head.projection {
                    *value = 0;
                }
                for value in &mut head.a_src {
                    *value = 0;
                }
                for value in &mut head.a_dst {
                    *value = 0;
                }
            }
        }
        for head in &mut self.spec.model.output_heads {
            for value in &mut head.projection {
                *value = 0;
            }
            for value in &mut head.a_src {
                *value = 0;
            }
            for value in &mut head.a_dst {
                *value = 0;
            }
            for value in &mut head.bias {
                *value = 0;
            }
        }
        self.spec.private_params_redacted = true;
    }

    fn unknown_or_map<FN>(input: &ValTensor<Fp>, dims: &[usize], map: FN,) -> Result<ValTensor<Fp>, CircuitError>
    where
        FN: Fn(IntegerRep) -> Result<IntegerRep, String>,
    {
        let vals = if input.any_unknowns()? {
            vec![ValType::Value(Value::unknown()); dims.iter().product()]
        } else {
            input
                .int_evals()?
                .into_iter()
                .map(|x| {
                    map(x)
                        .map(|v| ValType::Value(Value::known(integer_rep_to_felt(v))))
                        .map_err(|e| { CircuitError::TensorError(crate::tensor::TensorError::DimError(e)) }) })
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut tensor: ValTensor<Fp> = Tensor::from_vec(vals, dims)?.into();
        tensor.set_scale(input.scale());
        Ok(tensor)
    }

    fn constant_i128(values: &[i128], dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        let vals = values.iter().map(|x| ValType::Constant(integer_rep_to_felt(*x))).collect::<Vec<_>>();
        Ok(Tensor::from_vec(vals, dims)?.into())
    }

    /// Create witness/advice values for data-dependent intermediates.
    ///
    /// A known value here is still a `Value::known`, not a circuit constant;
    /// key generation can therefore use unknown advice without baking a
    /// prover's witness into the compiled graph or changing the relation's
    /// fixed layout.
    fn value_i128(values: &[i128], dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        let vals = values.iter().map(|x| ValType::Value(Value::known(integer_rep_to_felt(*x)))).collect::<Vec<_>>();
        Ok(Tensor::from_vec(vals, dims)?.into())
    }

    fn unknown_tensor(dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        let vals = vec![ValType::Value(Value::unknown()); dims.iter().product()];
        Ok(Tensor::from_vec(vals, dims)?.into())
    }

    fn checked_signed_field(value: Fp) -> Option<i128> {
        let negative = value > Fp::from_u128(i128::MAX as u128);
        let magnitude = if negative { -value } else { value };
        let repr = magnitude.to_repr();
        let bytes = repr.as_ref();
        if bytes.len() < 16 || bytes[16..].iter().any(|byte| *byte != 0) {
            return None;
        }
        let magnitude = u128::from_le_bytes(bytes[..16].try_into().ok()?);
        if !negative {
            return i128::try_from(magnitude).ok();
        }
        if magnitude == (1_u128 << 127) {
            return Some(i128::MIN);
        }
        i128::try_from(magnitude).ok()?.checked_neg()
    }

    fn range_check_bounded(config: &BaseConfig<Fp>, region: &mut RegionCtx<Fp>, values: &[&ValTensor<Fp>; 1], range: &crate::circuit::table::Range) -> Result<(), CircuitError> {
        let value = values[0];
        if range.0 > range.1 {
            return Err(CircuitError::InvalidMinMaxRange(range.0, range.1));
        }
        let span = range.1.checked_sub(range.0).ok_or(CircuitError::UnsupportedOp)?;
        if span <= 1023 {
            layouts::range_check(config, region, &[value], range)?;
            return Ok(());
        }
        let significant_bits = 128 - (span as u128).leading_zeros() as usize;
        let limbs = significant_bits.checked_add(7).ok_or(CircuitError::UnsupportedOp)? / 8;
        if limbs > 15 {
            return Err(CircuitError::UnsupportedOp);
        }
        let no_wrap_bits = limbs.checked_mul(8).and_then(|bits| bits.checked_add(1)).ok_or(CircuitError::UnsupportedOp)?;
        // The BN254 scalar modulus is greater than 2^253; the two limb sums are below 2^no_wrap_bits.
        if no_wrap_bits > 253 {
            return Err(CircuitError::UnsupportedOp);
        }
        let maximum_difference = 1_i128.checked_shl((limbs * 8) as u32).and_then(|value| value.checked_sub(1)).ok_or(CircuitError::UnsupportedOp)?;
        let lower_constant = Self::constant_i128(&vec![range.0; value.len()], value.dims())?;
        let upper_constant = Self::constant_i128(&vec![range.1; value.len()], value.dims())?;
        let lower_diff = layouts::pairwise(config, region, &[value, &lower_constant], BaseOp::Sub)?;
        let upper_diff = layouts::pairwise(config, region, &[&upper_constant, value], BaseOp::Sub)?;
        for difference in [&lower_diff, &upper_diff] {
            if !difference.any_unknowns()? {
                for item in difference.get_inner_tensor()?.iter() {
                    let integer = Self::checked_signed_field(item.get_felt_eval().ok_or(CircuitError::UnsupportedOp)?).ok_or(CircuitError::UnsupportedOp)?;
                    if integer.checked_abs().map_or(true, |magnitude| magnitude > maximum_difference) {
                        return Err(CircuitError::UnsupportedOp);
                    }
                }
            }
        }
        let base = 256;
        let (lower_parts, lower_value) = layouts::decompose(config, region, &[&lower_diff], &base, &limbs, true)?;
        let (upper_parts, upper_value) = layouts::decompose(config, region, &[&upper_diff], &base, &limbs, true)?;
        let mut lower_sign = lower_parts.clone();
        lower_sign.get_every_n(limbs + 1)?;
        lower_sign.reshape(value.dims())?;
        layouts::range_check(config, region, &[&lower_sign], &(0, 1))?;
        let mut upper_sign = upper_parts;
        upper_sign.get_every_n(limbs + 1)?;
        upper_sign.reshape(value.dims())?;
        layouts::range_check(config, region, &[&upper_sign], &(0, 1))?;
        let difference_sum = layouts::pairwise(config, region, &[&lower_value, &upper_value], BaseOp::Add)?;
        let span_constant = Self::constant_i128(&vec![span; value.len()], value.dims())?;
        layouts::enforce_equality(config, region, &[&difference_sum, &span_constant])?;
        Ok(())
    }

    fn constrain_projected_bound(
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        projected: &ValTensor<Fp>,
        bounds: &BoundFamily,
        max_degree: usize,
    ) -> Result<(), CircuitError> {
        let bound = bounds
            .projected_abs_bound(max_degree)
            .map_err(CircuitError::DimMismatch)?;
        if bound == 0 {
            let zero = Self::constant_i128(&vec![0; projected.len()], projected.dims())?;
            layouts::enforce_equality(config, region, &[projected, &zero])?;
            return Ok(());
        }
        let lower = bound.checked_neg().ok_or(CircuitError::UnsupportedOp)?;
        Self::range_check_bounded(config, region, &[projected], &(lower, bound))
    }

    fn table(table: &ExactTable) -> Result<(ValTensor<Fp>, ValTensor<Fp>), CircuitError> {
        let keys = table.rows.iter().map(|(x, _)| ValType::Constant(integer_rep_to_felt(*x))).collect::<Vec<_>>();
        let values = table.rows.iter().map(|(_, y)| ValType::Constant(integer_rep_to_felt(*y))).collect::<Vec<_>>();
        let key_len = keys.len();
        let value_len = values.len();
        Ok((Tensor::from_vec(keys, &[key_len])?.into(), Tensor::from_vec(values, &[value_len])?.into(),)) }

    fn exact_lookup(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        input: &ValTensor<Fp>,
        table: &ExactTable,
        assigned_tables: &mut HashMap<String, layouts::AssignedDynamicLookupTable>,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        Self::require_scale(
            input,
            table.input_scale,
            &format!("exact table {} input", table.table_id),
        )?;
        let output = Self::unknown_or_map(input, input.dims(), |x| table.lookup(x))?;
        let identity = format!("{}:{}", table.table_id, table.digest);
        let token = if let Some(token) = assigned_tables.get(&identity) {
            *token
        } else {
            let assignment_started = std::time::Instant::now();
            println!(
                "EZKL_NATIVE_PHASE START stage=exact_table_assignment model_arch_id={} table_id={} rows={}",
                self.spec.model.model_arch_id,
                table.table_id,
                table.rows.len(),
            );
            let (keys, values) = Self::table(table)?;
            let token = layouts::assign_dynamic_lookup_table(config, region, &[&keys, &values])?;
            assigned_tables.insert(identity, token);
            println!(
                "EZKL_NATIVE_PHASE DONE stage=exact_table_assignment model_arch_id={} table_id={} rows={} elapsed_ms={:.3}",
                self.spec.model.model_arch_id,
                table.table_id,
                table.rows.len(),
                assignment_started.elapsed().as_secs_f64() * 1000.0,
            );
            token
        };
        let (_, mut checked_output) = layouts::dynamic_lookup_from_table(config, region, &[input, &output], token)?;
        if checked_output.len() != input.len() {
            return Err(CircuitError::DimMismatch("GAT exact lookup changed the canonical tensor length".into()));
        }
        checked_output.reshape(input.dims())?;
        checked_output.set_scale(table.output_scale);
        Ok(checked_output)
    }

    fn require_scale(tensor: &ValTensor<Fp>, expected: i32, label: &str,) -> Result<(), CircuitError> {
        if tensor.scale() != expected {
            return Err(CircuitError::DimMismatch(format!("{} scale {} does not match canonical scale {}", label, tensor.scale(), expected)));
        }
        Ok(())
    }

    fn require_dims(tensor: &ValTensor<Fp>, expected: &[usize], label: &str,) -> Result<(), CircuitError> {
        if tensor.dims() != expected {
            return Err(CircuitError::DimMismatch(format!("{} shape {:?} does not match canonical shape {:?}", label, tensor.dims(), expected)));
        }
        Ok(())
    }

    fn topology_constant(&self, which: usize) -> Result<ValTensor<Fp>, CircuitError> {
        let graph = &self.spec.graph;
        let (data, dims): (Vec<i128>, Vec<usize>) = match which {
            NODE_ID_INPUT => (graph.node_ids.iter().map(|x| *x as i128).collect(), vec![graph.n],),
            SRC_INPUT => (graph.src.iter().map(|x| *x as i128).collect(), vec![graph.e],),
            DST_INPUT => (graph.dst.iter().map(|x| *x as i128).collect(), vec![graph.e],),
            NODE_PTR_INPUT => (graph.node_ptr.iter().map(|x| *x as i128).collect(), vec![graph.g_batch + 1],),
            EDGE_PTR_INPUT => (graph.edge_ptr.iter().map(|x| *x as i128).collect(), vec![graph.g_batch + 1],),
            _ => return Err(CircuitError::UnsupportedOp),
        };
        Self::constant_i128(&data, &dims)
    }

    fn bind_topology(&self, config: &BaseConfig<Fp>, region: &mut RegionCtx<Fp>, input: &ValTensor<Fp>, slot: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        let public = layouts::identity(config, region, &[input])?;
        let expected = self.topology_constant(slot)?;
        layouts::enforce_equality(config, region, &[&public, &expected])
    }

    fn weight_tensors<'a>(&self, values: &'a [&ValTensor<Fp>],) -> Result<Vec<ValTensor<Fp>>, CircuitError> {
        let flat = values.get(FIRST_WEIGHT_INPUT).ok_or_else(|| CircuitError::DimMismatch("missing flat GAT parameter input".into()))?;
        if values.len() != FIRST_WEIGHT_INPUT + 1 {
            return Err(CircuitError::DimMismatch("canonical GAT requires one flat runtime parameter input".into(),));
        }
        Self::require_scale(
            flat,
            self.spec.scales.s_w_hidden[0][0],
            "flat GAT parameter input",
        )?;
        let expected_len = self.spec.model.flattened_params().len();
        if flat.len() != expected_len || flat.dims() != [expected_len] {
            return Err(CircuitError::DimMismatch(format!("flat GAT parameter length {} does not match canonical closure {}", flat.len(), expected_len)));
        }
        let mut at = 0usize;
        let mut out = Vec::new();
        for (layer_idx, layer) in self.spec.model.hidden_layers.iter().enumerate() {
            for (head_idx, _head) in layer.heads.iter().enumerate() {
                for (dims, scale) in [
                    (vec![layer.d_in, layer.d_head], self.spec.scales.s_w_hidden[layer_idx][head_idx],),(vec![layer.d_head], self.spec.scales.s_a_src_hidden[layer_idx][head_idx],),(vec![layer.d_head], self.spec.scales.s_a_dst_hidden[layer_idx][head_idx],),
                ] {
                    let len = dims.iter().product::<usize>();
                    let tensor = Self::flatten_slice(flat, at, at + len, &dims)?;
                    let mut tensor = tensor;
                    tensor.set_scale(scale);
                    out.push(tensor);
                    at += len;
                }
            }
        }
        for (head_idx, _head) in self.spec.model.output_heads.iter().enumerate() {
            for (dims, scale) in [
                (
                    vec![
                        self.spec.model.hidden_layers.last().unwrap().heads.len() * self.spec.model.hidden_layers.last().unwrap().d_head,
                        self.spec.model.classes,
                    ],
                    self.spec.scales.s_w_output[head_idx],
                ),
                (vec![self.spec.model.classes], self.spec.scales.s_a_src_output[head_idx],),(vec![self.spec.model.classes], self.spec.scales.s_a_dst_output[head_idx],),(vec![self.spec.model.classes], self.spec.scales.s_b_output[head_idx],),
            ] {
                let len = dims.iter().product::<usize>();
                let tensor = Self::flatten_slice(flat, at, at + len, &dims)?;
                let mut tensor = tensor;
                tensor.set_scale(scale);
                out.push(tensor);
                at += len;
            }
        }
        if at != flat.len() {
            return Err(CircuitError::DimMismatch("GAT flat parameter closure was not consumed exactly".into(),));
        }
        Ok(out)
    }

    /// Validate the real formal graph/tensor metadata before GraphCircuit
    /// allocates its expensive dummy layout.  This deliberately checks the
    /// adapter's rank-preserving broadcast plan, not just element counts.
    pub fn validate_pre_synthesis(
        spec: &CanonicalGat,
        input_shapes: &[Vec<usize>],
        output_shapes: &[Vec<usize>],
        param_shapes: &[Vec<usize>],
        input_scales: &[crate::Scale],
        param_scales: &[crate::Scale],
        output_scales: &[crate::Scale],
        inputs: &[Tensor<Fp>],
        params: &Tensor<Fp>,
        formal_input_scale: crate::Scale,
        formal_param_scale: crate::Scale,
    ) -> Result<(), String> {
        spec.validate_for_compile()?;
        let graph = &spec.graph;
        let public_shapes = vec![
            vec![graph.n_total, spec.feature_dim],
            vec![graph.n],
            vec![graph.e],
            vec![graph.e],
            vec![graph.g_batch + 1],
            vec![graph.g_batch + 1],
        ];
        if input_shapes != public_shapes.as_slice() || inputs.len() != public_shapes.len() {
            return Err(format!("formal GAT input shape metadata {:?} does not match {:?}", input_shapes, public_shapes));
        }
        let expected_input_scales = [spec.scales.s_input, 0, 0, 0, 0, 0];
        if input_scales != expected_input_scales.as_slice() || formal_input_scale != spec.scales.s_input {
            return Err("formal GAT input scale metadata disagrees with canonical scale chain".into());
        }
        for (index, (tensor, shape)) in inputs.iter().zip(&public_shapes).enumerate() {
            if tensor.dims() != shape.as_slice() || tensor.scale() != Some(expected_input_scales[index]) {
                return Err(format!("formal GAT input {} tensor metadata has shape {:?}, scale {:?}; expected {:?}, {}", index, tensor.dims(), tensor.scale(), shape, expected_input_scales[index]));
            }
        }
        let parameter_len = spec.model.flattened_params().len();
        let parameter_shape = vec![parameter_len];
        if parameter_len == 0
            || param_shapes != [parameter_shape.clone()]
            || param_scales != [formal_param_scale]
            || params.dims() != parameter_shape.as_slice()
            || params.scale() != Some(formal_param_scale)
            || formal_param_scale != spec.scales.s_w_hidden[0][0]
        {
            return Err("formal GAT parameter tensor shape/scale metadata is not canonical".into());
        }
        let output_shape = vec![graph.n, spec.model.classes];
        if !is_exact_single_shape(output_shapes, &output_shape) || output_scales != [spec.scales.s_out] {
            return Err(format!("formal GAT output metadata {:?}/{:?} does not match {:?}/{}", output_shapes, output_scales, output_shape, spec.scales.s_out));
        }
        let formal_output_shape = output_shapes[0].as_slice();
        // The actual layout repeats this scalar later.  This early probe only
        // needs to prove that the target tensor size is representable; building
        // and expanding a full dummy output here duplicates an O(nodes ×
        // classes) allocation before synthesis.
        checked_shape_len(formal_output_shape)
            .ok_or_else(|| "formal output scalar repeat shape overflow".to_string())?;
        let shape = |label: &str, actual: &[usize], expected: &[usize]| -> Result<(), String> {
            if actual != expected {
                return Err(format!("{} shape {:?} does not match {:?}", label, actual, expected));
            }
            Ok(())
        };
        let edge = [graph.e];
        let groups = graph.destination_groups()?;
        if groups.is_empty() {
            return Err("formal GAT has no destination groups".into());
        }
        let mut current_dim = spec.feature_dim;
        for (layer_idx, layer) in spec.model.hidden_layers.iter().enumerate() {
            shape("hidden input", &[graph.n, current_dim], &[graph.n, layer.d_in])?;
            for (head_idx, _) in layer.heads.iter().enumerate() {
                let edge_channels = [graph.e, layer.d_head];
                let node_channels = [graph.n, layer.d_head];
                shape("hidden projection", &[layer.d_in, layer.d_head], &[layer.d_in, layer.d_head])?;
                shape("hidden projected", &[graph.n, layer.d_head], &[graph.n, layer.d_head])?;
                shape("hidden gathered", &edge, &[graph.e])?;
                shape("hidden attention vectors", &[layer.d_head], &[layer.d_head])?;
                shape("hidden scores", &edge, &[graph.e])?;
                shape("batched max claims", &[graph.n], &[graph.n])?;
                shape("batched max edge repeat", &edge, &[graph.e])?;
                shape("batched max indicators", &edge, &[graph.e])?;
                shape("batched local max prefix", &edge, &[graph.e])?;
                shape("batched exponentials", &edge, &[graph.e])?;
                shape("batched denominator local prefix", &edge, &[graph.e])?;
                shape("batched denominators", &[graph.n], &[graph.n])?;
                shape("batched denominator inverses", &[graph.n], &[graph.n])?;
                shape("batched alpha/source values", &edge_channels, &[graph.e, layer.d_head])?;
                shape("batched exponential/source values", &edge_channels, &[graph.e, layer.d_head])?;
                shape("batched field local prefixes", &edge_channels, &[graph.e, layer.d_head])?;
                shape("batched numerator local prefixes", &edge_channels, &[graph.e, layer.d_head])?;
                shape("batched aggregation numerators", &node_channels, &[graph.n, layer.d_head])?;
                shape("batched signed dividend", &node_channels, &[graph.n, layer.d_head])?;
                shape("batched row divisor", &[graph.n], &[graph.n])?;
                let _ = (layer_idx, head_idx);
            }
            current_dim = layer.heads.len().checked_mul(layer.d_head).ok_or_else(|| "hidden concat shape overflow".to_string())?;
            shape("hidden head concat", &[graph.n, current_dim], &[graph.n, current_dim])?;
        }
        for (head_idx, _) in spec.model.output_heads.iter().enumerate() {
            let edge_channels = [graph.e, spec.model.classes];
            let node_channels = [graph.n, spec.model.classes];
            shape("output projection", &[current_dim, spec.model.classes], &[current_dim, spec.model.classes])?;
            shape("output projected", &[graph.n, spec.model.classes], &[graph.n, spec.model.classes])?;
            shape("output attention vectors", &[spec.model.classes], &[spec.model.classes])?;
            shape("output scores", &edge, &[graph.e])?;
            shape("output batched max claims", &[graph.n], &[graph.n])?;
            shape("output batched max edge repeat", &edge, &[graph.e])?;
            shape("output batched max indicators", &edge, &[graph.e])?;
            shape("output batched local max prefix", &edge, &[graph.e])?;
            shape("output batched exponentials", &edge, &[graph.e])?;
            shape("output batched denominator local prefix", &edge, &[graph.e])?;
            shape("output batched denominators", &[graph.n], &[graph.n])?;
            shape("output batched denominator inverses", &[graph.n], &[graph.n])?;
            shape("output batched alpha/source values", &edge_channels, &[graph.e, spec.model.classes])?;
            shape("output batched exponential/source values", &edge_channels, &[graph.e, spec.model.classes])?;
            shape("output batched field local prefixes", &edge_channels, &[graph.e, spec.model.classes])?;
            shape("output batched numerator local prefixes", &edge_channels, &[graph.e, spec.model.classes])?;
            shape("output batched aggregation numerators", &node_channels, &[graph.n, spec.model.classes])?;
            shape("output batched signed dividend", &node_channels, &[graph.n, spec.model.classes])?;
            shape("output batched row divisor", &[graph.n], &[graph.n])?;
            shape("output alignment dividend", &output_shape, &[graph.n, spec.model.classes])?;
            shape("output alignment scalar repeat", &output_shape, &[graph.n, spec.model.classes])?;
            shape("output head", &[graph.n, spec.model.classes], &[graph.n, spec.model.classes])?;
            let _ = head_idx;
        }
        shape("head average scalar repeat", &output_shape, &[graph.n, spec.model.classes])?;
        shape("head average", &output_shape, &output_shape)?;
        shape("output reshape", &output_shape, &[graph.n, spec.model.classes])?;
        Ok(())
    }

    fn repeated_scalar(scalar: &ValTensor<Fp>, dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        if scalar.len() != 1 {
            return Err(CircuitError::DimMismatch("GAT repeat source must be scalar".into()));
        }
        let value = scalar.get_inner_tensor()?.first()?.get_scalar();
        let len = dims.iter().product::<usize>();
        let mut repeated: ValTensor<Fp> = Tensor::from_vec(vec![value; len], dims)?.into();
        repeated.set_scale(scalar.scale());
        Ok(repeated)
    }

    fn flatten_slice(value: &ValTensor<Fp>, start: usize, end: usize, dims: &[usize],) -> Result<ValTensor<Fp>, CircuitError> {
        match value {
            ValTensor::Value { .. } if !value.dims().is_empty() => {
                let flat = value.get_inner_tensor()?.get_flat_slice(start..end)?;
                let mut slice: ValTensor<Fp> = flat.into();
                slice.set_scale(value.scale());
                slice.reshape(dims)?;
                Ok(slice)
            }
            // Preserve the established behavior for instance-backed or
            // scalar-shaped tensors, which do not have a normal flat value
            // buffer. GAT's gathered internal tensors use the range-copy
            // path above.
            ValTensor::Instance { .. } => {
                let mut flat = value.clone();
                flat.flatten();
                let mut slice = flat.get_slice(&[start..end])?;
                slice.reshape(dims)?;
                Ok(slice)
            }
            ValTensor::Value { .. } => {
                let mut flat = value.clone();
                flat.flatten();
                let mut slice = flat.get_slice(&[start..end])?;
                slice.reshape(dims)?;
                Ok(slice)
            }
        }
    }

    /// Route row-major tensor cells through compile-time public topology.
    /// Cloning `ValType` entries preserves unknown witnesses and existing
    /// assigned cells; downstream native layouts make the required cell copies.
    fn fixed_route_rows(value: &ValTensor<Fp>, rows: &[usize], width: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        if value.dims().len() != 2 || value.dims()[1] != width || width == 0 {
            return Err(CircuitError::DimMismatch("fixed GAT route requires a rank-two tensor with the canonical row width".into()));
        }
        let source_rows = value.dims()[0];
        let output_len = rows.len().checked_mul(width).ok_or(CircuitError::UnsupportedOp)?;
        let source = value.get_inner_tensor()?;
        let mut routed = Vec::with_capacity(output_len);
        for row in rows {
            if *row >= source_rows {
                return Err(CircuitError::DimMismatch("fixed GAT route index is outside the compiled tensor".into()));
            }
            let start = row.checked_mul(width).ok_or(CircuitError::UnsupportedOp)?;
            for column in 0..width {
                routed.push(source.get_flat_index(start.checked_add(column).ok_or(CircuitError::UnsupportedOp)?));
            }
        }
        let mut output: ValTensor<Fp> = Tensor::from_vec(routed, &[rows.len(), width])?.into();
        output.set_scale(value.scale());
        Ok(output)
    }

    fn fixed_route_rows_owned(value: ValTensor<Fp>, rows: &[usize], width: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        if value.dims().len() == 2 && value.dims()[0] == rows.len() && value.dims()[1] == width && rows.iter().enumerate().all(|(expected, actual)| expected == *actual) {
            return Ok(value);
        }
        Self::fixed_route_rows(&value, rows, width)
    }

    fn repeat_tensor_rows(tensor: &ValTensor<Fp>, rows: usize, dims: &[usize]) -> Result<ValTensor<Fp>, CircuitError> {
        if dims.first().copied() != Some(rows) || dims.get(1..) != Some(tensor.dims()) {
            return Err(CircuitError::DimMismatch(format!(
                "GAT repeated row shape {:?} does not match {:?} rows of {:?}",
                dims,
                rows,
                tensor.dims(),
            )));
        }
        let expected = checked_shape_len(dims).ok_or(CircuitError::UnsupportedOp)?;
        let source = tensor.get_inner_tensor()?;
        let repeated_len = source.len().checked_mul(rows).ok_or(CircuitError::UnsupportedOp)?;
        if repeated_len != expected {
            return Err(CircuitError::DimMismatch(format!(
                "GAT repeated row has {} values, expected {} for {:?}",
                repeated_len, expected, dims
            )));
        }
        let mut values = Vec::with_capacity(expected);
        for _ in 0..rows {
            values.extend(source.iter().cloned());
        }
        let mut output: ValTensor<Fp> = Tensor::from_vec(values, dims)?.into();
        output.set_scale(tensor.scale());
        Ok(output)
    }

    /// Interleave hidden-head outputs into the canonical `[node, head*width]`
    /// layout.  A plain concatenation would produce head-major storage and
    /// silently change the next layer's projection relation.
    fn interleave_heads(tensors: &[ValTensor<Fp>], nodes: usize, channels: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        if tensors.is_empty() {
            return Err(CircuitError::UnsupportedOp);
        }
        if tensors.iter().any(|tensor| tensor.len() != nodes * channels || tensor.dims() != [nodes, channels]) {
            return Err(CircuitError::DimMismatch("hidden head output shape does not match graph".into(),));
        }
        let mut vals = Vec::with_capacity(nodes * tensors.len() * channels);
        for node in 0..nodes {
            for tensor in tensors {
                let inner = tensor.get_inner_tensor()?;
                let start = node * channels;
                vals.extend(inner[start..start + channels].iter().cloned());
            }
        }
        Ok(Tensor::from_vec(vals, &[nodes, tensors.len() * channels])?.into())
    }

    fn tile_rows(value: &ValTensor<Fp>, rows: usize, cols: usize,) -> Result<ValTensor<Fp>, CircuitError> {
        if value.len() != rows || value.dims() != [rows] {
            return Err(CircuitError::DimMismatch("cannot tile tensor with incompatible rows".into(),));
        }
        let mut vals = Vec::with_capacity(rows * cols);
        for item in value.get_inner_tensor()?.iter() {
            vals.extend(std::iter::repeat(item.clone()).take(cols));
        }
        Ok(Tensor::from_vec(vals, &[rows, cols])?.into())
    }

    fn canonicalize_edge_channels(
        value: &mut ValTensor<Fp>,
        edge_count: usize,
        channels: usize,
        label: &str,
    ) -> Result<(), CircuitError> {
        let expected_dims = if channels == 1 {
            vec![edge_count]
        } else {
            vec![edge_count, channels]
        };
        if value.dims() == expected_dims {
            return Ok(());
        }
        if channels == 1 && value.len() == edge_count && value.dims() == [edge_count, 1] {
            value.reshape(&expected_dims)?;
            return Ok(());
        }
        Err(CircuitError::DimMismatch(format!(
            "{} shape {:?} with {} values is not canonical for {} edges and {} channels",
            label,
            value.dims(),
            value.len(),
            edge_count,
            channels,
        )))
    }

    fn canonicalize_destination_channels(
        value: &mut ValTensor<Fp>,
        node_count: usize,
        channels: usize,
        label: &str,
    ) -> Result<(), CircuitError> {
        let expected_dims = vec![node_count, channels];
        if value.dims() == expected_dims {
            return Ok(());
        }
        if channels == 1 && value.len() == node_count && value.dims() == [node_count] {
            value.reshape(&expected_dims)?;
            return Ok(());
        }
        Err(CircuitError::DimMismatch(format!(
            "{} shape {:?} with {} values is not canonical for {} destinations and {} channels",
            label,
            value.dims(),
            value.len(),
            node_count,
            channels,
        )))
    }

    fn repeat_by_destination_groups(
        value: &ValTensor<Fp>,
        destination_groups: &[(usize, usize, usize)],
        edge_count: usize,
        channels: usize,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        if value.dims() != [destination_groups.len()] || channels == 0 {
            return Err(CircuitError::DimMismatch(
                "GAT destination-group repeat shape is not canonical".into(),
            ));
        }
        let mut cursor = 0usize;
        let inner = value.get_inner_tensor()?;
        let mut repeated = Vec::with_capacity(edge_count.checked_mul(channels).ok_or(CircuitError::UnsupportedOp)?);
        for (expected_node, (node, start, end)) in destination_groups.iter().copied().enumerate() {
            if node != expected_node || start != cursor || start >= end || end > edge_count {
                return Err(CircuitError::DimMismatch(
                    "GAT destination-group schedule is not a contiguous public partition".into(),
                ));
            }
            let item = inner.get(&[node]).clone();
            for _ in start..end {
                repeated.extend(std::iter::repeat(item.clone()).take(channels));
            }
            cursor = end;
        }
        if cursor != edge_count {
            return Err(CircuitError::DimMismatch(
                "GAT destination-group schedule does not cover every edge".into(),
            ));
        }
        let dims = if channels == 1 {
            vec![edge_count]
        } else {
            vec![edge_count, channels]
        };
        let mut output: ValTensor<Fp> = Tensor::from_vec(repeated, &dims)?.into();
        output.set_scale(value.scale());
        Ok(output)
    }

    fn destination_group_endpoints(
        value: &ValTensor<Fp>,
        destination_groups: &[(usize, usize, usize)],
        edge_count: usize,
        channels: usize,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        let expected_dims = if channels == 1 {
            vec![edge_count]
        } else {
            vec![edge_count, channels]
        };
        if value.dims() != expected_dims || channels == 0 {
            return Err(CircuitError::DimMismatch(
                "GAT segmented-prefix endpoint input shape is not canonical".into(),
            ));
        }
        let inner = value.get_inner_tensor()?;
        let mut endpoints = Vec::with_capacity(destination_groups.len().checked_mul(channels).ok_or(CircuitError::UnsupportedOp)?);
        for (expected_node, (node, start, end)) in destination_groups.iter().copied().enumerate() {
            if node != expected_node || start >= end || end > edge_count {
                return Err(CircuitError::DimMismatch(
                    "GAT destination-group endpoint is outside its public partition".into(),
                ));
            }
            let last_edge = end - 1;
            for channel in 0..channels {
                endpoints.push(inner[last_edge * channels + channel].clone());
            }
        }
        let dims = if channels == 1 {
            vec![destination_groups.len()]
        } else {
            vec![destination_groups.len(), channels]
        };
        let mut output: ValTensor<Fp> = Tensor::from_vec(endpoints, &dims)?.into();
        output.set_scale(value.scale());
        Ok(output)
    }

    /// Assign and constrain one bounded slice of a segmented prefix. If a
    /// chunk starts inside a destination group, `carry` is the previous
    /// chunk's final assigned prefix cell for each channel; the first
    /// recurrence therefore remains linked to the native AssignedCell rather
    /// than to a host-computed value.
    fn segmented_prefix_chunk(
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        values: &ValTensor<Fp>,
        destination_groups: &[(usize, usize, usize)],
        chunk_start: usize,
        edge_count: usize,
        channels: usize,
        carry_cells: &[ValType<Fp>],
        carry_witness: &[Option<Fp>],
        all_unknown: bool,
    ) -> Result<(ValTensor<Fp>, Vec<ValType<Fp>>, Vec<Option<Fp>>), CircuitError> {
        let chunk_edges = values.dims().first().copied().unwrap_or(0);
        let chunk_end = chunk_start
            .checked_add(chunk_edges)
            .ok_or(CircuitError::UnsupportedOp)?;
        let expected_dims = if channels == 1 {
            vec![chunk_edges]
        } else {
            vec![chunk_edges, channels]
        };
        if channels == 0
            || chunk_edges == 0
            || values.dims() != expected_dims
            || chunk_end > edge_count
        {
            return Err(CircuitError::DimMismatch(
                "GAT segmented-prefix chunk shape or range is not canonical".into(),
            ));
        }

        let first_group = destination_groups.partition_point(|(_, _, end)| *end <= chunk_start);
        let mut group_index = first_group;
        let mut accumulators = vec![Fp::ZERO; channels];
        if let Some((node, group_start, group_end)) = destination_groups.get(first_group).copied() {
            if node != first_group || group_start > chunk_start || chunk_start >= group_end {
                return Err(CircuitError::DimMismatch(
                    "GAT segmented-prefix chunk does not begin in its public destination group".into(),
                ));
            }
            if group_start < chunk_start {
                if carry_cells.len() != channels || carry_witness.len() != channels {
                    return Err(CircuitError::DimMismatch(
                        "GAT segmented-prefix chunk is missing its per-channel assigned/witness carry".into(),
                    ));
                }
                if !all_unknown {
                    for (channel, value) in carry_witness.iter().enumerate() {
                        accumulators[channel] = value.ok_or(CircuitError::UnsupportedOp)?;
                    }
                }
            }
        } else {
            return Err(CircuitError::DimMismatch(
                "GAT segmented-prefix chunk starts beyond the destination schedule".into(),
            ));
        }

        let input_inner = values.get_inner_tensor()?;
        let mut prefix_witness = Vec::with_capacity(values.len());
        for row in chunk_start..chunk_end {
            while group_index < destination_groups.len()
                && row >= destination_groups[group_index].2
            {
                group_index += 1;
                accumulators.fill(Fp::ZERO);
            }
            let (node, group_start, group_end) = destination_groups
                .get(group_index)
                .copied()
                .ok_or(CircuitError::UnsupportedOp)?;
            if node != group_index || group_start > row || row >= group_end {
                return Err(CircuitError::DimMismatch(
                    "GAT segmented-prefix chunk crossed an invalid destination boundary".into(),
                ));
            }
            for channel in 0..channels {
                let item = if all_unknown {
                    ValType::Value(Value::unknown())
                } else {
                        let current = input_inner[(row - chunk_start) * channels + channel]
                            .get_felt_eval()
                            .ok_or(CircuitError::UnsupportedOp)?;
                    accumulators[channel] += current;
                    ValType::Value(Value::known(accumulators[channel]))
                };
                prefix_witness.push(item);
            }
        }

        let mut claimed: ValTensor<Fp> = Tensor::from_vec(prefix_witness, &expected_dims)?.into();
        claimed.set_scale(values.scale());
        let claimed = region.assign(&config.custom_gates.inputs[1], &claimed)?;
        let claimed_inner = claimed.get_inner_tensor()?;
        region.increment(claimed.len());
        let mut group_index = first_group;
        let mut previous_values = Vec::with_capacity(values.len());
        for row in chunk_start..chunk_end {
            while group_index < destination_groups.len()
                && row >= destination_groups[group_index].2
            {
                group_index += 1;
            }
            let (_, group_start, _) = destination_groups
                .get(group_index)
                .copied()
                .ok_or(CircuitError::UnsupportedOp)?;
            for channel in 0..channels {
                let previous = if row == group_start {
                    ValType::Constant(Fp::ZERO)
                } else if row == chunk_start {
                    carry_cells
                        .get(channel)
                        .cloned()
                        .ok_or(CircuitError::UnsupportedOp)?
                } else {
                    // `claimed_inner` contains the assigned previous prefix
                    // cell, preserving the recurrence's native equality link.
                    claimed_inner[(row - chunk_start - 1) * channels + channel].clone()
                };
                previous_values.push(previous);
            }
        }
        let mut previous: ValTensor<Fp> = Tensor::from_vec(previous_values, &expected_dims)?.into();
        previous.set_scale(values.scale());
        let recurrence = layouts::pairwise(config, region, &[&previous, values], BaseOp::Add)?;
        layouts::enforce_equality(config, region, &[&recurrence, &claimed])?;

        let last_row = (chunk_edges - 1) * channels;
        let carry_out_cells = (0..channels)
            .map(|channel| claimed_inner[last_row + channel].clone())
            .collect();
        let carry_out_witness = if all_unknown {
            vec![None; channels]
        } else {
            accumulators.into_iter().map(Some).collect()
        };
        Ok((claimed, carry_out_cells, carry_out_witness))
    }

    /// Append only complete destination endpoints represented in this edge
    /// chunk. Chunks and the public destination schedule are both ordered, so
    /// the resulting N x C endpoint vector stays canonical without retaining
    /// the edge-prefix tensor.
    fn append_chunk_endpoints(
        chunk_prefix: &ValTensor<Fp>,
        destination_groups: &[(usize, usize, usize)],
        chunk_start: usize,
        chunk_end: usize,
        channels: usize,
        next_group: &mut usize,
        endpoints: &mut Vec<ValType<Fp>>,
    ) -> Result<(), CircuitError> {
        let inner = chunk_prefix.get_inner_tensor()?;
        while let Some((node, group_start, group_end)) = destination_groups.get(*next_group).copied() {
            if group_end > chunk_end {
                break;
            }
            if node != *next_group || group_start >= group_end || group_end <= chunk_start {
                return Err(CircuitError::DimMismatch(
                    "GAT chunk endpoint does not match its ordered public destination group".into(),
                ));
            }
            let local_row = group_end - 1 - chunk_start;
            let offset = local_row
                .checked_mul(channels)
                .ok_or(CircuitError::UnsupportedOp)?;
            for channel in 0..channels {
                endpoints.push(
                    inner
                        .get_flat_index(offset + channel)
                        .clone(),
                );
            }
            *next_group += 1;
        }
        Ok(())
    }

    /// Constrain one field prefix recurrence per edge/channel, resetting at
    /// every public destination boundary. The public schedule only chooses
    /// predecessor cells; every recurrence is enforced by `pairwise` and an
    /// equality to the claimed prefix advice tensor.
    fn segmented_prefix(
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        values: &ValTensor<Fp>,
        destination_groups: &[(usize, usize, usize)],
        edge_count: usize,
        channels: usize,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        let expected_dims = if channels == 1 {
            vec![edge_count]
        } else {
            vec![edge_count, channels]
        };
        if channels == 0 || values.dims() != expected_dims {
            return Err(CircuitError::DimMismatch(format!(
                "GAT segmented-prefix tensor shape {:?} is not canonical; expected {:?}",
                values.dims(),
                expected_dims,
            )));
        }
        let all_unknown = values.any_unknowns()?;
        let input_inner = values.get_inner_tensor()?;
        let mut prefix_witness = Vec::with_capacity(values.len());
        let mut cursor = 0usize;
        let mut accumulators = vec![Fp::ZERO; channels];
        for (expected_node, (node, start, end)) in destination_groups.iter().copied().enumerate() {
            if node != expected_node || start != cursor || start >= end || end > edge_count {
                return Err(CircuitError::DimMismatch(
                    "GAT segmented-prefix schedule is not a contiguous public partition".into(),
                ));
            }
            accumulators.fill(Fp::ZERO);
            for row in start..end {
                for channel in 0..channels {
                    let item = if all_unknown {
                        ValType::Value(Value::unknown())
                    } else {
                        let current = input_inner[row * channels + channel]
                            .get_felt_eval()
                            .ok_or(CircuitError::UnsupportedOp)?;
                        accumulators[channel] += current;
                        ValType::Value(Value::known(accumulators[channel]))
                    };
                    prefix_witness.push(item);
                }
            }
            cursor = end;
        }
        if cursor != edge_count {
            return Err(CircuitError::DimMismatch(
                "GAT segmented-prefix schedule does not cover every edge".into(),
            ));
        }
        let mut claimed: ValTensor<Fp> = Tensor::from_vec(prefix_witness, &expected_dims)?.into();
        claimed.set_scale(values.scale());
        let claimed = region.assign(&config.custom_gates.inputs[1], &claimed)?;
        let claimed_inner = claimed.get_inner_tensor()?;
        region.increment(claimed.len());

        let mut previous_values = Vec::with_capacity(values.len());
        for (_, start, end) in destination_groups.iter().copied() {
            for row in start..end {
                for channel in 0..channels {
                    if row == start {
                        previous_values.push(ValType::Constant(Fp::ZERO));
                    } else {
                        previous_values.push(claimed_inner[(row - 1) * channels + channel].clone());
                    }
                }
            }
        }
        let mut previous: ValTensor<Fp> = Tensor::from_vec(previous_values, &expected_dims)?.into();
        previous.set_scale(values.scale());
        let recurrence = layouts::pairwise(config, region, &[&previous, values], BaseOp::Add)?;
        layouts::enforce_equality(config, region, &[&recurrence, &claimed])?;
        Ok(claimed)
    }

    fn broadcast_destination_rows(
        value: &ValTensor<Fp>,
        dividend: &ValTensor<Fp>,
        destination_count: usize,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        if value.len() == 1 {
            return Self::repeated_scalar(value, dividend.dims());
        }
        if dividend.dims().len() == 2
            && dividend.dims()[0] == destination_count
            && value.dims() == [destination_count]
        {
            return Self::tile_rows(value, destination_count, dividend.dims()[1]);
        }
        if value.dims() == dividend.dims() {
            return Ok(value.clone());
        }
        Err(CircuitError::DimMismatch(
            "GAT divisor must be scalar, one value per destination row, or match the dividend exactly".into(),
        ))
    }

    fn inverse_exact(&self, config: &BaseConfig<Fp>, region: &mut RegionCtx<Fp>, denominator: &ValTensor<Fp>,) -> Result<ValTensor<Fp>, CircuitError> {
        let inv = if denominator.any_unknowns()? {
            Self::unknown_tensor(denominator.dims())?
        } else {
            let inverse_values = denominator
                .int_evals()?
                .into_iter()
                .map(|d| {
                    if d <= 0 {
                        return Err(CircuitError::UnsupportedOp);
                    }
                    Ok(ValType::Value(Value::known(
                        Option::from(integer_rep_to_felt::<Fp>(d).invert())
                            .ok_or(CircuitError::UnsupportedOp)?,
                    )))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::from_vec(inverse_values, denominator.dims())?.into()
        };
        let inv = region.assign(&config.custom_gates.inputs[1], &inv)?;
        region.increment(inv.len());
        let product = layouts::pairwise(config, region, &[denominator, &inv], BaseOp::Mult)?;
        let one = Self::constant_i128(&vec![1; denominator.len()], denominator.dims())?;
        layouts::enforce_equality(config, region, &[&product, &one])?;
        Ok(inv)
    }

    fn signed_round_div_layout(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        dividend: &ValTensor<Fp>,
        divisor: &ValTensor<Fp>,
        bounds: Option<&BoundFamily>,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        self.signed_round_div_layout_mode(config, region, dividend, divisor, self.rounding, bounds)
    }

    fn signed_round_div_layout_mode(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        dividend: &ValTensor<Fp>,
        divisor: &ValTensor<Fp>,
        rounding: RoundingMode,
        bounds: Option<&BoundFamily>,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        if dividend.dims().is_empty() || dividend.len() == 0 {
            return Err(CircuitError::DimMismatch("GAT dividend must be a nonempty tensor".into()));
        }
        let destination_count = dividend.dims()[0];
        let d_repeated = Self::broadcast_destination_rows(divisor, dividend, destination_count)?;
        if d_repeated.len() != dividend.len() || d_repeated.dims() != dividend.dims() {
            return Err(CircuitError::DimMismatch("GAT divisor broadcast did not exactly match the dividend shape".into()));
        }
        let (q, q_floor, rem, abs, sigma, zero, rounded_up) = if dividend.any_unknowns()? || divisor.any_unknowns()? {
            let dims = dividend.dims();
            (
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
                Self::unknown_tensor(dims)?,
            )
        } else {
            let divisors = d_repeated.int_evals()?;
            if divisors.len() != dividend.len() || divisors.iter().any(|d| *d <= 0) {
                return Err(CircuitError::DimMismatch("GAT divisor rows must be positive and match the dividend".into()));
            }
            let mut qv = Vec::new();
            let mut q_floor_v = Vec::new();
            let mut rv = Vec::new();
            let mut av = Vec::new();
            let mut sv = Vec::new();
            let mut zv = Vec::new();
            let mut rounded_up_v = Vec::new();
            for (x, d) in dividend.int_evals()?.iter().copied().zip(divisors) {
                let signed = super::relation::evaluate_signed_division(x, d, rounding).map_err(|_| CircuitError::UnsupportedOp)?;
                qv.push(signed.quotient);
                q_floor_v.push(signed.quotient_floor);
                rv.push(signed.remainder);
                av.push(signed.abs_dividend);
                sv.push(if signed.sigma { 1 } else { 0 });
                zv.push(if signed.zero { 1 } else { 0 });
                rounded_up_v.push(if signed.rounded_up { 1 } else { 0 });
            }
            (
                Self::value_i128(&qv, dividend.dims())?,
                Self::value_i128(&q_floor_v, dividend.dims())?,
                Self::value_i128(&rv, dividend.dims())?,
                Self::value_i128(&av, dividend.dims())?,
                Self::value_i128(&sv, dividend.dims())?,
                Self::value_i128(&zv, dividend.dims())?,
                Self::value_i128(&rounded_up_v, dividend.dims())?,
            )
        };
        let q = region.assign(&config.custom_gates.inputs[1], &q)?;
        region.increment(q.len());
        let q_floor = region.assign(&config.custom_gates.inputs[1], &q_floor)?;
        region.increment(q_floor.len());
        let rem = region.assign(&config.custom_gates.inputs[1], &rem)?;
        region.increment(rem.len());
        let abs = region.assign(&config.custom_gates.inputs[1], &abs)?;
        region.increment(abs.len());
        let sigma = region.assign(&config.custom_gates.inputs[1], &sigma)?;
        region.increment(sigma.len());
        let zero = region.assign(&config.custom_gates.inputs[1], &zero)?;
        region.increment(zero.len());
        let rounded_up = region.assign(&config.custom_gates.inputs[1], &rounded_up)?;
        region.increment(rounded_up.len());
        let sigma = layouts::boolean_identity(config, region, &[&sigma], true)?;
        let zero = layouts::boolean_identity(config, region, &[&zero], true)?;
        let rounded_up = layouts::boolean_identity(config, region, &[&rounded_up], true)?;
        let two = Self::constant_i128(&vec![2; dividend.len()], dividend.dims())?;
        let two_sigma = layouts::pairwise(config, region, &[&sigma, &two], BaseOp::Mult)?;
        let one = Self::constant_i128(&vec![1; dividend.len()], dividend.dims())?;
        let sign_factor = layouts::pairwise(config, region, &[&one, &two_sigma], BaseOp::Sub)?;
        let signed_abs = layouts::pairwise(config, region, &[&sign_factor, &abs], BaseOp::Mult)?;
        layouts::enforce_equality(config, region, &[dividend, &signed_abs])?;
        let q_abs = layouts::pairwise(config, region, &[&q_floor, &rounded_up], BaseOp::Add)?;
        let signed_q = layouts::pairwise(config, region, &[&sign_factor, &q_abs], BaseOp::Mult)?;
        let q = layouts::enforce_equality(config, region, &[&q, &signed_q])?;
        let max_divisor = bounds.map(|family| family.max_divisor).unwrap_or(self.spec.ledger.max_divisor);
        let max_aux_abs = bounds.map(|family| family.max_aux_abs).unwrap_or(self.spec.ledger.max_aux_abs);
        let max_dividend_abs = bounds.map(|family| family.max_dividend_abs).unwrap_or(self.spec.ledger.max_dividend_abs);
        Self::range_check_bounded(config, region, &[&q_abs], &(0, max_dividend_abs))?;
        Self::range_check_bounded(config, region, &[&q], &(-max_dividend_abs, max_dividend_abs),)?;
        Self::range_check_bounded(config, region, &[&sign_factor], &(-1, 1))?;
        Self::range_check_bounded(config, region, &[divisor], &(1, max_divisor))?;
        let d_inv_scalar = self.inverse_exact(config, region, divisor)?;
        let d_inv = Self::broadcast_destination_rows(&d_inv_scalar, dividend, destination_count)?;
        let d_product = layouts::pairwise(config, region, &[&d_repeated, &d_inv], BaseOp::Mult)?;
        let d_one = Self::constant_i128(&vec![1; dividend.len()], dividend.dims())?;
        layouts::enforce_equality(config, region, &[&d_product, &d_one])?;
        let abs_inv = if abs.any_unknowns()? {
            Self::unknown_tensor(abs.dims())?
        } else {
            let vals = abs
                .int_evals()?
                .into_iter()
                .map(|x| -> Result<ValType<Fp>, CircuitError> {
                    if x == 0 {
                        Ok(ValType::Value(Value::known(Fp::ZERO)))
                    } else {
                        Ok(ValType::Value(Value::known(Option::from(integer_rep_to_felt::<Fp>(x).invert()).ok_or(CircuitError::UnsupportedOp)?,))) } })
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::from_vec(vals, abs.dims())?.into()
        };
        let abs_inv = region.assign(&config.custom_gates.inputs[1], &abs_inv)?;
        region.increment(abs_inv.len());
        let abs_product = layouts::pairwise(config, region, &[&abs, &abs_inv], BaseOp::Mult)?;
        let one_minus_zero = layouts::pairwise(config, region, &[&one, &zero], BaseOp::Sub)?;
        layouts::enforce_equality(config, region, &[&abs_product, &one_minus_zero])?;
        let zero_from_abs = layouts::equals_zero(config, region, &[&abs])?;
        layouts::enforce_equality(config, region, &[&zero, &zero_from_abs])?;
        let zero_abs = layouts::pairwise(config, region, &[&zero, &abs], BaseOp::Mult)?;
        let zero_sigma = layouts::pairwise(config, region, &[&zero, &sigma], BaseOp::Mult)?;
        let zero_inv = layouts::pairwise(config, region, &[&zero, &abs_inv], BaseOp::Mult)?;
        let zero_q = layouts::pairwise(config, region, &[&zero, &q_abs], BaseOp::Mult)?;
        let zeros = Self::constant_i128(&vec![0; dividend.len()], dividend.dims())?;
        layouts::enforce_equality(config, region, &[&zero_abs, &zeros])?;
        layouts::enforce_equality(config, region, &[&zero_sigma, &zeros])?;
        layouts::enforce_equality(config, region, &[&zero_inv, &zeros])?;
        layouts::enforce_equality(config, region, &[&zero_q, &zeros])?;
        let d_minus_one = layouts::pairwise(config, region, &[&d_repeated, &one], BaseOp::Sub)?;
        let delta_vals = if rem.any_unknowns()? {
            Self::unknown_tensor(rem.dims())?
        } else {
            let divisors = d_repeated.int_evals()?;
            let values = rem.int_evals()?;
            if divisors.len() != values.len() {
                return Err(CircuitError::DimMismatch("GAT divisor and remainder lengths differ".into()));
            }
            let deltas = values
                .iter()
                .zip(divisors)
                .map(|(r, d)| match rounding {
                    RoundingMode::TowardZero => d.checked_sub(1).and_then(|v| v.checked_sub(*r)),
                    RoundingMode::NearestTiesAway => d.checked_mul(2).and_then(|v| v.checked_sub(1)).and_then(|v| v.checked_sub(*r)),
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(CircuitError::UnsupportedOp)?;
            Self::value_i128(&deltas, rem.dims())?
        };
        let delta_vals = region.assign(&config.custom_gates.inputs[1], &delta_vals)?;
        region.increment(delta_vals.len());

        match rounding {
            RoundingMode::TowardZero => {
                let dq_floor = layouts::pairwise(config, region, &[&d_repeated, &q_floor], BaseOp::Mult)?;
                let abs_reconstructed = layouts::pairwise(config, region, &[&dq_floor, &rem], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&abs, &abs_reconstructed])?;
                let r_delta = layouts::pairwise(config, region, &[&rem, &delta_vals], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&r_delta, &d_minus_one])?;
            }
            RoundingMode::NearestTiesAway => {
                let two_abs = layouts::pairwise(config, region, &[&two, &abs], BaseOp::Mult)?;
                let lhs = layouts::pairwise(config, region, &[&two_abs, &d_repeated], BaseOp::Add)?;
                let dq_abs = layouts::pairwise(config, region, &[&d_repeated, &q_abs], BaseOp::Mult)?;
                let two_dq_abs = layouts::pairwise(config, region, &[&two, &dq_abs], BaseOp::Mult)?;
                let rhs = layouts::pairwise(config, region, &[&two_dq_abs, &rem], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&lhs, &rhs])?;

                let two_d = layouts::pairwise(config, region, &[&two, &d_repeated], BaseOp::Mult)?;
                let two_d_minus_one = layouts::pairwise(config, region, &[&two_d, &one], BaseOp::Sub)?;
                let r_delta = layouts::pairwise(config, region, &[&rem, &delta_vals], BaseOp::Add)?;
                layouts::enforce_equality(config, region, &[&r_delta, &two_d_minus_one])?;

                let dq_floor = layouts::pairwise(config, region, &[&d_repeated, &q_floor], BaseOp::Mult)?;
                let floor_rem = layouts::pairwise(config, region, &[&abs, &dq_floor], BaseOp::Sub)?;
                let two_floor_rem = layouts::pairwise(config, region, &[&two, &floor_rem], BaseOp::Mult)?;
                let low = layouts::pairwise(config, region, &[&d_minus_one, &two_floor_rem], BaseOp::Sub,)?;
                let high = layouts::pairwise(config, region, &[&two_floor_rem, &d_repeated], BaseOp::Sub)?;
                let one_minus_round = layouts::pairwise(config, region, &[&one, &rounded_up], BaseOp::Sub)?;
                let low_selected = layouts::pairwise(config, region, &[&one_minus_round, &low], BaseOp::Mult)?;
                let high_selected = layouts::pairwise(config, region, &[&rounded_up, &high], BaseOp::Mult)?;
                let selected = layouts::pairwise(config, region, &[&low_selected, &high_selected], BaseOp::Add,)?;
                Self::range_check_bounded(config, region, &[&floor_rem], &(0, max_aux_abs))?;
                Self::range_check_bounded(config, region, &[&selected], &(0, max_aux_abs))?;
            }
        }

        Self::range_check_bounded(config, region, &[&rem], &(0, max_aux_abs))?;
        Self::range_check_bounded(config, region, &[&delta_vals], &(0, max_aux_abs))?;
        Self::range_check_bounded(config, region, &[&q_floor], &(0, max_dividend_abs))?;
        match rounding {
            RoundingMode::TowardZero => {
                let zero_round = Self::constant_i128(&vec![0; dividend.len()], dividend.dims())?;
                layouts::enforce_equality(config, region, &[&rounded_up, &zero_round])?;
            }
            RoundingMode::NearestTiesAway => {}
        }
        Self::range_check_bounded(config, region, &[&abs], &(0, max_dividend_abs))?;
        Ok(q)
    }

    /// Compute raw attention logits once per node, then route the two node
    /// scores through the authenticated public edge topology. The callback
    /// consumes each raw edge chunk before the next one is built, preserving
    /// bounded memory in production and exposing the same raw chunks to the
    /// edge-wise equivalence regression.
    fn node_shared_attention_score_chunks<F>(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        projected: &ValTensor<Fp>,
        attention_src: &ValTensor<Fp>,
        attention_dst: &ValTensor<Fp>,
        channels: usize,
        edge_chunk_rows: usize,
        mut consume_raw_chunk: F,
    ) -> Result<Vec<ValType<Fp>>, CircuitError>
    where
        F: FnMut(
            &BaseConfig<Fp>,
            &mut RegionCtx<Fp>,
            &ValTensor<Fp>,
        ) -> Result<Vec<ValType<Fp>>, CircuitError>,
    {
        let graph = &self.spec.graph;
        let mut score_values = Vec::with_capacity(graph.e);
        let mut source_node_score =
            layouts::einsum(config, region, &[projected, attention_src], "ij,j->i")?;
        let mut destination_node_score =
            layouts::einsum(config, region, &[projected, attention_dst], "ij,j->i")?;
        source_node_score.set_scale(self.spec.scales.s_delta);
        destination_node_score.set_scale(self.spec.scales.s_delta);
        source_node_score.reshape(&[graph.n, 1])?;
        destination_node_score.reshape(&[graph.n, 1])?;
        let mut source_gather_ms = 0.0;
        let mut destination_gather_ms = 0.0;
        let mut score_chunk_count = 0usize;
        println!("EZKL_NATIVE_PHASE START stage=graph_gather direction=source_score nodes={} edges={} channels={} input_values={} node_scores={} max_chunk_cells={}", graph.n, graph.e, channels, projected.len(), source_node_score.len(), MAX_GAT_EDGE_CHUNK_CELLS);
        println!("EZKL_NATIVE_PHASE START stage=graph_gather direction=destination_score nodes={} edges={} channels={} input_values={} node_scores={} max_chunk_cells={}", graph.n, graph.e, channels, projected.len(), destination_node_score.len(), MAX_GAT_EDGE_CHUNK_CELLS);
        for chunk_start in (0..graph.e).step_by(edge_chunk_rows) {
            let chunk_end = chunk_start.saturating_add(edge_chunk_rows).min(graph.e);
            {
                let source_rows = graph.src.get(chunk_start..chunk_end).ok_or_else(|| CircuitError::DimMismatch("compiled GAT source topology does not cover an edge chunk".into()))?;
                let gather_started = std::time::Instant::now();
                let src_score = Self::fixed_route_rows(&source_node_score, source_rows, 1)?;
                source_gather_ms += gather_started.elapsed().as_secs_f64() * 1000.0;

                let destination_rows = graph.dst.get(chunk_start..chunk_end).ok_or_else(|| CircuitError::DimMismatch("compiled GAT destination topology does not cover an edge chunk".into()))?;
                let gather_started = std::time::Instant::now();
                let dst_score = Self::fixed_route_rows(&destination_node_score, destination_rows, 1)?;
                destination_gather_ms += gather_started.elapsed().as_secs_f64() * 1000.0;

                // Preserve the original integer relation: both dot products
                // are formed at node scope, then added on each public edge
                // before the LeakyReLU lookup in aggregate_head.
                let mut score_chunk = layouts::pairwise(config, region, &[&src_score, &dst_score], BaseOp::Add)?;
                score_chunk.set_scale(self.spec.scales.s_delta);
                score_chunk.reshape(&[chunk_end - chunk_start])?;
                score_values.extend(consume_raw_chunk(config, region, &score_chunk)?);
            }
            score_chunk_count += 1;
        }
        println!("EZKL_NATIVE_PHASE DONE stage=graph_gather direction=source_score nodes={} edges={} channels={} input_values={} output_values={} chunks={} elapsed_ms={:.3}", graph.n, graph.e, channels, projected.len(), graph.e, score_chunk_count, source_gather_ms);
        println!("EZKL_NATIVE_PHASE DONE stage=graph_gather direction=destination_score nodes={} edges={} channels={} input_values={} output_values={} chunks={} elapsed_ms={:.3}", graph.n, graph.e, channels, projected.len(), graph.e, score_chunk_count, destination_gather_ms);
        drop(source_node_score);
        drop(destination_node_score);
        Ok(score_values)
    }

    fn aggregate_head(
        &self,
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        projected: &ValTensor<Fp>,
        source: &ValTensor<Fp>,
        dst: &ValTensor<Fp>,
        attention_src: &ValTensor<Fp>,
        attention_dst: &ValTensor<Fp>,
        channels: usize,
        use_elu: bool,
        bounds: &BoundFamily,
        recovery_scale: i32,
        elu_output_scale: i32,
        projected_scale: i32,
        attention_src_scale: i32,
        attention_dst_scale: i32,
        destination_groups: &[(usize, usize, usize)],
        assigned_tables: &mut HashMap<String, layouts::AssignedDynamicLookupTable>,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        let graph = &self.spec.graph;
        Self::require_dims(projected, &[graph.n, channels], "projected tensor")?;
        Self::require_dims(source, &[graph.e], "source edge indices")?;
        Self::require_dims(dst, &[graph.e], "destination edge indices")?;
        Self::require_dims(attention_src, &[channels], "source attention vector")?;
        Self::require_dims(attention_dst, &[channels], "destination attention vector")?;
        Self::require_scale(projected, projected_scale, "projected tensor")?;
        Self::require_scale(attention_src, attention_src_scale, "source attention vector",)?;
        Self::require_scale(attention_dst, attention_dst_scale, "destination attention vector",)?;
        if channels == 0
            || channels > MAX_GAT_EDGE_CHUNK_CELLS
            || destination_groups.len() != graph.n
        {
            return Err(CircuitError::DimMismatch(
                "GAT destination schedule or score shape is not canonical".into(),
            ));
        }
        let max_degree = destination_groups
            .iter()
            .map(|(_, start, end)| end - start)
            .max()
            .ok_or(CircuitError::UnsupportedOp)?;
        // The authenticated ledger validates Dmax*max_exp <= max_sum < Fr/2
        // and max_aggregation_num < Fr/2.  The checked projected bound below
        // therefore bounds each recurrence prefix by max_aggregation_num,
        // with a unique field lift. Keep the explicit endpoint check below;
        // only E-cell prefix checks are redundant.
        Self::constrain_projected_bound(config, region, projected, bounds, max_degree)?;
        let edge_chunk_rows = (MAX_GAT_EDGE_CHUNK_CELLS / channels).max(1);
        let score_values = self.node_shared_attention_score_chunks(
            config,
            region,
            projected,
            attention_src,
            attention_dst,
            channels,
            edge_chunk_rows,
            |config, region, score_chunk| {
                Self::range_check_bounded(
                    config,
                    region,
                    &[score_chunk],
                    &(-bounds.max_score_abs, bounds.max_score_abs),
                )?;
                let mut activated = self.exact_lookup(
                    config,
                    region,
                    score_chunk,
                    &self.spec.tables.lrelu,
                    assigned_tables,
                )?;
                activated.set_scale(self.spec.scales.s_delta);
                Self::range_check_bounded(
                    config,
                    region,
                    &[&activated],
                    &(-bounds.max_table_output_abs, bounds.max_table_output_abs),
                )?;
                Ok(activated.get_inner_tensor()?.iter().cloned().collect())
            },
        )?;
        let mut score: ValTensor<Fp> = Tensor::from_vec(score_values, &[graph.e])?.into();
        score.set_scale(self.spec.scales.s_delta);
        let mut max_claim = if score.any_unknowns()? {
            Self::unknown_tensor(&[graph.n])?
        } else {
            let scores = score.int_evals()?;
            let mut maxima = Vec::with_capacity(graph.n);
            for (expected_node, (node, start, end)) in destination_groups.iter().copied().enumerate() {
                if node != expected_node || start >= end || end > graph.e {
                    return Err(CircuitError::DimMismatch("invalid public destination group for GAT max".into()));
                }
                let maximum = scores[start..end].iter().copied().max().ok_or(CircuitError::UnsupportedOp)?;
                maxima.push(ValType::Value(Value::known(integer_rep_to_felt(maximum))));
            }
            Tensor::from_vec(maxima, &[graph.n])?.into()
        };
        max_claim.set_scale(score.scale());
        let max_claim = region.assign(&config.custom_gates.inputs[1], &max_claim)?;
        region.increment(max_claim.len());
        Self::range_check_bounded(
            config,
            region,
            &[&max_claim],
            &(-bounds.max_table_output_abs, bounds.max_table_output_abs),
        )?;
        let max_repeated = Self::repeat_by_destination_groups(&max_claim, destination_groups, graph.e, 1)?;
        drop(max_claim);
        let mut one_edge = Self::constant_i128(&vec![1; graph.e], &[graph.e])?;
        one_edge.set_scale(score.scale());
        let mut max_minus_one = layouts::pairwise(config, region, &[&max_repeated, &one_edge], BaseOp::Sub)?;
        drop(one_edge);
        max_minus_one.set_scale(score.scale());
        max_minus_one.reshape(&[graph.e])?;
        let mut max_test_input = layouts::pairwise(config, region, &[&score, &max_minus_one], BaseOp::Sub)?;
        drop(max_minus_one);
        max_test_input.set_scale(score.scale());
        max_test_input.reshape(&[graph.e])?;
        let mut max_indicator = self.exact_lookup(
            config,
            region,
            &max_test_input,
            &self.spec.tables.max_relu,
            assigned_tables,
        )?;
        drop(max_test_input);
        Self::require_scale(&max_indicator, score.scale(), "max-ReLU indicator")?;
        max_indicator.reshape(&[graph.e])?;
        let mut max_indicator = layouts::boolean_identity(config, region, &[&max_indicator], true)?;
        Self::require_scale(&max_indicator, score.scale(), "Boolean max-ReLU indicator")?;
        max_indicator.reshape(&[graph.e])?;
        let max_indicator_prefix = Self::segmented_prefix(
            config,
            region,
            &max_indicator,
            destination_groups,
            graph.e,
            1,
        )?;
        drop(max_indicator);
        let public_max_degree = i128::try_from(max_degree).map_err(|_| CircuitError::UnsupportedOp)?;
        // Each indicator is constrained to {0,1}; the public group reset and
        // recurrence therefore imply every local prefix is in [0,Dmax]. Since
        // Dmax is bounded by the usize-sized public edge count, these prefixes
        // have a unique field lift. Range-check only the N endpoint claims.
        let max_indicator_sum = Self::destination_group_endpoints(
            &max_indicator_prefix,
            destination_groups,
            graph.e,
            1,
        )?;
        drop(max_indicator_prefix);
        Self::range_check_bounded(
            config,
            region,
            &[&max_indicator_sum],
            &(0, public_max_degree),
        )?;
        let mut one_node = Self::constant_i128(&vec![1; graph.n], &[graph.n])?;
        one_node.set_scale(score.scale());
        let max_deficit = layouts::pairwise(config, region, &[&one_node, &max_indicator_sum], BaseOp::Sub)?;
        drop(one_node);
        drop(max_indicator_sum);
        let mut max_deficit = max_deficit;
        max_deficit.set_scale(score.scale());
        let max_deficit_relu = self.exact_lookup(
            config,
            region,
            &max_deficit,
            &self.spec.tables.max_relu,
            assigned_tables,
        )?;
        drop(max_deficit);
        Self::require_scale(&max_deficit_relu, score.scale(), "max-ReLU deficit output")?;
        let mut zero_node = Self::constant_i128(&vec![0; graph.n], &[graph.n])?;
        zero_node.set_scale(score.scale());
        layouts::enforce_equality(config, region, &[&max_deficit_relu, &zero_node])?;
        drop(max_deficit_relu);
        drop(zero_node);

        let mut delta = layouts::pairwise(config, region, &[&max_repeated, &score], BaseOp::Sub)?;
        delta.reshape(&[graph.e])?;
        delta.set_scale(self.spec.scales.s_delta);
        Self::range_check_bounded(config, region, &[&delta], &(0, bounds.max_delta))?;
        drop(max_repeated);
        drop(score);
        let mut exp = self.exact_lookup(config, region, &delta, &self.spec.tables.exp, assigned_tables)?;
        drop(delta);
        exp.set_scale(self.spec.scales.s_exp);
        Self::range_check_bounded(config, region, &[&exp], &(1, bounds.max_exp))?;

        let denominator_prefix = Self::segmented_prefix(
            config,
            region,
            &exp,
            destination_groups,
            graph.e,
            1,
        )?;
        // Each term is already constrained to [1, max_exp], and the ordered
        // per-graph recurrence resets at zero. Since the ledger validates
        // Dmax*max_exp <= max_sum < Fr/2, every interior prefix has that
        // canonical integer lift; only the N destination endpoints need an
        // explicit range check below.
        let mut denominator = Self::destination_group_endpoints(
            &denominator_prefix,
            destination_groups,
            graph.e,
            1,
        )?;
        drop(denominator_prefix);
        denominator.set_scale(self.spec.scales.s_exp);
        Self::range_check_bounded(config, region, &[&denominator], &(1, bounds.max_sum))?;
        let inverse = self.inverse_exact(config, region, &denominator)?;
        let inverse_edges = Self::repeat_by_destination_groups(&inverse, destination_groups, graph.e, 1)?;
        let mut alpha = layouts::pairwise(config, region, &[&exp, &inverse_edges], BaseOp::Mult)?;
        drop(inverse_edges);
        alpha.set_scale(self.spec.scales.s_exp);

        let projected_unknown = projected.any_unknowns()?;
        let field_prefix_unknown = projected_unknown || alpha.any_unknowns()?;
        let numerator_prefix_unknown = projected_unknown || exp.any_unknowns()?;
        let endpoint_capacity = graph.n.checked_mul(channels).ok_or(CircuitError::UnsupportedOp)?;
        let mut field_endpoint_values = Vec::with_capacity(endpoint_capacity);
        let mut numerator_endpoint_values = Vec::with_capacity(endpoint_capacity);
        let mut field_carry_cells = Vec::new();
        let mut field_carry_witness = Vec::new();
        let mut numerator_carry_cells = Vec::new();
        let mut numerator_carry_witness = Vec::new();
        let mut field_next_group = 0usize;
        let mut numerator_next_group = 0usize;
        let mut field_scale = None;
        for chunk_start in (0..graph.e).step_by(edge_chunk_rows) {
            let chunk_end = chunk_start.saturating_add(edge_chunk_rows).min(graph.e);
            let chunk_rows = chunk_end - chunk_start;
            {
                let source_rows = graph.src.get(chunk_start..chunk_end).ok_or_else(|| CircuitError::DimMismatch("compiled GAT source topology does not cover an edge chunk".into()))?;
                let mut src_values = Self::fixed_route_rows(projected, source_rows, channels)?;
                src_values.set_scale(projected_scale);

                let alpha_chunk = Self::flatten_slice(&alpha, chunk_start, chunk_end, &[chunk_rows])?;
                let alpha_rows = Self::tile_rows(&alpha_chunk, chunk_rows, channels)?;
                let mut field_product = layouts::pairwise(config, region, &[&alpha_rows, &src_values], BaseOp::Mult)?;
                Self::canonicalize_edge_channels(
                    &mut field_product,
                    chunk_rows,
                    channels,
                    "GAT field-product chunk",
                )?;
                field_scale.get_or_insert(field_product.scale());
                let (field_prefix, next_field_carry_cells, next_field_carry_witness) = Self::segmented_prefix_chunk(
                    config,
                    region,
                    &field_product,
                    destination_groups,
                    chunk_start,
                    graph.e,
                    channels,
                    &field_carry_cells,
                    &field_carry_witness,
                    field_prefix_unknown,
                )?;
                Self::append_chunk_endpoints(
                    &field_prefix,
                    destination_groups,
                    chunk_start,
                    chunk_end,
                    channels,
                    &mut field_next_group,
                    &mut field_endpoint_values,
                )?;
                field_carry_cells = next_field_carry_cells;
                field_carry_witness = next_field_carry_witness;
                drop(field_prefix);
                drop(field_product);
                drop(alpha_rows);

                let exp_chunk = Self::flatten_slice(&exp, chunk_start, chunk_end, &[chunk_rows])?;
                let exp_rows = Self::tile_rows(&exp_chunk, chunk_rows, channels)?;
                let mut numerator_product = layouts::pairwise(config, region, &[&exp_rows, &src_values], BaseOp::Mult)?;
                Self::canonicalize_edge_channels(
                    &mut numerator_product,
                    chunk_rows,
                    channels,
                    "GAT numerator-product chunk",
                )?;
                let (numerator_prefix, next_numerator_carry_cells, next_numerator_carry_witness) = Self::segmented_prefix_chunk(
                    config,
                    region,
                    &numerator_product,
                    destination_groups,
                    chunk_start,
                    graph.e,
                    channels,
                    &numerator_carry_cells,
                    &numerator_carry_witness,
                    numerator_prefix_unknown,
                )?;
                Self::append_chunk_endpoints(
                    &numerator_prefix,
                    destination_groups,
                    chunk_start,
                    chunk_end,
                    channels,
                    &mut numerator_next_group,
                    &mut numerator_endpoint_values,
                )?;
                numerator_carry_cells = next_numerator_carry_cells;
                numerator_carry_witness = next_numerator_carry_witness;
                drop(numerator_prefix);
                drop(numerator_product);
                drop(exp_rows);
                drop(src_values);
            }
        }
        drop(alpha);
        drop(exp);
        drop(field_carry_cells);
        drop(field_carry_witness);
        drop(numerator_carry_cells);
        drop(numerator_carry_witness);
        if field_next_group != destination_groups.len()
            || numerator_next_group != destination_groups.len()
            || field_endpoint_values.len() != endpoint_capacity
            || numerator_endpoint_values.len() != endpoint_capacity
        {
            return Err(CircuitError::DimMismatch(
                "GAT chunked aggregation did not emit every canonical destination endpoint".into(),
            ));
        }
        let endpoint_dims = if channels == 1 {
            vec![graph.n]
        } else {
            vec![graph.n, channels]
        };
        let mut field_sum: ValTensor<Fp> =
            Tensor::from_vec(field_endpoint_values, &endpoint_dims)?.into();
        Self::canonicalize_destination_channels(
            &mut field_sum,
            graph.n,
            channels,
            "GAT field endpoints",
        )?;
        field_sum.set_scale(field_scale.ok_or(CircuitError::UnsupportedOp)?);
        let mut numerator: ValTensor<Fp> =
            Tensor::from_vec(numerator_endpoint_values, &[graph.n, channels])?.into();
        Self::canonicalize_destination_channels(&mut numerator, graph.n, channels, "GAT numerator endpoints")?;
        let max_numerator = bounds.max_aggregation_num;
        let min_numerator = max_numerator.checked_neg().ok_or(CircuitError::UnsupportedOp)?;
        Self::range_check_bounded(
            config,
            region,
            &[&numerator],
            &(min_numerator, max_numerator),
        )?;
        numerator.set_scale(
            self.spec
                .scales
                .s_exp
                .checked_add(projected_scale)
                .ok_or(CircuitError::UnsupportedOp)?,
        );
        let inverse_rows = Self::tile_rows(&inverse, graph.n, channels)?;
        drop(inverse);
        let field_from_numerator = layouts::pairwise(config, region, &[&numerator, &inverse_rows], BaseOp::Mult)?;
        layouts::enforce_equality(config, region, &[&field_sum, &field_from_numerator])?;
        drop(field_sum);
        drop(field_from_numerator);
        drop(inverse_rows);

        let recovery_multiplier = scale_exponent_to_multiplier(recovery_scale)
            .map_err(|_| CircuitError::UnsupportedOp)
            .and_then(|value| i128::try_from(value).map_err(|_| CircuitError::UnsupportedOp))?;
        let denominator_multiplier = scale_exponent_to_multiplier(projected_scale)
            .map_err(|_| CircuitError::UnsupportedOp)
            .and_then(|value| i128::try_from(value).map_err(|_| CircuitError::UnsupportedOp))?;
        let numerator_scale = Self::constant_i128(&vec![recovery_multiplier; numerator.len()], numerator.dims())?;
        let mut scaled_numerator = layouts::pairwise(config, region, &[&numerator, &numerator_scale], BaseOp::Mult)?;
        let denominator_scale = Self::constant_i128(&vec![denominator_multiplier; graph.n], &[graph.n])?;
        let mut scaled_denominator = layouts::pairwise(config, region, &[&denominator, &denominator_scale], BaseOp::Mult)?;
        drop(numerator_scale);
        drop(denominator_scale);
        drop(numerator);
        drop(denominator);
        let divisor_scale = self.spec.scales.s_exp.checked_add(projected_scale).ok_or(CircuitError::UnsupportedOp)?;
        let dividend_scale = divisor_scale.checked_add(recovery_scale).ok_or(CircuitError::UnsupportedOp)?;
        scaled_numerator.set_scale(dividend_scale);
        scaled_denominator.set_scale(divisor_scale);
        Self::require_dims(&scaled_numerator, &[graph.n, channels], "batched GAT signed-dividend")?;
        Self::require_dims(&scaled_denominator, &[graph.n], "batched GAT destination divisor")?;
        Self::require_scale(&scaled_numerator, dividend_scale, "batched GAT signed-dividend")?;
        Self::require_scale(&scaled_denominator, divisor_scale, "batched GAT destination divisor")?;
        let mut q = self.signed_round_div_layout(
            config,
            region,
            &scaled_numerator,
            &scaled_denominator,
            Some(bounds),
        )?;
        drop(scaled_numerator);
        drop(scaled_denominator);
        if use_elu {
            q.set_scale(recovery_scale);
            let mut elu = self.exact_lookup(config, region, &q, &self.spec.tables.elu, assigned_tables)?;
            drop(q);
            elu.set_scale(elu_output_scale);
            Ok(elu)
        } else {
            q.set_scale(self.spec.scales.s_out_pre);
            Ok(q)
        }
    }
}

impl Op<Fp> for GatOp {
    fn as_string(&self) -> String {
        format!("CanonicalGAT({})", self.spec.model.model_arch_id)
    }

    fn layout(&self, config: &mut BaseConfig<Fp>, region: &mut RegionCtx<Fp>, values: &[&ValTensor<Fp>],) -> Result<Option<ValTensor<Fp>>, CircuitError> {
        // The adapter's geometry planner and actual live synthesis both walk
        // edge/head loops in source order.  The legacy generic dummy loop
        // executes closures concurrently and merges cursor deltas, which can
        // change duplication offsets at column boundaries.  Keep its default
        // behavior for all other operations and scope ordered accounting to
        // this GAT layout only.
        let previous_ordered_dummy_loop = if region.is_dummy() {
            Some(region.set_ordered_dummy_loop(true))
        } else {
            None
        };
        let result = (|| -> Result<Option<ValTensor<Fp>>, CircuitError> {
        if values.len() < FIRST_WEIGHT_INPUT {
            return Err(CircuitError::DimMismatch("GAT requires six public inputs".into(),));
        }
        self.spec.validate_for_compile().map_err(|e| CircuitError::DimMismatch(format!("invalid canonical GAT: {}", e)))?;
        let mut assigned_tables = HashMap::new();
        let graph = &self.spec.graph;
        let destination_groups = graph.destination_groups_validated();
        for (value, dims, label) in [
            (values[FEATURE_INPUT], &[graph.n_total, self.spec.feature_dim][..], "feature input"),
            (values[NODE_ID_INPUT], &[graph.n][..], "node ids"),
            (values[SRC_INPUT], &[graph.e][..], "source edges"),
            (values[DST_INPUT], &[graph.e][..], "destination edges"),
            (values[NODE_PTR_INPUT], &[graph.g_batch + 1][..], "node pointers"),
            (values[EDGE_PTR_INPUT], &[graph.g_batch + 1][..], "edge pointers"),
        ] {
            Self::require_dims(value, dims, label)?;
        }
        Self::require_scale(values[FEATURE_INPUT], self.spec.scales.s_input, "feature input",)?;
        let mut features = layouts::identity(config, region, &[values[FEATURE_INPUT]])?;
        features.set_scale(self.spec.scales.s_input);
        let _node_ids = self.bind_topology(config, region, values[NODE_ID_INPUT], NODE_ID_INPUT)?;
        let mut src = self.bind_topology(config, region, values[SRC_INPUT], SRC_INPUT)?;
        let mut dst = self.bind_topology(config, region, values[DST_INPUT], DST_INPUT)?;
        src.set_scale(0);
        dst.set_scale(0);
        let _node_ptr = self.bind_topology(config, region, values[NODE_PTR_INPUT], NODE_PTR_INPUT)?;
        let _edge_ptr = self.bind_topology(config, region, values[EDGE_PTR_INPUT], EDGE_PTR_INPUT)?;
        let weights = self.weight_tensors(values)?;
        let mut weight_idx = 0;
        println!(
            "EZKL_NATIVE_PHASE START stage=graph_gather direction=feature model_arch_id={} nodes={} feature_rows={} feature_dim={} input_values={}",
            self.spec.model.model_arch_id,
            graph.n,
            graph.n_total,
            self.spec.feature_dim,
            features.len(),
        );
        let feature_gather_started = std::time::Instant::now();
        let mut current = Self::fixed_route_rows_owned(features, &graph.node_ids, self.spec.feature_dim)?;
        println!("EZKL_NATIVE_WITNESS stage=feature_gather unknown={}", current.any_unknowns()?);
        println!(
            "EZKL_NATIVE_PHASE DONE stage=graph_gather direction=feature model_arch_id={} output_values={} elapsed_ms={:.3}",
            self.spec.model.model_arch_id,
            current.len(),
            feature_gather_started.elapsed().as_secs_f64() * 1000.0,
        );
        Self::require_dims(&current, &[graph.n, self.spec.feature_dim], "gathered features")?;
        current.set_scale(self.spec.scales.s_h[0]);
        let mut current_dim = self.spec.feature_dim;
        for (layer_idx, layer) in self.spec.model.hidden_layers.iter().enumerate() {
            let mut heads = Vec::new();
            for (head_idx, _head) in layer.heads.iter().enumerate() {
                let bounds = self.bound_family(layer_idx, head_idx, false)?;
                let projection = layouts::identity(config, region, &[&weights[weight_idx]])
                    .map_err(|error| {
                        log::error!(
                            "GAT hidden projection input assignment failed: layer={} head={} linear_coord={} error={:?}",
                            layer_idx,
                            head_idx,
                            region.linear_coord(),
                            error
                        );
                        error
                    })?;
                let attention_src = layouts::identity(config, region, &[&weights[weight_idx + 1]])?;
                let attention_dst = layouts::identity(config, region, &[&weights[weight_idx + 2]])?;
                weight_idx += 3;
                Self::require_scale(&projection, self.spec.scales.s_w_hidden[layer_idx][head_idx], "hidden projection weight",)?;
                Self::require_scale(&attention_src, self.spec.scales.s_a_src_hidden[layer_idx][head_idx], "hidden source attention weight",)?;
                Self::require_scale(&attention_dst, self.spec.scales.s_a_dst_hidden[layer_idx][head_idx], "hidden destination attention weight",)?;
                if projection.dims() != [current_dim, layer.d_head] {
                    return Err(CircuitError::DimMismatch("GAT projection shape".into()));
                }
                let mut projected = layouts::einsum(config, region, &[&current, &projection], "ij,jk->ik")
                    .map_err(|error| {
                        log::error!(
                            "GAT hidden projection einsum failed: layer={} head={} input_dims={:?} weight_dims={:?} linear_coord={} error={:?}",
                            layer_idx,
                            head_idx,
                            current.dims(),
                            projection.dims(),
                            region.linear_coord(),
                            error
                        );
                        error
                    })?;
                println!(
                    "EZKL_NATIVE_WITNESS stage=hidden_projection layer={} head={} input_unknown={} weight_unknown={} output_unknown={}",
                    layer_idx,
                    head_idx,
                    current.any_unknowns()?,
                    projection.any_unknowns()?,
                    projected.any_unknowns()?
                );
                projected.set_scale(self.spec.scales.s_h_prime[layer_idx][head_idx]);
                let aggregated = self.aggregate_head(
                    config,
                    region,
                    &projected,
                    &src,
                    &dst,
                    &attention_src,
                    &attention_dst,
                    layer.d_head,
                    true,
                    bounds,
                    self.spec.scales.s_agg_pre[layer_idx],
                    self.spec.scales.s_agg[layer_idx],
                    self.spec.scales.s_h_prime[layer_idx][head_idx],
                    self.spec.scales.s_a_src_hidden[layer_idx][head_idx],
                    self.spec.scales.s_a_dst_hidden[layer_idx][head_idx],
                    &destination_groups,
                    &mut assigned_tables,
                ).map_err(|error| {
                    log::error!(
                        "GAT aggregate_head failed: phase=hidden layer={} head={} linear_coord={} error={:?}",
                        layer_idx,
                        head_idx,
                        region.linear_coord(),
                        error
                    );
                    error
                })?;
                println!(
                    "EZKL_NATIVE_WITNESS stage=hidden_aggregate layer={} head={} output_unknown={}",
                    layer_idx,
                    head_idx,
                    aggregated.any_unknowns()?
                );
                heads.push(aggregated);
            }
            current = Self::interleave_heads(&heads, self.spec.graph.n, layer.d_head).map_err(|error| {
                log::error!(
                    "GAT hidden head interleave failed: layer={} heads={} channels={} linear_coord={} error={:?}",
                    layer_idx,
                    heads.len(),
                    layer.d_head,
                    region.linear_coord(),
                    error
                );
                error
            })?;
            println!(
                "EZKL_NATIVE_WITNESS stage=hidden_interleave layer={} output_unknown={}",
                layer_idx,
                current.any_unknowns()?
            );
            current.set_scale(self.spec.scales.s_h[layer_idx + 1]);
            current_dim = layer.heads.len() * layer.d_head;
        }
        let mut output_heads = Vec::new();
        for (head_idx, _head) in self.spec.model.output_heads.iter().enumerate() {
            let bounds = self.bound_family(usize::MAX, head_idx, true)?;
            let projection = layouts::identity(config, region, &[&weights[weight_idx]])
                .map_err(|error| {
                    log::error!(
                        "GAT output projection input assignment failed: head={} linear_coord={} error={:?}",
                        head_idx,
                        region.linear_coord(),
                        error
                    );
                    error
                })?;
            let attention_src = layouts::identity(config, region, &[&weights[weight_idx + 1]])?;
            let attention_dst = layouts::identity(config, region, &[&weights[weight_idx + 2]])?;
            let bias = layouts::identity(config, region, &[&weights[weight_idx + 3]])?;
            weight_idx += 4;
            Self::require_scale(&projection, self.spec.scales.s_w_output[head_idx], "output projection weight",)?;
            Self::require_scale(&attention_src, self.spec.scales.s_a_src_output[head_idx], "output source attention weight",)?;
            Self::require_scale(&attention_dst, self.spec.scales.s_a_dst_output[head_idx], "output destination attention weight",)?;
            Self::require_scale(&bias, self.spec.scales.s_b_output[head_idx], "output bias")?;
            let mut projected = layouts::einsum(config, region, &[&current, &projection], "ij,jk->ik")
                .map_err(|error| {
                    log::error!(
                        "GAT output projection einsum failed: head={} input_dims={:?} weight_dims={:?} linear_coord={} error={:?}",
                        head_idx,
                        current.dims(),
                        projection.dims(),
                        region.linear_coord(),
                        error
                    );
                    error
                    })?;
            println!(
                "EZKL_NATIVE_WITNESS stage=output_projection head={} input_unknown={} weight_unknown={} output_unknown={}",
                head_idx,
                current.any_unknowns()?,
                projection.any_unknowns()?,
                projected.any_unknowns()?
            );
            projected.set_scale(self.spec.scales.s_y_prime[head_idx]);
            let aggregated = self.aggregate_head(
                config,
                region,
                &projected,
                &src,
                &dst,
                &attention_src,
                &attention_dst,
                self.spec.model.classes,
                false,
                bounds,
                self.spec.scales.s_out_pre,
                self.spec.scales.s_out_pre,
                self.spec.scales.s_y_prime[head_idx],
                self.spec.scales.s_a_src_output[head_idx],
                self.spec.scales.s_a_dst_output[head_idx],
                &destination_groups,
                &mut assigned_tables,
            ).map_err(|error| {
                log::error!(
                    "GAT aggregate_head failed: phase=output head={} linear_coord={} error={:?}",
                    head_idx,
                    region.linear_coord(),
                    error
                );
                error
            })?;
            let out_head_multiplier = scale_exponent_to_multiplier(self.spec.scales.s_out_head).map_err(|_| CircuitError::UnsupportedOp)?;
            let out_head_multiplier = i128::try_from(out_head_multiplier).map_err(|_| CircuitError::UnsupportedOp)?;
            let out_pre_multiplier = scale_exponent_to_multiplier(self.spec.scales.s_out_pre).map_err(|_| CircuitError::UnsupportedOp)?;
            let out_pre_multiplier = i128::try_from(out_pre_multiplier).map_err(|_| CircuitError::UnsupportedOp)?;
            let scale_const = Self::constant_i128(&vec![out_head_multiplier; aggregated.len()], aggregated.dims(),)?;
            let aligned_dividend = layouts::pairwise(config, region, &[&aggregated, &scale_const], BaseOp::Mult)?;
            let divisor = Self::constant_i128(&[out_pre_multiplier], &[1])?;
            let mut aligned = self
                .signed_round_div_layout(config, region, &aligned_dividend, &divisor, Some(bounds))
                .map_err(|error| {
                    log::error!(
                        "GAT output head scale recovery failed: head={} linear_coord={} error={:?}",
                        head_idx,
                        region.linear_coord(),
                        error
                    );
                    error
                })?;
            aligned.set_scale(self.spec.scales.s_out_head);
            let bias_repeated = Self::repeat_tensor_rows(
                &bias,
                self.spec.graph.n,
                &[self.spec.graph.n, self.spec.model.classes],
            )?;
            Self::require_scale(&bias_repeated, self.spec.scales.s_b_output[head_idx], "repeated output bias")?;
            let mut head_output = layouts::pairwise(config, region, &[&aligned, &bias_repeated], BaseOp::Add)?;
            head_output.set_scale(self.spec.scales.s_out_head);
            output_heads.push(head_output);
        }
        let mut output = output_heads.first().cloned().ok_or(CircuitError::UnsupportedOp)?;
        for head in output_heads.iter().skip(1) {
            output = layouts::pairwise(config, region, &[&output, head], BaseOp::Add)?;
            output.set_scale(self.spec.scales.s_out_head);
        }
        output.set_scale(self.spec.scales.s_out_head);
        let count = Self::constant_i128(&[i128::try_from(self.spec.model.output_heads.len()).map_err(|_| CircuitError::UnsupportedOp)?], &[1],)?;
        output = self.signed_round_div_layout_mode(config, region, &output, &count, RoundingMode::NearestTiesAway, None,)?;
        output.set_scale(self.spec.scales.s_out);
        Self::require_dims(&output, &[graph.n, self.spec.model.classes], "final GAT output")?;
        output.reshape(&[graph.n, self.spec.model.classes])?;
        Ok(Some(output))
        })();
        if let Some(previous) = previous_ordered_dummy_loop {
            region.set_ordered_dummy_loop(previous);
        }
        result
    }

    fn out_scale(&self, _in_scales: Vec<crate::Scale>) -> Result<crate::Scale, CircuitError> {
        Ok(self.spec.scales.s_out)
    }

    fn clone_dyn(&self) -> Box<dyn Op<Fp>> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
#[path = "../tests/bounded_range_tests.rs"]
mod bounded_range_tests;

/// Shared bounded-range and chunked-prefix MockProver regressions for the
/// one root-crate formal integration test.
#[cfg(test)]
pub(crate) fn run_gat_bounded_mock_regressions() {
    bounded_range_tests::run_aggregation_budget_bounds_nonzero_projection_and_zero_projection_families();
    segmented_batch_tests::run_chunked_prefix_mockprover_checks_carries_endpoints_and_rounding();
}

/// Exercise nonzero node-shared attention against the edge-wise integer
/// relation on the existing tiny two-graph fixture.
#[cfg(test)]
pub(crate) fn run_nonzero_attention_route_equivalence() {
    segmented_batch_tests::run_nonzero_attention_route_equivalence();
}

#[cfg(test)]
mod flat_slice_tests {
    use super::*;

    #[test]
    fn pre_synthesis_shape_probe_checks_rank_zero_and_overflow_without_allocating() {
        assert_eq!(checked_shape_len(&[2, 3]), Some(6));
        assert_eq!(checked_shape_len(&[2, 0]), Some(0));
        assert_eq!(checked_shape_len(&[usize::MAX, 2]), None);
        assert!(is_exact_single_shape(&[vec![2, 3]], &[2, 3]));
        assert!(!is_exact_single_shape(&[vec![2]], &[2, 3]));
        assert!(!is_exact_single_shape(&[vec![2, 0]], &[2, 3]));
    }

    #[test]
    fn flat_slice_matches_existing_multidimensional_range_and_keeps_values() {
        let source_values = (0..24)
            .map(|value| {
                if value % 2 == 0 {
                    ValType::Constant(Fp::from(value as u64))
                } else {
                    ValType::Value(Value::known(Fp::from(value as u64)))
                }
            })
            .collect::<Vec<_>>();
        let mut source: ValTensor<Fp> = Tensor::new(Some(&source_values), &[2, 3, 4])
            .unwrap()
            .into();
        source.set_scale(13);

        let mut old_flatten_then_slice = source.clone();
        old_flatten_then_slice.flatten();
        let mut expected = old_flatten_then_slice.get_slice(&[3..11]).unwrap();
        expected.reshape(&[4, 2]).unwrap();

        let actual = GatOp::flatten_slice(&source, 3, 11, &[4, 2]).unwrap();
        assert_eq!(actual.dims(), &[4, 2]);
        assert_eq!(actual.scale(), 13);
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual
            .get_inner_tensor()
            .unwrap()
            .iter()
            .zip(expected.get_inner_tensor().unwrap().iter())
        {
            assert_eq!(actual.get_felt_eval(), expected.get_felt_eval());
            assert_eq!(
                std::mem::discriminant(actual),
                std::mem::discriminant(expected)
            );
        }
    }

    #[test]
    fn flat_slice_rejects_invalid_bounds_and_shape() {
        let source: ValTensor<Fp> = Tensor::new(
            Some(&[ValType::Constant(Fp::from(1)), ValType::Constant(Fp::from(2))]),
            &[2],
        )
        .unwrap()
        .into();
        assert!(GatOp::flatten_slice(&source, 3, 2, &[0]).is_err());
        assert!(GatOp::flatten_slice(&source, 0, 3, &[3]).is_err());
        assert!(GatOp::flatten_slice(&source, 0, 2, &[3]).is_err());
    }
}

#[cfg(test)]
mod segmented_batch_tests {
    use super::*;
    use crate::circuit::region::{RegionCtx, RegionSettings};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    fn dummy_layout() -> (BaseConfig<Fp>, RegionCtx<'static, Fp>) {
        (
            BaseConfig::dummy(12, 2),
            RegionCtx::new_dummy(0, 2, RegionSettings::all_true(256, 4)),
        )
    }

    fn dummy_layout_for_gat_tables(
        spec: &CanonicalGat,
    ) -> (BaseConfig<Fp>, RegionCtx<'static, Fp>) {
        let max_table_rows = [
            spec.tables.lrelu.rows.len(),
            spec.tables.exp.rows.len(),
            spec.tables.elu.rows.len(),
            spec.tables.max_relu.rows.len(),
        ]
        .into_iter()
        .max()
        .expect("GAT has authenticated exact tables");
        let table_column_rows = max_table_rows
            .checked_next_power_of_two()
            .expect("test GAT table row count fits a dummy column");
        (
            BaseConfig::dummy(table_column_rows, 2),
            RegionCtx::new_dummy(0, 2, RegionSettings::all_true(256, 4)),
        )
    }

    fn integers(value: &ValTensor<Fp>) -> Vec<IntegerRep> {
        value.int_evals().unwrap().iter().copied().collect()
    }

    fn nonzero_attention_spec() -> CanonicalGat {
        let mut spec = crate::gat_adapter::geometry::tiny_multigraph_spec_for_tests();
        spec.features = vec![1, 2, -1, 3, 2, -2, 3, 1, -2, 2];

        let first = &mut spec.model.hidden_layers[0].heads;
        first[0].projection = vec![1, 1, 1, -1];
        first[0].a_src = vec![1, -1];
        first[0].a_dst = vec![-1, 1];
        first[1].projection = vec![1, -1, -1, -1];
        first[1].a_src = vec![-1, 1];
        first[1].a_dst = vec![1, 1];

        let second = &mut spec.model.hidden_layers[1].heads[0];
        second.projection = vec![1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, 1];
        second.a_src = vec![1, -1, 1];
        second.a_dst = vec![-1, 1, 1];

        spec.model.output_heads[0].projection = vec![1, 1, -1, 1, 1, -1];
        spec.model.output_heads[0].a_src = vec![1, -1];
        spec.model.output_heads[0].a_dst = vec![-1, 1];
        spec.model.output_heads[0].bias = vec![1, -1];
        spec.model.output_heads[1].projection = vec![1, -1, 1, 1, -1, 1];
        spec.model.output_heads[1].a_src = vec![1, 1];
        spec.model.output_heads[1].a_dst = vec![-1, -1];
        spec.model.output_heads[1].bias = vec![-1, 1];

        // A small monotone exact table makes the softmax denominator depend
        // on nonzero attention scores while remaining an authenticated toy
        // table over the fixture's full declared domain.
        for (input, output) in &mut spec.tables.exp.rows {
            *output = match *input {
                0 => 4,
                1 => 3,
                2 => 2,
                _ => 1,
            };
        }
        spec.tables.exp.refresh_digest().unwrap();
        spec.tables.refresh_digest().unwrap();

        spec.ledger.max_exp = 4;
        spec.ledger.max_sum = 12;
        spec.ledger.max_divisor = 24;
        for family in &mut spec.ledger.families {
            family.max_exp = 4;
            family.max_sum = 12;
            family.max_aggregation_num = 3_600;
            family.max_divisor = 24;
        }
        spec.ledger.refresh_digest().unwrap();
        spec.ledger_identity.digest = crate::gat_adapter::schema::sha256_digest(
            &spec.ledger.canonical_bytes(),
        )
        .unwrap();
        let field_params = spec
            .model
            .flattened_params()
            .into_iter()
            .map(integer_rep_to_felt::<Fp>)
            .collect::<Vec<_>>();
        spec.model_binding.hash =
            crate::graph::modules::native_poseidon_hash_string(&field_params).unwrap();
        spec.model_binding.canonical_model_digest = spec.canonical_model_digest().unwrap();
        spec.validate_for_compile().unwrap();
        spec
    }

    fn edgewise_attention_scores(
        projected: &ValTensor<Fp>,
        attention_src: &[i128],
        attention_dst: &[i128],
        graph: &crate::gat_adapter::CanonicalGraph,
        channels: usize,
    ) -> Vec<i128> {
        let projected = integers(projected);
        (0..graph.e)
            .map(|edge| {
                let src_start = graph.src[edge] * channels;
                let dst_start = graph.dst[edge] * channels;
                let source = projected[src_start..src_start + channels]
                    .iter()
                    .zip(attention_src)
                    .try_fold(0_i128, |sum, (value, weight)| {
                        sum.checked_add(value.checked_mul(*weight)?)
                    })
                    .unwrap();
                let destination = projected[dst_start..dst_start + channels]
                    .iter()
                    .zip(attention_dst)
                    .try_fold(0_i128, |sum, (value, weight)| {
                        sum.checked_add(value.checked_mul(*weight)?)
                    })
                    .unwrap();
                source.checked_add(destination).unwrap()
            })
            .collect()
    }

    fn trace_values_by_node(
        traces: &[crate::gat_adapter::relation::AggregationTrace],
        nodes: usize,
        channels: usize,
    ) -> Vec<i128> {
        let mut values = vec![0; nodes * channels];
        for trace in traces {
            assert_eq!(trace.values.len(), channels);
            let start = trace.dst * channels;
            values[start..start + channels].copy_from_slice(&trace.values);
        }
        values
    }

    pub(super) fn run_nonzero_attention_route_equivalence() {
        let spec = nonzero_attention_spec();
        let rounding = RoundingMode::NearestTiesAway;
        let reference = crate::gat_adapter::relation::evaluate_for_compile_for_tests(
            &spec,
            rounding,
        )
        .unwrap();
        let operation = GatOp {
            spec: spec.clone(),
            rounding,
        };
        let graph = &spec.graph;
        let groups = graph.destination_groups_validated();
        assert_eq!(graph.node_ptr, [0, 3, 5]);
        assert_eq!(graph.edge_ptr, [0, 6, 9]);
        for graph_idx in 0..graph.g_batch {
            let node_start = graph.node_ptr[graph_idx];
            let node_end = graph.node_ptr[graph_idx + 1];
            for edge in graph.edge_ptr[graph_idx]..graph.edge_ptr[graph_idx + 1] {
                assert!((node_start..node_end).contains(&graph.src[edge]));
                assert!((node_start..node_end).contains(&graph.dst[edge]));
            }
        }

        let mut nonuniform_softmax_seen = false;
        let (config, mut region) = dummy_layout_for_gat_tables(&spec);
        region.set_ordered_dummy_loop(true);
        let source = GatOp::value_i128(
            &graph.src.iter().map(|node| *node as i128).collect::<Vec<_>>(),
            &[graph.e],
        )
        .unwrap();
        let destination = GatOp::value_i128(
            &graph.dst.iter().map(|node| *node as i128).collect::<Vec<_>>(),
            &[graph.e],
        )
        .unwrap();
        let gathered_features = graph
            .node_ids
            .iter()
            .flat_map(|node| {
                let start = node * spec.feature_dim;
                spec.features[start..start + spec.feature_dim].iter().copied()
            })
            .collect::<Vec<_>>();
        let mut current = GatOp::value_i128(
            &gathered_features,
            &[graph.n, spec.feature_dim],
        )
        .unwrap();
        current.set_scale(spec.scales.s_h[0]);
        let mut current_dim = spec.feature_dim;
        let mut assigned_tables = HashMap::new();

        for (layer_idx, layer) in spec.model.hidden_layers.iter().enumerate() {
            let mut head_outputs = Vec::with_capacity(layer.heads.len());
            for (head_idx, head) in layer.heads.iter().enumerate() {
                let mut projection = GatOp::value_i128(
                    &head.projection,
                    &[layer.d_in, layer.d_head],
                )
                .unwrap();
                projection.set_scale(spec.scales.s_w_hidden[layer_idx][head_idx]);
                let mut projected = layouts::einsum(
                    &config,
                    &mut region,
                    &[&current, &projection],
                    "ij,jk->ik",
                )
                .unwrap();
                projected.set_scale(spec.scales.s_h_prime[layer_idx][head_idx]);
                let mut attention_src =
                    GatOp::value_i128(&head.a_src, &[layer.d_head]).unwrap();
                attention_src.set_scale(spec.scales.s_a_src_hidden[layer_idx][head_idx]);
                let mut attention_dst =
                    GatOp::value_i128(&head.a_dst, &[layer.d_head]).unwrap();
                attention_dst.set_scale(spec.scales.s_a_dst_hidden[layer_idx][head_idx]);

                let raw_score_values = operation
                    .node_shared_attention_score_chunks(
                        &config,
                        &mut region,
                        &projected,
                        &attention_src,
                        &attention_dst,
                        layer.d_head,
                        (MAX_GAT_EDGE_CHUNK_CELLS / layer.d_head).max(1),
                        |_, _, score_chunk| {
                            Ok(score_chunk.get_inner_tensor()?.iter().cloned().collect())
                        },
                    )
                    .unwrap();
                let raw_scores: ValTensor<Fp> =
                    Tensor::from_vec(raw_score_values, &[graph.e]).unwrap().into();
                let edgewise = edgewise_attention_scores(
                    &projected,
                    &head.a_src,
                    &head.a_dst,
                    graph,
                    layer.d_head,
                );
                assert_eq!(
                    integers(&raw_scores),
                    edgewise,
                    "node-shared raw source+destination logits differ from edge-wise dots at hidden layer {layer_idx} head {head_idx}"
                );

                let bounds = operation.bound_family(layer_idx, head_idx, false).unwrap();
                let aggregated = operation
                    .aggregate_head(
                        &config,
                        &mut region,
                        &projected,
                        &source,
                        &destination,
                        &attention_src,
                        &attention_dst,
                        layer.d_head,
                        true,
                        bounds,
                        spec.scales.s_agg_pre[layer_idx],
                        spec.scales.s_agg[layer_idx],
                        spec.scales.s_h_prime[layer_idx][head_idx],
                        spec.scales.s_a_src_hidden[layer_idx][head_idx],
                        spec.scales.s_a_dst_hidden[layer_idx][head_idx],
                        &groups,
                        &mut assigned_tables,
                    )
                    .unwrap();
                let traces = &reference.hidden[layer_idx].heads[head_idx];
                let baseline_denominator = 4_i128;
                for trace in traces {
                    let (_, start, end) = groups[trace.dst];
                    let uniform = i128::try_from(end - start).unwrap() * baseline_denominator;
                    nonuniform_softmax_seen |= trace.denominator != uniform;
                }
                assert_eq!(
                    integers(&aggregated),
                    trace_values_by_node(traces, graph.n, layer.d_head),
                    "production hidden softmax/aggregation differs from edge-wise relation at layer {layer_idx} head {head_idx}"
                );
                head_outputs.push(aggregated);
            }
            current = GatOp::interleave_heads(&head_outputs, graph.n, layer.d_head).unwrap();
            current.set_scale(spec.scales.s_h[layer_idx + 1]);
            current_dim = layer.heads.len() * layer.d_head;
            assert_eq!(
                integers(&current),
                reference.hidden[layer_idx].node_values,
                "hidden head interleave differs from the canonical edge-wise relation at layer {layer_idx}"
            );
        }

        for (head_idx, head) in spec.model.output_heads.iter().enumerate() {
            let mut projection = GatOp::value_i128(
                &head.projection,
                &[current_dim, spec.model.classes],
            )
            .unwrap();
            projection.set_scale(spec.scales.s_w_output[head_idx]);
            let mut projected = layouts::einsum(
                &config,
                &mut region,
                &[&current, &projection],
                "ij,jk->ik",
            )
            .unwrap();
            projected.set_scale(spec.scales.s_y_prime[head_idx]);
            let mut attention_src =
                GatOp::value_i128(&head.a_src, &[spec.model.classes]).unwrap();
            attention_src.set_scale(spec.scales.s_a_src_output[head_idx]);
            let mut attention_dst =
                GatOp::value_i128(&head.a_dst, &[spec.model.classes]).unwrap();
            attention_dst.set_scale(spec.scales.s_a_dst_output[head_idx]);

            let raw_score_values = operation
                .node_shared_attention_score_chunks(
                    &config,
                    &mut region,
                    &projected,
                    &attention_src,
                    &attention_dst,
                    spec.model.classes,
                    (MAX_GAT_EDGE_CHUNK_CELLS / spec.model.classes).max(1),
                    |_, _, score_chunk| {
                        Ok(score_chunk.get_inner_tensor()?.iter().cloned().collect())
                    },
                )
                .unwrap();
            let raw_scores: ValTensor<Fp> =
                Tensor::from_vec(raw_score_values, &[graph.e]).unwrap().into();
            let edgewise = edgewise_attention_scores(
                &projected,
                &head.a_src,
                &head.a_dst,
                graph,
                spec.model.classes,
            );
            assert_eq!(
                integers(&raw_scores),
                edgewise,
                "node-shared raw source+destination logits differ from edge-wise dots at output head {head_idx}"
            );

            let bounds = operation.bound_family(usize::MAX, head_idx, true).unwrap();
            let aggregated = operation
                .aggregate_head(
                    &config,
                    &mut region,
                    &projected,
                    &source,
                    &destination,
                    &attention_src,
                    &attention_dst,
                    spec.model.classes,
                    false,
                    bounds,
                    spec.scales.s_out_pre,
                    spec.scales.s_out_pre,
                    spec.scales.s_y_prime[head_idx],
                    spec.scales.s_a_src_output[head_idx],
                    spec.scales.s_a_dst_output[head_idx],
                    &groups,
                    &mut assigned_tables,
                )
                .unwrap();
            let traces = &reference.output_heads[head_idx];
            for trace in traces {
                let (_, start, end) = groups[trace.dst];
                let uniform = i128::try_from(end - start).unwrap() * 4;
                nonuniform_softmax_seen |= trace.denominator != uniform;
            }
            assert_eq!(
                integers(&aggregated),
                trace_values_by_node(traces, graph.n, spec.model.classes),
                "production output softmax/aggregation differs from edge-wise relation at head {head_idx}"
            );
        }
        assert!(
            nonuniform_softmax_seen,
            "nonzero attention fixture must exercise score-dependent softmax weights"
        );

        let (mut proof_config, mut proof_region) = dummy_layout_for_gat_tables(&spec);
        let mut public_and_private = vec![
            {
                let mut tensor = GatOp::value_i128(
                    &spec.features,
                    &[graph.n_total, spec.feature_dim],
                )
                .unwrap();
                tensor.set_scale(spec.scales.s_input);
                tensor
            },
            GatOp::value_i128(
                &graph.node_ids.iter().map(|node| *node as i128).collect::<Vec<_>>(),
                &[graph.n],
            )
            .unwrap(),
            GatOp::value_i128(
                &graph.src.iter().map(|node| *node as i128).collect::<Vec<_>>(),
                &[graph.e],
            )
            .unwrap(),
            GatOp::value_i128(
                &graph.dst.iter().map(|node| *node as i128).collect::<Vec<_>>(),
                &[graph.e],
            )
            .unwrap(),
            GatOp::value_i128(
                &graph.node_ptr.iter().map(|node| *node as i128).collect::<Vec<_>>(),
                &[graph.g_batch + 1],
            )
            .unwrap(),
            GatOp::value_i128(
                &graph.edge_ptr.iter().map(|edge| *edge as i128).collect::<Vec<_>>(),
                &[graph.g_batch + 1],
            )
            .unwrap(),
        ];
        let parameter_len = spec.model.flattened_params_len().unwrap();
        let mut parameters = GatOp::value_i128(
            &spec.model.flattened_params(),
            &[parameter_len],
        )
        .unwrap();
        parameters.set_scale(spec.scales.s_w_hidden[0][0]);
        public_and_private.push(parameters);
        let value_refs = public_and_private.iter().collect::<Vec<_>>();
        let actual_logits = operation
            .layout(&mut proof_config, &mut proof_region, &value_refs)
            .unwrap()
            .unwrap();
        assert_eq!(
            integers(&actual_logits),
            reference.output,
            "production multilayer/multihead logits differ from the edge-wise integer relation"
        );
    }

    fn exact_table(table_id: &str, rows: &[(i128, i128)]) -> ExactTable {
        ExactTable {
            table_id: table_id.into(),
            input_scale: 0,
            output_scale: 0,
            function_id: table_id.into(),
            domain_id: format!("{}-domain", table_id),
            generator_id: format!("{}-generator", table_id),
            rows: rows.to_vec(),
            input_min: rows.first().map(|row| row.0).unwrap_or(0),
            input_max: rows.last().map(|row| row.0).unwrap_or(0),
            digest: format!("{}-digest", table_id),
        }
    }

    fn refresh_table_registry(spec: &mut CanonicalGat) {
        spec.tables.max_relu.refresh_digest().unwrap();
        spec.tables.refresh_digest().unwrap();
    }

    const MOCK_CHUNK_K: usize = 12;
    const MOCK_CHUNK_CAPACITY: usize = 4096;
    const MOCK_CHUNK_GROUPS: [(usize, usize, usize); 3] = [(0, 0, 2), (1, 2, 5), (2, 5, 7)];
    const MOCK_FIELD_PRODUCTS: [i128; 14] = [1, -3, 2, 4, -1, 5, 7, 8, -4, 2, 6, -2, 3, 9];
    const MOCK_NUMERATOR_PRODUCTS: [i128; 14] = [2, 1, 3, -1, 3, 4, 5, 2, 2, 0, 9, -1, 6, -1];
    const MOCK_FIELD_ENDPOINTS: [i128; 6] = [3, 1, 2, 15, 9, 7];
    const MOCK_NUMERATOR_ENDPOINTS: [i128; 6] = [5, 0, 10, 6, 15, -2];
    const MOCK_ROUNDED_QUOTIENT: [i128; 6] = [3, 0, 3, 2, 3, 0];

    #[derive(Clone, Default)]
    struct ChunkPrefixMockParams;

    #[derive(Clone)]
    struct ChunkPrefixMockCircuit {
        chunk_rows: usize,
        tamper_carry: bool,
        tamper_quotient: bool,
        check_rounding: bool,
        check_expected_quotient: bool,
    }

    impl Circuit<Fp> for ChunkPrefixMockCircuit {
        type Config = BaseConfig<Fp>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = ChunkPrefixMockParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(meta: &mut ConstraintSystem<Fp>) -> Self::Config {
            let input = crate::tensor::VarTensor::new_advice(
                meta,
                MOCK_CHUNK_K,
                2,
                MOCK_CHUNK_CAPACITY,
            );
            let index = crate::tensor::VarTensor::new_advice(
                meta,
                MOCK_CHUNK_K,
                2,
                MOCK_CHUNK_CAPACITY,
            );
            let output = crate::tensor::VarTensor::new_advice(
                meta,
                MOCK_CHUNK_K,
                2,
                MOCK_CHUNK_CAPACITY,
            );
            crate::tensor::VarTensor::constant_cols(meta, MOCK_CHUNK_K, 8, false);
            let mut config = BaseConfig::configure(
                meta,
                &[input.clone(), index.clone()],
                &output,
                crate::circuit::ops::chip::CheckMode::SAFE,
            );
            for range in [(-20, 20), (-1, 1), (0, 1), (0, 20), (1, 5), (0, 255)] {
                config
                    .configure_range_check(meta, &input, &index, range, MOCK_CHUNK_K)
                    .unwrap();
            }
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<Fp>,
        ) -> Result<(), Error> {
            config
                .layout_range_checks(&mut layouter)
                .map_err(|error| {
                    eprintln!("chunk mock range-table layout failed: {error:?}");
                    Error::Synthesis
                })?;
            layouter.assign_region(
                || "chunked GAT prefix and canonical rounding",
                |region| {
                    let settings = RegionSettings::new(true, false, 256, 4);
                    let mut region = RegionCtx::new_with_settings(region, 0, 2, settings);
                    let field_values = constant_i128_tensor(&MOCK_FIELD_PRODUCTS, &[7, 2]);
                    let numerator_values = constant_i128_tensor(&MOCK_NUMERATOR_PRODUCTS, &[7, 2]);
                    let field_endpoints = chunked_prefix_endpoints(
                        &config,
                        &mut region,
                        &field_values,
                        self.chunk_rows,
                        self.tamper_carry,
                    )
                    .map_err(|error| {
                        eprintln!("chunk mock field recurrence failed: {error:?}");
                        Error::Synthesis
                    })?;
                    let numerator_endpoints = chunked_prefix_endpoints(
                        &config,
                        &mut region,
                        &numerator_values,
                        self.chunk_rows,
                        false,
                    )
                    .map_err(|error| {
                        eprintln!("chunk mock numerator recurrence failed: {error:?}");
                        Error::Synthesis
                    })?;
                    let expected_field = constant_i128_tensor(&MOCK_FIELD_ENDPOINTS, &[3, 2]);
                    let expected_numerator =
                        constant_i128_tensor(&MOCK_NUMERATOR_ENDPOINTS, &[3, 2]);
                    layouts::enforce_equality(
                        &config,
                        &mut region,
                        &[&field_endpoints, &expected_field],
                    )
                    .map_err(|_| Error::Synthesis)?;
                    layouts::enforce_equality(
                        &config,
                        &mut region,
                        &[&numerator_endpoints, &expected_numerator],
                    )
                    .map_err(|_| Error::Synthesis)?;

                    if !self.check_rounding {
                        return Ok(());
                    }

                    let divisor = constant_i128_tensor(&[2, 3, 5], &[3]);
                    let bounds = BoundFamily {
                        max_dividend_abs: 20,
                        max_aux_abs: 20,
                        max_divisor: 5,
                        max_aggregation_num: 20,
                        ..BoundFamily::default()
                    };
                    let op = GatOp {
                        spec: CanonicalGat::default(),
                        rounding: RoundingMode::NearestTiesAway,
                    };
                    let quotient = op
                        .signed_round_div_layout_mode(
                            &config,
                            &mut region,
                            &numerator_endpoints,
                            &divisor,
                            RoundingMode::NearestTiesAway,
                            Some(&bounds),
                        )
                        .map_err(|error| {
                            eprintln!("chunk mock signed division failed: {error:?}");
                            Error::Synthesis
                        })?;
                    if !self.check_expected_quotient {
                        return Ok(());
                    }
                    let mut expected_quotient = MOCK_ROUNDED_QUOTIENT;
                    if self.tamper_quotient {
                        expected_quotient[0] = expected_quotient[0]
                            .checked_add(1)
                            .ok_or(Error::Synthesis)?;
                    }
                    let expected_quotient = constant_i128_tensor(&expected_quotient, &[3, 2]);
                    layouts::enforce_equality(
                        &config,
                        &mut region,
                        &[&quotient, &expected_quotient],
                    )
                    .map_err(|_| Error::Synthesis)?;
                    Ok(())
                },
            )
        }
    }

    fn constant_i128_tensor(values: &[i128], dims: &[usize]) -> ValTensor<Fp> {
        let values = values
            .iter()
            .map(|value| ValType::Constant(integer_rep_to_felt(*value)))
            .collect::<Vec<_>>();
        Tensor::new(Some(&values), dims).unwrap().into()
    }

    fn chunked_prefix_endpoints(
        config: &BaseConfig<Fp>,
        region: &mut RegionCtx<Fp>,
        values: &ValTensor<Fp>,
        chunk_rows: usize,
        tamper_carry: bool,
    ) -> Result<ValTensor<Fp>, CircuitError> {
        let edge_count = 7;
        let channels = 2;
        if chunk_rows == 0 || values.dims() != [edge_count, channels] {
            return Err(CircuitError::DimMismatch(
                "mock GAT chunk fixture has an invalid shape or chunk size".into(),
            ));
        }
        let mut carry_cells = Vec::new();
        let mut carry_witness = Vec::new();
        let mut next_group = 0;
        let mut endpoint_values = Vec::with_capacity(MOCK_FIELD_ENDPOINTS.len());
        for chunk_start in (0..edge_count).step_by(chunk_rows) {
            if tamper_carry && chunk_start == chunk_rows {
                let wrong_carry = constant_i128_tensor(&[99, -99], &[channels]);
            let assigned = region
                .assign(&config.custom_gates.inputs[1], &wrong_carry)
                .map_err(|error| {
                    eprintln!("chunk mock injected carry assignment failed: {error:?}");
                    error
                })?;
                region.increment(channels);
                carry_cells = assigned.get_inner_tensor()?.iter().cloned().collect();
            }
            let chunk_end = (chunk_start + chunk_rows).min(edge_count);
            let chunk = GatOp::flatten_slice(
                values,
                chunk_start * channels,
                chunk_end * channels,
                &[chunk_end - chunk_start, channels],
            )?;
            let (prefix, next_carry_cells, next_carry_witness) = GatOp::segmented_prefix_chunk(
                config,
                region,
                &chunk,
                &MOCK_CHUNK_GROUPS,
                chunk_start,
                edge_count,
                channels,
                &carry_cells,
                &carry_witness,
                false,
            )?;
            GatOp::append_chunk_endpoints(
                &prefix,
                &MOCK_CHUNK_GROUPS,
                chunk_start,
                chunk_end,
                channels,
                &mut next_group,
                &mut endpoint_values,
            )?;
            carry_cells = next_carry_cells;
            carry_witness = next_carry_witness;
        }
        if next_group != MOCK_CHUNK_GROUPS.len() || endpoint_values.len() != MOCK_FIELD_ENDPOINTS.len() {
            return Err(CircuitError::DimMismatch(
                "mock GAT chunk fixture did not emit every destination endpoint".into(),
            ));
        }
        let mut endpoints: ValTensor<Fp> =
            Tensor::new(Some(&endpoint_values), &[MOCK_CHUNK_GROUPS.len(), channels])?.into();
        drop(endpoint_values);
        endpoints.set_scale(values.scale());
        Ok(endpoints)
    }

    pub(super) fn run_chunked_prefix_mockprover_checks_carries_endpoints_and_rounding() {
        for chunk_rows in [2, 3] {
            let circuit = ChunkPrefixMockCircuit {
                chunk_rows,
                tamper_carry: false,
                tamper_quotient: false,
                check_rounding: chunk_rows == 3,
                check_expected_quotient: chunk_rows == 3,
            };
            let prover = MockProver::run(MOCK_CHUNK_K as u32, &circuit, vec![]).unwrap();
            let verification = prover.verify();
            assert!(verification.is_ok(), "valid chunk size {chunk_rows} failed: {verification:?}");
        }

        let bad_carry = ChunkPrefixMockCircuit {
            chunk_rows: 3,
            tamper_carry: true,
            tamper_quotient: false,
            check_rounding: false,
            check_expected_quotient: false,
        };
        let prover = MockProver::run(MOCK_CHUNK_K as u32, &bad_carry, vec![]).unwrap();
        assert!(prover.verify().is_err(), "a wrong cross-chunk carry must violate constrained endpoints");

        let bad_quotient = ChunkPrefixMockCircuit {
            chunk_rows: 3,
            tamper_carry: false,
            tamper_quotient: true,
            check_rounding: true,
            check_expected_quotient: true,
        };
        let prover = MockProver::run(MOCK_CHUNK_K as u32, &bad_quotient, vec![]).unwrap();
        assert!(prover.verify().is_err(), "a quotient that violates the canonical rounded relation must be rejected");
    }

    #[test]
    fn chunked_prefix_mockprover_checks_carries_endpoints_and_rounding() {
        run_chunked_prefix_mockprover_checks_carries_endpoints_and_rounding();
    }

    #[test]
    fn authenticated_max_relu_scale_closure_rejects_mismatched_resource_metadata() {
        let baseline_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let loaded = crate::gat_adapter::load_dataset_with_manifest(
            &baseline_root,
            std::path::Path::new("resources/resource_manifest.json"),
            "cora",
        )
        .unwrap();
        let expected_scale = loaded.spec.scales.s_delta;
        assert_eq!(loaded.spec.tables.max_relu.input_scale, expected_scale);
        assert_eq!(loaded.spec.tables.max_relu.output_scale, expected_scale);
        loaded.spec.validate_for_compile().unwrap();

        for mismatch_input in [true, false] {
            let mut altered = loaded.spec.clone();
            if mismatch_input {
                altered.tables.max_relu.input_scale = expected_scale + 1;
            } else {
                altered.tables.max_relu.output_scale = expected_scale + 1;
            }
            refresh_table_registry(&mut altered);
            let error = altered.validate_for_compile().unwrap_err();
            assert!(
                error.contains("max_relu table scale closure"),
                "unexpected max-ReLU scale rejection: {}",
                error,
            );
        }
    }

    #[test]
    fn ragged_prefix_resets_at_public_groups_and_keeps_signed_channels() {
        let (config, mut region) = dummy_layout();
        let groups = [(0, 0, 2), (1, 2, 3), (2, 3, 5)];
        let values = GatOp::value_i128(&[1, -3, 2, 4, -1, 5, 7, 8, -4, 2], &[5, 2]).unwrap();
        let prefix = GatOp::segmented_prefix(&config, &mut region, &values, &groups, 5, 2).unwrap();
        assert_eq!(prefix.dims(), &[5, 2]);
        assert_eq!(integers(&prefix), vec![1, -3, 3, 1, -1, 5, 7, 8, 3, 10]);

        let endpoints = GatOp::destination_group_endpoints(&prefix, &groups, 5, 2).unwrap();
        assert_eq!(endpoints.dims(), &[3, 2]);
        assert_eq!(integers(&endpoints), vec![3, 1, -1, 5, 3, 10]);
    }

    #[test]
    fn chunked_prefix_matches_dense_reference_across_groups_and_carries() {
        let groups = [(0, 0, 3), (1, 3, 4), (2, 4, 7)];
        let values = GatOp::value_i128(
            &[1, -3, 2, 4, -1, 5, 7, 8, -4, 2, 6, -2, 3, 9],
            &[7, 2],
        )
        .unwrap();
        let (config, mut dense_region) = dummy_layout();
        let dense = GatOp::segmented_prefix(&config, &mut dense_region, &values, &groups, 7, 2)
            .unwrap();
        let dense_endpoints = GatOp::destination_group_endpoints(&dense, &groups, 7, 2).unwrap();

        let (config, mut chunk_region) = dummy_layout();
        let mut carry_cells = Vec::new();
        let mut carry_witness = Vec::new();
        let mut next_group = 0;
        let mut endpoint_values = Vec::new();
        let mut chunk_values = Vec::new();
        for chunk_start in (0..7).step_by(2) {
            let chunk_end = (chunk_start + 2).min(7);
            let chunk_edges = chunk_end - chunk_start;
            let chunk = GatOp::flatten_slice(
                &values,
                chunk_start * 2,
                chunk_end * 2,
                &[chunk_edges, 2],
            )
            .unwrap();
            let (prefix, next_carry_cells, next_carry_witness) = GatOp::segmented_prefix_chunk(
                &config,
                &mut chunk_region,
                &chunk,
                &groups,
                chunk_start,
                7,
                2,
                &carry_cells,
                &carry_witness,
                false,
            )
            .unwrap();
            chunk_values.extend(integers(&prefix));
            GatOp::append_chunk_endpoints(
                &prefix,
                &groups,
                chunk_start,
                chunk_end,
                2,
                &mut next_group,
                &mut endpoint_values,
            )
            .unwrap();
            carry_cells = next_carry_cells;
            carry_witness = next_carry_witness;
        }

        assert_eq!(chunk_values, integers(&dense));
        let chunk_endpoints: ValTensor<Fp> = Tensor::new(Some(&endpoint_values), &[3, 2])
            .unwrap()
            .into();
        assert_eq!(integers(&chunk_endpoints), integers(&dense_endpoints));
        assert_eq!(next_group, groups.len());
    }

    #[test]
    fn group_repeats_follow_ragged_edge_offsets_without_padding() {
        let groups = [(0, 0, 2), (1, 2, 3), (2, 3, 5)];
        let by_node = GatOp::value_i128(&[3, 7, 11], &[3]).unwrap();
        let repeated = GatOp::repeat_by_destination_groups(&by_node, &groups, 5, 1).unwrap();
        assert_eq!(repeated.dims(), &[5]);
        assert_eq!(integers(&repeated), vec![3, 3, 7, 11, 11]);

        let padded_schedule = [(0, 0, 2), (1, 3, 5)];
        assert!(GatOp::repeat_by_destination_groups(&by_node, &padded_schedule, 5, 1).is_err());
    }

    #[test]
    fn signed_division_batches_per_destination_divisors_and_rejects_bad_shapes() {
        let (config, mut region) = dummy_layout();
        let op = GatOp {
            spec: CanonicalGat::default(),
            rounding: RoundingMode::NearestTiesAway,
        };
        let dividend = GatOp::value_i128(&[5, -5, 0, 1], &[2, 2]).unwrap();
        let divisor = GatOp::value_i128(&[2, 3], &[2]).unwrap();
        let bounds = BoundFamily {
            max_dividend_abs: 16,
            max_aux_abs: 32,
            max_divisor: 4,
            ..BoundFamily::default()
        };
        let quotient = op
            .signed_round_div_layout_mode(
                &config,
                &mut region,
                &dividend,
                &divisor,
                RoundingMode::NearestTiesAway,
                Some(&bounds),
            )
            .unwrap();
        assert_eq!(quotient.dims(), &[2, 2]);
        assert_eq!(integers(&quotient), vec![3, -3, 0, 0]);

        let bad_divisor = GatOp::value_i128(&[2, 3, 4], &[3]).unwrap();
        assert!(op
            .signed_round_div_layout_mode(
                &config,
                &mut region,
                &dividend,
                &bad_divisor,
                RoundingMode::NearestTiesAway,
                Some(&bounds),
            )
            .is_err());
    }

    #[test]
    fn batched_attention_keeps_ties_singleton_groups_and_negative_values() {
        let (config, mut region) = dummy_layout();
        let mut spec = CanonicalGat::default();
        spec.graph.n = 2;
        spec.graph.e = 3;
        spec.graph.src = vec![0, 1, 0];
        spec.graph.dst = vec![0, 0, 1];
        spec.tables.lrelu = exact_table("lrelu", &[(0, 0)]);
        spec.tables.exp = exact_table("exp", &[(0, 1)]);
        spec.tables.max_relu = exact_table("max_relu", &[(-1, 0), (0, 0), (1, 1)]);
        let op = GatOp {
            spec,
            rounding: RoundingMode::NearestTiesAway,
        };
        let mut projected = GatOp::value_i128(&[-3, 5], &[2, 1]).unwrap();
        projected.set_scale(0);
        let source = GatOp::value_i128(&[0, 1, 0], &[3]).unwrap();
        let destination = GatOp::value_i128(&[0, 0, 1], &[3]).unwrap();
        let mut attention_src = GatOp::value_i128(&[0], &[1]).unwrap();
        let mut attention_dst = GatOp::value_i128(&[0], &[1]).unwrap();
        attention_src.set_scale(0);
        attention_dst.set_scale(0);
        let bounds = BoundFamily {
            max_score_abs: 10,
            max_table_output_abs: 10,
            max_delta: 4,
            max_exp: 1,
            max_sum: 2,
            max_aggregation_num: 10,
            max_dividend_abs: 10,
            max_aux_abs: 20,
            max_divisor: 2,
            ..BoundFamily::default()
        };
        let groups = [(0, 0, 2), (1, 2, 3)];
        let mut tables = HashMap::new();
        let output = op
            .aggregate_head(
                &config,
                &mut region,
                &projected,
                &source,
                &destination,
                &attention_src,
                &attention_dst,
                1,
                false,
                &bounds,
                0,
                0,
                0,
                0,
                0,
                &groups,
                &mut tables,
            )
            .unwrap();
        // The first destination has a tied maximum (both edges are active);
        // the second has degree one. Negative source values remain in the
        // exact numerator and are rounded only by the native signed divider.
        assert_eq!(output.dims(), &[2, 1]);
        assert_eq!(integers(&output), vec![1, -3]);
    }
}
