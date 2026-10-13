//! Shape-only counters for native ezkl/Halo2 assignment primitives.
//!
//! This module mirrors only assignment geometry. Equation strategy and reduction ordering come
//! from the native Einsums analyzer and reduction planner; no witness values or GAT arithmetic are
//! evaluated here.

use std::collections::HashMap;

use crate::circuit::{einsum::{analysis::{analyze_single_equation, EinsumStrategy}, native_input_reductions, Reduction}, CircuitError};
use crate::tensor::duplicate_every_n_output_len;

/// Shape-only assignment and cursor summary for one native einsum operation.
#[derive(Debug, Clone)]
pub struct EinsumGeometryPlan {
    /// Strategy selected by the native einsum analyzer for this equation.
    pub strategy: EinsumStrategy,
    /// Whether the selected strategy assigns a Freivalds check tensor.
    pub uses_freivalds: bool,
    /// Canonicalized einsum equation used by the native analyzer.
    pub normalized_equation: String,
    /// Number of values in the main Freivalds assignment, or zero otherwise.
    pub main_assignment_len: usize,
    /// Final einsum cursor after accounting for this operation's assignments.
    pub final_einsum_col: usize,
    /// Change in the einsum assignment index produced by this operation.
    pub einsum_index_delta: usize,
    /// Original equation and the dimensions of its named axes.
    pub equation_metadata: (String, HashMap<char, usize>),
}

struct EinsumCursor { col: usize, width: usize, col_size: usize }

/// Return the exact assigned length produced by native `Tensor::duplicate_every_n` geometry.
pub fn native_duplicated_assignment_len(len: usize, frequency: usize, repeats: usize, initial_offset: usize) -> Result<usize, CircuitError> {
    duplicate_every_n_output_len(len, frequency, repeats, initial_offset).map_err(|_| CircuitError::InvalidEinsum)
}

/// Return the dummy-region cursor after repeated native unconstrained assignments of one shape.
pub fn native_dummy_input_cursor_end(start: usize, width: usize, col_size: usize, input_len: usize, count: usize) -> Result<usize, CircuitError> {
    let mut cursor = EinsumCursor::new(start, width, col_size)?;
    cursor.advance_input_repeated(input_len, count)?;
    Ok(cursor.col)
}

/// Exercise the native/live versus dummy duplication phase at a column boundary that
/// is not reached by the tiny settings-parity fixture.
#[cfg(test)]
pub(crate) fn run_dummy_duplication_boundary_regression() {
    use crate::fieldutils::IntegerRep;
    use crate::tensor::Tensor;

    // With width=2 and four rows per column, cursor 10 is at the second row
    // phase. Native duplication sees that full offset and duplicates the last
    // element of this six-value assignment once (two extra copies).
    let input = Tensor::<IntegerRep>::new(Some(&[1, 2, 3, 4, 5, 6]), &[6]).unwrap();
    let live_len = input.duplicate_every_n(8, 2, 10).unwrap().len();
    assert_eq!(live_len, 8);

    // The dummy schedule must advance by the same live assignment footprint.
    let dummy_end = native_dummy_input_cursor_end(10, 2, 4, 6, 1).unwrap();
    assert_eq!(dummy_end, 10 + live_len);
    assert_eq!(dummy_end, 18);
}

impl EinsumCursor {
    fn new(col: usize, width: usize, col_size: usize) -> Result<Self, CircuitError> {
        if width == 0 || col_size == 0 { return Err(CircuitError::InvalidEinsum); }
        Ok(Self { col, width, col_size })
    }
    fn flush(&mut self) -> Result<(), CircuitError> {
        let rem = self.col % self.width;
        if rem != 0 { self.col = self.col.checked_add(self.width - rem).ok_or(CircuitError::InvalidEinsum)?; }
        Ok(())
    }
    fn add(&mut self, n: usize) -> Result<(), CircuitError> {
        self.col = self.col.checked_add(n).ok_or(CircuitError::InvalidEinsum)?;
        Ok(())
    }
    fn padded(&self, len: usize) -> Result<usize, CircuitError> {
        let rem = len % self.width;
        if rem == 0 { Ok(len) } else { len.checked_add(self.width - rem).ok_or(CircuitError::InvalidEinsum) }
    }
    fn advance_input(&mut self, len: usize) -> Result<(), CircuitError> {
        self.flush()?;
        let padded = self.padded(len)?;
        let frequency = self.col_size.checked_mul(self.width).ok_or(CircuitError::InvalidEinsum)?;
        let assigned = native_duplicated_assignment_len(padded, frequency, self.width, self.col)?;
        self.add(assigned)
    }
    fn advance_input_repeated(&mut self, len: usize, count: usize) -> Result<(), CircuitError> {
        if count == 0 || len == 0 { return Ok(()); }
        self.flush()?;
        let padded = self.padded(len)?;
        let frequency = self.col_size.checked_mul(self.width).ok_or(CircuitError::InvalidEinsum)?;
        let max_offset = (self.col_size - 1).checked_mul(self.width).ok_or(CircuitError::InvalidEinsum)?;
        let min_len = native_duplicated_assignment_len(padded, frequency, self.width, 0)?;
        let max_len = native_duplicated_assignment_len(padded, frequency, self.width, max_offset)?;
        if min_len == max_len {
            let assigned = min_len.checked_mul(count).ok_or(CircuitError::InvalidEinsum)?;
            return self.add(assigned);
        }
        let mut units = self.col / self.width;
        let mut iteration = 0usize;
        let mut seen: HashMap<usize, (usize, usize)> = HashMap::new();
        while iteration < count {
            let row_phase = units % self.col_size;
            if let Some((previous_iteration, previous_units)) = seen.get(&row_phase).copied() {
                let cycle_len = iteration - previous_iteration;
                let cycle_units = units - previous_units;
                let cycles = (count - iteration) / cycle_len;
                if cycles > 0 {
                    let skipped_units = cycles.checked_mul(cycle_units).ok_or(CircuitError::InvalidEinsum)?;
                    let skipped_iterations = cycles.checked_mul(cycle_len).ok_or(CircuitError::InvalidEinsum)?;
                    units = units.checked_add(skipped_units).ok_or(CircuitError::InvalidEinsum)?;
                    iteration = iteration.checked_add(skipped_iterations).ok_or(CircuitError::InvalidEinsum)?;
                    continue;
                }
            } else { seen.insert(row_phase, (iteration, units)); }
            let offset = row_phase.checked_mul(self.width).ok_or(CircuitError::InvalidEinsum)?;
            let assigned = native_duplicated_assignment_len(padded, frequency, self.width, offset)?;
            units = units.checked_add(assigned / self.width).ok_or(CircuitError::InvalidEinsum)?;
            iteration = iteration.checked_add(1).ok_or(CircuitError::InvalidEinsum)?;
        }
        self.col = units.checked_mul(self.width).ok_or(CircuitError::InvalidEinsum)?;
        Ok(())
    }
    fn advance_sum_repeated(&mut self, len: usize, count: usize) -> Result<(), CircuitError> {
        if len <= 1 || count == 0 { return Ok(()); }
        self.advance_input_repeated(len, count)
    }
    fn advance_direct_repeated(&mut self, len: usize, count: usize) -> Result<(), CircuitError> {
        if count == 0 || len == 0 { return Ok(()); }
        self.flush()?;
        let pad = (self.width - len % self.width) % self.width;
        let interstitial = pad.checked_mul(count.saturating_sub(1)).ok_or(CircuitError::InvalidEinsum)?;
        self.add(len.checked_mul(count).and_then(|n| n.checked_add(interstitial)).ok_or(CircuitError::InvalidEinsum)?)
    }
}

fn product(dims: &[usize]) -> Result<usize, CircuitError> {
    dims.iter().try_fold(1usize, |acc, dim| acc.checked_mul(*dim).ok_or(CircuitError::InvalidEinsum))
}

fn input_axes(equation: &str, input_dims: &[Vec<usize>]) -> Result<(HashMap<char, usize>, Vec<String>), CircuitError> {
    let (inputs, _) = equation.split_once("->").ok_or(CircuitError::InvalidEinsum)?;
    let inputs = inputs.split(',').map(str::to_owned).collect::<Vec<_>>();
    if inputs.len() != input_dims.len() { return Err(CircuitError::DimMismatch("einsum inputs".into())); }
    let mut axes: HashMap<char, usize> = HashMap::new();
    for (expr, dims) in inputs.iter().zip(input_dims) {
        if expr.chars().count() != dims.len() { return Err(CircuitError::DimMismatch("einsum input shape".into())); }
        for (axis, dim) in expr.chars().zip(dims) {
            if *dim == 0 { return Err(CircuitError::InvalidEinsum); }
            if let Some(previous) = axes.get(&axis) { if previous != dim { return Err(CircuitError::DimMismatch("einsum axis dimension".into())); } }
            else { axes.insert(axis, *dim); }
        }
    }
    Ok((axes, inputs))
}

fn shape_len_for_expr(expr: &str, axes: &HashMap<char, usize>, fixed: &[char]) -> Result<usize, CircuitError> {
    expr.chars().try_fold(1usize, |acc, axis| {
        let dim = if fixed.contains(&axis) { 1 } else { *axes.get(&axis).ok_or(CircuitError::InvalidEinsum)? };
        acc.checked_mul(dim).ok_or(CircuitError::InvalidEinsum)
    })
}

fn reduce_cursor(equation: &str, input_dims: &[Vec<usize>], axes: &HashMap<char, usize>, cursor: &mut EinsumCursor) -> Result<(), CircuitError> {
    let normalized_inputs = equation.split_once("->").ok_or(CircuitError::InvalidEinsum)?.0.split(',').collect::<Vec<_>>();
    let mut tensor_exprs = normalized_inputs.iter().map(|expr| expr.to_string()).collect::<Vec<_>>();
    let mut tensor_lens = input_dims.iter().map(|dims| product(dims)).collect::<Result<Vec<_>, _>>()?;
    let reductions = native_input_reductions(equation)?;
    for reduction in reductions {
        let (input_expr, output_expr) = reduction.expression().split_once("->").ok_or(CircuitError::InvalidEinsum)?;
        let reduction_exprs = input_expr.split(',').collect::<Vec<_>>();
        let indices = reduction.input_indices();
        let remaining_axes = output_expr.chars().collect::<Vec<_>>();
        let combinations = product(&remaining_axes.iter().map(|axis| *axes.get(axis).unwrap_or(&1)).collect::<Vec<_>>())?;
        let mut flat_lens = Vec::with_capacity(indices.len());
        for (position, index) in indices.iter().enumerate() {
            let tensor_index = index.index();
            let expr = tensor_exprs.get(tensor_index).ok_or(CircuitError::InvalidEinsum)?;
            let expected_expr = reduction_exprs.get(position).copied().ok_or(CircuitError::InvalidEinsum)?;
            if expr.as_str() != expected_expr { return Err(CircuitError::InvalidEinsum); }
            let mut len = shape_len_for_expr(expr, axes, &remaining_axes)?;
            len = len.checked_mul(combinations).ok_or(CircuitError::InvalidEinsum)?;
            flat_lens.push(len);
        }
        if flat_lens.is_empty() { return Err(CircuitError::InvalidEinsum); }
        let output_dims = output_expr.chars().map(|axis| axes.get(&axis).copied().unwrap_or(1)).collect::<Vec<_>>();
        let output_len = product(&output_dims)?;
        match reduction {
            Reduction::RLC { axis, .. } => {
                if flat_lens.len() != 1 { return Err(CircuitError::InvalidEinsum); }
                let rlc_len = *axes.get(&axis).ok_or(CircuitError::InvalidEinsum)?;
                if rlc_len == 0 || flat_lens[0] % rlc_len != 0 { return Err(CircuitError::InvalidEinsum); }
                cursor.advance_input_repeated(rlc_len, flat_lens[0] / rlc_len)?;
            }
            Reduction::Contraction { axis, .. } => match axis {
                None => {
                    for _ in 1..flat_lens.len() { cursor.advance_direct_repeated(flat_lens[0], 1)?; }
                }
                Some(axis) => {
                    let dot_len = *axes.get(&axis).ok_or(CircuitError::InvalidEinsum)?;
                    if dot_len == 0 || flat_lens.iter().any(|len| len % dot_len != 0) { return Err(CircuitError::InvalidEinsum); }
                    let dot_count = output_len;
                    match flat_lens.len() {
                        1 => cursor.advance_sum_repeated(dot_len, dot_count)?,
                        2 => cursor.advance_input_repeated(dot_len, dot_count)?,
                        _ => return Err(CircuitError::InvalidEinsum),
                    }
                }
            },
        }
        tensor_exprs.push(output_expr.to_string());
        tensor_lens.push(output_len);
    }
    let scalar_count = tensor_exprs.iter().zip(tensor_lens.iter()).filter(|(_, len)| **len == 1).count();
    cursor.advance_input(scalar_count)
}

/// Plan the native dummy `Einsums::assign_einsum` bookkeeping for a known GAT tensor shape.
pub fn plan_einsum_geometry(equation: &str, input_dims: &[Vec<usize>], output_dims: &[usize], block_width: usize, col_size: usize, start_einsum_col: usize) -> Result<EinsumGeometryPlan, CircuitError> {
    let (axes, _) = input_axes(equation, input_dims)?;
    let (input_expr, output_expr) = equation.split_once("->").ok_or(CircuitError::InvalidEinsum)?;
    let output_indices = output_expr.chars().collect::<Vec<_>>();
    let raw_output_dims = output_indices.iter().map(|axis| axes.get(axis).copied().ok_or(CircuitError::InvalidEinsum)).collect::<Result<Vec<_>, _>>()?;
    let raw_axes = axes.clone();
    let analysis = analyze_single_equation(equation, &axes)?;
    let compact_output_dims = analysis.output_indices.iter().map(|axis| axes.get(axis).copied().ok_or(CircuitError::InvalidEinsum)).collect::<Result<Vec<_>, _>>()?;
    // Native einsum removes trivial axes before assigning. The caller may therefore supply
    // either its original output shape or the compact shape used by `assign_einsum`.
    let output_shape_matches = output_dims == raw_output_dims || output_dims == compact_output_dims || (output_indices.is_empty() && output_dims == [1]);
    if !output_shape_matches { return Err(CircuitError::DimMismatch("einsum output shape".into())); }
    let uses_freivalds = matches!(&analysis.strategy, EinsumStrategy::Freivalds);
    let main_assignment_len = if uses_freivalds { product(&compact_output_dims)? } else { 0 };
    let mut cursor = EinsumCursor::new(start_einsum_col, block_width, col_size)?;
    if uses_freivalds {
        let compact_axes = analysis.output_indices.clone();
        let mut current_len = product(&compact_axes.iter().map(|axis| axes.get(axis).copied().unwrap_or(1)).collect::<Vec<_>>())?;
        for axis in compact_axes.iter().rev() {
            let rlc_len = *axes.get(axis).ok_or(CircuitError::InvalidEinsum)?;
            if rlc_len == 0 || current_len % rlc_len != 0 { return Err(CircuitError::InvalidEinsum); }
            cursor.advance_input_repeated(rlc_len, current_len / rlc_len)?;
            current_len /= rlc_len;
        }
        cursor.add(1)?;
        let normalized_input_dims = input_dims.iter().zip(input_expr.split(',')).map(|(dims, expr)| dims.iter().zip(expr.chars()).filter_map(|(dim, axis)| if axes.get(&axis).copied().unwrap_or(1) > 1 { Some(*dim) } else { None }).collect::<Vec<_>>()).collect::<Vec<_>>();
        reduce_cursor(&analysis.equation, &normalized_input_dims, &axes, &mut cursor)?;
    }
    let equation_metadata = (equation.to_string(), raw_axes);
    Ok(EinsumGeometryPlan { strategy: analysis.strategy, uses_freivalds, normalized_equation: analysis.equation, main_assignment_len, final_einsum_col: cursor.col, einsum_index_delta: 1, equation_metadata })
}
