//! Shape-only interpreter for the canonical GAT adapter's native layout.
//!
//! The interpreter follows GatOp::layout and its helpers without constructing
//! feature, parameter, witness, or intermediate value buffers. Its counters
//! are assignment geometry, not a substitute for Halo2's later configured
//! constraint-system census.

use std::collections::{HashMap, HashSet};

use halo2curves::bn256::Fr as Fp;
use halo2curves::ff::Field;

use crate::circuit::table::Range;
use crate::fieldutils::integer_rep_to_felt;
use crate::graph::{
    modules::{GraphModules, ModuleSizes},
    DynamicLookupParams, EinsumParams, GraphSettings, ShuffleParams, VarVisibility,
};
use crate::RunArgs;

use super::{
    builder::build_model_for_run_args,
    geometry_primitives::{native_dummy_input_cursor_end, plan_einsum_geometry},
    op::MAX_GAT_EDGE_CHUNK_CELLS,
    relation::RoundingMode,
    schema::{BoundFamily, CanonicalGat, ExactTable},
};

/// Reuse the bounded synthetic geometry and MockProver regressions from the
/// single root-crate formal test entrypoint.
#[cfg(test)]
pub(crate) fn run_tiny_settings_parity_fixture() {
    tests::run_shape_only_settings_match_the_native_tiny_multigraph_dummy_layout();
}

/// The witness-bearing two-graph fixture used by geometry parity and native
/// proof tests. Kept in one place so callers do not recreate its GAT relation.
#[cfg(test)]
pub(crate) fn tiny_multigraph_spec_for_tests() -> CanonicalGat {
    tests::tiny_multigraph_spec()
}

#[derive(Clone, Debug)]
struct ShapeTensor {
    dims: Vec<usize>,
    scale: i32,
    constants: HashSet<i128>,
}

impl ShapeTensor {
    fn unknown(dims: &[usize], scale: i32) -> Result<Self, String> {
        checked_product(dims)?;
        Ok(Self { dims: dims.to_vec(), scale, constants: HashSet::new() })
    }

    fn constant(dims: &[usize], scale: i32, values: impl IntoIterator<Item = i128>) -> Result<Self, String> {
        checked_product(dims)?;
        Ok(Self { dims: dims.to_vec(), scale, constants: values.into_iter().collect() })
    }

    fn len(&self) -> Result<usize, String> {
        checked_product(&self.dims)
    }
}

fn checked_product(dims: &[usize]) -> Result<usize, String> {
    dims.iter().try_fold(1usize, |total, dimension| total.checked_mul(*dimension))
        .ok_or_else(|| format!("GAT geometry shape product overflow: {:?}", dims))
}

fn checked_add(lhs: usize, rhs: usize, label: &str) -> Result<usize, String> {
    lhs.checked_add(rhs).ok_or_else(|| format!("GAT geometry {} overflow", label))
}

fn canonicalize_edge_channels(
    value: &mut ShapeTensor,
    edge_count: usize,
    channels: usize,
    label: &str,
) -> Result<(), String> {
    let expected_dims = if channels == 1 {
        vec![edge_count]
    } else {
        vec![edge_count, channels]
    };
    if value.dims == expected_dims {
        return Ok(());
    }
    if channels == 1 && value.len()? == edge_count && value.dims == [edge_count, 1] {
        value.dims = expected_dims;
        return Ok(());
    }
    Err(format!(
        "{} shape {:?} with {} cells is not canonical for {} edges and {} channels",
        label,
        value.dims,
        value.len()?,
        edge_count,
        channels,
    ))
}

fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, String> {
    let rank = lhs.len().max(rhs.len());
    let mut result = vec![1; rank];
    for offset in 0..rank {
        let left = lhs.get(lhs.len().wrapping_sub(offset + 1)).copied().unwrap_or(1);
        let right = rhs.get(rhs.len().wrapping_sub(offset + 1)).copied().unwrap_or(1);
        if left != right && left != 1 && right != 1 {
            return Err(format!("GAT geometry cannot broadcast {:?} with {:?}", lhs, rhs));
        }
        result[rank - offset - 1] = left.max(right);
    }
    Ok(result)
}

/// Native assignment ledgers for one witnessless GAT relation.
struct GeometryRegion {
    linear: usize,
    row: usize,
    num_inner_cols: usize,
    col_size: usize,
    dynamic_col: usize,
    dynamic_index: usize,
    max_dynamic_input_len: usize,
    einsum_col: usize,
    einsum_index: usize,
    equations: Vec<(String, HashMap<char, usize>)>,
    assigned_tables: HashMap<String, usize>,
    constants: HashSet<Fp>,
    ranges: HashSet<Range>,
}

impl GeometryRegion {
    fn new(run_args: &RunArgs) -> Result<Self, String> {
        let domain = 1usize.checked_shl(run_args.logrows)
            .ok_or_else(|| format!("GAT geometry logrows {} overflows usize", run_args.logrows))?;
        let col_size = domain.checked_sub(6)
            .ok_or_else(|| "GAT geometry domain has fewer than six reserved rows".to_string())?;
        if run_args.num_inner_cols == 0 || col_size == 0 {
            return Err("GAT geometry requires positive inner-column and native column sizes".into());
        }
        Ok(Self {
            linear: 0,
            row: 0,
            num_inner_cols: run_args.num_inner_cols,
            col_size,
            dynamic_col: 0,
            dynamic_index: 0,
            max_dynamic_input_len: 0,
            einsum_col: 0,
            einsum_index: 0,
            equations: Vec::new(),
            assigned_tables: HashMap::new(),
            constants: HashSet::new(),
            ranges: HashSet::new(),
        })
    }

    fn advance(&mut self, cells: usize) -> Result<(), String> {
        self.linear = checked_add(self.linear, cells, "main linear cursor")?;
        self.row = self.linear / self.num_inner_cols;
        Ok(())
    }

    fn flush_main(&mut self) -> Result<(), String> {
        let rem = self.linear % self.num_inner_cols;
        if rem != 0 {
            self.advance(self.num_inner_cols - rem)?;
        }
        Ok(())
    }

    fn record_constants(&mut self, tensor: &ShapeTensor) {
        self.constants.extend(tensor.constants.iter().map(|value| integer_rep_to_felt::<Fp>(*value)));
    }

    fn identity(&mut self, tensor: &ShapeTensor) -> Result<ShapeTensor, String> {
        self.record_constants(tensor);
        self.advance(tensor.len()?)?;
        ShapeTensor::unknown(&tensor.dims, tensor.scale)
    }

    fn direct_assignment(&mut self, tensor: &ShapeTensor) -> Result<ShapeTensor, String> {
        self.record_constants(tensor);
        self.advance(tensor.len()?)?;
        ShapeTensor::unknown(&tensor.dims, tensor.scale)
    }

    fn pairwise(&mut self, lhs: &ShapeTensor, rhs: &ShapeTensor) -> Result<ShapeTensor, String> {
        let dims = broadcast_shape(&lhs.dims, &rhs.dims)?;
        self.record_constants(lhs);
        self.record_constants(rhs);
        self.advance(checked_product(&dims)?)?;
        ShapeTensor::unknown(&dims, lhs.scale.max(rhs.scale))
    }

    fn equality(&mut self, lhs: &ShapeTensor, rhs: &ShapeTensor) -> Result<ShapeTensor, String> {
        let lhs_len = lhs.len()?;
        let rhs_len = rhs.len()?;
        if lhs_len != rhs_len {
            return Err(format!("GAT geometry equality length mismatch: {} vs {}", lhs_len, rhs_len));
        }
        self.record_constants(lhs);
        self.record_constants(rhs);
        self.advance(lhs_len)?;
        ShapeTensor::unknown(&rhs.dims, rhs.scale)
    }

    fn range_check(&mut self, value: &ShapeTensor, range: Range) -> Result<(), String> {
        if range.0 > range.1 {
            return Err(format!("invalid GAT geometry range {:?}", range));
        }
        self.ranges.insert(range);
        self.record_constants(value);
        self.advance(value.len()?)
    }

    fn boolean_identity(&mut self, value: &ShapeTensor) -> Result<ShapeTensor, String> {
        self.record_constants(value);
        self.advance(value.len()?)?;
        self.range_check(value, (0, 1))?;
        ShapeTensor::unknown(&value.dims, value.scale)
    }

    fn dynamic_lookup(&mut self, input: &ShapeTensor, table: &ExactTable) -> Result<ShapeTensor, String> {
        let identity = format!("{}:{}", table.table_id, table.digest);
        let table_index = if let Some(index) = self.assigned_tables.get(&identity).copied() {
            index
        } else {
            let table_len = table.rows.len();
            if table_len == 0 || table_len > self.col_size {
                return Err(format!(
                    "GAT exact table {} has {} rows, native column capacity is {}",
                    table.table_id, table_len, self.col_size
                ));
            }
            let index = self.dynamic_index;
            let flush = dynamic_column_flush(self.dynamic_col, table_len, self.col_size)?;
            self.constants.extend(table.rows.iter().flat_map(|(key, value)| {
                [integer_rep_to_felt::<Fp>(*key), integer_rep_to_felt::<Fp>(*value)]
            }));
            self.constants.insert(Fp::from(index as u64));
            self.dynamic_col = checked_add(
                self.dynamic_col,
                checked_add(table_len, flush, "dynamic table extent")?,
                "dynamic table cursor",
            )?;
            self.dynamic_index = checked_add(self.dynamic_index, 1, "dynamic table index")?;
            self.max_dynamic_input_len = self.max_dynamic_input_len.max(table_len);
            self.assigned_tables.insert(identity, index);
            index
        };
        self.record_constants(input);
        // Dynamic query assignment uses the cached table index. Query tensors
        // do not contribute to the table-column maximum-input statistic.
        self.constants.insert(Fp::from(table_index as u64));
        // `dynamic_lookup_from_table` assigns input, output, and repeated
        // table index into three parallel lookup-input columns at the same
        // main linear cursor, then advances that cursor once by query_len.
        let query_len = input.len()?;
        self.advance(query_len)?;
        ShapeTensor::unknown(&input.dims, table.output_scale)
    }

    fn einsum(
        &mut self,
        run_args: &RunArgs,
        equation: &str,
        inputs: &[&ShapeTensor],
        output_dims: &[usize],
    ) -> Result<ShapeTensor, String> {
        let input_dims = inputs.iter().map(|tensor| tensor.dims.clone()).collect::<Vec<_>>();
        let plan = plan_einsum_geometry(
            equation,
            &input_dims,
            output_dims,
            run_args.num_inner_cols,
            self.col_size,
            self.einsum_col,
        ).map_err(|error| format!("native einsum geometry: {:?}", error))?;
        self.equations.push(plan.equation_metadata.clone());
        for tensor in inputs {
            self.record_constants(tensor);
        }

        if run_args.disable_freivalds || !plan.uses_freivalds {
            let rows = checked_product(output_dims)?;
            let reduction = einsum_reduction_width(equation, &input_dims)?;
            self.base_dots(rows, reduction, inputs)?;
        } else {
            self.advance(plan.main_assignment_len)?;
            self.einsum_col = plan.final_einsum_col;
        }
        self.einsum_index = checked_add(self.einsum_index, plan.einsum_index_delta, "einsum index")?;
        ShapeTensor::unknown(output_dims, inputs.first().map_or(0, |input| input.scale))
    }

    fn base_dots(&mut self, rows: usize, reduction_width: usize, inputs: &[&ShapeTensor]) -> Result<(), String> {
        if inputs.len() != 2 || reduction_width == 0 {
            return Err("GAT BaseOps einsum does not match a nonempty two-input dot contraction".into());
        }
        let padded = reduction_width
            .checked_add(self.num_inner_cols - 1)
            .ok_or_else(|| "GAT dot input padding overflow".to_string())?
            / self.num_inner_cols
            * self.num_inner_cols;
        if padded != reduction_width {
            self.constants.insert(Fp::ZERO);
        }
        for input in inputs {
            self.record_constants(input);
        }
        if rows == 0 {
            return Ok(());
        }
        self.flush_main()?;
        self.linear = native_dummy_input_cursor_end(
            self.linear,
            self.num_inner_cols,
            self.col_size,
            padded,
            rows,
        ).map_err(|error| format!("native repeated dot geometry: {:?}", error))?;
        self.row = self.linear / self.num_inner_cols;
        Ok(())
    }
}

fn dynamic_column_flush(start: usize, len: usize, col_size: usize) -> Result<usize, String> {
    if len > col_size {
        return Err(format!("dynamic table length {} exceeds native column size {}", len, col_size));
    }
    let start_block = start / col_size;
    let end_block = checked_add(start, len, "dynamic table endpoint")? / col_size;
    if start_block != end_block {
        Ok(col_size - (start % col_size))
    } else {
        Ok(0)
    }
}

fn einsum_reduction_width(equation: &str, input_dims: &[Vec<usize>]) -> Result<usize, String> {
    let (input_expr, output_expr) = equation.split_once("->")
        .ok_or_else(|| format!("invalid GAT einsum equation {}", equation))?;
    let input_exprs = input_expr.split(',').collect::<Vec<_>>();
    if input_exprs.len() != input_dims.len() {
        return Err(format!("GAT einsum {} input count mismatch", equation));
    }
    let mut axes = HashMap::<char, usize>::new();
    for (expr, dims) in input_exprs.iter().zip(input_dims) {
        if expr.chars().count() != dims.len() {
            return Err(format!("GAT einsum {} rank mismatch", equation));
        }
        for (axis, dim) in expr.chars().zip(dims) {
            if let Some(previous) = axes.insert(axis, *dim) {
                if previous != *dim {
                    return Err(format!("GAT einsum {} axis {} dimension mismatch", equation, axis));
                }
            }
        }
    }
    axes.keys().copied().filter(|axis| !output_expr.contains(*axis)).try_fold(1usize, |width, axis| {
        width.checked_mul(axes[&axis])
            .ok_or_else(|| format!("GAT einsum {} reduction size overflow", equation))
    })
}

struct GatGeometry<'a> {
    spec: &'a CanonicalGat,
    run_args: &'a RunArgs,
    region: GeometryRegion,
    groups: Vec<(usize, usize, usize)>,
}

impl<'a> GatGeometry<'a> {
    fn new(spec: &'a CanonicalGat, run_args: &'a RunArgs) -> Result<Self, String> {
        spec.validate_for_compile()?;
        run_args.validate()?;
        let groups = spec.graph.destination_groups()?;
        Ok(Self { spec, run_args, region: GeometryRegion::new(run_args)?, groups })
    }

    fn unknown(&self, dims: &[usize], scale: i32) -> Result<ShapeTensor, String> {
        ShapeTensor::unknown(dims, scale)
    }

    fn constant(&self, dims: &[usize], scale: i32, values: impl IntoIterator<Item = i128>) -> Result<ShapeTensor, String> {
        ShapeTensor::constant(dims, scale, values)
    }

    fn constant_repeated(&self, dims: &[usize], scale: i32, value: i128) -> Result<ShapeTensor, String> {
        self.constant(dims, scale, [value])
    }

    fn bind_topology(&mut self, dims: &[usize], values: &[usize]) -> Result<(), String> {
        if checked_product(dims)? != values.len() {
            return Err(format!("GAT topology values length {} does not match {:?}", values.len(), dims));
        }
        let public = self.unknown(dims, 0)?;
        let assigned = self.region.identity(&public)?;
        let expected = self.constant(dims, 0, values.iter().map(|value| *value as i128))?;
        self.region.equality(&assigned, &expected)?;
        Ok(())
    }

    fn exact_lookup(&mut self, input: &ShapeTensor, table: &ExactTable) -> Result<ShapeTensor, String> {
        self.region.dynamic_lookup(input, table)
    }

    fn bounded_range(&mut self, value: &ShapeTensor, range: Range) -> Result<(), String> {
        if range.0 > range.1 {
            return Err(format!("invalid canonical GAT range {:?}", range));
        }
        let span = range.1.checked_sub(range.0).ok_or_else(|| "GAT bounded range span overflow".to_string())?;
        if span <= 1023 {
            return self.region.range_check(value, range);
        }
        let significant_bits = 128 - (span as u128).leading_zeros() as usize;
        let limbs = significant_bits.checked_add(7).ok_or_else(|| "GAT range limb count overflow".to_string())? / 8;
        let no_wrap_bits = limbs.checked_mul(8).and_then(|bits| bits.checked_add(1))
            .ok_or_else(|| "GAT range limb width overflow".to_string())?;
        if limbs == 0 || limbs > 15 || no_wrap_bits > 253 {
            return Err("GAT bounded range cannot be represented by the canonical limb relation".into());
        }
        let len = value.len()?;
        let lower = self.constant_repeated(&value.dims, value.scale, range.0)?;
        let upper = self.constant_repeated(&value.dims, value.scale, range.1)?;
        let lower_diff = self.region.pairwise(value, &lower)?;
        let upper_diff = self.region.pairwise(&upper, value)?;
        let (lower_parts, lower_value) = self.decompose(&lower_diff, 256, limbs, true)?;
        let (upper_parts, upper_value) = self.decompose(&upper_diff, 256, limbs, true)?;
        let lower_sign = self.unknown(&value.dims, value.scale)?;
        let upper_sign = self.unknown(&value.dims, value.scale)?;
        let _ = (lower_parts, upper_parts);
        self.region.range_check(&lower_sign, (0, 1))?;
        self.region.range_check(&upper_sign, (0, 1))?;
        let difference_sum = self.region.pairwise(&lower_value, &upper_value)?;
        let span_constant = self.constant_repeated(&value.dims, value.scale, span)?;
        self.region.equality(&difference_sum, &span_constant)?;
        if lower_value.len()? != len {
            return Err("GAT bounded range decomposition changed its tensor shape".into());
        }
        Ok(())
    }

    fn equals_zero(&mut self, value: &ShapeTensor) -> Result<ShapeTensor, String> {
        let inverse = self.unknown(&value.dims, value.scale)?;
        let product = self.region.pairwise(value, &inverse)?;
        let one = self.constant_repeated(&value.dims, value.scale, 1)?;
        let is_zero = self.region.pairwise(&one, &product)?;
        let prod_check = self.region.pairwise(value, &is_zero)?;
        let zeros = self.constant_repeated(&value.dims, value.scale, 0)?;
        self.region.equality(&prod_check, &zeros)?;
        Ok(is_zero)
    }

    fn decompose(&mut self, value: &ShapeTensor, base: usize, legs: usize, zero_sign_matters: bool) -> Result<(ShapeTensor, ShapeTensor), String> {
        if base < 2 {
            return Err("GAT decomposition base must be at least two".into());
        }
        let len = value.len()?;
        self.region.record_constants(value);
        let mut claimed_dims = value.dims.clone();
        claimed_dims.push(legs.checked_add(1).ok_or_else(|| "GAT decomposition shape overflow".to_string())?);
        let claimed = self.region.direct_assignment(&self.unknown(&claimed_dims, value.scale)?)?;

        let sign = self.unknown(&value.dims, value.scale)?;
        self.region.range_check(&sign, (-1, 1))?;
        if zero_sign_matters {
            let is_zero = self.equals_zero(value)?;
            let sign_zero = self.region.pairwise(&sign, &is_zero)?;
            let zeros = self.constant_repeated(&value.dims, value.scale, 0)?;
            self.region.equality(&sign_zero, &zeros)?;
        }

        let mut rest_dims = value.dims.clone();
        rest_dims.push(legs);
        let rest = self.unknown(&rest_dims, value.scale)?;
        self.region.range_check(&rest, (0, i128::try_from(base - 1).map_err(|_| "GAT decomposition base does not fit i128")?))?;
        let mut powers = HashSet::new();
        for power in (0..legs).rev() {
            let exponent = u32::try_from(power).map_err(|_| "GAT decomposition exponent overflow")?;
            powers.insert((base as i128).checked_pow(exponent).ok_or_else(|| "GAT decomposition base power overflow".to_string())?);
        }
        let bases = self.constant(&rest_dims, value.scale, powers)?;
        let alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
        let rank = rest_dims.len();
        if rank == 0 || rank > alphabet.len() {
            return Err("GAT decomposition rank exceeds native einsum alphabet".into());
        }
        let lhs = alphabet.chars().take(rank).collect::<String>();
        let rhs = alphabet.chars().take(rank - 1).collect::<String>();
        let equation = format!("{},{}->{}", lhs, lhs, rhs);
        let prod_decomp = self.region.einsum(self.run_args, &equation, &[&rest, &bases], &value.dims)?;
        let signed = self.region.pairwise(&prod_decomp, &sign)?;
        self.region.equality(value, &signed)?;
        if len != checked_product(&value.dims)? {
            return Err("GAT decomposition geometry length drift".into());
        }
        Ok((claimed, value.clone()))
    }

    fn inverse_exact(&mut self, denominator: &ShapeTensor) -> Result<ShapeTensor, String> {
        let inverse = self.region.direct_assignment(&self.unknown(&denominator.dims, denominator.scale)?)?;
        let product = self.region.pairwise(denominator, &inverse)?;
        let one = self.constant_repeated(&denominator.dims, denominator.scale, 1)?;
        self.region.equality(&product, &one)?;
        Ok(inverse)
    }

    fn signed_round_div(
        &mut self,
        dividend: &ShapeTensor,
        divisor: &ShapeTensor,
        rounding: RoundingMode,
        bounds: Option<&BoundFamily>,
    ) -> Result<ShapeTensor, String> {
        if dividend.dims.is_empty() || dividend.len()? == 0 {
            return Err("GAT division requires a nonempty dividend".into());
        }
        let destination_count = dividend.dims[0];
        if divisor.len()? != 1 && divisor.dims.as_slice() != [destination_count] && divisor.dims != dividend.dims {
            return Err(format!("GAT divisor shape {:?} does not broadcast to {:?}", divisor.dims, dividend.dims));
        }

        let q = self.region.direct_assignment(&self.unknown(&dividend.dims, dividend.scale)?)?;
        let q_floor = self.region.direct_assignment(&self.unknown(&dividend.dims, dividend.scale)?)?;
        let rem = self.region.direct_assignment(&self.unknown(&dividend.dims, dividend.scale)?)?;
        let abs = self.region.direct_assignment(&self.unknown(&dividend.dims, dividend.scale)?)?;
        let sigma = self.region.direct_assignment(&self.unknown(&dividend.dims, 0)?)?;
        let zero = self.region.direct_assignment(&self.unknown(&dividend.dims, 0)?)?;
        let rounded_up = self.region.direct_assignment(&self.unknown(&dividend.dims, 0)?)?;
        let sigma = self.region.boolean_identity(&sigma)?;
        let zero = self.region.boolean_identity(&zero)?;
        let rounded_up = self.region.boolean_identity(&rounded_up)?;

        let two = self.constant_repeated(&dividend.dims, 0, 2)?;
        let two_sigma = self.region.pairwise(&sigma, &two)?;
        let one = self.constant_repeated(&dividend.dims, 0, 1)?;
        let sign_factor = self.region.pairwise(&one, &two_sigma)?;
        let signed_abs = self.region.pairwise(&sign_factor, &abs)?;
        self.region.equality(dividend, &signed_abs)?;
        let q_abs = self.region.pairwise(&q_floor, &rounded_up)?;
        let signed_q = self.region.pairwise(&sign_factor, &q_abs)?;
        self.region.equality(&q, &signed_q)?;

        let max_divisor = bounds.map_or(self.spec.ledger.max_divisor, |family| family.max_divisor);
        let max_aux_abs = bounds.map_or(self.spec.ledger.max_aux_abs, |family| family.max_aux_abs);
        let max_dividend_abs = bounds.map_or(self.spec.ledger.max_dividend_abs, |family| family.max_dividend_abs);
        self.bounded_range(&q_abs, (0, max_dividend_abs))?;
        self.bounded_range(&q, (-max_dividend_abs, max_dividend_abs))?;
        self.bounded_range(&sign_factor, (-1, 1))?;
        self.bounded_range(divisor, (1, max_divisor))?;

        let d_inv = self.inverse_exact(divisor)?;
        let repeated_divisor = self.unknown(&dividend.dims, divisor.scale)?;
        let repeated_inverse = self.unknown(&dividend.dims, d_inv.scale)?;
        let d_product = self.region.pairwise(&repeated_divisor, &repeated_inverse)?;
        let d_one = self.constant_repeated(&dividend.dims, dividend.scale, 1)?;
        self.region.equality(&d_product, &d_one)?;

        let abs_inv = self.region.direct_assignment(&self.unknown(&dividend.dims, dividend.scale)?)?;
        let abs_product = self.region.pairwise(&abs, &abs_inv)?;
        let one_minus_zero = self.region.pairwise(&one, &zero)?;
        self.region.equality(&abs_product, &one_minus_zero)?;
        let zero_from_abs = self.equals_zero(&abs)?;
        self.region.equality(&zero, &zero_from_abs)?;
        let zero_constant = self.constant_repeated(&dividend.dims, dividend.scale, 0)?;
        for checked in [&abs, &sigma, &abs_inv, &q_abs] {
            let product = self.region.pairwise(&zero, checked)?;
            self.region.equality(&product, &zero_constant)?;
        }

        let d_minus_one = self.region.pairwise(&repeated_divisor, &one)?;
        let delta = self.region.direct_assignment(&self.unknown(&dividend.dims, dividend.scale)?)?;
        match rounding {
            RoundingMode::TowardZero => {
                let dq_floor = self.region.pairwise(&repeated_divisor, &q_floor)?;
                let abs_reconstructed = self.region.pairwise(&dq_floor, &rem)?;
                self.region.equality(&abs, &abs_reconstructed)?;
                let r_delta = self.region.pairwise(&rem, &delta)?;
                self.region.equality(&r_delta, &d_minus_one)?;
            }
            RoundingMode::NearestTiesAway => {
                let two_abs = self.region.pairwise(&two, &abs)?;
                let lhs = self.region.pairwise(&two_abs, &repeated_divisor)?;
                let dq_abs = self.region.pairwise(&repeated_divisor, &q_abs)?;
                let two_dq_abs = self.region.pairwise(&two, &dq_abs)?;
                let rhs = self.region.pairwise(&two_dq_abs, &rem)?;
                self.region.equality(&lhs, &rhs)?;

                let two_d = self.region.pairwise(&two, &repeated_divisor)?;
                let two_d_minus_one = self.region.pairwise(&two_d, &one)?;
                let r_delta = self.region.pairwise(&rem, &delta)?;
                self.region.equality(&r_delta, &two_d_minus_one)?;

                let dq_floor = self.region.pairwise(&repeated_divisor, &q_floor)?;
                let floor_rem = self.region.pairwise(&abs, &dq_floor)?;
                let two_floor_rem = self.region.pairwise(&two, &floor_rem)?;
                let low = self.region.pairwise(&d_minus_one, &two_floor_rem)?;
                let high = self.region.pairwise(&two_floor_rem, &repeated_divisor)?;
                let one_minus_round = self.region.pairwise(&one, &rounded_up)?;
                let low_selected = self.region.pairwise(&one_minus_round, &low)?;
                let high_selected = self.region.pairwise(&rounded_up, &high)?;
                let selected = self.region.pairwise(&low_selected, &high_selected)?;
                self.bounded_range(&floor_rem, (0, max_aux_abs))?;
                self.bounded_range(&selected, (0, max_aux_abs))?;
            }
        }
        self.bounded_range(&rem, (0, max_aux_abs))?;
        self.bounded_range(&delta, (0, max_aux_abs))?;
        self.bounded_range(&q_floor, (0, max_dividend_abs))?;
        if matches!(rounding, RoundingMode::TowardZero) {
            let zero_round = self.constant_repeated(&dividend.dims, dividend.scale, 0)?;
            self.region.equality(&rounded_up, &zero_round)?;
        }
        self.bounded_range(&abs, (0, max_dividend_abs))?;
        Ok(q)
    }

    fn segmented_prefix(&mut self, values: &ShapeTensor, groups: &[(usize, usize, usize)], channels: usize) -> Result<ShapeTensor, String> {
        let edge_count = values.dims.first().copied().unwrap_or(0);
        let expected = if channels == 1 { vec![edge_count] } else { vec![edge_count, channels] };
        if channels == 0 || values.dims != expected {
            return Err(format!("GAT prefix shape {:?} is not {:?}", values.dims, expected));
        }
        validate_groups(groups, edge_count)?;
        let claimed = self.region.direct_assignment(&self.unknown(&values.dims, values.scale)?)?;
        let previous = ShapeTensor::constant(&values.dims, values.scale, [0])?;
        let recurrence = self.region.pairwise(&previous, values)?;
        self.region.equality(&recurrence, &claimed)?;
        Ok(claimed)
    }

    fn segmented_prefix_chunk(
        &mut self,
        values: &ShapeTensor,
        groups: &[(usize, usize, usize)],
        chunk_start: usize,
        edge_count: usize,
        channels: usize,
    ) -> Result<ShapeTensor, String> {
        let chunk_edges = values.dims.first().copied().unwrap_or(0);
        let chunk_end = checked_add(chunk_start, chunk_edges, "prefix chunk end")?;
        let expected = if channels == 1 { vec![chunk_edges] } else { vec![chunk_edges, channels] };
        if channels == 0 || chunk_edges == 0 || values.dims != expected || chunk_end > edge_count {
            return Err(format!("invalid GAT prefix chunk {:?} at {} for {} edges", values.dims, chunk_start, edge_count));
        }
        validate_groups(groups, edge_count)?;
        let claimed = self.region.direct_assignment(&self.unknown(&values.dims, values.scale)?)?;
        let has_group_start = groups.iter().any(|(_, start, _)| *start >= chunk_start && *start < chunk_end);
        let previous = if has_group_start {
            ShapeTensor::constant(&values.dims, values.scale, [0])?
        } else {
            self.unknown(&values.dims, values.scale)?
        };
        let recurrence = self.region.pairwise(&previous, values)?;
        self.region.equality(&recurrence, &claimed)?;
        Ok(claimed)
    }

    fn destination_endpoints(&self, values: &ShapeTensor, groups: &[(usize, usize, usize)], channels: usize) -> Result<ShapeTensor, String> {
        validate_groups(groups, self.spec.graph.e)?;
        let dims = if channels == 1 { vec![groups.len()] } else { vec![groups.len(), channels] };
        self.unknown(&dims, values.scale)
    }

    fn aggregate_head(
        &mut self,
        projected: &ShapeTensor,
        attention_src: &ShapeTensor,
        attention_dst: &ShapeTensor,
        channels: usize,
        use_elu: bool,
        bounds: &BoundFamily,
        recovery_scale: i32,
        elu_output_scale: i32,
        projected_scale: i32,
    ) -> Result<ShapeTensor, String> {
        let graph = &self.spec.graph;
        // The interpreter methods below mutate their counters. Keep the
        // validated per-destination schedule in a small metadata-only local
        // so those calls do not borrow `self.groups` across `&mut self`.
        let groups = self.groups.clone();
        if projected.dims != [graph.n, channels]
            || attention_src.dims != [channels]
            || attention_dst.dims != [channels]
            || channels == 0
            || channels > MAX_GAT_EDGE_CHUNK_CELLS
            || groups.len() != graph.n
        {
            return Err("GAT geometry received noncanonical projected or attention shapes".into());
        }
        let max_degree = groups.iter().map(|(_, start, end)| end - start).max()
            .ok_or_else(|| "GAT graph has no destination groups".to_string())?;
        // Mirror the live circuit's authenticated integer lift: Dmax*max_exp
        // is ledger-bounded, the raw projection receives the checked quotient
        // bound, and only destination endpoints receive explicit numerator
        // range checks.
        let projected_bound = bounds.projected_abs_bound(max_degree)?;
        if projected_bound == 0 {
            let zero = self.constant_repeated(&projected.dims, projected.scale, 0)?;
            self.region.equality(projected, &zero)?;
        } else {
            self.bounded_range(
                projected,
                (projected_bound.checked_neg().ok_or("projected bound negation overflow")?, projected_bound),
            )?;
        }
        let edge_chunk_rows = (MAX_GAT_EDGE_CHUNK_CELLS / channels).max(1);
        let mut source_node_score = self.region.einsum(self.run_args, "ij,j->i", &[projected, attention_src], &[graph.n])?;
        let mut destination_node_score = self.region.einsum(self.run_args, "ij,j->i", &[projected, attention_dst], &[graph.n])?;
        source_node_score.scale = self.spec.scales.s_delta;
        destination_node_score.scale = self.spec.scales.s_delta;

        for chunk_start in (0..graph.e).step_by(edge_chunk_rows) {
            let chunk_len = graph.e.saturating_sub(chunk_start).min(edge_chunk_rows);
            let routed_src = self.unknown(&[chunk_len], 0)?;
            let routed_dst = self.unknown(&[chunk_len], 0)?;
            let score_chunk = self.region.pairwise(&routed_src, &routed_dst)?;
            self.bounded_range(&score_chunk, (-bounds.max_score_abs, bounds.max_score_abs))?;
            let score_chunk = self.exact_lookup(&score_chunk, &self.spec.tables.lrelu)?;
            self.bounded_range(&score_chunk, (-bounds.max_table_output_abs, bounds.max_table_output_abs))?;
        }
        let score = self.unknown(&[graph.e], self.spec.scales.s_delta)?;

        let max_claim = self.region.direct_assignment(&self.unknown(&[graph.n], score.scale)?)?;
        self.bounded_range(&max_claim, (-bounds.max_table_output_abs, bounds.max_table_output_abs))?;
        let max_repeated = self.unknown(&[graph.e], score.scale)?;
        let one_edge = self.constant_repeated(&[graph.e], score.scale, 1)?;
        let max_minus_one = self.region.pairwise(&max_repeated, &one_edge)?;
        let max_test_input = self.region.pairwise(&score, &max_minus_one)?;
        let max_indicator = self.exact_lookup(&max_test_input, &self.spec.tables.max_relu)?;
        let max_indicator = self.region.boolean_identity(&max_indicator)?;
        let max_prefix = self.segmented_prefix(&max_indicator, &groups, 1)?;
        let max_indicator_sum = self.destination_endpoints(&max_prefix, &groups, 1)?;
        let max_degree = i128::try_from(max_degree).map_err(|_| "GAT max degree overflows i128")?;
        self.bounded_range(&max_indicator_sum, (0, max_degree))?;
        let one_node = self.constant_repeated(&[graph.n], score.scale, 1)?;
        let max_deficit = self.region.pairwise(&one_node, &max_indicator_sum)?;
        let max_deficit_relu = self.exact_lookup(&max_deficit, &self.spec.tables.max_relu)?;
        let zero_node = self.constant_repeated(&[graph.n], score.scale, 0)?;
        self.region.equality(&max_deficit_relu, &zero_node)?;

        let delta = self.region.pairwise(&max_repeated, &score)?;
        self.bounded_range(&delta, (0, bounds.max_delta))?;
        let exp = self.exact_lookup(&delta, &self.spec.tables.exp)?;
        self.bounded_range(&exp, (1, bounds.max_exp))?;
        let denominator_prefix = self.segmented_prefix(&exp, &groups, 1)?;
        let denominator = self.destination_endpoints(&denominator_prefix, &groups, 1)?;
        self.bounded_range(&denominator, (1, bounds.max_sum))?;
        let inverse = self.inverse_exact(&denominator)?;
        let alpha = self.region.pairwise(&exp, &self.unknown(&[graph.e], inverse.scale)?)?;

        let edge_chunks = graph.e.checked_add(edge_chunk_rows - 1)
            .ok_or_else(|| "GAT aggregation chunk count overflow".to_string())? / edge_chunk_rows;
        for chunk_start in (0..graph.e).step_by(edge_chunk_rows) {
            let chunk_rows = graph.e.saturating_sub(chunk_start).min(edge_chunk_rows);
            let edge_channels = self.unknown(&[chunk_rows, channels], projected_scale)?;
            let alpha_rows = self.unknown(&[chunk_rows, channels], alpha.scale)?;
            let mut field_product = self.region.pairwise(&alpha_rows, &edge_channels)?;
            canonicalize_edge_channels(
                &mut field_product,
                chunk_rows,
                channels,
                "GAT field-product chunk",
            )?;
            let _field_prefix = self.segmented_prefix_chunk(&field_product, &groups, chunk_start, graph.e, channels)?;

            let exp_rows = self.unknown(&[chunk_rows, channels], exp.scale)?;
            let mut numerator_product = self.region.pairwise(&exp_rows, &edge_channels)?;
            canonicalize_edge_channels(
                &mut numerator_product,
                chunk_rows,
                channels,
                "GAT numerator-product chunk",
            )?;
            let _numerator_prefix = self.segmented_prefix_chunk(&numerator_product, &groups, chunk_start, graph.e, channels)?;
        }
        if edge_chunks == 0 {
            return Err("GAT aggregation requires at least one edge chunk".into());
        }

        let field_endpoints = self.unknown(&[graph.n, channels], projected_scale)?;
        let numerator_endpoints = self.unknown(
            &[graph.n, channels],
            self.spec.scales.s_exp.checked_add(projected_scale).ok_or_else(|| "GAT aggregation scale overflow".to_string())?,
        )?;
        self.bounded_range(
            &numerator_endpoints,
            (
                bounds.max_aggregation_num.checked_neg().ok_or("aggregation numerator bound negation overflow")?,
                bounds.max_aggregation_num,
            ),
        )?;
        let inverse_rows = self.unknown(&[graph.n, channels], inverse.scale)?;
        let field_from_numerator = self.region.pairwise(&numerator_endpoints, &inverse_rows)?;
        self.region.equality(&field_endpoints, &field_from_numerator)?;

        let recovery_multiplier = scale_exponent_to_multiplier(recovery_scale)?;
        let denominator_multiplier = scale_exponent_to_multiplier(projected_scale)?;
        let numerator_scale = self.constant_repeated(&[graph.n, channels], 0, recovery_multiplier)?;
        let scaled_numerator = self.region.pairwise(&numerator_endpoints, &numerator_scale)?;
        let denominator_scale = self.constant_repeated(&[graph.n], 0, denominator_multiplier)?;
        let scaled_denominator = self.region.pairwise(&denominator, &denominator_scale)?;
        self.signed_round_div(
            &scaled_numerator,
            &scaled_denominator,
            RoundingMode::NearestTiesAway,
            Some(bounds),
        )?;
        if use_elu {
            self.exact_lookup(&self.unknown(&[graph.n, channels], recovery_scale)?, &self.spec.tables.elu)
        } else {
            let _ = elu_output_scale;
            self.unknown(&[graph.n, channels], self.spec.scales.s_out_pre)
        }
    }

    fn run(&mut self) -> Result<(), String> {
        let graph = &self.spec.graph;
        // Model::layout_nodes first runs six Input operations and one flat
        // runtime-parameter Input.  GatOp then independently assigns its
        // feature and topology values again, and assigns each sliced private
        // tensor before use.  Keep these two layers of native assignments
        // distinct in the geometry ledger.
        for (dims, scale) in [
            (vec![graph.n_total, self.spec.feature_dim], self.spec.scales.s_input),
            (vec![graph.n], 0),
            (vec![graph.e], 0),
            (vec![graph.e], 0),
            (vec![graph.g_batch + 1], 0),
            (vec![graph.g_batch + 1], 0),
        ] {
            self.region.identity(&self.unknown(&dims, scale)?)?;
        }
        let flat_parameter_len = self.spec.model.flattened_params_len()?;
        if flat_parameter_len == 0 {
            return Err("GAT model has no flattened runtime parameters".into());
        }
        self.region.identity(&self.unknown(
            &[flat_parameter_len],
            self.spec.scales.s_w_hidden[0][0],
        )?)?;

        let feature_input = self.unknown(&[graph.n_total, self.spec.feature_dim], self.spec.scales.s_input)?;
        self.region.identity(&feature_input)?;
        self.bind_topology(&[graph.n], &graph.node_ids)?;
        self.bind_topology(&[graph.e], &graph.src)?;
        self.bind_topology(&[graph.e], &graph.dst)?;
        self.bind_topology(&[graph.g_batch + 1], &graph.node_ptr)?;
        self.bind_topology(&[graph.g_batch + 1], &graph.edge_ptr)?;

        let mut current_dim = self.spec.feature_dim;
        let mut current = self.unknown(&[graph.n, current_dim], self.spec.scales.s_h[0])?;
        for (layer_idx, layer) in self.spec.model.hidden_layers.iter().enumerate() {
            if layer.d_in != current_dim {
                return Err(format!("GAT layer {} input dimension {} does not match {}", layer_idx, layer.d_in, current_dim));
            }
            let mut heads = Vec::with_capacity(layer.heads.len());
            for (head_idx, _) in layer.heads.iter().enumerate() {
                let bounds = self.bound_family(layer_idx, head_idx, false)?.clone();
                let projection = self.region.identity(&self.unknown(
                    &[layer.d_in, layer.d_head],
                    self.spec.scales.s_w_hidden[layer_idx][head_idx],
                )?)?;
                let attention_src = self.region.identity(&self.unknown(
                    &[layer.d_head],
                    self.spec.scales.s_a_src_hidden[layer_idx][head_idx],
                )?)?;
                let attention_dst = self.region.identity(&self.unknown(
                    &[layer.d_head],
                    self.spec.scales.s_a_dst_hidden[layer_idx][head_idx],
                )?)?;
                let mut projected = self.region.einsum(
                    self.run_args,
                    "ij,jk->ik",
                    &[&current, &projection],
                    &[graph.n, layer.d_head],
                )?;
                projected.scale = self.spec.scales.s_h_prime[layer_idx][head_idx];
                let aggregate = self.aggregate_head(
                    &projected,
                    &attention_src,
                    &attention_dst,
                    layer.d_head,
                    true,
                    &bounds,
                    self.spec.scales.s_agg_pre[layer_idx],
                    self.spec.scales.s_agg[layer_idx],
                    self.spec.scales.s_h_prime[layer_idx][head_idx],
                )?;
                heads.push(aggregate);
            }
            current_dim = layer.heads.len().checked_mul(layer.d_head)
                .ok_or_else(|| "GAT hidden layer dimension overflow".to_string())?;
            current = self.unknown(&[graph.n, current_dim], self.spec.scales.s_h[layer_idx + 1])?;
        }

        for (head_idx, _) in self.spec.model.output_heads.iter().enumerate() {
            let bounds = self.bound_family(usize::MAX, head_idx, true)?.clone();
            let projection = self.region.identity(&self.unknown(
                &[current_dim, self.spec.model.classes],
                self.spec.scales.s_w_output[head_idx],
            )?)?;
            let attention_src = self.region.identity(&self.unknown(
                &[self.spec.model.classes],
                self.spec.scales.s_a_src_output[head_idx],
            )?)?;
            let attention_dst = self.region.identity(&self.unknown(
                &[self.spec.model.classes],
                self.spec.scales.s_a_dst_output[head_idx],
            )?)?;
            let bias = self.region.identity(&self.unknown(
                &[self.spec.model.classes],
                self.spec.scales.s_b_output[head_idx],
            )?)?;
            let mut projected = self.region.einsum(
                self.run_args,
                "ij,jk->ik",
                &[&current, &projection],
                &[graph.n, self.spec.model.classes],
            )?;
            projected.scale = self.spec.scales.s_y_prime[head_idx];
            let aggregated = self.aggregate_head(
                &projected,
                &attention_src,
                &attention_dst,
                self.spec.model.classes,
                false,
                &bounds,
                self.spec.scales.s_out_pre,
                self.spec.scales.s_out_pre,
                self.spec.scales.s_y_prime[head_idx],
            )?;

            let head_multiplier = scale_exponent_to_multiplier(self.spec.scales.s_out_head)?;
            let pre_multiplier = scale_exponent_to_multiplier(self.spec.scales.s_out_pre)?;
            let scale_constant = self.constant_repeated(&[graph.n, self.spec.model.classes], 0, head_multiplier)?;
            let aligned_dividend = self.region.pairwise(&aggregated, &scale_constant)?;
            let divisor = self.constant(&[1], 0, [pre_multiplier])?;
            self.signed_round_div(&aligned_dividend, &divisor, RoundingMode::NearestTiesAway, Some(&bounds))?;
            let _ = self.region.pairwise(
                &self.unknown(&[graph.n, self.spec.model.classes], self.spec.scales.s_out_head)?,
                &self.unknown(&[graph.n, self.spec.model.classes], bias.scale)?,
            )?;
        }

        let output_heads = self.spec.model.output_heads.len();
        if output_heads == 0 {
            return Err("GAT model must contain at least one output attention head".into());
        }
        let output_dims = [graph.n, self.spec.model.classes];
        let mut output = self.unknown(&output_dims, self.spec.scales.s_out_head)?;
        for _ in 1..output_heads {
            output = self.region.pairwise(&output, &self.unknown(&output_dims, self.spec.scales.s_out_head)?)?;
        }
        let divisor = self.constant(&[1], 0, [i128::try_from(output_heads).map_err(|_| "GAT output head count does not fit i128")?])?;
        let output = self.signed_round_div(&output, &divisor, RoundingMode::NearestTiesAway, None)?;

        if !self.run_args.ignore_range_check_inputs_outputs {
            let output = self.decompose(
                &output,
                self.run_args.decomp_base,
                self.run_args.decomp_legs,
                false,
            )?.1;
            let comparator = self.unknown(&output_dims, output.scale)?;
            let comparator = self.decompose(
                &comparator,
                self.run_args.decomp_base,
                self.run_args.decomp_legs,
                false,
            )?.1;
            self.region.equality(&output, &comparator)?;
        } else {
            // GraphCircuit's public-output module assigns the external
            // comparator before enforcing equality.  The GAT result is
            // already assigned by its final signed division.
            let comparator = self.region.direct_assignment(&self.unknown(&output_dims, output.scale)?)?;
            self.region.equality(&output, &comparator)?;
        }
        Ok(())
    }

    fn bound_family(&self, layer: usize, head: usize, is_output: bool) -> Result<&BoundFamily, String> {
        self.spec.ledger.families.iter()
            .find(|family| family.layer == layer && family.head == head && family.is_output == is_output)
            .ok_or_else(|| format!("missing GAT geometry bound family layer={} head={} output={}", layer, head, is_output))
    }
}

fn validate_groups(groups: &[(usize, usize, usize)], edge_count: usize) -> Result<(), String> {
    let mut cursor = 0usize;
    for (expected_node, (node, start, end)) in groups.iter().copied().enumerate() {
        if node != expected_node || start != cursor || start >= end || end > edge_count {
            return Err("GAT geometry destination groups are not a contiguous graph partition".into());
        }
        cursor = end;
    }
    if cursor != edge_count {
        return Err("GAT geometry destination groups do not cover the graph edges".into());
    }
    Ok(())
}

fn scale_exponent_to_multiplier(scale: i32) -> Result<i128, String> {
    super::schema::scale_exponent_to_multiplier(scale)
        .and_then(|value| i128::try_from(value).map_err(|_| "GAT scale multiplier does not fit i128".into()))
}

/// Build full GraphSettings from the canonical GAT shape and real topology
/// without materializing any full feature or model-value tensor.
pub fn gat_settings_for_run_args(spec: &CanonicalGat, run_args: &RunArgs) -> Result<GraphSettings, String> {
    let mut geometry = GatGeometry::new(spec, run_args)?;
    geometry.run()?;
    let model = build_model_for_run_args(spec.clone(), RoundingMode::NearestTiesAway, run_args)?;
    let parameter_shapes = model.graph.param_shapes().map_err(|error| format!("GAT geometry parameter shapes: {:?}", error))?;
    let num_params = parameter_shapes.iter().try_fold(0usize, |total, shape| {
        let len = checked_product(shape)?;
        total.checked_add(len).ok_or_else(|| "GAT geometry runtime parameter size overflow".to_string())
    })?;
    let input_shapes = model.graph.input_shapes().map_err(|error| format!("GAT geometry input shapes: {:?}", error))?;
    let output_shapes = model.graph.output_shapes().map_err(|error| format!("GAT geometry output shapes: {:?}", error))?;
    let module_sizes: ModuleSizes = GraphModules::num_constraints_and_instances(
        input_shapes,
        vec![vec![num_params]],
        output_shapes,
        VarVisibility::from_args(run_args).map_err(|error| format!("GAT geometry visibility: {:?}", error))?,
    );
    let settings = GraphSettings {
        run_args: run_args.clone(),
        num_rows: geometry.region.row.max(module_sizes.max_constraints()),
        total_assignments: geometry.region.linear,
        total_const_size: geometry.region.constants.len(),
        dynamic_lookup_params: DynamicLookupParams {
            total_dynamic_col_size: geometry.region.dynamic_col,
            max_dynamic_input_len: geometry.region.max_dynamic_input_len,
            num_dynamic_lookups: geometry.region.dynamic_index,
        },
        shuffle_params: ShuffleParams::default(),
        einsum_params: EinsumParams {
            equations: geometry.region.equations,
            total_einsum_col_size: geometry.region.einsum_col,
        },
        model_instance_shapes: model.instance_shapes().map_err(|error| format!("GAT geometry instance shapes: {:?}", error))?,
        model_output_scales: model.graph.get_output_scales().map_err(|error| format!("GAT geometry output scales: {:?}", error))?,
        model_input_scales: model.graph.get_input_scales(),
        module_sizes,
        // Every nonlinearity in this GAT relation is an authenticated dynamic
        // exact-table lookup; no static LookupOp is configured by GatOp.
        required_lookups: Vec::new(),
        required_range_checks: {
            let mut ranges = geometry.region.ranges.into_iter().collect::<Vec<_>>();
            ranges.sort_unstable();
            ranges
        },
        check_mode: run_args.check_mode,
        version: env!("CARGO_PKG_VERSION").to_string(),
        num_blinding_factors: None,
        timestamp: Some(instant::SystemTime::now().duration_since(instant::SystemTime::UNIX_EPOCH)
            .map_err(|error| format!("GAT geometry timestamp: {:?}", error))?.as_millis()),
        input_types: model.get_input_types().ok(),
        output_types: Some(model.get_output_types()),
    };
    Ok(settings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gat_adapter::schema::{
        sha256_digest, BoundFamily, CanonicalGraph, CanonicalModel, DigestIdentity, ExactTable,
        HeadWeights, HiddenLayerSpec, IntegerBoundLedger, ModelBinding, ScaleChain, StaticTables,
        GAT_SCHEMA_VERSION, MODEL_HASH_DOMAIN,
    };
    use crate::graph::{GraphCircuit, Visibility};
    use halo2curves::ff::PrimeField;

    fn head(input: usize, output: usize, with_bias: bool) -> HeadWeights {
        HeadWeights {
            projection: vec![0; input * output],
            a_src: vec![0; output],
            a_dst: vec![0; output],
            bias: if with_bias { vec![0; output] } else { Vec::new() },
        }
    }

    fn exact_table(
        table_id: &str,
        function_id: &str,
        input_min: i128,
        input_max: i128,
        input_scale: i32,
        output_scale: i32,
        output: impl Fn(i128) -> i128,
    ) -> ExactTable {
        let mut table = ExactTable {
            table_id: table_id.into(),
            input_scale,
            output_scale,
            function_id: function_id.into(),
            domain_id: format!("{}-domain", table_id),
            generator_id: format!("{}-generator", table_id),
            rows: (input_min..=input_max)
                .map(|input| (input, output(input)))
                .collect(),
            input_min,
            input_max,
            digest: String::new(),
        };
        table.refresh_digest().unwrap();
        table
    }

    fn family(layer: usize, head: usize, is_output: bool) -> BoundFamily {
        // One family carries a wide, nonzero PPI-like projection budget;
        // the remaining families exercise the authenticated zero-bound path.
        let max_aggregation_num = if layer == 0 && head == 0 && !is_output {
            3_600
        } else {
            0
        };
        BoundFamily {
            layer,
            head,
            is_output,
            max_score_abs: 600,
            max_weight_abs: 1,
            max_attention_abs: 1,
            max_bias_abs: 1,
            max_table_output_abs: 3_600,
            max_delta: 1_200,
            max_exp: 1,
            max_sum: 3,
            max_aggregation_num,
            max_dividend_abs: 7_200,
            max_aux_abs: 43_212,
            max_divisor: 6,
        }
    }

    pub(super) fn tiny_multigraph_spec() -> CanonicalGat {
        // Two graph segments, five nodes, and destination degrees 1, 2, 3,
        // 1, 2. This exercises graph-boundary resets and unequal group sizes.
        let mut graph = CanonicalGraph {
            n_total: 5,
            n: 5,
            e: 9,
            g_batch: 2,
            node_ids: vec![0, 1, 2, 3, 4],
            src: vec![0, 1, 0, 2, 1, 0, 3, 4, 3],
            dst: vec![0, 1, 1, 2, 2, 2, 3, 4, 4],
            node_ptr: vec![0, 3, 5],
            edge_ptr: vec![0, 6, 9],
            topology_id: "tiny-two-graph-topology".into(),
            topology_digest: String::new(),
        };
        graph.refresh_digest().unwrap();

        let model = CanonicalModel {
            hidden_layers: vec![
                HiddenLayerSpec {
                    d_in: 2,
                    heads: vec![head(2, 2, false), head(2, 2, false)],
                    d_head: 2,
                },
                HiddenLayerSpec {
                    d_in: 4,
                    heads: vec![head(4, 3, false)],
                    d_head: 3,
                },
            ],
            output_heads: vec![head(3, 2, true), head(3, 2, true)],
            classes: 2,
            model_arch_id: "tiny-two-layer-multihead".into(),
        };
        let scales = ScaleChain {
            s_input: 1,
            s_delta: 2,
            s_exp: 1,
            s_h: vec![1, 1, 1],
            s_w_hidden: vec![vec![0, 0], vec![0]],
            s_h_prime: vec![vec![1, 1], vec![1]],
            s_a_src_hidden: vec![vec![1, 1], vec![1]],
            s_a_dst_hidden: vec![vec![1, 1], vec![1]],
            s_agg_pre: vec![1, 1],
            s_agg: vec![1, 1],
            s_cat: vec![1, 1],
            s_w_output: vec![0, 0],
            s_y_prime: vec![1, 1],
            s_a_src_output: vec![1, 1],
            s_a_dst_output: vec![1, 1],
            s_b_output: vec![2, 2],
            s_out_pre: 1,
            s_out_head: 2,
            s_out: 2,
        };
        let tables = StaticTables {
            lrelu: exact_table("tiny-lrelu", "lrelu", -600, 600, 2, 2, |value| value),
            exp: exact_table("tiny-exp", "exp_neg", 0, 1_200, 2, 1, |_| 1),
            elu: exact_table("tiny-elu", "elu", -3_600, 3_600, 1, 1, |value| value),
            max_relu: exact_table("tiny-max-relu", "max_relu", -1_200, 1_200, 2, 2, |value| value.max(0)),
            static_tables_id: "tiny-static-tables".into(),
            backend_aux_tables_id: "tiny-backend-aux".into(),
            static_tables_digest: String::new(),
        };
        let mut tables = tables;
        tables.refresh_digest().unwrap();

        let mut families = vec![family(0, 0, false), family(0, 1, false), family(1, 0, false)];
        families.push(family(usize::MAX, 0, true));
        families.push(family(usize::MAX, 1, true));
        let mut ledger = IntegerBoundLedger {
            max_score_abs: 600,
            max_table_output_abs: 3_600,
            max_delta: 1_200,
            max_exp: 1,
            max_sum: 3,
            max_aggregation_num: 3_600,
            max_dividend_abs: 7_200,
            max_aux_abs: 43_212,
            max_divisor: 6,
            field_modulus: halo2curves::bn256::Fr::MODULUS.into(),
            ledger_id: "tiny-bounds".into(),
            ledger_digest: String::new(),
            families,
        };
        ledger.refresh_digest().unwrap();
        let topology = DigestIdentity {
            id: graph.topology_id.clone(),
            digest: graph.topology_digest.clone(),
        };
        let quant_cfg_id = "tiny-quantization".to_string();
        let quantization = DigestIdentity {
            id: quant_cfg_id.clone(),
            digest: sha256_digest(&scales.canonical_bytes()).unwrap(),
        };
        let ledger_identity = DigestIdentity {
            id: ledger.ledger_id.clone(),
            digest: sha256_digest(&ledger.canonical_bytes()).unwrap(),
        };
        let mut spec = CanonicalGat {
            schema_version: GAT_SCHEMA_VERSION,
            graph,
            features: vec![0; 10],
            feature_dim: 2,
            model,
            scales,
            tables,
            ledger,
            quant_cfg_id,
            ezkl_security_profile_id: "tiny-native-profile".into(),
            topology,
            quantization,
            ledger_identity,
            circuit_ref: None,
            model_binding: ModelBinding {
                hash: "1".into(),
                hash_domain: MODEL_HASH_DOMAIN.into(),
                canonical_model_digest: String::new(),
            },
            private_params_redacted: false,
        };
        // The bound ledger above intentionally assigns a nonzero bound to
        // this family's first projected cell while keeping other families
        // at B=0.  These nonzero canonical identity values accompany the
        // separate MockProver bound fixture; shape planning itself remains
        // witnessless.
        spec.features[0] = 1;
        spec.model.hidden_layers[0].heads[0].projection[0] = 1;
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

    pub(super) fn run_shape_only_settings_match_the_native_tiny_multigraph_dummy_layout() {
        let spec = tiny_multigraph_spec();
        let mut run_args = RunArgs::default();
        run_args.param_visibility = Visibility::Hashed {
            hash_is_public: true,
            outlets: Vec::new(),
        };
        run_args.logrows = 17;
        run_args.num_inner_cols = 2;
        run_args.decomp_base = 256;
        run_args.decomp_legs = 2;

        let model = build_model_for_run_args(
            spec.clone(),
            RoundingMode::NearestTiesAway,
            &run_args,
        )
        .unwrap();
        let actual = GraphCircuit::new(model, &run_args).unwrap();
        let mut expected = actual.settings().clone();
        let planned = gat_settings_for_run_args(&spec, &run_args).unwrap();

        // Required ranges are semantically a set in the source dummy pass;
        // canonicalize its HashSet-derived vector before checking every
        // GraphSettings field. The timestamps are generated independently.
        expected.required_range_checks.sort_unstable();
        let mut planned = planned;
        planned.required_range_checks.sort_unstable();
        expected.timestamp = planned.timestamp;
        assert_eq!(planned, expected);
        let max_degree = spec
            .graph
            .destination_groups_validated()
            .iter()
            .map(|(_, start, end)| end - start)
            .max()
            .unwrap();
        eprintln!(
            "EZKL_GAT_GEOMETRY_DIFF_OK k={} columns={} nodes={} edges={} graph_segments={} max_degree={} rows={} assignments={} constants={} dynamic_rows={} dynamic_tables={} max_lookup_input={} einsum_rows={} einsum_equations={} model_instances={:?} module_sizes={:?} ranges={}",
            run_args.logrows,
            run_args.num_inner_cols,
            spec.graph.n,
            spec.graph.e,
            spec.graph.g_batch,
            max_degree,
            actual.settings().num_rows,
            actual.settings().total_assignments,
            actual.settings().total_const_size,
            actual.settings().dynamic_lookup_params.total_dynamic_col_size,
            actual.settings().dynamic_lookup_params.num_dynamic_lookups,
            actual.settings().dynamic_lookup_params.max_dynamic_input_len,
            actual.settings().einsum_params.total_einsum_col_size,
            actual.settings().einsum_params.equations.len(),
            actual.settings().model_instance_shapes,
            actual.settings().module_sizes,
            actual.settings().required_range_checks.len(),
        );
        eprintln!(
            "EZKL_GAT_VALUE_LAYOUT valtype_bytes={} tensor_header_bytes={}",
            std::mem::size_of::<crate::tensor::ValType<Fp>>(),
            std::mem::size_of::<crate::tensor::Tensor<crate::tensor::ValType<Fp>>>(),
        );
    }

    #[test]
    fn shape_only_settings_match_the_native_tiny_multigraph_dummy_layout() {
        run_shape_only_settings_match_the_native_tiny_multigraph_dummy_layout();
    }
}
