use crate::plonk::{permutation, Any, FixedColumnSource, ProvingKey};

#[cfg(feature = "mv-lookup")]
use crate::plonk::mv_lookup as lookup;

#[cfg(not(feature = "mv-lookup"))]
use crate::plonk::lookup;

use crate::poly::Basis;
use crate::{
    arithmetic::{parallelize, CurveAffine},
    poly::{
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
    },
};

#[cfg(feature = "icicle_gpu")]
use crate::h_eval::{GateProgram, GateQuery, MvLookupProgram, GATE_OP_ADD, GATE_OP_COMMIT_BLOCK, GATE_OP_DOUBLE, GATE_OP_FOLD, GATE_OP_FOLD_BLOCK, GATE_OP_LOAD_CONST, GATE_OP_LOAD_QUERY, GATE_OP_MUL, GATE_OP_MV_LOOKUP_FOLD, GATE_OP_NEGATE, GATE_OP_SCALE_ACCUMULATOR, GATE_OP_SQUARE, GATE_OP_STORE_OUTPUT, GATE_OP_SUB};
use group::ff::{Field, PrimeField, WithSmallOrderMulGroup};
#[cfg(feature = "mv-lookup")]
use maybe_rayon::iter::{IndexedParallelIterator, ParallelIterator};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "icicle_gpu")]
use std::collections::HashMap;
use std::io;
use std::time::Instant;

use super::{shuffle, ConstraintSystem, Expression};

#[cfg(feature = "mv-lookup")]
use ff::BatchInvert;

#[cfg(feature = "mv-lookup")]
use maybe_rayon::slice::ParallelSliceMut;

/// Return the index in the polynomial of size `isize` after rotation `rot`.
fn get_rotation_idx(idx: usize, rot: i32, rot_scale: i32, isize: i32) -> usize {
    (((idx as i32) + (rot * rot_scale)).rem_euclid(isize)) as usize
}

fn permutation_term_count(set_count: usize) -> io::Result<usize> {
    set_count
        .checked_mul(2)
        .and_then(|count| count.checked_add(1))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "permutation H term count overflows host index width",
            )
        })
}

fn permutation_term_exponent(total_terms: usize, term_index: usize) -> io::Result<u64> {
    let exponent = total_terms
        .checked_sub(term_index.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "permutation H term index overflows host index width",
            )
        })?)
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "permutation H term index exceeds the fold length",
            )
        })?;
    u64::try_from(exponent).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "permutation H exponent exceeds field exponent width",
        )
    })
}

fn scale_permutation_h<F: Field + Send + Sync>(
    values: &mut [F],
    y: F,
    total_terms: usize,
) -> io::Result<()> {
    let exponent = u64::try_from(total_terms).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "permutation H fold length exceeds field exponent width",
        )
    })?;
    let scale = y.pow_vartime([exponent]);
    parallelize(values, move |chunk, _| {
        for value in chunk {
            *value *= scale;
        }
    });
    Ok(())
}

fn add_weighted_permutation_term<F, TermAtRow>(
    values: &mut [F],
    y: F,
    total_terms: usize,
    term_index: usize,
    term_at_row: TermAtRow,
) -> io::Result<()>
where
    F: Field + Send + Sync,
    TermAtRow: Fn(usize) -> F + Send + Sync + Clone,
{
    let exponent = permutation_term_exponent(total_terms, term_index)?;
    let scale = y.pow_vartime([exponent]);
    parallelize(values, move |chunk, start| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            *value += term_at_row(start + offset) * scale;
        }
    });
    Ok(())
}

fn add_permutation_set_h_terms<F: Field + Send + Sync>(
    values: &mut [F],
    z: &[F],
    l0: &[F],
    l_last: &[F],
    y: F,
    total_terms: usize,
    set_index: usize,
    set_count: usize,
    last_rotation: Rotation,
    part_index: usize,
    part_stride: usize,
    rot_scale: i32,
    isize: i32,
) -> io::Result<(usize, usize)> {
    if set_count < 2 || set_index >= set_count || values.len() != z.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "permutation set H fold geometry is invalid",
        ));
    }

    let has_first_boundary = set_index == 0;
    let has_last_boundary = set_index + 1 == set_count;
    let has_incoming_transition = set_index != 0;
    let has_outgoing_transition = set_index + 1 < set_count;

    let first_boundary_scale = has_first_boundary
        .then(|| permutation_term_exponent(total_terms, 0))
        .transpose()?
        .map(|exponent| y.pow_vartime([exponent]));
    let last_boundary_scale = has_last_boundary
        .then(|| permutation_term_exponent(total_terms, 1))
        .transpose()?
        .map(|exponent| y.pow_vartime([exponent]));
    let incoming_transition_scale = has_incoming_transition
        .then(|| {
            let term_index = set_index.checked_add(1).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "permutation incoming transition index overflows host index width",
                )
            })?;
            permutation_term_exponent(total_terms, term_index)
        })
        .transpose()?
        .map(|exponent| y.pow_vartime([exponent]));
    let outgoing_transition_scale = has_outgoing_transition
        .then(|| {
            let term_index = set_index.checked_add(2).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "permutation outgoing transition index overflows host index width",
                )
            })?;
            permutation_term_exponent(total_terms, term_index)
        })
        .transpose()?
        .map(|exponent| y.pow_vartime([exponent]));

    parallelize(values, |chunk, start| {
        for (offset, value) in chunk.iter_mut().enumerate() {
            let row = start + offset;
            let idx = row * part_stride + part_index;
            let mut contribution = F::ZERO;

            if let Some(scale) = first_boundary_scale {
                contribution += (F::ONE - z[row]) * l0[idx] * scale;
            }
            if let Some(scale) = last_boundary_scale {
                let z_value = z[row];
                contribution += (z_value.square() - z_value) * l_last[idx] * scale;
            }
            if let Some(scale) = incoming_transition_scale {
                contribution += z[row] * l0[idx] * scale;
            }
            if let Some(scale) = outgoing_transition_scale {
                let r_last = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);
                contribution -= z[r_last / part_stride] * l0[idx] * scale;
            }

            *value += contribution;
        }
    });

    let logical_terms = has_first_boundary as usize
        + has_last_boundary as usize
        + has_incoming_transition as usize;
    let split_contributions = logical_terms + has_outgoing_transition as usize;
    Ok((logical_terms, split_contributions))
}

/// Value used in a calculation
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd)]
pub enum ValueSource {
    /// This is a constant value
    Constant(usize),
    /// This is an intermediate value
    Intermediate(usize),
    /// This is a fixed column
    Fixed(usize, usize),
    /// This is an advice (witness) column
    Advice(usize, usize),
    /// This is an instance (external) column
    Instance(usize, usize),
    /// This is a challenge
    Challenge(usize),
    /// beta
    Beta(),
    /// gamma
    Gamma(),
    /// theta
    Theta(),
    /// y
    Y(),
    /// Previous value
    PreviousValue(),
}

impl Default for ValueSource {
    fn default() -> Self {
        ValueSource::Constant(0)
    }
}

impl ValueSource {
    /// Get the value for this source
    #[allow(clippy::too_many_arguments)]
    pub fn get<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        match self {
            ValueSource::Constant(idx) => constants[*idx],
            ValueSource::Intermediate(idx) => intermediates[*idx],
            ValueSource::Fixed(column_index, rotation) => {
                fixed_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Advice(column_index, rotation) => {
                advice_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Instance(column_index, rotation) => {
                instance_values[*column_index][rotations[*rotation]]
            }
            ValueSource::Challenge(index) => challenges[*index],
            ValueSource::Beta() => *beta,
            ValueSource::Gamma() => *gamma,
            ValueSource::Theta() => *theta,
            ValueSource::Y() => *y,
            ValueSource::PreviousValue() => *previous_value,
        }
    }
}

/// Calculation
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Calculation {
    /// This is an addition
    Add(ValueSource, ValueSource),
    /// This is a subtraction
    Sub(ValueSource, ValueSource),
    /// This is a product
    Mul(ValueSource, ValueSource),
    /// This is a square
    Square(ValueSource),
    /// This is a double
    Double(ValueSource),
    /// This is a negation
    Negate(ValueSource),
    /// This is Horner's rule: `val = a; val = val * c + b[]`
    Horner(ValueSource, Vec<ValueSource>, ValueSource),
    /// This is a simple assignment
    Store(ValueSource),
}

impl Calculation {
    /// Get the resulting value of this calculation
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<F: Field, B: Basis>(
        &self,
        rotations: &[usize],
        constants: &[F],
        intermediates: &[F],
        fixed_values: &[Polynomial<F, B>],
        advice_values: &[Polynomial<F, B>],
        instance_values: &[Polynomial<F, B>],
        challenges: &[F],
        beta: &F,
        gamma: &F,
        theta: &F,
        y: &F,
        previous_value: &F,
    ) -> F {
        let get_value = |value: &ValueSource| {
            value.get(
                rotations,
                constants,
                intermediates,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            )
        };
        match self {
            Calculation::Add(a, b) => get_value(a) + get_value(b),
            Calculation::Sub(a, b) => get_value(a) - get_value(b),
            Calculation::Mul(a, b) => get_value(a) * get_value(b),
            Calculation::Square(v) => get_value(v).square(),
            Calculation::Double(v) => get_value(v).double(),
            Calculation::Negate(v) => -get_value(v),
            Calculation::Horner(start_value, parts, factor) => {
                let factor = get_value(factor);
                let mut value = get_value(start_value);
                for part in parts.iter() {
                    value = value * factor + get_value(part);
                }
                value
            }
            Calculation::Store(v) => get_value(v),
        }
    }
}

/// Evaluator
#[derive(Clone, Default, Debug)]
pub struct Evaluator<C: CurveAffine> {
    /// Custom gate constraints, kept in native gate/polynomial order so the
    /// prover can evaluate and release one constraint group's cosets at a time.
    pub custom_gates: Vec<GraphEvaluator<C>>,
    ///  Lookups evalution
    #[cfg(feature = "mv-lookup")]
    pub lookups: Vec<(Vec<GraphEvaluator<C>>, GraphEvaluator<C>)>,
    #[cfg(not(feature = "mv-lookup"))]
    pub lookups: Vec<GraphEvaluator<C>>,
    ///  Shuffle evalution
    pub shuffles: Vec<GraphEvaluator<C>>,
}

/// GraphEvaluator
#[derive(Clone, Debug)]
pub struct GraphEvaluator<C: CurveAffine> {
    /// Constants
    pub constants: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<i32>,
    /// Calculations
    pub calculations: Vec<CalculationInfo>,
    /// Number of intermediates
    pub num_intermediates: usize,
}

/// EvaluationData
#[derive(Default, Debug)]
pub struct EvaluationData<C: CurveAffine> {
    /// Intermediates
    pub intermediates: Vec<C::ScalarExt>,
    /// Rotations
    pub rotations: Vec<usize>,
}

/// CaluclationInfo
#[derive(Clone, Debug)]
pub struct CalculationInfo {
    /// Calculation
    pub calculation: Calculation,
    /// Target
    pub target: usize,
}

#[derive(Clone, Debug, Default)]
struct ColumnDependencies {
    fixed: BTreeSet<usize>,
    advice: BTreeSet<usize>,
    instance: BTreeSet<usize>,
}

#[cfg(feature = "mv-lookup")]
fn lookup_rows_per_chunk(max_rows: usize, input_count: usize) -> Option<usize> {
    if input_count == 0 {
        return None;
    }
    let legacy_field_bound = max_rows.checked_mul(input_count)?;
    let rows = (legacy_field_bound / 2).min(max_rows);
    (rows != 0).then_some(rows)
}

#[cfg(feature = "mv-lookup")]
struct LookupInputValues<F: Field> {
    inputs_products: Vec<F>,
    products_except_sum: Vec<F>,
}

#[cfg(feature = "mv-lookup")]
impl<F: Field> LookupInputValues<F> {
    fn inputs_product(&self, row: usize) -> F {
        self.inputs_products[row]
    }

    fn products_except_sum(&self, row: usize) -> F {
        self.products_except_sum[row]
    }
}

#[cfg(feature = "mv-lookup")]
fn evaluate_lookup_input_values<F: Field>(
    row_count: usize,
    input_count: usize,
    mut evaluate_input: impl FnMut(usize, usize) -> F,
) -> LookupInputValues<F> {
    let mut inputs_products = vec![F::ZERO; row_count];
    let mut products_except_sum = vec![F::ZERO; row_count];
    for row in 0..row_count {
        let mut inputs_product = F::ONE;
        let mut row_products_except_sum = F::ZERO;
        let mut has_zero_input = false;
        for input_index in 0..input_count {
            let input_value = evaluate_input(row, input_index);
            row_products_except_sum = row_products_except_sum * input_value + inputs_product;
            has_zero_input |= bool::from(input_value.is_zero());
            inputs_product *= input_value;
        }
        inputs_products[row] = inputs_product;
        // batch_invert maps each zero input to zero. The old quotient then
        // multiplies the inverse sum by the row product, so any zero input
        // makes that weighted sum zero. Preserve that convention exactly.
        products_except_sum[row] = if has_zero_input {
            F::ZERO
        } else {
            row_products_except_sum
        };
    }
    LookupInputValues {
        inputs_products,
        products_except_sum,
    }
}

#[cfg(feature = "mv-lookup")]
fn fold_lookup_h_row<F: Field>(
    h_value: &mut F,
    y: F,
    l0: F,
    l_last: F,
    l_active_row: F,
    phi: F,
    phi_next: F,
    table_value: F,
    inputs_product: F,
    products_except_sum: F,
    multiplicity: F,
) {
    let lhs = table_value * inputs_product * (phi_next - phi);
    let rhs = table_value * products_except_sum - inputs_product * multiplicity;
    *h_value = *h_value * y + l0 * phi;
    *h_value = *h_value * y + l_last * phi;
    *h_value = *h_value * y + (lhs - rhs) * l_active_row;
}

/// Exact column dependency counts for the native evaluator's sequential groups.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeEvaluationShape {
    /// Distinct fixed/advice/instance columns loaded by each custom-gate group.
    pub custom_gate_dependency_columns: Vec<usize>,
    /// Distinct columns loaded by each lookup argument after chunking.
    pub lookup_dependency_columns: Vec<usize>,
    /// Number of query polynomials evaluated by each lookup argument.
    pub lookup_query_polynomial_counts: Vec<usize>,
    /// Distinct columns loaded by each shuffle argument.
    pub shuffle_dependency_columns: Vec<usize>,
}

/// Source-derived SHPLONK commitment counts grouped by the exact rotation set
/// that `construct_intermediate_sets` uses for each polynomial commitment.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NativeShplonkShape {
    /// Number of polynomial commitments in each distinct rotation-set class.
    pub rotation_set_commitment_counts: Vec<usize>,
    /// Number of opened polynomial commitments before SHPLONK groups them.
    pub query_polynomial_count: usize,
    /// Number of distinct polynomial rotation-set classes.
    pub rotation_set_count: usize,
    /// Maximum number of queried rotation points attached to one commitment.
    pub max_rotation_count: usize,
    /// Number of unique native query rotation points across all commitments.
    pub unique_opening_point_count: usize,
    /// Maximum number of polynomial contributions in one rotation set.
    pub max_commitments_per_rotation_set: usize,
    /// Number of permutation product polynomials opened by the native prover.
    pub permutation_product_set_count: usize,
    /// Number of queried advice polynomials across proof instances.
    pub advice_query_polynomial_count: usize,
    /// Number of queried fixed polynomials shared across proof instances.
    pub fixed_query_polynomial_count: usize,
    /// Number of post-chunk multi-lookup arguments per proof instance.
    pub lookup_argument_count: usize,
    /// Number of shuffle product polynomials per proof instance.
    pub shuffle_argument_count: usize,
    /// Number of circuit instances passed to native proof construction.
    pub proof_instance_count: usize,
    /// Number of committed quotient h pieces.
    pub quotient_piece_count: usize,
}

fn canonical_shplonk_rotation(rotation: i32, domain_size: u64) -> i64 {
    if domain_size <= i64::MAX as u64 {
        i64::from(rotation).rem_euclid(domain_size as i64)
    } else {
        i64::from(rotation)
    }
}

fn insert_shplonk_rotation_class(
    classes: &mut BTreeMap<Vec<i64>, usize>,
    rotations: impl IntoIterator<Item = i32>,
    domain_size: u64,
    count: usize,
) {
    if count == 0 {
        return;
    }
    let rotations = rotations
        .into_iter()
        .map(|rotation| canonical_shplonk_rotation(rotation, domain_size))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    *classes.entry(rotations).or_default() += count;
}

fn queried_column_rotations<C: crate::plonk::ColumnType>(
    queries: &[(crate::plonk::Column<C>, crate::poly::Rotation)],
) -> BTreeMap<usize, BTreeSet<i32>> {
    let mut columns = BTreeMap::<usize, BTreeSet<i32>>::new();
    for (column, rotation) in queries {
        columns
            .entry(column.index())
            .or_default()
            .insert(rotation.0);
    }
    columns
}

/// Derives SHPLONK's per-rotation-set contribution width from the same
/// constraint-system queries and protocol object counts used by native proof
/// construction. `lookup_argument_count` is the post-chunk native mv-lookup
/// count; `proof_instance_count` is the number of circuit instances passed to
/// `plonk::create_proof`; `quotient_piece_count` is the native domain's exact
/// number of committed h pieces.
pub fn native_shplonk_resource_shape<F: Field>(
    cs: &ConstraintSystem<F>,
    domain_size: u64,
    lookup_argument_count: usize,
    proof_instance_count: usize,
    quotient_piece_count: usize,
) -> NativeShplonkShape {
    let mut classes = BTreeMap::<Vec<i64>, usize>::new();
    let advice = queried_column_rotations(&cs.advice_queries);
    let fixed = queried_column_rotations(&cs.fixed_queries);
    let cur = crate::poly::Rotation::cur().0;
    let next = crate::poly::Rotation::next().0;
    let prev = crate::poly::Rotation::prev().0;

    for _ in 0..proof_instance_count {
        for rotations in advice.values() {
            insert_shplonk_rotation_class(
                &mut classes,
                rotations.iter().copied(),
                domain_size,
                1,
            );
        }

        let permutation_chunk_size = cs.degree().saturating_sub(2).max(1);
    let permutation_product_set_count =
        cs.num_permutation_columns().div_ceil(permutation_chunk_size);
    let permutation_nonfinal_rotations = [prev, cur, next];
    let permutation_final_rotations = [cur, next];
    for set_index in 0..permutation_product_set_count {
        let rotations = if set_index + 1 < permutation_product_set_count {
            &permutation_nonfinal_rotations[..]
        } else {
            &permutation_final_rotations[..]
        };
            insert_shplonk_rotation_class(
                &mut classes,
                rotations.iter().copied(),
                domain_size,
                1,
            );
        }

        for _ in 0..lookup_argument_count {
            // mv-lookup opens phi at x and omega*x and m at x.
            insert_shplonk_rotation_class(
                &mut classes,
                [cur, next],
                domain_size,
                1,
            );
            insert_shplonk_rotation_class(&mut classes, [cur], domain_size, 1);
        }

        for _ in cs.shuffles.iter() {
            // Each native shuffle opens one product polynomial at x and omega*x.
            insert_shplonk_rotation_class(
                &mut classes,
                [cur, next],
                domain_size,
                1,
            );
        }
    }

    for rotations in fixed.values() {
        insert_shplonk_rotation_class(
            &mut classes,
            rotations.iter().copied(),
            domain_size,
            1,
        );
    }

    // The native permutation verifying key opens one sigma polynomial per
    // permutation column. These and vanishing h/random polynomials are shared
    // across proof instances in plonk::create_proof.
    insert_shplonk_rotation_class(
        &mut classes,
        [cur],
        domain_size,
        cs.num_permutation_columns(),
    );
    insert_shplonk_rotation_class(
        &mut classes,
        [cur],
        domain_size,
        quotient_piece_count.saturating_add(1),
    );

    let rotation_set_count = classes.len();
    let max_rotation_count = classes.keys().map(Vec::len).max().unwrap_or(0);
    let unique_opening_point_count = classes
        .keys()
        .flat_map(|rotations| rotations.iter().copied())
        .collect::<BTreeSet<_>>()
        .len();
    let rotation_set_commitment_counts = classes.into_values().collect::<Vec<_>>();
    let query_polynomial_count = rotation_set_commitment_counts.iter().sum();
    let max_commitments_per_rotation_set =
        rotation_set_commitment_counts.iter().copied().max().unwrap_or(0);
    let permutation_chunk_size = cs.degree().saturating_sub(2).max(1);
    NativeShplonkShape {
        rotation_set_commitment_counts,
        query_polynomial_count,
        rotation_set_count,
        max_rotation_count,
        unique_opening_point_count,
        max_commitments_per_rotation_set,
        permutation_product_set_count: cs
            .num_permutation_columns()
            .div_ceil(permutation_chunk_size),
        advice_query_polynomial_count: advice.len() * proof_instance_count,
        fixed_query_polynomial_count: fixed.len(),
        lookup_argument_count,
        shuffle_argument_count: cs.shuffles.len() * proof_instance_count,
        proof_instance_count,
        quotient_piece_count,
    }
}

impl ColumnDependencies {
    fn add(&mut self, source: &ValueSource) {
        match source {
            ValueSource::Fixed(column, _) => {
                self.fixed.insert(*column);
            }
            ValueSource::Advice(column, _) => {
                self.advice.insert(*column);
            }
            ValueSource::Instance(column, _) => {
                self.instance.insert(*column);
            }
            _ => {}
        }
    }

    fn extend(&mut self, other: &Self) {
        self.fixed.extend(other.fixed.iter().copied());
        self.advice.extend(other.advice.iter().copied());
        self.instance.extend(other.instance.iter().copied());
    }

    fn additional_column_count(&self, existing: &Self) -> usize { self.fixed.difference(&existing.fixed).count() + self.advice.difference(&existing.advice).count() + self.instance.difference(&existing.instance).count() }

    fn additional_fft_load_count(
        &self,
        existing: &Self,
        fixed_constant_columns: &BTreeSet<usize>,
    ) -> usize {
        self.fixed
            .difference(&existing.fixed)
            .filter(|column| !fixed_constant_columns.contains(column))
            .count()
            + self.advice.difference(&existing.advice).count()
            + self.instance.difference(&existing.instance).count()
    }

    fn visit_columns(&self, mut visit: impl FnMut(u8, usize)) {
        for column in &self.fixed {
            visit(0, *column);
        }
        for column in &self.advice {
            visit(1, *column);
        }
        for column in &self.instance {
            visit(2, *column);
        }
    }

    fn column_count(&self) -> usize {
        self.fixed.len() + self.advice.len() + self.instance.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct DependencyColumnKey(u8, usize);

#[derive(Clone, Debug)]
struct CustomGateGroup {
    start: usize,
    end: usize,
    dependencies: ColumnDependencies,
}

#[derive(Clone, Copy)]
struct WeightedGateGroup<F: Field> {
    initial_scale: F,
    contribution_scale: F,
}

fn custom_gate_groups<C: CurveAffine>(graphs: &[GraphEvaluator<C>]) -> Vec<CustomGateGroup> {
    let max_dependencies = graphs
        .iter()
        .map(|graph| graph.column_dependencies().column_count())
        .max()
        .unwrap_or(0);
    let mut groups = Vec::new();
    let mut start = 0;
    while start < graphs.len() {
        let mut dependencies = graphs[start].column_dependencies();
        let mut end = start + 1;
        while end < graphs.len() {
            let next = graphs[end].column_dependencies();
            if dependencies
                .column_count()
                .saturating_add(next.additional_column_count(&dependencies))
                > max_dependencies
            {
                break;
            }
            dependencies.extend(&next);
            end += 1;
        }
        groups.push(CustomGateGroup {
            start,
            end,
            dependencies,
        });
        start = end;
    }
    groups
}

fn estimate_gate_group_fft_loads(
    order: &[usize],
    groups: &[CustomGateGroup],
    fixed_constant_columns: &BTreeSet<usize>,
) -> usize {
    let mut previous: Option<&ColumnDependencies> = None;
    order
        .iter()
        .map(|index| {
            let dependencies = &groups[*index].dependencies;
            let loads = previous.map_or_else(
                || {
                    dependencies
                        .fixed
                        .iter()
                        .filter(|column| !fixed_constant_columns.contains(column))
                        .count()
                        + dependencies.advice.len()
                        + dependencies.instance.len()
                },
                |previous| {
                    dependencies.additional_fft_load_count(previous, fixed_constant_columns)
                },
            );
            previous = Some(dependencies);
            loads
        })
        .sum()
}

/// Reorders only the existing contiguous dependency-bounded gate groups.
/// Candidate generation is bounded by a 16-group original-order window plus
/// at most 16 distinct groups found by scanning at most 32 posting entries for
/// each of at most 16 current-group columns. At most 32 candidates are scored;
/// each score is O(columns in one group). Posting cursors skip scheduled
/// entries monotonically, so cursor advancement is amortized over all
/// dependency postings rather than rescanned from the start.
fn schedule_custom_gate_groups<C: CurveAffine>(
    graphs: &[GraphEvaluator<C>],
    groups: &[CustomGateGroup],
    fixed_constant_columns: &BTreeSet<usize>,
) -> (Vec<usize>, usize, usize, bool) {
    let original = (0..groups.len()).collect::<Vec<_>>();
    let original_loads = estimate_gate_group_fft_loads(&original, groups, fixed_constant_columns);
    if groups.len() < 2 || graphs.iter().any(GraphEvaluator::uses_previous_value) {
        return (original, original_loads, original_loads, false);
    }

    const MAX_CANDIDATES: usize = 32;
    const ORIGINAL_ORDER_CANDIDATES: usize = 16;
    const PER_COLUMN_NEIGHBORS: usize = 8;
    const MAX_POSTING_SCANS_PER_COLUMN: usize = 32;
    let mut postings = BTreeMap::<DependencyColumnKey, Vec<usize>>::new();
    for (group_index, group) in groups.iter().enumerate() {
        group.dependencies.visit_columns(|kind, column| {
            postings
                .entry(DependencyColumnKey(kind, column))
                .or_default()
                .push(group_index);
        });
    }
    let mut posting_positions = postings
        .keys()
        .copied()
        .map(|key| (key, 0usize))
        .collect::<BTreeMap<_, _>>();
    let mut scheduled = vec![false; groups.len()];
    let mut order = Vec::with_capacity(groups.len());
    let mut original_cursor = 0usize;
    let mut previous_index: Option<usize> = None;

    while order.len() < groups.len() {
        let mut candidates = BTreeSet::new();
        let mut scanned_original = 0usize;
        let mut cursor = original_cursor;
        while cursor < groups.len() && scanned_original < ORIGINAL_ORDER_CANDIDATES {
            if !scheduled[cursor] {
                candidates.insert(cursor);
                scanned_original += 1;
            }
            cursor += 1;
        }
        while original_cursor < groups.len() && scheduled[original_cursor] {
            original_cursor += 1;
        }

        if let Some(previous_index) = previous_index {
            let mut neighbor_count = 0usize;
            let mut neighbor_keys = Vec::new();
            groups[previous_index]
                .dependencies
                .visit_columns(|kind, column| neighbor_keys.push(DependencyColumnKey(kind, column)));
            for key in neighbor_keys {
                if candidates.len() >= MAX_CANDIDATES || neighbor_count >= 16 {
                    break;
                }
                let Some(group_indices) = postings.get(&key) else {
                    continue;
                };
                let Some(position) = posting_positions.get_mut(&key) else {
                    continue;
                };
                while *position < group_indices.len() && scheduled[group_indices[*position]] {
                    *position += 1;
                }
                let mut scan = *position;
                let mut added_for_column = 0usize;
                let mut scanned_for_column = 0usize;
                while scan < group_indices.len()
                    && added_for_column < PER_COLUMN_NEIGHBORS
                    && scanned_for_column < MAX_POSTING_SCANS_PER_COLUMN
                    && neighbor_count < 16
                    && candidates.len() < MAX_CANDIDATES
                {
                    let candidate = group_indices[scan];
                    scan += 1;
                    scanned_for_column += 1;
                    if !scheduled[candidate] && candidates.insert(candidate) {
                        added_for_column += 1;
                        neighbor_count += 1;
                    }
                }
            }
        }

        let next_index = candidates
            .into_iter()
            .min_by_key(|candidate| {
                let dependencies = &groups[*candidate].dependencies;
                let added = previous_index.map_or_else(
                    || {
                        dependencies
                            .fixed
                            .iter()
                            .filter(|column| !fixed_constant_columns.contains(column))
                            .count()
                            + dependencies.advice.len()
                            + dependencies.instance.len()
                    },
                    |previous| {
                        dependencies.additional_fft_load_count(
                            &groups[previous].dependencies,
                            fixed_constant_columns,
                        )
                    },
                );
                (added, *candidate)
            })
            .expect("at least one unscheduled group is a candidate");
        scheduled[next_index] = true;
        order.push(next_index);
        previous_index = Some(next_index);
    }

    let candidate_loads = estimate_gate_group_fft_loads(&order, groups, fixed_constant_columns);
    if candidate_loads < original_loads {
        (order, original_loads, candidate_loads, true)
    } else {
        (original, original_loads, original_loads, false)
    }
}

/// Reconstruct only the cosets referenced by the active native constraint
/// group. Unused entries are zero-length sentinels; graph dependency discovery
/// guarantees that evaluation never indexes one of them.
fn load_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    polys: &[Polynomial<F, Coeff>],
    columns: &BTreeSet<usize>,
    extended_omega_factor: F,
) -> io::Result<Vec<Polynomial<F, LagrangeCoeff>>> {
    let mut cosets = (0..polys.len())
        .map(|_| Polynomial::from_vec(Vec::new()))
        .collect::<Vec<_>>();
    for &column in columns {
        let poly = polys.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent polynomial",
            )
        })?;
        cosets[column] = domain.coeff_to_extended_part_spilled(poly, extended_omega_factor)?;
    }
    Ok(cosets)
}

/// Expand only fixed columns used by the active gate batch; compact singleton columns directly fill their coset, while other sources pass through Lagrange and coefficient form.
fn load_fixed_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    sources: &[FixedColumnSource<F>],
    columns: &BTreeSet<usize>,
    extended_omega_factor: F,
) -> io::Result<Vec<Polynomial<F, LagrangeCoeff>>> {
    let mut cosets = (0..sources.len())
        .map(|_| Polynomial::from_vec(Vec::new()))
        .collect::<Vec<_>>();
    for &column in columns {
        let source = sources.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent fixed source",
            )
        })?;
        if let Some(value) = source.constant_value() {
            let n = domain.get_n() as usize;
            if source.len() != n {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "constant fixed source has inconsistent dimensions",
                ));
            }
            cosets[column] = Polynomial::mapped_filled(n, value)?;
            continue;
        }
        let lagrange = source.materialize_lagrange()?;
        let coeff = domain.lagrange_to_coeff(lagrange).spill()?;
        cosets[column] = domain.coeff_to_extended_part(coeff, extended_omega_factor);
    }
    Ok(cosets)
}

#[derive(Default)]
struct CosetRefreshCounts {
    transforms_loaded: usize,
    transforms_reused: usize,
}

#[derive(Default)]
struct FixedCosetRefreshCounts {
    transforms_loaded: usize,
    transforms_reused: usize,
    constant_fills_loaded: usize,
    constant_fills_reused: usize,
}

#[derive(Default)]
struct GateCosetCacheSummary {
    groups: usize,
    advice: CosetRefreshCounts,
    instance: CosetRefreshCounts,
    fixed: FixedCosetRefreshCounts,
}

#[derive(Default)]
struct GateGroupEvalSummary {
    rows: usize,
    gpu_rows: usize,
    cpu_rows: usize,
    gpu_fallback_rows: usize,
    gpu_groups: usize,
    cpu_groups: usize,
    gpu_fallback_groups: usize,
    coset_prepare_ms: f64,
    program_compile_ms: f64,
    gpu_or_cpu_eval_wall_ms: f64,
    gpu_h_call_ms: f64,
    cpu_eval_ms: f64,
}

struct GateGroupEvalTiming {
    gpu_executed: bool,
    gpu_fallback: bool,
    program_compile_ms: f64,
    gpu_h_call_ms: f64,
    cpu_eval_ms: f64,
}

fn empty_coset_cache<F>(column_count: usize) -> Vec<Polynomial<F, LagrangeCoeff>> {
    (0..column_count)
        .map(|_| Polynomial::from_vec(Vec::new()))
        .collect()
}

/// Drop the previous group's inactive columns before loading any columns for
/// the next group. The three source classes are evicted together by the caller
/// so the live coset union never exceeds the current dependency budget.
fn evict_inactive_cosets<F>(
    cosets: &mut [Polynomial<F, LagrangeCoeff>],
    previous_columns: &BTreeSet<usize>,
    next_columns: &BTreeSet<usize>,
) -> io::Result<()> {
    for &column in previous_columns.difference(next_columns) {
        let coset = cosets.get_mut(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coset cache has inconsistent dimensions",
            )
        })?;
        *coset = Polynomial::from_vec(Vec::new());
    }
    Ok(())
}

fn load_missing_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    polys: &[Polynomial<F, Coeff>],
    columns: &BTreeSet<usize>,
    extended_omega_factor: F,
    cosets: &mut [Polynomial<F, LagrangeCoeff>],
) -> io::Result<CosetRefreshCounts> {
    load_missing_cosets_with_factors(
        domain,
        polys,
        columns,
        extended_omega_factor,
        cosets,
        None,
        None,
    )
}

fn load_missing_cosets_with_factors<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    polys: &[Polynomial<F, Coeff>],
    columns: &BTreeSet<usize>,
    extended_omega_factor: F,
    cosets: &mut [Polynomial<F, LagrangeCoeff>],
    factors: Option<&[F]>,
    deferred_column: Option<usize>,
) -> io::Result<CosetRefreshCounts> {
    let mut counts = CosetRefreshCounts::default();
    for &column in columns {
        let poly = polys.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent polynomial",
            )
        })?;
        let coset = cosets.get_mut(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coset cache has inconsistent dimensions",
            )
        })?;
        if coset.is_empty() {
            if deferred_column == Some(column) {
                continue;
            }
            *coset = match factors {
                Some(factors) => domain.coeff_to_extended_part_spilled_with_factors(
                    poly,
                    factors,
                    extended_omega_factor,
                )?,
                None => domain.coeff_to_extended_part_spilled(poly, extended_omega_factor)?,
            };
            counts.transforms_loaded += 1;
        } else {
            counts.transforms_reused += 1;
        }
    }
    Ok(counts)
}

fn load_missing_fixed_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    sources: &[FixedColumnSource<F>],
    columns: &BTreeSet<usize>,
    extended_omega_factor: F,
    cosets: &mut [Polynomial<F, LagrangeCoeff>],
) -> io::Result<FixedCosetRefreshCounts> {
    load_missing_fixed_cosets_with_factors(
        domain,
        sources,
        columns,
        extended_omega_factor,
        cosets,
        None,
    )
}

fn load_missing_fixed_cosets_with_factors<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    sources: &[FixedColumnSource<F>],
    columns: &BTreeSet<usize>,
    extended_omega_factor: F,
    cosets: &mut [Polynomial<F, LagrangeCoeff>],
    factors: Option<&[F]>,
) -> io::Result<FixedCosetRefreshCounts> {
    let mut counts = FixedCosetRefreshCounts::default();
    for &column in columns {
        let source = sources.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent fixed source",
            )
        })?;
        let coset = cosets.get_mut(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coset cache has inconsistent dimensions",
            )
        })?;
        if let Some(value) = source.constant_value() {
            let n = domain.get_n() as usize;
            if source.len() != n {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "constant fixed source has inconsistent dimensions",
                ));
            }
            if coset.is_empty() {
                *coset = Polynomial::mapped_filled(n, value)?;
                counts.constant_fills_loaded += 1;
            } else {
                counts.constant_fills_reused += 1;
            }
        } else if coset.is_empty() {
            let lagrange = source.materialize_lagrange()?;
            let coeff = domain.lagrange_to_coeff(lagrange).spill()?;
            *coset = match factors {
                Some(factors) => domain.coeff_to_extended_part_with_factors(
                    coeff,
                    factors,
                    extended_omega_factor,
                )?,
                None => domain.coeff_to_extended_part(coeff, extended_omega_factor),
            };
            counts.transforms_loaded += 1;
        } else {
            counts.transforms_reused += 1;
        }
    }
    Ok(counts)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedCosetTarget {
    Advice(usize),
    Instance(usize),
}

fn select_shared_coset_target<F: Copy>(
    expected_len: usize,
    fixed_sources: &[FixedColumnSource<F>],
    advice: &[Polynomial<F, Coeff>],
    instance: &[Polynomial<F, Coeff>],
    fixed_columns: &BTreeSet<usize>,
    advice_columns: &BTreeSet<usize>,
    instance_columns: &BTreeSet<usize>,
    fixed_cosets: &[Polynomial<F, LagrangeCoeff>],
    advice_cosets: &[Polynomial<F, LagrangeCoeff>],
    instance_cosets: &[Polynomial<F, LagrangeCoeff>],
) -> io::Result<Option<SharedCosetTarget>> {
    let mut missing_transforms = 0usize;
    for &column in fixed_columns {
        let source = fixed_sources.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent fixed source",
            )
        })?;
        let coset = fixed_cosets.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coset cache has inconsistent dimensions",
            )
        })?;
        if source.len() != expected_len || (!coset.is_empty() && coset.len() != expected_len) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "fixed source or cached coset has inconsistent dimensions",
            ));
        }
        if coset.is_empty() && source.constant_value().is_none() {
            missing_transforms += 1;
        }
    }
    for &column in advice_columns {
        let source = advice.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent polynomial",
            )
        })?;
        let coset = advice_cosets.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coset cache has inconsistent dimensions",
            )
        })?;
        if source.len() != expected_len || (!coset.is_empty() && coset.len() != expected_len) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "advice polynomial or cached coset has inconsistent dimensions",
            ));
        }
        if coset.is_empty() {
            missing_transforms += 1;
        }
    }
    for &column in instance_columns {
        let source = instance.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "constraint references an absent polynomial",
            )
        })?;
        let coset = instance_cosets.get(column).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "coset cache has inconsistent dimensions",
            )
        })?;
        if source.len() != expected_len || (!coset.is_empty() && coset.len() != expected_len) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "instance polynomial or cached coset has inconsistent dimensions",
            ));
        }
        if coset.is_empty() {
            missing_transforms += 1;
        }
    }

    // Instance sources follow advice sources in the original load order. Use
    // the last missing dense source as the reserved buffer so every other
    // source, including nonconstant fixed columns, is converted first. Fixed
    // sources are not selected as the reserved output because materializing
    // their Lagrange input already needs a separate mapped N-element buffer.
    if missing_transforms < 2 {
        return Ok(None);
    }
    if let Some(&column) = instance_columns
        .iter()
        .rev()
        .find(|&&column| instance_cosets[column].is_empty())
    {
        return Ok(Some(SharedCosetTarget::Instance(column)));
    }
    if let Some(&column) = advice_columns
        .iter()
        .rev()
        .find(|&&column| advice_cosets[column].is_empty())
    {
        return Ok(Some(SharedCosetTarget::Advice(column)));
    }
    Ok(None)
}

/// Refresh all three source classes as one group so a single output-sized
/// mapped buffer can hold the shared coefficient powers and then become the
/// final missing advice/instance coset. The target is always loaded last.
fn load_gate_group_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    fixed_sources: &[FixedColumnSource<F>],
    advice: &[Polynomial<F, Coeff>],
    instance: &[Polynomial<F, Coeff>],
    fixed_columns: &BTreeSet<usize>,
    advice_columns: &BTreeSet<usize>,
    instance_columns: &BTreeSet<usize>,
    extended_omega_factor: F,
    fixed_cosets: &mut [Polynomial<F, LagrangeCoeff>],
    advice_cosets: &mut [Polynomial<F, LagrangeCoeff>],
    instance_cosets: &mut [Polynomial<F, LagrangeCoeff>],
) -> io::Result<(
    FixedCosetRefreshCounts,
    CosetRefreshCounts,
    CosetRefreshCounts,
)> {
    let expected_len = usize::try_from(domain.get_n()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "base domain length does not fit in memory index space",
        )
    })?;
    let target = select_shared_coset_target(
        expected_len,
        fixed_sources,
        advice,
        instance,
        fixed_columns,
        advice_columns,
        instance_columns,
        fixed_cosets,
        advice_cosets,
        instance_cosets,
    )?;
    let Some(target) = target else {
        let fixed_counts = load_missing_fixed_cosets(
            domain,
            fixed_sources,
            fixed_columns,
            extended_omega_factor,
            fixed_cosets,
        )?;
        let advice_counts = load_missing_cosets(
            domain,
            advice,
            advice_columns,
            extended_omega_factor,
            advice_cosets,
        )?;
        let instance_counts = load_missing_cosets(
            domain,
            instance,
            instance_columns,
            extended_omega_factor,
            instance_cosets,
        )?;
        return Ok((fixed_counts, advice_counts, instance_counts));
    };

    let target_source = match target {
        SharedCosetTarget::Advice(column) => advice.get(column),
        SharedCosetTarget::Instance(column) => instance.get(column),
    }
    .ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "constraint references an absent polynomial",
        )
    })?;

    if domain.extended_part_gpu_coset_is_eligible(
        &target_source[..],
        extended_omega_factor,
    ) {
        // Keep the existing single mapped output slot and defer the selected
        // target until last, but do not fill that slot with host powers. The
        // fused GPU transform generates and caches powers on device.
        let mut output_buffer = match Polynomial::<F, Coeff>::mapped_filled(expected_len, F::ZERO)
        {
            Ok(buffer) => buffer,
            Err(_) => {
                // Match the existing shared-factor allocation fallback: make
                // all conversions independently before touching a reusable
                // group factor/output buffer.
                let fixed_counts = load_missing_fixed_cosets(
                    domain,
                    fixed_sources,
                    fixed_columns,
                    extended_omega_factor,
                    fixed_cosets,
                )?;
                let advice_counts = load_missing_cosets(
                    domain,
                    advice,
                    advice_columns,
                    extended_omega_factor,
                    advice_cosets,
                )?;
                let instance_counts = load_missing_cosets(
                    domain,
                    instance,
                    instance_columns,
                    extended_omega_factor,
                    instance_cosets,
                )?;
                return Ok((fixed_counts, advice_counts, instance_counts));
            }
        };

        let (fixed_counts, mut advice_counts, mut instance_counts) = {
            let fixed_counts = load_missing_fixed_cosets(
                domain,
                fixed_sources,
                fixed_columns,
                extended_omega_factor,
                fixed_cosets,
            )?;
            let advice_counts = load_missing_cosets_with_factors(
                domain,
                advice,
                advice_columns,
                extended_omega_factor,
                advice_cosets,
                None,
                match target {
                    SharedCosetTarget::Advice(column) => Some(column),
                    SharedCosetTarget::Instance(_) => None,
                },
            )?;
            let instance_counts = load_missing_cosets_with_factors(
                domain,
                instance,
                instance_columns,
                extended_omega_factor,
                instance_cosets,
                None,
                match target {
                    SharedCosetTarget::Instance(column) => Some(column),
                    SharedCosetTarget::Advice(_) => None,
                },
            )?;
            (fixed_counts, advice_counts, instance_counts)
        };

        output_buffer.copy_from_slice(target_source);
        match target {
            SharedCosetTarget::Advice(column) => {
                let coset = advice_cosets.get_mut(column).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "coset cache has inconsistent dimensions",
                    )
                })?;
                if !coset.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "deferred advice coset was loaded before finalization",
                    ));
                }
                *coset = domain.coeff_to_extended_part(output_buffer, extended_omega_factor);
                advice_counts.transforms_loaded += 1;
            }
            SharedCosetTarget::Instance(column) => {
                let coset = instance_cosets.get_mut(column).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "coset cache has inconsistent dimensions",
                    )
                })?;
                if !coset.is_empty() {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "deferred instance coset was loaded before finalization",
                    ));
                }
                *coset = domain.coeff_to_extended_part(output_buffer, extended_omega_factor);
                instance_counts.transforms_loaded += 1;
            }
        }
        return Ok((fixed_counts, advice_counts, instance_counts));
    }

    // Factor storage has exactly the size of the deferred output slot. If its
    // mapped allocation cannot be created, fall back before mutating any cache.
    let factors = match domain.extended_part_factors_spilled(extended_omega_factor) {
        Ok(factors) => factors,
        Err(_) => {
            let fixed_counts = load_missing_fixed_cosets(
                domain,
                fixed_sources,
                fixed_columns,
                extended_omega_factor,
                fixed_cosets,
            )?;
            let advice_counts = load_missing_cosets(
                domain,
                advice,
                advice_columns,
                extended_omega_factor,
                advice_cosets,
            )?;
            let instance_counts = load_missing_cosets(
                domain,
                instance,
                instance_columns,
                extended_omega_factor,
                instance_cosets,
            )?;
            return Ok((fixed_counts, advice_counts, instance_counts));
        }
    };

    let (fixed_counts, mut advice_counts, mut instance_counts) = {
        let factors = &factors[..];
        let fixed_counts = load_missing_fixed_cosets_with_factors(
            domain,
            fixed_sources,
            fixed_columns,
            extended_omega_factor,
            fixed_cosets,
            Some(factors),
        )?;
        let advice_counts = load_missing_cosets_with_factors(
            domain,
            advice,
            advice_columns,
            extended_omega_factor,
            advice_cosets,
            Some(factors),
            match target {
                SharedCosetTarget::Advice(column) => Some(column),
                SharedCosetTarget::Instance(_) => None,
            },
        )?;
        let instance_counts = load_missing_cosets_with_factors(
            domain,
            instance,
            instance_columns,
            extended_omega_factor,
            instance_cosets,
            Some(factors),
            match target {
                SharedCosetTarget::Instance(column) => Some(column),
                SharedCosetTarget::Advice(_) => None,
            },
        )?;
        (fixed_counts, advice_counts, instance_counts)
    };

    match target {
        SharedCosetTarget::Advice(column) => {
            let source = advice.get(column).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "constraint references an absent polynomial",
                )
            })?;
            let coset = advice_cosets.get_mut(column).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "coset cache has inconsistent dimensions",
                )
            })?;
            if !coset.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "deferred advice coset was loaded before finalization",
                ));
            }
            *coset = domain.coeff_to_extended_part_in_factor_buffer(
                factors,
                source,
                extended_omega_factor,
            )?;
            advice_counts.transforms_loaded += 1;
        }
        SharedCosetTarget::Instance(column) => {
            let source = instance.get(column).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "constraint references an absent polynomial",
                )
            })?;
            let coset = instance_cosets.get_mut(column).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "coset cache has inconsistent dimensions",
                )
            })?;
            if !coset.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "deferred instance coset was loaded before finalization",
                ));
            }
            *coset = domain.coeff_to_extended_part_in_factor_buffer(
                factors,
                source,
                extended_omega_factor,
            )?;
            instance_counts.transforms_loaded += 1;
        }
    }
    Ok((fixed_counts, advice_counts, instance_counts))
}

#[allow(clippy::too_many_arguments)]
fn refresh_lookup_argument_cosets<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
    fixed_sources: &[FixedColumnSource<F>],
    advice: &[Polynomial<F, Coeff>],
    instance: &[Polynomial<F, Coeff>],
    dependencies: &ColumnDependencies,
    extended_omega_factor: F,
    previous_dependencies: &mut ColumnDependencies,
    fixed_cosets: &mut [Polynomial<F, LagrangeCoeff>],
    advice_cosets: &mut [Polynomial<F, LagrangeCoeff>],
    instance_cosets: &mut [Polynomial<F, LagrangeCoeff>],
) -> io::Result<(
    FixedCosetRefreshCounts,
    CosetRefreshCounts,
    CosetRefreshCounts,
)> {
    evict_inactive_cosets(
        fixed_cosets,
        &previous_dependencies.fixed,
        &dependencies.fixed,
    )?;
    evict_inactive_cosets(
        advice_cosets,
        &previous_dependencies.advice,
        &dependencies.advice,
    )?;
    evict_inactive_cosets(
        instance_cosets,
        &previous_dependencies.instance,
        &dependencies.instance,
    )?;

    let counts = load_gate_group_cosets(
        domain,
        fixed_sources,
        advice,
        instance,
        &dependencies.fixed,
        &dependencies.advice,
        &dependencies.instance,
        extended_omega_factor,
        fixed_cosets,
        advice_cosets,
        instance_cosets,
    )?;
    *previous_dependencies = dependencies.clone();
    Ok(counts)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_graph_group<C: CurveAffine>(graphs: &[GraphEvaluator<C>], fixed: &[Polynomial<C::ScalarExt, LagrangeCoeff>], advice: &[Polynomial<C::ScalarExt, LagrangeCoeff>], instance: &[Polynomial<C::ScalarExt, LagrangeCoeff>], challenges: &[C::ScalarExt], values: &mut Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>, y: C::ScalarExt, beta: C::ScalarExt, gamma: C::ScalarExt, theta: C::ScalarExt, weighted: Option<WeightedGateGroup<C::ScalarExt>>, rot_scale: i32, isize: i32, part_index: usize, part_stride: usize) -> Result<GateGroupEvalTiming, super::Error> {
    #[cfg(feature = "icicle_gpu")]
    let (program_compile_ms, gpu_h_call_ms, gpu_fallback) = {
        if crate::icicle::native_prove_ntt_scope_active()
            && std::env::var_os("ENABLE_ICICLE_GPU").is_some()
        {
            crate::icicle::record_gpu_h_eval_group();
            let compile_started = Instant::now();
            let previous_value = C::ScalarExt::ZERO;
            let program = compile_gate_program(graphs, challenges, beta, gamma, theta, y, previous_value, weighted, rot_scale, isize).map_err(super::Error::NativeStorage)?;
            let program_compile_ms = compile_started.elapsed().as_secs_f64() * 1000.0;
            if let Some(program) = program {
                let gpu_call_started = Instant::now();
                if crate::icicle::try_evaluate_h_program(&program, fixed, advice, instance, part_index, part_stride, rot_scale, isize, y, &mut values[..]).map_err(super::Error::NativeStorage)? {
                    return Ok(GateGroupEvalTiming { gpu_executed: true, gpu_fallback: false, program_compile_ms, gpu_h_call_ms: gpu_call_started.elapsed().as_secs_f64() * 1000.0, cpu_eval_ms: 0.0 });
                }
                (program_compile_ms, gpu_call_started.elapsed().as_secs_f64() * 1000.0, true)
            } else {
                crate::icicle::record_gpu_h_eval_fallback();
                (program_compile_ms, 0.0, true)
            }
        } else {
            (0.0, 0.0, false)
        }
    };
    #[cfg(not(feature = "icicle_gpu"))]
    let (program_compile_ms, gpu_h_call_ms, gpu_fallback) = (0.0, 0.0, false);
    let cpu_eval_started = Instant::now();
    let scratch_intermediates = graphs
        .iter()
        .map(|graph| graph.num_intermediates)
        .max()
        .unwrap_or(0);
    let scratch_rotations = graphs
        .iter()
        .map(|graph| graph.rotations.len())
        .max()
        .unwrap_or(0);
    parallelize(&mut values[..], |h, start| {
        let mut data = EvaluationData {
            intermediates: vec![C::ScalarExt::ZERO; scratch_intermediates],
            rotations: vec![0usize; scratch_rotations],
        };
        for (offset, value) in h.iter_mut().enumerate() {
            let row = start + offset;
            let idx = row * part_stride + part_index;
            let mut accumulated = *value;
            let mut block_accumulator = C::ScalarExt::ZERO;
            if let Some(weighted) = weighted {
                accumulated *= weighted.initial_scale;
            }
            for graph in graphs {
                let contribution = graph.evaluate_part(
                    &mut data,
                    fixed,
                    advice,
                    instance,
                    challenges,
                    &beta,
                    &gamma,
                    &theta,
                    &y,
                    &C::ScalarExt::ZERO,
                    idx,
                    rot_scale,
                    isize,
                    part_stride,
                );
                if weighted.is_some() {
                    block_accumulator = block_accumulator * y + contribution;
                } else {
                    accumulated = accumulated * y + contribution;
                }
            }
            if let Some(weighted) = weighted {
                accumulated += block_accumulator * weighted.contribution_scale;
            }
            *value = accumulated;
        }
    });
    Ok(GateGroupEvalTiming { gpu_executed: false, gpu_fallback, program_compile_ms, gpu_h_call_ms, cpu_eval_ms: cpu_eval_started.elapsed().as_secs_f64() * 1000.0 })
}

#[cfg(feature = "icicle_gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum GateProgramNodeKey {
    Query(u32),
    Constant(u32),
    Unary(u32, usize),
    Binary(u32, usize, usize),
}

#[cfg(feature = "icicle_gpu")]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GateProgramQueryKey {
    kind: u32,
    column: usize,
    rotation: i32,
}

#[cfg(feature = "icicle_gpu")]
#[derive(Clone, Copy, Debug)]
enum GateProgramStep {
    Compute(usize),
    Fold(usize),
    StoreOutput(usize, usize),
    MvLookupFold(usize),
}

#[cfg(feature = "icicle_gpu")]
const MAX_GPU_H_PROGRAM_METADATA_BYTES: u128 = 64 * 1024 * 1024;
#[cfg(feature = "icicle_gpu")]
const MAX_GPU_H_PROGRAM_NODES: usize = 65_536;
#[cfg(feature = "icicle_gpu")]
const MAX_GPU_H_PROGRAM_INSTRUCTIONS: usize = 65_536;
#[cfg(feature = "icicle_gpu")]
const MAX_GPU_H_PROGRAM_QUERIES: usize = 4_096;
#[cfg(feature = "icicle_gpu")]
const MAX_GPU_H_PROGRAM_CONSTANTS: usize = 8_192;
#[cfg(feature = "icicle_gpu")]
const MAX_GPU_H_PROGRAM_SCRATCH_SLOTS: usize = 128;

#[cfg(feature = "icicle_gpu")]
fn add_gate_program_bound(bound: &mut usize, value: usize) -> bool {
    match bound.checked_add(value) { Some(total) => { *bound = total; true }, None => false }
}

#[cfg(feature = "icicle_gpu")]
fn count_gate_program_source(source: &ValueSource, queries: &mut usize, constants: &mut usize) -> bool {
    match source {
        ValueSource::Fixed(_, _) | ValueSource::Advice(_, _) | ValueSource::Instance(_, _) => add_gate_program_bound(queries, 1),
        ValueSource::Constant(_) | ValueSource::Challenge(_) | ValueSource::Beta() | ValueSource::Gamma() | ValueSource::Theta() | ValueSource::Y() | ValueSource::PreviousValue() => add_gate_program_bound(constants, 1),
        ValueSource::Intermediate(_) => true,
    }
}

#[cfg(feature = "icicle_gpu")]
fn gate_program_bounds<C: CurveAffine>(graphs: &[&GraphEvaluator<C>], weighted: bool, mv_lookup_input_counts: Option<&[usize]>) -> Option<(usize, usize, usize, usize, usize)> {
    let mut queries = 0;
    let mut constants = 0;
    let mut operations = 0;
    let mut max_intermediates = 0;
    for graph in graphs {
        max_intermediates = max_intermediates.max(graph.num_intermediates);
        for calc in &graph.calculations {
            let valid = match &calc.calculation {
                Calculation::Add(a, b) | Calculation::Sub(a, b) | Calculation::Mul(a, b) => count_gate_program_source(a, &mut queries, &mut constants) && count_gate_program_source(b, &mut queries, &mut constants) && add_gate_program_bound(&mut operations, 1),
                Calculation::Square(value) | Calculation::Double(value) | Calculation::Negate(value) => count_gate_program_source(value, &mut queries, &mut constants) && add_gate_program_bound(&mut operations, 1),
                Calculation::Horner(start, parts, factor) => {
                    let mut valid = count_gate_program_source(start, &mut queries, &mut constants) && count_gate_program_source(factor, &mut queries, &mut constants);
                    for part in parts { valid &= count_gate_program_source(part, &mut queries, &mut constants); }
                    parts.len().checked_mul(2).map(|count| valid && add_gate_program_bound(&mut operations, count)).unwrap_or(false)
                }
                Calculation::Store(value) => count_gate_program_source(value, &mut queries, &mut constants),
            };
            if !valid { return None; }
        }
    }
    let lookup_folds = if let Some(input_counts) = mv_lookup_input_counts {
        if input_counts.is_empty() || input_counts.iter().any(|count| *count == 0) { return None; }
        let expected_graphs = input_counts.iter().try_fold(0usize, |total, count| total.checked_add(count.checked_add(1)?))?;
        if expected_graphs != graphs.len() { return None; }
        input_counts.len()
    } else { 0 };
    if weighted && lookup_folds != 0 { return None; }
    constants = constants.checked_add(if weighted { 3 } else { 1 })?;
    let nodes = queries.checked_add(constants)?.checked_add(operations)?;
    let instructions = nodes.checked_add(graphs.len())?.checked_add(lookup_folds)?.checked_add(if weighted { 2 } else { 0 })?;
    Some((nodes, instructions, queries, constants, max_intermediates))
}

#[cfg(feature = "icicle_gpu")]
fn gate_program_metadata_upper<F: PrimeField>(nodes: usize, instructions: usize, queries: usize, constants: usize, graph_count: usize, max_intermediates: usize) -> u128 {
    let n = nodes as u128;
    let i = instructions as u128;
    let q = queries as u128;
    let c = constants as u128;
    let g = graph_count as u128;
    let m = max_intermediates as u128;
    let vector_bytes = n * std::mem::size_of::<GateProgramNodeKey>() as u128 + i * (std::mem::size_of::<GateProgramStep>() + std::mem::size_of::<[u32; 8]>()) as u128 + q * std::mem::size_of::<GateQuery>() as u128 * 2 + c * std::mem::size_of::<F>() as u128 * 2 + n * (std::mem::size_of::<usize>() * 4) as u128 + g * (std::mem::size_of::<usize>() * 2) as u128 + m * std::mem::size_of::<Option<usize>>() as u128;
    let hash_bytes = n * (std::mem::size_of::<GateProgramNodeKey>() + std::mem::size_of::<usize>() + 32) as u128 * 2 + q * (std::mem::size_of::<GateProgramQueryKey>() + std::mem::size_of::<u32>() + 32) as u128 * 2;
    vector_bytes + hash_bytes
}

#[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
fn mv_lookup_program_allocated_bytes<F: PrimeField>(program: &GateProgram<F>) -> Option<usize> {
    let mut bytes = program.instructions.capacity().checked_mul(std::mem::size_of::<[u32; 8]>())?;
    bytes = bytes.checked_add(program.constants.capacity().checked_mul(std::mem::size_of::<F>())?)?;
    bytes = bytes.checked_add(program.queries.capacity().checked_mul(std::mem::size_of::<GateQuery>())?)?;
    bytes = bytes.checked_add(program.scratch_slots.checked_mul(std::mem::size_of::<bool>())?)?;
    let lookup = program.mv_lookup.as_ref()?;
    bytes = bytes.checked_add(lookup.input_counts.capacity().checked_mul(std::mem::size_of::<usize>())?)?;
    bytes = bytes.checked_add(lookup.graph_query_count.checked_mul(std::mem::size_of::<usize>())?)?;
    Some(bytes)
}

#[cfg(feature = "icicle_gpu")]
fn gate_program_index(index: usize, what: &str) -> io::Result<u32> {
    u32::try_from(index).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("GPU H program {what} exceeds u32")))
}

#[cfg(feature = "icicle_gpu")]
fn compile_gate_program<C: CurveAffine>(graphs: &[GraphEvaluator<C>], challenges: &[C::ScalarExt], beta: C::ScalarExt, gamma: C::ScalarExt, theta: C::ScalarExt, y: C::ScalarExt, previous_value: C::ScalarExt, weighted: Option<WeightedGateGroup<C::ScalarExt>>, rot_scale: i32, isize: i32) -> io::Result<Option<GateProgram<C::ScalarExt>>> {
    let graph_refs = graphs.iter().collect::<Vec<_>>();
    compile_gate_program_with_lookup(&graph_refs, challenges, beta, gamma, theta, y, previous_value, weighted, rot_scale, isize, None)
}

#[cfg(feature = "icicle_gpu")]
fn compile_gate_program_with_lookup<C: CurveAffine>(graphs: &[&GraphEvaluator<C>], challenges: &[C::ScalarExt], beta: C::ScalarExt, gamma: C::ScalarExt, theta: C::ScalarExt, y: C::ScalarExt, previous_value: C::ScalarExt, weighted: Option<WeightedGateGroup<C::ScalarExt>>, rot_scale: i32, isize: i32, mv_lookup_input_counts: Option<&[usize]>) -> io::Result<Option<GateProgram<C::ScalarExt>>> {
    struct Compiler<F: PrimeField> {
        constants: Vec<F>,
        queries: Vec<GateQuery>,
        query_indices: HashMap<GateProgramQueryKey, u32>,
        nodes: Vec<GateProgramNodeKey>,
        node_indices: HashMap<GateProgramNodeKey, usize>,
    }
    impl<F: PrimeField> Compiler<F> {
        fn intern(&mut self, key: GateProgramNodeKey) -> usize {
            if let Some(index) = self.node_indices.get(&key) { return *index; }
            let index = self.nodes.len();
            self.nodes.push(key);
            self.node_indices.insert(key, index);
            index
        }
        fn constant_index(&mut self, value: F) -> io::Result<u32> {
            let index = match self.constants.iter().position(|candidate| *candidate == value) {
                Some(index) => index,
                None => { let index = self.constants.len(); self.constants.push(value); index }
            };
            gate_program_index(index, "constant table")
        }
        fn constant(&mut self, value: F) -> io::Result<usize> {
            let index = self.constant_index(value)?;
            Ok(self.intern(GateProgramNodeKey::Constant(index)))
        }
        fn query(&mut self, kind: u32, column: usize, rotation: i32) -> io::Result<usize> {
            let key = GateProgramQueryKey { kind, column, rotation };
            let index = match self.query_indices.get(&key) {
                Some(index) => *index,
                None => { let index = gate_program_index(self.queries.len(), "query table")?; self.queries.push(GateQuery { kind, column, rotation }); self.query_indices.insert(key, index); index }
            };
            Ok(self.intern(GateProgramNodeKey::Query(index)))
        }
        fn unary(&mut self, opcode: u32, source: usize) -> usize { self.intern(GateProgramNodeKey::Unary(opcode, source)) }
        fn binary(&mut self, opcode: u32, mut lhs: usize, mut rhs: usize) -> usize {
            if (opcode == GATE_OP_ADD || opcode == GATE_OP_MUL) && rhs < lhs { std::mem::swap(&mut lhs, &mut rhs); }
            self.intern(GateProgramNodeKey::Binary(opcode, lhs, rhs))
        }
        fn source<C: CurveAffine<ScalarExt = F>>(&mut self, source: &ValueSource, graph: &GraphEvaluator<C>, intermediates: &[Option<usize>], challenges: &[F], beta: F, gamma: F, theta: F, y: F, previous_value: F, rot_scale: i32, isize: i32) -> io::Result<usize> {
            match source {
                ValueSource::Constant(index) => self.constant(*graph.constants.get(*index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program constant index is invalid"))?),
                ValueSource::Intermediate(index) => intermediates.get(*index).and_then(|node| *node).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program intermediate is not topologically available")),
                ValueSource::Fixed(column, rotation) | ValueSource::Advice(column, rotation) | ValueSource::Instance(column, rotation) => {
                    if rot_scale <= 0 || isize <= 0 || isize % rot_scale != 0 { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program rotation geometry is invalid")); }
                    let rotation = graph.rotations.get(*rotation).copied().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program rotation index is invalid"))?.rem_euclid(isize / rot_scale);
                    let kind = match source { ValueSource::Fixed(_, _) => 0, ValueSource::Advice(_, _) => 1, _ => 2 };
                    self.query(kind, *column, rotation)
                }
                ValueSource::Challenge(index) => self.constant(*challenges.get(*index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program challenge index is invalid"))?),
                ValueSource::Beta() => self.constant(beta),
                ValueSource::Gamma() => self.constant(gamma),
                ValueSource::Theta() => self.constant(theta),
                ValueSource::Y() => self.constant(y),
                ValueSource::PreviousValue() => self.constant(previous_value),
            }
        }
    }
    if rot_scale <= 0 || isize <= 0 || isize % rot_scale != 0 { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program rotation geometry is invalid")); }
    let Some((node_upper, instruction_upper, query_upper, constant_upper, max_intermediates)) = gate_program_bounds(graphs, weighted.is_some(), mv_lookup_input_counts) else { return Ok(None); };
    if node_upper > MAX_GPU_H_PROGRAM_NODES || instruction_upper > MAX_GPU_H_PROGRAM_INSTRUCTIONS || query_upper > MAX_GPU_H_PROGRAM_QUERIES || constant_upper > MAX_GPU_H_PROGRAM_CONSTANTS || max_intermediates > MAX_GPU_H_PROGRAM_NODES || gate_program_metadata_upper::<C::ScalarExt>(node_upper, instruction_upper, query_upper, constant_upper, graphs.len(), max_intermediates) > MAX_GPU_H_PROGRAM_METADATA_BYTES { return Ok(None); }
    let mut compiler = Compiler { constants: Vec::with_capacity(constant_upper), queries: Vec::with_capacity(query_upper), query_indices: HashMap::with_capacity(query_upper), nodes: Vec::with_capacity(node_upper), node_indices: HashMap::with_capacity(node_upper) };
    let mut roots = Vec::with_capacity(graphs.len());
    let mut graph_node_ends = Vec::with_capacity(graphs.len());
    for graph in graphs {
        let mut intermediates = vec![None; graph.num_intermediates];
        for calc in &graph.calculations {
            let node = match &calc.calculation {
                Calculation::Add(a, b) => { let lhs = compiler.source(a, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; let rhs = compiler.source(b, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; compiler.binary(GATE_OP_ADD, lhs, rhs) }
                Calculation::Sub(a, b) => { let lhs = compiler.source(a, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; let rhs = compiler.source(b, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; compiler.binary(GATE_OP_SUB, lhs, rhs) }
                Calculation::Mul(a, b) => { let lhs = compiler.source(a, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; let rhs = compiler.source(b, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; compiler.binary(GATE_OP_MUL, lhs, rhs) }
                Calculation::Square(value) => { let value = compiler.source(value, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; compiler.unary(GATE_OP_SQUARE, value) }
                Calculation::Double(value) => { let value = compiler.source(value, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; compiler.unary(GATE_OP_DOUBLE, value) }
                Calculation::Negate(value) => { let value = compiler.source(value, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; compiler.unary(GATE_OP_NEGATE, value) }
                Calculation::Horner(start, parts, factor) => {
                    let mut value = compiler.source(start, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?;
                    let factor = compiler.source(factor, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?;
                    for part in parts { let part = compiler.source(part, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?; let product = compiler.binary(GATE_OP_MUL, value, factor); value = compiler.binary(GATE_OP_ADD, product, part); }
                    value
                }
                Calculation::Store(value) => compiler.source(value, graph, &intermediates, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize)?,
            };
            let target = intermediates.get_mut(calc.target).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program calculation target is invalid"))?;
            *target = Some(node);
        }
        let root = match graph.calculations.last() { Some(calc) => intermediates.get(calc.target).and_then(|node| *node).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program root is unavailable"))?, None => compiler.constant(C::ScalarExt::ZERO)? };
        roots.push(root);
        graph_node_ends.push(compiler.nodes.len());
    }
    let weighted_constants = weighted
        .map(|group| {
            Ok::<_, io::Error>((
                compiler.constant_index(group.initial_scale)?,
                compiler.constant_index(group.contribution_scale)?,
            ))
        })
        .transpose()?;
    let mut steps = Vec::with_capacity(instruction_upper);
    let mut node_start = 0;
    let mut lookup_output_offsets = Vec::new();
    if let Some(input_counts) = mv_lookup_input_counts {
        let mut graph_index = 0usize;
        let mut output_index = 0usize;
        for (lookup_index, input_count) in input_counts.iter().copied().enumerate() {
            lookup_output_offsets.push(output_index);
            for _ in 0..=input_count {
                let root = *roots.get(graph_index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup graph grouping is incomplete"))?;
                let node_end = *graph_node_ends.get(graph_index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup graph boundary is absent"))?;
                steps.extend((node_start..node_end).map(GateProgramStep::Compute));
                steps.push(GateProgramStep::StoreOutput(root, output_index));
                node_start = node_end;
                graph_index += 1;
                output_index += 1;
            }
            steps.push(GateProgramStep::MvLookupFold(lookup_index));
        }
        if graph_index != roots.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup graph grouping has trailing graphs"));
        }
    } else {
        for (root, node_end) in roots.iter().copied().zip(graph_node_ends.iter().copied()) {
            steps.extend((node_start..node_end).map(GateProgramStep::Compute));
            steps.push(GateProgramStep::Fold(root));
            node_start = node_end;
        }
    }
    let mut last_use = vec![0usize; compiler.nodes.len()];
    let mut remaining_uses = vec![0usize; compiler.nodes.len()];
    for (step_index, step) in steps.iter().enumerate() {
        match step {
            GateProgramStep::Compute(node) => { last_use[*node] = step_index; match compiler.nodes[*node] {
                GateProgramNodeKey::Unary(_, source) => { last_use[source] = step_index; remaining_uses[source] += 1; }
                GateProgramNodeKey::Binary(_, lhs, rhs) => { last_use[lhs] = step_index; last_use[rhs] = step_index; remaining_uses[lhs] += 1; remaining_uses[rhs] += 1; }
                GateProgramNodeKey::Query(_) | GateProgramNodeKey::Constant(_) => {}
            } }
            GateProgramStep::Fold(root) | GateProgramStep::StoreOutput(root, _) => { last_use[*root] = step_index; remaining_uses[*root] += 1; }
            GateProgramStep::MvLookupFold(_) => {}
        }
    }
    let mut node_slots = vec![0usize; compiler.nodes.len()];
    let mut free_slots = Vec::with_capacity(compiler.nodes.len());
    let mut instructions = Vec::with_capacity(instruction_upper);
    if let Some((initial_scale, _)) = weighted_constants {
        instructions.push([GATE_OP_SCALE_ACCUMULATOR, initial_scale, 0, 0, 0, 0, 0, 0]);
    }
    let mut scratch_slots = 0usize;
    let mut output_instructions = Vec::new();
    let mut lookup_fold_instructions = Vec::new();
    for (step_index, step) in steps.iter().enumerate() {
        match step {
            GateProgramStep::Compute(node) => {
                let key = compiler.nodes[*node];
                let sources = match key { GateProgramNodeKey::Unary(_, source) => [Some(source), None], GateProgramNodeKey::Binary(_, lhs, rhs) => [Some(lhs), Some(rhs)], _ => [None, None] };
                let mut destination = None;
                for source in sources.iter().flatten().copied() {
                    if sources.iter().flatten().filter(|candidate| **candidate == source).count() == remaining_uses[source]
                        && last_use[source] == step_index
                    {
                        destination = Some(node_slots[source]);
                        break;
                    }
                }
                let destination = destination.or_else(|| free_slots.pop()).unwrap_or_else(|| { let slot = scratch_slots; scratch_slots += 1; slot });
                let dst = gate_program_index(destination, "scratch slot")?;
                let instruction = match key {
                    GateProgramNodeKey::Query(query) => [GATE_OP_LOAD_QUERY, dst, query, 0, 0, 0, 0, 0],
                    GateProgramNodeKey::Constant(constant) => [GATE_OP_LOAD_CONST, dst, constant, 0, 0, 0, 0, 0],
                    GateProgramNodeKey::Unary(opcode, source) => [opcode, dst, gate_program_index(node_slots[source], "scratch slot")?, 0, 0, 0, 0, 0],
                    GateProgramNodeKey::Binary(opcode, lhs, rhs) => [opcode, dst, gate_program_index(node_slots[lhs], "scratch slot")?, gate_program_index(node_slots[rhs], "scratch slot")?, 0, 0, 0, 0],
                };
                instructions.push(instruction);
                node_slots[*node] = destination;
                let mut released = [None, None];
                let mut released_len = 0;
                for source in sources.iter().flatten().copied() {
                    if released[..released_len].contains(&Some(source)) { continue; }
                    released[released_len] = Some(source);
                    released_len += 1;
                    let uses_here = sources.iter().flatten().filter(|candidate| **candidate == source).count();
                    remaining_uses[source] = remaining_uses[source].checked_sub(uses_here).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program use count underflow"))?;
                    if remaining_uses[source] == 0 && node_slots[source] != destination {
                        free_slots.push(node_slots[source]);
                    }
                }
                if remaining_uses[*node] == 0 { free_slots.push(destination); }
            }
            GateProgramStep::Fold(root) => {
                let slot = node_slots[*root];
                let opcode = if weighted.is_some() { GATE_OP_FOLD_BLOCK } else { GATE_OP_FOLD };
                instructions.push([opcode, gate_program_index(slot, "scratch slot")?, 0, 0, 0, 0, 0, 0]);
                remaining_uses[*root] = remaining_uses[*root].checked_sub(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program fold use count underflow"))?;
                if remaining_uses[*root] == 0 { free_slots.push(slot); }
            }
            GateProgramStep::StoreOutput(root, output_index) => {
                let slot = node_slots[*root];
                let instruction_index = instructions.len();
                instructions.push([GATE_OP_STORE_OUTPUT, gate_program_index(*output_index, "lookup output slot")?, gate_program_index(slot, "scratch slot")?, 0, 0, 0, 0, 0]);
                output_instructions.push((instruction_index, *output_index));
                remaining_uses[*root] = remaining_uses[*root].checked_sub(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup output use count underflow"))?;
                if remaining_uses[*root] == 0 { free_slots.push(slot); }
            }
            GateProgramStep::MvLookupFold(lookup_index) => {
                let input_count = *mv_lookup_input_counts.and_then(|counts| counts.get(*lookup_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup fold index is invalid"))?;
                let instruction_index = instructions.len();
                instructions.push([GATE_OP_MV_LOOKUP_FOLD, 0, gate_program_index(input_count, "multi-lookup input count")?, gate_program_index(*lookup_index, "multi-lookup index")?, 0, 0, 0, 0]);
                let graph_offset = *lookup_output_offsets.get(*lookup_index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup output offset is missing"))?;
                lookup_fold_instructions.push((instruction_index, *lookup_index, graph_offset));
            }
        }
    }
    if let Some((_, contribution_scale)) = weighted_constants {
        instructions.push([GATE_OP_COMMIT_BLOCK, contribution_scale, 0, 0, 0, 0, 0, 0]);
    }
    if remaining_uses.iter().any(|uses| *uses != 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GPU H program liveness scan left unconsumed node uses",
        ));
    }
    if scratch_slots > MAX_GPU_H_PROGRAM_SCRATCH_SLOTS { return Ok(None); }
    let mv_lookup = if let Some(input_counts) = mv_lookup_input_counts {
        let output_base = scratch_slots;
        let output_count = roots.len();
        let total_scratch = scratch_slots.checked_add(output_count).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup scratch size overflows"))?;
        if output_count > 8192 || total_scratch > 16_384 { return Ok(None); }
        for (instruction_index, output_index) in output_instructions {
            instructions[instruction_index][1] = gate_program_index(output_base.checked_add(output_index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup output slot overflows"))?, "multi-lookup output slot")?;
        }
        for (instruction_index, _lookup_index, graph_offset) in lookup_fold_instructions {
            instructions[instruction_index][1] = gate_program_index(output_base.checked_add(graph_offset).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup graph output offset overflows"))?, "multi-lookup output base")?;
        }
        Some(MvLookupProgram { output_base, input_counts: input_counts.to_vec(), graph_query_count: compiler.queries.len() })
    } else { None };
    let scratch_slots = if mv_lookup.is_some() { scratch_slots + roots.len() } else { scratch_slots };
    let fold_count = if let Some(program) = &mv_lookup { program.input_counts.len().checked_mul(3).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU multi-lookup fold count overflows"))? } else { roots.len() };
    Ok(Some(GateProgram { instructions, constants: compiler.constants, queries: compiler.queries, scratch_slots, fold_count, mv_lookup }))
}

#[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
fn compile_mv_lookup_program<C: CurveAffine>(lookups: &[(&[GraphEvaluator<C>], &GraphEvaluator<C>)], challenges: &[C::ScalarExt], beta: C::ScalarExt, gamma: C::ScalarExt, theta: C::ScalarExt, y: C::ScalarExt, rot_scale: i32, isize: i32) -> io::Result<Option<GateProgram<C::ScalarExt>>> {
    let lookup_count = lookups.len();
    let graph_count = lookups.iter().try_fold(0usize, |total, (inputs, _)| {
        if inputs.is_empty() { return None; }
        total.checked_add(inputs.len().checked_add(1)?)
    });
    let Some(graph_count) = graph_count else { return Ok(None); };
    let mut graphs = Vec::new();
    let mut input_counts = Vec::new();
    if graphs.try_reserve_exact(graph_count).is_err() || input_counts.try_reserve_exact(lookup_count).is_err() {
        return Ok(None);
    }
    for (inputs, table) in lookups {
        input_counts.push(inputs.len());
        graphs.extend(inputs.iter());
        graphs.push(*table);
    }
    compile_gate_program_with_lookup(
        &graphs,
        challenges,
        beta,
        gamma,
        theta,
        y,
        C::ScalarExt::ZERO,
        None,
        rot_scale,
        isize,
        Some(&input_counts),
    )
}

impl<C: CurveAffine> Evaluator<C> {
    /// Creates a new evaluation structure
    pub fn new(cs: &ConstraintSystem<C::ScalarExt>) -> Self {
        let mut ev = Evaluator::default();

        // Custom gates
        let mut custom_gates = Vec::new();
        for gate in cs.gates.iter() {
            for poly in gate.polynomials() {
                let mut graph = GraphEvaluator::default();
                let root = graph.add_expression(poly);
                graph.add_calculation(Calculation::Store(root));
                custom_gates.push(graph);
            }
        }
        ev.custom_gates = custom_gates;

        // Lookups
        #[cfg(feature = "mv-lookup")]
        for lookup in cs.lookups.iter() {
            let mut graph_table = GraphEvaluator::default();
            let mut graph_inputs: Vec<_> = (0..lookup.inputs_expressions.len())
                .map(|_| GraphEvaluator::default())
                .collect();

            let evaluate_lc = |graph: &mut GraphEvaluator<C>, expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Inputs cosets
            for (input_expressions, graph_input) in lookup
                .inputs_expressions
                .iter()
                .zip(graph_inputs.iter_mut())
            {
                let compressed_input_coset = evaluate_lc(graph_input, input_expressions);

                graph_input.add_calculation(Calculation::Add(
                    compressed_input_coset,
                    ValueSource::Beta(),
                ));
            }

            // table coset
            let compressed_table_coset = evaluate_lc(&mut graph_table, &lookup.table_expressions);

            graph_table.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Beta(),
            ));

            /*
                a) f_i + beta
                b) t + beta
            */
            ev.lookups.push((graph_inputs.to_vec(), graph_table));
        }

        #[cfg(not(feature = "mv-lookup"))]
        // Lookups
        for lookup in cs.lookups.iter() {
            let mut graph = GraphEvaluator::default();

            let mut evaluate_lc = |expressions: &Vec<Expression<_>>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            // Input coset
            let compressed_input_coset = evaluate_lc(&lookup.input_expressions);
            // table coset
            let compressed_table_coset = evaluate_lc(&lookup.table_expressions);
            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            let right_gamma = graph.add_calculation(Calculation::Add(
                compressed_table_coset,
                ValueSource::Gamma(),
            ));
            let lc = graph.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Beta(),
            ));
            graph.add_calculation(Calculation::Mul(lc, right_gamma));

            ev.lookups.push(graph);
        }

        // Shuffles
        for shuffle in cs.shuffles.iter() {
            let evaluate_lc = |expressions: &Vec<Expression<_>>, graph: &mut GraphEvaluator<C>| {
                let parts = expressions
                    .iter()
                    .map(|expr| graph.add_expression(expr))
                    .collect();
                graph.add_calculation(Calculation::Horner(
                    ValueSource::Constant(0),
                    parts,
                    ValueSource::Theta(),
                ))
            };

            let mut graph_input = GraphEvaluator::default();
            let compressed_input_coset = evaluate_lc(&shuffle.input_expressions, &mut graph_input);
            let _ = graph_input.add_calculation(Calculation::Add(
                compressed_input_coset,
                ValueSource::Gamma(),
            ));

            let mut graph_shuffle = GraphEvaluator::default();
            let compressed_shuffle_coset =
                evaluate_lc(&shuffle.shuffle_expressions, &mut graph_shuffle);
            let _ = graph_shuffle.add_calculation(Calculation::Add(
                compressed_shuffle_coset,
                ValueSource::Gamma(),
            ));

            ev.shuffles.push(graph_input);
            ev.shuffles.push(graph_shuffle);
        }

        ev
    }

    fn native_resource_shape(&self) -> NativeEvaluationShape {
        let custom_gate_dependency_columns = self
            .custom_gates
            .iter()
            .map(|graph| graph.column_dependencies().column_count())
            .collect();

        #[cfg(feature = "mv-lookup")]
        let (lookup_dependency_columns, lookup_query_polynomial_counts) = {
            let mut dependencies = Vec::with_capacity(self.lookups.len());
            let mut query_counts = Vec::with_capacity(self.lookups.len());
            for (input_graphs, table_graph) in &self.lookups {
                let mut group = ColumnDependencies::default();
                for graph in input_graphs.iter().chain(std::iter::once(table_graph)) {
                    group.extend(&graph.column_dependencies());
                }
                dependencies.push(group.column_count());
                query_counts.push(input_graphs.len());
            }
            (dependencies, query_counts)
        };

        #[cfg(not(feature = "mv-lookup"))]
        let (lookup_dependency_columns, lookup_query_polynomial_counts) = {
            let dependencies = self
                .lookups
                .iter()
                .map(|graph| graph.column_dependencies().column_count())
                .collect();
            (dependencies, vec![1; self.lookups.len()])
        };

        let shuffle_dependency_columns = self
            .shuffles
            .chunks_exact(2)
            .map(|graphs| {
                let mut group = graphs[0].column_dependencies();
                group.extend(&graphs[1].column_dependencies());
                group.column_count()
            })
            .collect();

        NativeEvaluationShape {
            custom_gate_dependency_columns,
            lookup_dependency_columns,
            lookup_query_polynomial_counts,
            shuffle_dependency_columns,
        }
    }

    /// Evaluate the native quotient polynomial one extended-domain part at a
    /// time. Within each part the Horner order remains gate -> permutation ->
    /// lookup -> shuffle; inputs use the native part FFT and the completed
    /// values are scattered into the full extended-domain output.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn evaluate_h(
        &self,
        pk: &ProvingKey<C>,
        advice_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        instance_polys: &[&[Polynomial<C::ScalarExt, Coeff>]],
        challenges: &[C::ScalarExt],
        y: C::ScalarExt,
        beta: C::ScalarExt,
        gamma: C::ScalarExt,
        theta: C::ScalarExt,
        lookups: &[Vec<lookup::prover::Committed<C>>],
        shuffles: &[Vec<shuffle::prover::Committed<C>>],
        permutations: &[permutation::prover::Committed<C>],
    ) -> Result<Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>, super::Error> {
        let domain = &pk.vk.domain;
        let size = domain.extended_len();
        let rot_scale = 1 << (domain.extended_k() - domain.k());
        let isize = size as i32;
        let one = C::ScalarExt::ONE;
        let l0 = &pk.l0;
        let l_last = &pk.l_last;
        let l_active_row = &pk.l_active_row;
        let p = &pk.vk.cs.permutation;
        let extended_omega = domain.get_extended_omega();
        let mut values: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
            Polynomial::mapped_filled(size, C::ScalarExt::ZERO)
                .map_err(super::Error::NativeStorage)?;
        assert!(rot_scale > 0);
        let part_stride = rot_scale as usize;
        assert_eq!(size % part_stride, 0);
        let part_size = size / part_stride;
        let gate_groups = custom_gate_groups(&self.custom_gates);
        let fixed_constant_columns = pk
            .fixed_values
            .iter()
            .enumerate()
            .filter_map(|(column, source)| source.constant_value().map(|_| column))
            .collect::<BTreeSet<_>>();
        let (gate_group_order, original_fft_loads, scheduled_fft_loads, reordered_gate_groups) =
            schedule_custom_gate_groups(
                &self.custom_gates,
                &gate_groups,
                &fixed_constant_columns,
            );
        if reordered_gate_groups {
            eprintln!(
                "EZKL_NATIVE_GATE_GROUP_SCHEDULE custom_gates={} groups={} original_estimated_fft_loads={} scheduled_estimated_fft_loads={} max_live_dependency_columns={} candidate_bound=32",
                self.custom_gates.len(),
                gate_groups.len(),
                original_fft_loads,
                scheduled_fft_loads,
                self.custom_gates
                    .iter()
                    .map(|graph| graph.column_dependencies().column_count())
                    .max()
                    .unwrap_or(0),
            );
        }
        let initial_gate_scale = y.pow_vartime([self.custom_gates.len() as u64]);

        for part_index in 0..part_stride {
            let extended_omega_factor = extended_omega.pow_vartime([part_index as u64]);
            let mut part_values: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
                Polynomial::mapped_filled(part_size, C::ScalarExt::ZERO)
                    .map_err(super::Error::NativeStorage)?;

            for (proof_instance_index, (((advice, instance), lookup_sets), (shuffle_sets, permutation))) in advice_polys
                .iter()
                .zip(instance_polys.iter())
                .zip(lookups.iter())
                .zip(shuffles.iter().zip(permutations.iter()))
                .enumerate()
            {
                // Custom gate polynomials are folded in their existing native order.
                let mut fixed_cosets = empty_coset_cache::<C::ScalarExt>(pk.fixed_values.len());
                let mut advice_cosets = empty_coset_cache::<C::ScalarExt>(advice.len());
                let mut instance_cosets = empty_coset_cache::<C::ScalarExt>(instance.len());
                let mut previous_fixed_columns = BTreeSet::new();
                let mut previous_advice_columns = BTreeSet::new();
                let mut previous_instance_columns = BTreeSet::new();
                let mut cache_summary = GateCosetCacheSummary::default();
                let mut gate_eval_summary = GateGroupEvalSummary::default();
                for (group_position, group_index) in gate_group_order.iter().copied().enumerate() {
                    let group = &gate_groups[group_index];
                    let dependencies = &group.dependencies;
                    {
                        // Evict all source classes before loading any missing
                        // columns, keeping the transient live union within the
                        // existing dependency budget.
                        evict_inactive_cosets(&mut fixed_cosets, &previous_fixed_columns, &dependencies.fixed).map_err(super::Error::NativeStorage)?;
                        evict_inactive_cosets(&mut advice_cosets, &previous_advice_columns, &dependencies.advice).map_err(super::Error::NativeStorage)?;
                        evict_inactive_cosets(&mut instance_cosets, &previous_instance_columns, &dependencies.instance).map_err(super::Error::NativeStorage)?;

                        let source_conversion_started = Instant::now();
                        let (fixed_counts, advice_counts, instance_counts) = load_gate_group_cosets(
                            domain,
                            &pk.fixed_values,
                            advice,
                            instance,
                            &dependencies.fixed,
                            &dependencies.advice,
                            &dependencies.instance,
                            extended_omega_factor,
                            &mut fixed_cosets,
                            &mut advice_cosets,
                            &mut instance_cosets,
                        )
                        .map_err(super::Error::NativeStorage)?;
                        gate_eval_summary.coset_prepare_ms += source_conversion_started.elapsed().as_secs_f64() * 1000.0;

                        cache_summary.groups += 1;
                        cache_summary.fixed.transforms_loaded += fixed_counts.transforms_loaded;
                        cache_summary.fixed.transforms_reused += fixed_counts.transforms_reused;
                        cache_summary.fixed.constant_fills_loaded += fixed_counts.constant_fills_loaded;
                        cache_summary.fixed.constant_fills_reused += fixed_counts.constant_fills_reused;
                        cache_summary.advice.transforms_loaded += advice_counts.transforms_loaded;
                        cache_summary.advice.transforms_reused += advice_counts.transforms_reused;
                        cache_summary.instance.transforms_loaded += instance_counts.transforms_loaded;
                        cache_summary.instance.transforms_reused += instance_counts.transforms_reused;

                        let weighted = reordered_gate_groups.then_some(WeightedGateGroup {
                            initial_scale: if group_position == 0 {
                                initial_gate_scale
                            } else {
                                one
                            },
                            contribution_scale: y.pow_vartime([
                                (self.custom_gates.len() - group.end) as u64,
                            ]),
                        });
                        let timing = evaluate_graph_group(&self.custom_gates[group.start..group.end], &fixed_cosets, &advice_cosets, &instance_cosets, challenges, &mut part_values, y, beta, gamma, theta, weighted, rot_scale, isize, part_index, part_stride)?;
                        gate_eval_summary.rows += part_values.len();
                        if timing.gpu_executed { gate_eval_summary.gpu_groups += 1; gate_eval_summary.gpu_rows += part_values.len(); }
                        else { gate_eval_summary.cpu_groups += 1; gate_eval_summary.cpu_rows += part_values.len(); }
                        if timing.gpu_fallback { gate_eval_summary.gpu_fallback_groups += 1; gate_eval_summary.gpu_fallback_rows += part_values.len(); }
                        gate_eval_summary.program_compile_ms += timing.program_compile_ms;
                        gate_eval_summary.gpu_or_cpu_eval_wall_ms += timing.gpu_h_call_ms + timing.cpu_eval_ms;
                        gate_eval_summary.gpu_h_call_ms += timing.gpu_h_call_ms;
                        gate_eval_summary.cpu_eval_ms += timing.cpu_eval_ms;
                    }
                    previous_fixed_columns = dependencies.fixed.clone();
                    previous_advice_columns = dependencies.advice.clone();
                    previous_instance_columns = dependencies.instance.clone();
                }
                if cache_summary.groups != 0 {
                    eprintln!(
                        "EZKL_NATIVE_CACHE_SUMMARY stage=custom_gate_cosets part_index={} instance_index={} groups={} advice_coeff_to_extended_loaded={} advice_coeff_to_extended_reused={} instance_coeff_to_extended_loaded={} instance_coeff_to_extended_reused={} fixed_nonconstant_cosets_loaded={} fixed_nonconstant_cosets_reused={} fixed_constant_fills_loaded={} fixed_constant_fills_reused={} rows={} gpu_h_groups={} gpu_h_rows={} cpu_groups={} cpu_rows={} gpu_fallback_groups={} gpu_fallback_rows={} coset_prepare_ms={:.3} program_compile_ms={:.3} gpu_or_cpu_eval_wall_ms={:.3} gpu_h_call_ms={:.3} cpu_eval_ms={:.3}",
                        part_index,
                        proof_instance_index,
                        cache_summary.groups,
                        cache_summary.advice.transforms_loaded,
                        cache_summary.advice.transforms_reused,
                        cache_summary.instance.transforms_loaded,
                        cache_summary.instance.transforms_reused,
                        cache_summary.fixed.transforms_loaded,
                        cache_summary.fixed.transforms_reused,
                        cache_summary.fixed.constant_fills_loaded,
                        cache_summary.fixed.constant_fills_reused,
                        gate_eval_summary.rows,
                        gate_eval_summary.gpu_groups,
                        gate_eval_summary.gpu_rows,
                        gate_eval_summary.cpu_groups,
                        gate_eval_summary.cpu_rows,
                        gate_eval_summary.gpu_fallback_groups,
                        gate_eval_summary.gpu_fallback_rows,
                        gate_eval_summary.coset_prepare_ms,
                        gate_eval_summary.program_compile_ms,
                        gate_eval_summary.gpu_or_cpu_eval_wall_ms,
                        gate_eval_summary.gpu_h_call_ms,
                        gate_eval_summary.cpu_eval_ms,
                    );
                }
                drop(fixed_cosets);
                drop(advice_cosets);
                drop(instance_cosets);

                #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                let mut gpu_lookup_programs = None;
                #[cfg(feature = "icicle_gpu")]
                let mut gpu_lookup_session = None;
                #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                let mut gpu_lookup_kernel_capacity = None;
                #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                let mut gpu_lookup_preflight_supported = false;
                #[cfg(feature = "icicle_gpu")]
                let gpu_h_requested = crate::icicle::native_prove_ntt_scope_active()
                    && std::env::var_os("ENABLE_ICICLE_GPU").is_some();
                #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                if gpu_h_requested && !lookup_sets.is_empty() {
                    for _ in lookup_sets.iter() {
                        crate::icicle::record_gpu_h_eval_group();
                    }
                    let mut programs = Vec::new();
                    let mut aggregate_metadata_bytes = 0usize;
                    let workspace_target = part_size.max(crate::icicle::FORMAL_GPU_NTT_MIN_SIZE).checked_next_power_of_two();
                    let mut supported = advice_polys.len() == 1
                        && lookup_sets.len() == self.lookups.len()
                        && workspace_target.is_some_and(|target| target <= crate::icicle::FORMAL_GPU_NTT_MAX_SIZE)
                        && l0.len() == size
                        && l_last.len() == size
                        && l_active_row.len() == size;
                    if supported && programs.try_reserve_exact(self.lookups.len()).is_err() { supported = false; }
                    if supported {
                        let target = workspace_target.unwrap();
                        for (input_graphs, table_graph) in &self.lookups {
                            if input_graphs.is_empty() { supported = false; break; }
                            let one_lookup = [(input_graphs.as_slice(), table_graph)];
                            let program = match compile_mv_lookup_program(&one_lookup, challenges, beta, gamma, theta, y, rot_scale, isize)
                                .map_err(super::Error::NativeStorage)? {
                                Some(program) => program,
                                None => { supported = false; break; }
                            };
                            let Some(program_bytes) = mv_lookup_program_allocated_bytes(&program) else { supported = false; break; };
                            let Some(total_metadata) = aggregate_metadata_bytes.checked_add(program_bytes) else { supported = false; break; };
                            aggregate_metadata_bytes = total_metadata;
                            if aggregate_metadata_bytes > crate::icicle::FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES {
                                supported = false;
                                break;
                            }
                            let Some(meta) = program.mv_lookup.as_ref() else { supported = false; break; };
                            let query_count = match meta.graph_query_count.checked_add(6) { Some(count) => count, None => { supported = false; break; } };
                            let fixed_input = match program.instructions.len().checked_add(program.constants.len()).and_then(|count| count.checked_add(1)) { Some(count) => count, None => { supported = false; break; } };
                            if query_count == 0 || fixed_input >= target || program.scratch_slots == 0 { supported = false; break; }
                            let rows_by_input = (target - fixed_input) / query_count;
                            let rows_by_output = target / program.scratch_slots;
                            let chunk = (rows_by_input.min(rows_by_output).min(part_size).min(1 << 16) / 32) * 32;
                            if chunk < 32 { supported = false; break; }
                            if program.queries.iter().any(|query| match query.kind {
                                0 => query.column >= pk.fixed_values.len(),
                                1 => query.column >= advice.len(),
                                2 => query.column >= instance.len(),
                                _ => true,
                            }) {
                                supported = false;
                                break;
                            }
                            programs.push(program);
                        }
                    }
                    if supported {
                        gpu_lookup_kernel_capacity = crate::icicle::resident_h_lookup_kernel_capacity()
                            .map_err(super::Error::NativeStorage)?;
                        supported = gpu_lookup_kernel_capacity.is_some();
                    }
                    if supported {
                        gpu_lookup_programs = Some(programs);
                        gpu_lookup_preflight_supported = true;
                    } else {
                        for _ in lookup_sets.iter() {
                            crate::icicle::record_gpu_h_eval_fallback();
                        }
                    }
                }

                // Permutation boundary and copy constraints.
                let sets = &permutation.sets;
                let permutation_h_started = Instant::now();
                #[cfg(feature = "icicle_gpu")]
                let permutation_gpu_stats_before = crate::icicle::execution_stats();
                #[cfg(feature = "icicle_gpu")]
                let mut gpu_permutation_used = false;
                #[cfg(not(feature = "icicle_gpu"))]
                let gpu_permutation_used = false;
                if !sets.is_empty() {
                    let sigma_omega_powers = permutation::omega_powers(domain)
                        .map_err(super::Error::NativeStorage)?;
                    let chunk_len = pk.vk.cs.degree() - 2;
                    let column_groups = p.columns.chunks(chunk_len).collect::<Vec<_>>();
                    if sets.len() != column_groups.len() {
                        return Err(super::Error::NativeStorage(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "permutation product set count does not match circuit columns",
                        )));
                    }
                    let last_rotation = Rotation(-((pk.vk.cs.blinding_factors() + 1) as i32));

                    #[cfg(feature = "icicle_gpu")]
                    if gpu_h_requested {
                        for _ in sets.iter() {
                            crate::icicle::record_gpu_h_eval_group();
                        }
                        let geometry_supported = chunk_len > 0
                            && part_size >= 32
                            && part_size <= crate::icicle::FORMAL_GPU_NTT_MAX_SIZE
                            && part_stride > 0
                            && part_index < part_stride
                            && isize > 0
                            && usize::try_from(isize).ok() == Some(size)
                            && l0.len() == size
                            && l_last.len() == size
                            && l_active_row.len() == size
                            && size % part_stride == 0
                            && size / part_stride == part_size;
                        let kernel_capacity = if geometry_supported {
                            crate::icicle::resident_h_lookup_kernel_capacity()
                                .map_err(super::Error::NativeStorage)?
                        } else {
                            None
                        };
                        if let Some(kernel_capacity) = kernel_capacity {
                            gpu_lookup_session = crate::icicle::try_begin_resident_h_session(
                                &part_values[..],
                                kernel_capacity,
                                true,
                            )
                            .map_err(super::Error::NativeStorage)?;
                        }
                        if let Some(session) = gpu_lookup_session.as_mut() {
                            let total_terms = permutation_term_count(sets.len())
                                .map_err(super::Error::NativeStorage)?;
                            session
                                .initialize_permutation_twiddles(
                                    extended_omega,
                                    part_stride,
                                    part_index,
                                )
                                .map_err(super::Error::NativeStorage)?;
                            session
                                .scale_permutation_h(y, total_terms)
                                .map_err(super::Error::NativeStorage)?;
                            let mut column_index = 0usize;
                            let mut delta_power = C::ScalarExt::ONE;
                            for (set_index, (set, columns)) in sets.iter().zip(column_groups.iter()).enumerate() {
                                let z = domain
                                    .coeff_to_extended_part_spilled(
                                        &set.permutation_product_poly,
                                        extended_omega_factor,
                                    )
                                    .map_err(super::Error::NativeStorage)?;
                                session
                                    .begin_permutation_set(
                                        &z[..],
                                        &l0[..],
                                        &l_last[..],
                                        y,
                                        total_terms,
                                        set_index,
                                        sets.len(),
                                        last_rotation.0,
                                        part_index,
                                        part_stride,
                                        rot_scale,
                                        isize,
                                    )
                                    .map_err(super::Error::NativeStorage)?;
                                drop(z);
                                for column in columns.iter() {
                                    let source_coset = match column.column_type() {
                                        Any::Advice(_) => domain
                                            .coeff_to_extended_part_spilled(
                                                &advice[column.index()],
                                                extended_omega_factor,
                                            )
                                            .map_err(super::Error::NativeStorage)?,
                                        Any::Fixed => {
                                            let lagrange = pk.fixed_values[column.index()]
                                                .materialize_lagrange()
                                                .map_err(super::Error::NativeStorage)?;
                                            let coeff = domain
                                                .lagrange_to_coeff(lagrange)
                                                .spill()
                                                .map_err(super::Error::NativeStorage)?;
                                            domain.coeff_to_extended_part(coeff, extended_omega_factor)
                                        }
                                        Any::Instance => domain
                                            .coeff_to_extended_part_spilled(
                                                &instance[column.index()],
                                                extended_omega_factor,
                                            )
                                            .map_err(super::Error::NativeStorage)?,
                                    };
                                    let sigma_poly = pk
                                        .permutation
                                        .sigma_coeff(domain, column_index, &sigma_omega_powers[..])
                                        .map_err(super::Error::NativeStorage)?;
                                    let sigma_coset =
                                        domain.coeff_to_extended_part(sigma_poly, extended_omega_factor);
                                    let delta_start = beta * C::ScalarExt::ZETA * delta_power;
                                    session
                                        .multiply_permutation_column(
                                            &source_coset[..],
                                            &sigma_coset[..],
                                            beta,
                                            gamma,
                                            delta_start,
                                        )
                                        .map_err(super::Error::NativeStorage)?;
                                    delta_power *= &C::ScalarExt::DELTA;
                                    column_index += 1;
                                    drop(sigma_coset);
                                    drop(source_coset);
                                }
                                session
                                    .finish_permutation_set(
                                        &l_active_row[..],
                                        y,
                                        total_terms,
                                        set_index,
                                        sets.len(),
                                        part_index,
                                        part_stride,
                                        isize,
                                    )
                                    .map_err(super::Error::NativeStorage)?;
                            }
                            if column_index != p.columns.len() {
                                return Err(super::Error::NativeStorage(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "resident permutation H fold did not cover every circuit column",
                                )));
                            }
                            gpu_permutation_used = true;
                        } else {
                            for _ in sets.iter() {
                                crate::icicle::record_gpu_h_eval_fallback();
                            }
                        }
                    }

                    if !gpu_permutation_used && sets.len() == 1 {
                        // Preserve the legacy evaluation order for a single
                        // product set. The streamed weighted fold below is
                        // intended for the multi-set case where repeated Z
                        // transforms dominate the permutation work.
                        let first_z = domain
                            .coeff_to_extended_part_spilled(
                                &sets[0].permutation_product_poly,
                                extended_omega_factor,
                            )
                            .map_err(super::Error::NativeStorage)?;
                        parallelize(&mut part_values[..], |h, start| {
                            for (offset, value) in h.iter_mut().enumerate() {
                                let row = start + offset;
                                let idx = row * part_stride + part_index;
                                *value = *value * y + ((one - first_z[row]) * l0[idx]);
                            }
                        });
                        drop(first_z);

                        let last_z = domain
                            .coeff_to_extended_part_spilled(
                                &sets.last().unwrap().permutation_product_poly,
                                extended_omega_factor,
                            )
                            .map_err(super::Error::NativeStorage)?;
                        parallelize(&mut part_values[..], |h, start| {
                            for (offset, value) in h.iter_mut().enumerate() {
                                let row = start + offset;
                                let idx = row * part_stride + part_index;
                                let z = last_z[row];
                                *value = *value * y + ((z.square() - z) * l_last[idx]);
                            }
                        });
                        drop(last_z);

                        // One set at a time, and one source/sigma pair at a
                        // time. Keep this path byte-for-byte in its previous
                        // Horner order for the degenerate one-set case.
                        let mut column_index = 0usize;
                        let mut delta_power = C::ScalarExt::ONE;
                        for (set, columns) in sets.iter().zip(column_groups.into_iter()) {
                            let z = domain
                                .coeff_to_extended_part_spilled(
                                    &set.permutation_product_poly,
                                    extended_omega_factor,
                                )
                                .map_err(super::Error::NativeStorage)?;
                            let mut left: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
                                Polynomial::mapped_filled(part_size, C::ScalarExt::ONE)
                                    .map_err(super::Error::NativeStorage)?;
                            let mut right: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
                                Polynomial::mapped_filled(part_size, C::ScalarExt::ONE)
                                    .map_err(super::Error::NativeStorage)?;
                            parallelize(&mut left[..], |out, start| {
                                for (offset, value) in out.iter_mut().enumerate() {
                                    let row = start + offset;
                                    let idx = row * part_stride + part_index;
                                    let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                                    *value = z[r_next / part_stride];
                                }
                            });
                            parallelize(&mut right[..], |out, start| {
                                out.copy_from_slice(&z[start..start + out.len()]);
                            });
                            drop(z);

                            for column in columns {
                                let source_coset = match column.column_type() {
                                    Any::Advice(_) => domain
                                        .coeff_to_extended_part_spilled(
                                            &advice[column.index()],
                                            extended_omega_factor,
                                        )
                                        .map_err(super::Error::NativeStorage)?,
                                    Any::Fixed => {
                                        let lagrange = pk.fixed_values[column.index()]
                                            .materialize_lagrange()
                                            .map_err(super::Error::NativeStorage)?;
                                        let coeff = domain
                                            .lagrange_to_coeff(lagrange)
                                            .spill()
                                            .map_err(super::Error::NativeStorage)?;
                                        domain.coeff_to_extended_part(coeff, extended_omega_factor)
                                    }
                                    Any::Instance => domain
                                        .coeff_to_extended_part_spilled(
                                            &instance[column.index()],
                                            extended_omega_factor,
                                        )
                                        .map_err(super::Error::NativeStorage)?,
                                };
                                let sigma_poly = pk
                                    .permutation
                                    .sigma_coeff(domain, column_index, &sigma_omega_powers[..])
                                    .map_err(super::Error::NativeStorage)?;
                                let sigma_coset =
                                    domain.coeff_to_extended_part(sigma_poly, extended_omega_factor);

                                parallelize(&mut left[..], |out, start| {
                                    for (offset, value) in out.iter_mut().enumerate() {
                                        let row = start + offset;
                                        *value *= source_coset[row] + beta * sigma_coset[row] + gamma;
                                    }
                                });
                                let delta_start = beta * C::ScalarExt::ZETA * delta_power;
                                let domain_omega = domain.get_omega();
                                parallelize(&mut right[..], |out, start| {
                                    let full_start = start * part_stride + part_index;
                                    let mut beta_term =
                                        delta_start * extended_omega.pow_vartime([full_start as u64]);
                                    for (offset, value) in out.iter_mut().enumerate() {
                                        let row = start + offset;
                                        *value *= source_coset[row] + beta_term + gamma;
                                        beta_term *= &domain_omega;
                                    }
                                });
                                delta_power *= &C::ScalarExt::DELTA;
                                column_index += 1;
                                drop(sigma_coset);
                                drop(source_coset);
                            }

                            parallelize(&mut part_values[..], |h, start| {
                                for (offset, value) in h.iter_mut().enumerate() {
                                    let row = start + offset;
                                    let idx = row * part_stride + part_index;
                                    *value = *value * y
                                        + ((left[row] - right[row]) * l_active_row[idx]);
                                }
                            });
                            drop(left);
                            drop(right);
                        }
                    } else if !gpu_permutation_used {
                        // The original Horner sequence contains two boundary
                        // terms, one transition term for every set after the
                        // first, and one copy term per set. Fold those terms
                        // with explicit powers of y while streaming each Z
                        // polynomial exactly once for this part. Each
                        // transition is split across its two adjacent sets.
                        let set_count = sets.len();
                        let total_terms = permutation_term_count(set_count)
                            .map_err(super::Error::NativeStorage)?;
                        scale_permutation_h(&mut part_values[..], y, total_terms)
                            .map_err(super::Error::NativeStorage)?;
                        let expected_split_contributions = set_count.checked_mul(3).ok_or_else(|| {
                            super::Error::NativeStorage(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "permutation H split contribution count overflows host index width",
                            ))
                        })?;
                        let mut logical_terms = 0usize;
                        let mut split_contributions = 0usize;

                        let mut current_z = Some(domain
                            .coeff_to_extended_part_spilled(
                                &sets[0].permutation_product_poly,
                                extended_omega_factor,
                            )
                            .map_err(super::Error::NativeStorage)?);

                        let mut column_index = 0usize;
                        let mut delta_power = C::ScalarExt::ONE;
                        for (set_index, (set, columns)) in
                            sets.iter().zip(column_groups.into_iter()).enumerate()
                        {
                            let z = current_z.as_ref().ok_or_else(|| {
                                super::Error::NativeStorage(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "permutation Z stream is missing the current set",
                                ))
                            })?;
                            let (set_logical_terms, set_split_contributions) =
                                add_permutation_set_h_terms(
                                    &mut part_values[..],
                                    &z[..],
                                    &l0[..],
                                    &l_last[..],
                                    y,
                                    total_terms,
                                    set_index,
                                    set_count,
                                    last_rotation,
                                    part_index,
                                    part_stride,
                                    rot_scale,
                                    isize,
                                )
                                .map_err(super::Error::NativeStorage)?;
                            logical_terms = logical_terms
                                .checked_add(set_logical_terms)
                                .ok_or_else(|| {
                                    super::Error::NativeStorage(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "permutation H logical term count overflows host index width",
                                    ))
                                })?;
                            split_contributions = split_contributions
                                .checked_add(set_split_contributions)
                                .ok_or_else(|| {
                                    super::Error::NativeStorage(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "permutation H split contribution count overflows host index width",
                                    ))
                                })?;

                            // At this point Z plus left/right are the three
                            // permutation product buffers. Drop Z immediately
                            // after initializing both products, before loading
                            // the mapped source and sigma cosets below.
                            let mut left: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
                                Polynomial::mapped_filled(part_size, C::ScalarExt::ONE)
                                    .map_err(super::Error::NativeStorage)?;
                            let mut right: Polynomial<C::ScalarExt, ExtendedLagrangeCoeff> =
                                Polynomial::mapped_filled(part_size, C::ScalarExt::ONE)
                                    .map_err(super::Error::NativeStorage)?;
                            parallelize(&mut left[..], |out, start| {
                                for (offset, value) in out.iter_mut().enumerate() {
                                    let row = start + offset;
                                    let idx = row * part_stride + part_index;
                                    let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                                    *value = z[r_next / part_stride];
                                }
                            });
                            parallelize(&mut right[..], |out, start| {
                                out.copy_from_slice(&z[start..start + out.len()]);
                            });
                            drop(current_z.take());

                            for column in columns {
                                let source_coset = match column.column_type() {
                                    Any::Advice(_) => domain
                                        .coeff_to_extended_part_spilled(
                                            &advice[column.index()],
                                            extended_omega_factor,
                                        )
                                        .map_err(super::Error::NativeStorage)?,
                                    Any::Fixed => {
                                        let lagrange = pk.fixed_values[column.index()]
                                            .materialize_lagrange()
                                            .map_err(super::Error::NativeStorage)?;
                                        let coeff = domain
                                            .lagrange_to_coeff(lagrange)
                                            .spill()
                                            .map_err(super::Error::NativeStorage)?;
                                        domain.coeff_to_extended_part(coeff, extended_omega_factor)
                                    }
                                    Any::Instance => domain
                                        .coeff_to_extended_part_spilled(
                                            &instance[column.index()],
                                            extended_omega_factor,
                                        )
                                        .map_err(super::Error::NativeStorage)?,
                                };
                                let sigma_poly = pk
                                    .permutation
                                    .sigma_coeff(domain, column_index, &sigma_omega_powers[..])
                                    .map_err(super::Error::NativeStorage)?;
                                let sigma_coset =
                                    domain.coeff_to_extended_part(sigma_poly, extended_omega_factor);

                                parallelize(&mut left[..], |out, start| {
                                    for (offset, value) in out.iter_mut().enumerate() {
                                        let row = start + offset;
                                        *value *= source_coset[row] + beta * sigma_coset[row] + gamma;
                                    }
                                });
                                let delta_start = beta * C::ScalarExt::ZETA * delta_power;
                                let domain_omega = domain.get_omega();
                                parallelize(&mut right[..], |out, start| {
                                    let full_start = start * part_stride + part_index;
                                    let mut beta_term =
                                        delta_start * extended_omega.pow_vartime([full_start as u64]);
                                    for (offset, value) in out.iter_mut().enumerate() {
                                        let row = start + offset;
                                        *value *= source_coset[row] + beta_term + gamma;
                                        beta_term *= &domain_omega;
                                    }
                                });
                                delta_power *= &C::ScalarExt::DELTA;
                                column_index += 1;
                                drop(sigma_coset);
                                drop(source_coset);
                            }

                            let copy_term_index = set_count
                                .checked_add(1)
                                .and_then(|index| index.checked_add(set_index))
                                .ok_or_else(|| {
                                    super::Error::NativeStorage(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "permutation H copy term index overflows host index width",
                                    ))
                                })?;
                            add_weighted_permutation_term(
                                &mut part_values[..],
                                y,
                                total_terms,
                                copy_term_index,
                                |row| {
                                    let idx = row * part_stride + part_index;
                                    (left[row] - right[row]) * l_active_row[idx]
                                },
                            )
                            .map_err(super::Error::NativeStorage)?;
                            logical_terms = logical_terms.checked_add(1).ok_or_else(|| {
                                super::Error::NativeStorage(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "permutation H logical term count overflows host index width",
                                ))
                            })?;
                            split_contributions = split_contributions
                                .checked_add(1)
                                .ok_or_else(|| {
                                    super::Error::NativeStorage(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "permutation H split contribution count overflows host index width",
                                    ))
                                })?;
                            drop(left);
                            drop(right);

                            if set_index + 1 < set_count {
                                let next_set_index = set_index.checked_add(1).ok_or_else(|| {
                                    super::Error::NativeStorage(io::Error::new(
                                        io::ErrorKind::InvalidData,
                                        "permutation set index overflows host index width",
                                    ))
                                })?;
                                current_z = Some(domain
                                    .coeff_to_extended_part_spilled(
                                        &sets[next_set_index].permutation_product_poly,
                                        extended_omega_factor,
                                    )
                                    .map_err(super::Error::NativeStorage)?);
                            }
                        }
                        drop(current_z);
                        if logical_terms != total_terms
                            || split_contributions != expected_split_contributions
                        {
                            return Err(super::Error::NativeStorage(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "permutation H weighted fold omitted or duplicated terms or split contributions",
                            )));
                        }
                        if column_index != p.columns.len() {
                            return Err(super::Error::NativeStorage(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "permutation H copy fold did not cover every circuit column",
                            )));
                        }
                    }
                }

                if !sets.is_empty() {
                    #[cfg(feature = "icicle_gpu")]
                    {
                        let after = crate::icicle::execution_stats();
                        eprintln!(
                            "EZKL_NATIVE_H_STAGE stage=permutation_h part_index={} part_stride={} sets={} columns={} requested_gpu={} path={} gpu_groups={} gpu_calls={} gpu_rows={} cpu_fallback_groups={} gpu_chunks={} gpu_eval_syncs={} gpu_transfer_syncs={} wall_elapsed_ms={:.3}",
                            part_index,
                            part_stride,
                            sets.len(),
                            p.columns.len(),
                            gpu_h_requested,
                            if gpu_permutation_used { "gpu" } else { "cpu" },
                            after.24.saturating_sub(permutation_gpu_stats_before.24),
                            after.17.saturating_sub(permutation_gpu_stats_before.17),
                            after.18.saturating_sub(permutation_gpu_stats_before.18),
                            after.19.saturating_sub(permutation_gpu_stats_before.19),
                            after.20.saturating_sub(permutation_gpu_stats_before.20),
                            after.21.saturating_sub(permutation_gpu_stats_before.21),
                            after.25.saturating_sub(permutation_gpu_stats_before.25),
                            permutation_h_started.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    #[cfg(not(feature = "icicle_gpu"))]
                    eprintln!(
                        "EZKL_NATIVE_H_STAGE stage=permutation_h part_index={} part_stride={} sets={} columns={} requested_gpu=false path=cpu wall_elapsed_ms={:.3}",
                        part_index,
                        part_stride,
                        sets.len(),
                        p.columns.len(),
                        permutation_h_started.elapsed().as_secs_f64() * 1000.0,
                    );
                }

                #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                let gpu_session_has_lookup_consumer = gpu_lookup_preflight_supported && !lookup_sets.is_empty();
                #[cfg(all(feature = "icicle_gpu", not(feature = "mv-lookup")))]
                let gpu_session_has_lookup_consumer = false;
                #[cfg(feature = "icicle_gpu")]
                if !gpu_session_has_lookup_consumer {
                    if let Some(session) = gpu_lookup_session.take() {
                        session
                            .finish(&mut part_values[..])
                            .map_err(super::Error::NativeStorage)?;
                    }
                }

                let mut lookup_coset_cache = if lookup_sets.is_empty() {
                    None
                } else {
                    Some((
                        empty_coset_cache::<C::ScalarExt>(pk.fixed_values.len()),
                        empty_coset_cache::<C::ScalarExt>(advice.len()),
                        empty_coset_cache::<C::ScalarExt>(instance.len()),
                        ColumnDependencies::default(),
                    ))
                };

                #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                if gpu_lookup_preflight_supported && gpu_lookup_session.is_none() {
                    let kernel_capacity = gpu_lookup_kernel_capacity.ok_or_else(|| {
                        super::Error::NativeStorage(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "resident H lookup capacity was not retained after preflight",
                        ))
                    })?;
                    gpu_lookup_session = crate::icicle::try_begin_resident_h_session(
                        &part_values[..],
                        kernel_capacity,
                        false,
                    )
                    .map_err(super::Error::NativeStorage)?;
                    if gpu_lookup_session.is_none() {
                        for _ in lookup_sets.iter() {
                            crate::icicle::record_gpu_h_eval_fallback();
                        }
                    }
                }

                #[cfg(feature = "mv-lookup")]
                for (lookup_index, lookup) in lookup_sets.iter().enumerate() {
                    let lookup_h_started = Instant::now();
                    #[cfg(feature = "icicle_gpu")]
                    let lookup_gpu_stats_before = crate::icicle::execution_stats();
                    let (input_graphs, table_graph) = &self.lookups[lookup_index];
                    if input_graphs.is_empty() {
                        return Err(super::Error::NativeStorage(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "multi-lookup argument has no input expressions",
                        )));
                    }
                    let mut dependencies = ColumnDependencies::default();
                    for graph in input_graphs.iter().chain(std::iter::once(table_graph)) {
                        dependencies.extend(&graph.column_dependencies());
                    }
                    let (
                        fixed_cache,
                        advice_cache,
                        instance_cache,
                        previous_dependencies,
                    ) = lookup_coset_cache.as_mut().ok_or_else(|| {
                        super::Error::NativeStorage(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "lookup argument coset cache is absent",
                        ))
                    })?;
                    let _ = refresh_lookup_argument_cosets(
                        domain,
                        &pk.fixed_values,
                        advice,
                        instance,
                        &dependencies,
                        extended_omega_factor,
                        previous_dependencies,
                        fixed_cache,
                        advice_cache,
                        instance_cache,
                    )
                    .map_err(super::Error::NativeStorage)?;
                    let fixed = fixed_cache.as_slice();
                    let advice_cosets = advice_cache.as_slice();
                    let instance_cosets = instance_cache.as_slice();
                    let phi_poly = lookup
                        .source
                        .materialize_coeff(lookup::prover::LookupPolynomialKind::GrandSum)
                        .map_err(super::Error::NativeStorage)?;
                    let phi_coset = domain
                        .coeff_to_extended_part(phi_poly, extended_omega_factor);
                    let m_poly = lookup
                        .source
                        .materialize_coeff(lookup::prover::LookupPolynomialKind::Multiplicity)
                        .map_err(super::Error::NativeStorage)?;
                    let m_coset = domain
                        .coeff_to_extended_part(m_poly, extended_omega_factor);
                    const LOOKUP_ROW_CHUNK: usize = 4096;
                    let lookup_input_count = input_graphs.len();
                    // Keep the existing per-worker scratch ceiling while the
                    // fused p/d recurrence stores only two field elements per
                    // row, independent of lookup width.
                    let rows_per_chunk = lookup_rows_per_chunk(
                        LOOKUP_ROW_CHUNK,
                        lookup_input_count,
                    )
                    .ok_or_else(|| {
                        super::Error::NativeStorage(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "multi-lookup row chunk size overflows host index width",
                        ))
                    })?;
                    let old_chunk_field_bound = LOOKUP_ROW_CHUNK
                        .checked_mul(lookup_input_count)
                        .ok_or_else(|| {
                            super::Error::NativeStorage(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "multi-lookup legacy scratch bound overflows host index width",
                            ))
                        })?;
                    let new_chunk_field_bound = rows_per_chunk
                        .checked_mul(2)
                        .ok_or_else(|| {
                            super::Error::NativeStorage(io::Error::new(
                                io::ErrorKind::InvalidData,
                                "multi-lookup p/d scratch bound overflows host index width",
                            ))
                        })?;
                    if new_chunk_field_bound > old_chunk_field_bound {
                        return Err(super::Error::NativeStorage(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "multi-lookup p/d scratch exceeds the existing chunk bound",
                        )));
                    }
                    #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                    let gpu_lookup_active = gpu_lookup_session.is_some();
                    #[cfg(not(all(feature = "icicle_gpu", feature = "mv-lookup")))]
                    let gpu_lookup_active = false;
                    #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                    if let Some(session) = gpu_lookup_session.as_mut() {
                        let program = gpu_lookup_programs
                            .as_ref()
                            .and_then(|programs| programs.get(lookup_index))
                            .ok_or_else(|| {
                                super::Error::NativeStorage(io::Error::new(
                                    io::ErrorKind::InvalidData,
                                    "resident H lookup program is missing after admission",
                                ))
                            })?;
                        session
                            .evaluate_mv_lookup_program(
                                program,
                                fixed,
                                advice_cosets,
                                instance_cosets,
                                &phi_coset,
                                &m_coset,
                                l0,
                                l_last,
                                l_active_row,
                                part_index,
                                part_stride,
                                rot_scale,
                                isize,
                                y,
                            )
                            .map_err(super::Error::NativeStorage)?;
                    }
                    if !gpu_lookup_active {
                        (&mut part_values[..])
                            .par_chunks_mut(rows_per_chunk)
                            .enumerate()
                            .for_each(|(chunk_index, h)| {
                            let start = chunk_index * rows_per_chunk;
                            let mut input_data = input_graphs
                                .iter()
                                .map(GraphEvaluator::instance)
                                .collect::<Vec<_>>();
                            let lookup_values = evaluate_lookup_input_values(
                                h.len(),
                                lookup_input_count,
                                |offset, input_index| {
                                    let row = start + offset;
                                    let idx = row * part_stride + part_index;
                                    input_graphs[input_index].evaluate_part(
                                        &mut input_data[input_index],
                                        fixed,
                                        advice_cosets,
                                        instance_cosets,
                                        challenges,
                                        &beta,
                                        &gamma,
                                        &theta,
                                        &y,
                                        &C::ScalarExt::ZERO,
                                        idx,
                                        rot_scale,
                                        isize,
                                        part_stride,
                                    )
                                },
                            );

                            let mut table_data = table_graph.instance();
                            for (offset, h_value) in h.iter_mut().enumerate() {
                                let row = start + offset;
                                let idx = row * part_stride + part_index;
                                let inputs_product = lookup_values.inputs_product(offset);
                                let products_except_sum = lookup_values.products_except_sum(offset);
                                let table_value = table_graph.evaluate_part(
                                    &mut table_data,
                                    fixed,
                                    advice_cosets,
                                    instance_cosets,
                                    challenges,
                                    &beta,
                                    &gamma,
                                    &theta,
                                    &y,
                                    &C::ScalarExt::ZERO,
                                    idx,
                                    rot_scale,
                                    isize,
                                    part_stride,
                                );
                                let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                                fold_lookup_h_row(
                                    h_value,
                                    y,
                                    l0[idx],
                                    l_last[idx],
                                    l_active_row[idx],
                                    phi_coset[row],
                                    phi_coset[r_next / part_stride],
                                    table_value,
                                    inputs_product,
                                    products_except_sum,
                                    m_coset[row],
                                );
                            }
                            });
                    }
                    drop(m_coset);
                    drop(phi_coset);
                    #[cfg(feature = "icicle_gpu")]
                    {
                        let after = crate::icicle::execution_stats();
                        eprintln!(
                            "EZKL_NATIVE_H_STAGE stage=mv_lookup_h part_index={} part_stride={} argument_index={} arguments={} input_expressions={} requested_gpu={} path={} gpu_calls={} gpu_rows={} gpu_fallback_groups={} gpu_chunks={} gpu_eval_syncs={} wall_elapsed_ms={:.3}",
                            part_index,
                            part_stride,
                            lookup_index,
                            lookup_sets.len(),
                            lookup_input_count,
                            gpu_h_requested,
                            if gpu_lookup_active { "gpu" } else { "cpu" },
                            after.17.saturating_sub(lookup_gpu_stats_before.17),
                            after.18.saturating_sub(lookup_gpu_stats_before.18),
                            after.19.saturating_sub(lookup_gpu_stats_before.19),
                            after.20.saturating_sub(lookup_gpu_stats_before.20),
                            after.21.saturating_sub(lookup_gpu_stats_before.21),
                            lookup_h_started.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                    #[cfg(not(feature = "icicle_gpu"))]
                    eprintln!(
                        "EZKL_NATIVE_H_STAGE stage=mv_lookup_h part_index={} part_stride={} argument_index={} arguments={} input_expressions={} requested_gpu=false path=cpu wall_elapsed_ms={:.3}",
                        part_index,
                        part_stride,
                        lookup_index,
                        lookup_sets.len(),
                        lookup_input_count,
                        lookup_h_started.elapsed().as_secs_f64() * 1000.0,
                    );
                }

                #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
                if let Some(session) = gpu_lookup_session.take() {
                    session
                        .finish(&mut part_values[..])
                        .map_err(super::Error::NativeStorage)?;
                }

                #[cfg(not(feature = "mv-lookup"))]
                for (lookup_index, lookup) in lookup_sets.iter().enumerate() {
                    let graph = &self.lookups[lookup_index];
                    let dependencies = graph.column_dependencies();
                    let (
                        fixed_cache,
                        advice_cache,
                        instance_cache,
                        previous_dependencies,
                    ) = lookup_coset_cache.as_mut().ok_or_else(|| {
                        super::Error::NativeStorage(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "lookup argument coset cache is absent",
                        ))
                    })?;
                    let _ = refresh_lookup_argument_cosets(
                        domain,
                        &pk.fixed_values,
                        advice,
                        instance,
                        &dependencies,
                        extended_omega_factor,
                        previous_dependencies,
                        fixed_cache,
                        advice_cache,
                        instance_cache,
                    )
                    .map_err(super::Error::NativeStorage)?;
                    let fixed = fixed_cache.as_slice();
                    let advice_cosets = advice_cache.as_slice();
                    let instance_cosets = instance_cache.as_slice();
                    let product_coset = domain
                        .coeff_to_extended_part_spilled(
                            &lookup.product_poly,
                            extended_omega_factor,
                        )
                        .map_err(super::Error::NativeStorage)?;
                    let input_coset = domain
                        .coeff_to_extended_part_spilled(
                            &lookup.permuted_input_poly,
                            extended_omega_factor,
                        )
                        .map_err(super::Error::NativeStorage)?;
                    let table_coset = domain
                        .coeff_to_extended_part_spilled(
                            &lookup.permuted_table_poly,
                            extended_omega_factor,
                        )
                        .map_err(super::Error::NativeStorage)?;
                    parallelize(&mut part_values[..], |h, start| {
                        let mut data = graph.instance();
                        for (offset, h_value) in h.iter_mut().enumerate() {
                            let row = start + offset;
                            let idx = row * part_stride + part_index;
                            let table_value = graph.evaluate_part(
                                &mut data,
                                fixed,
                                advice_cosets,
                                instance_cosets,
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &C::ScalarExt::ZERO,
                                idx,
                                rot_scale,
                                isize,
                                part_stride,
                            );
                            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                            let r_prev = get_rotation_idx(idx, -1, rot_scale, isize);
                            let input_minus_table = input_coset[row] - table_coset[row];
                            *h_value = *h_value * y + ((one - product_coset[row]) * l0[idx]);
                            *h_value = *h_value * y
                                + ((product_coset[row].square() - product_coset[row])
                                    * l_last[idx]);
                            *h_value = *h_value * y
                                + ((product_coset[r_next / part_stride]
                                    * (input_coset[row] + beta)
                                    * (table_coset[row] + gamma)
                                    - product_coset[row] * table_value)
                                    * l_active_row[idx]);
                            *h_value = *h_value * y + (input_minus_table * l0[idx]);
                            *h_value = *h_value * y
                                + (input_minus_table
                                    * (input_coset[row] - input_coset[r_prev / part_stride])
                                    * l_active_row[idx]);
                        }
                    });
                }

                drop(lookup_coset_cache);

                for (shuffle_index, shuffle) in shuffle_sets.iter().enumerate() {
                    let input_graph = &self.shuffles[2 * shuffle_index];
                    let shuffle_graph = &self.shuffles[2 * shuffle_index + 1];
                    let mut dependencies = input_graph.column_dependencies();
                    dependencies.extend(&shuffle_graph.column_dependencies());
                    let fixed = load_fixed_cosets(
                        domain,
                        &pk.fixed_values,
                        &dependencies.fixed,
                        extended_omega_factor,
                    )
                        .map_err(super::Error::NativeStorage)?;
                    let advice_cosets = load_cosets(
                        domain,
                        advice,
                        &dependencies.advice,
                        extended_omega_factor,
                    )
                        .map_err(super::Error::NativeStorage)?;
                    let instance_cosets = load_cosets(
                        domain,
                        instance,
                        &dependencies.instance,
                        extended_omega_factor,
                    )
                        .map_err(super::Error::NativeStorage)?;
                    let product_coset = domain
                        .coeff_to_extended_part_spilled(
                            &shuffle.product_poly,
                            extended_omega_factor,
                        )
                        .map_err(super::Error::NativeStorage)?;
                    parallelize(&mut part_values[..], |h, start| {
                        let mut input_data = input_graph.instance();
                        let mut shuffle_data = shuffle_graph.instance();
                        for (offset, h_value) in h.iter_mut().enumerate() {
                            let row = start + offset;
                            let idx = row * part_stride + part_index;
                            let input_value = input_graph.evaluate_part(
                                &mut input_data,
                                &fixed,
                                &advice_cosets,
                                &instance_cosets,
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &C::ScalarExt::ZERO,
                                idx,
                                rot_scale,
                                isize,
                                part_stride,
                            );
                            let shuffle_value = shuffle_graph.evaluate_part(
                                &mut shuffle_data,
                                &fixed,
                                &advice_cosets,
                                &instance_cosets,
                                challenges,
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &C::ScalarExt::ZERO,
                                idx,
                                rot_scale,
                                isize,
                                part_stride,
                            );
                            let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                            *h_value = *h_value * y + ((one - product_coset[row]) * l0[idx]);
                            *h_value = *h_value * y
                                + ((product_coset[row].square() - product_coset[row]) * l_last[idx]);
                            *h_value = *h_value * y
                                + l_active_row[idx]
                                    * (product_coset[r_next / part_stride] * shuffle_value
                                        - product_coset[row] * input_value);
                        }
                    });
                    drop(product_coset);
                    drop(instance_cosets);
                    drop(advice_cosets);
                    drop(fixed);
                }
            }

            // The source part is disjoint from the final extended polynomial,
            // and each row maps to exactly one strided output slot. The
            // parallel chunks partition `part_values`, so every destination is
            // written once without retaining more than one domain part.
            let output = values.as_mut_slice();
            for (row, value) in part_values.iter().enumerate() {
                output[row * part_stride + part_index] = *value;
            }
        }

        Ok(values)
    }
}

/// Build the same expression graphs used by the native prover and report each
/// sequential evaluator group's exact polynomial-column dependency count.
pub fn native_evaluation_resource_shape<C: CurveAffine>(
    cs: &ConstraintSystem<C::ScalarExt>,
) -> NativeEvaluationShape {
    Evaluator::<C>::new(cs).native_resource_shape()
}

impl<C: CurveAffine> Default for GraphEvaluator<C> {
    fn default() -> Self {
        Self {
            // Fixed positions to allow easy access
            constants: vec![
                C::ScalarExt::ZERO,
                C::ScalarExt::ONE,
                C::ScalarExt::from(2u64),
            ],
            rotations: Vec::new(),
            calculations: Vec::new(),
            num_intermediates: 0,
        }
    }
}

impl<C: CurveAffine> GraphEvaluator<C> {
    fn uses_previous_value(&self) -> bool {
        let is_previous_value = |source: &ValueSource| matches!(source, ValueSource::PreviousValue());
        self.calculations.iter().any(|calc| match &calc.calculation {
            Calculation::Add(a, b) | Calculation::Sub(a, b) | Calculation::Mul(a, b) => {
                is_previous_value(a) || is_previous_value(b)
            }
            Calculation::Square(value)
            | Calculation::Double(value)
            | Calculation::Negate(value)
            | Calculation::Store(value) => is_previous_value(value),
            Calculation::Horner(start, parts, factor) => {
                is_previous_value(start)
                    || is_previous_value(factor)
                    || parts.iter().any(is_previous_value)
            }
        })
    }

    fn column_dependencies(&self) -> ColumnDependencies {
        let mut dependencies = ColumnDependencies::default();
        for calc in &self.calculations {
            match &calc.calculation {
                Calculation::Add(a, b) | Calculation::Sub(a, b) | Calculation::Mul(a, b) => {
                    dependencies.add(a);
                    dependencies.add(b);
                }
                Calculation::Square(value)
                | Calculation::Double(value)
                | Calculation::Negate(value)
                | Calculation::Store(value) => dependencies.add(value),
                Calculation::Horner(start, parts, factor) => {
                    dependencies.add(start);
                    dependencies.add(factor);
                    for part in parts {
                        dependencies.add(part);
                    }
                }
            }
        }
        dependencies
    }

    /// Adds a rotation
    fn add_rotation(&mut self, rotation: &Rotation) -> usize {
        let position = self.rotations.iter().position(|&c| c == rotation.0);
        match position {
            Some(pos) => pos,
            None => {
                self.rotations.push(rotation.0);
                self.rotations.len() - 1
            }
        }
    }

    /// Adds a constant
    fn add_constant(&mut self, constant: &C::ScalarExt) -> ValueSource {
        let position = self.constants.iter().position(|&c| c == *constant);
        ValueSource::Constant(match position {
            Some(pos) => pos,
            None => {
                self.constants.push(*constant);
                self.constants.len() - 1
            }
        })
    }

    /// Adds a calculation.
    /// Currently does the simplest thing possible: just stores the
    /// resulting value so the result can be reused  when that calculation
    /// is done multiple times.
    fn add_calculation(&mut self, calculation: Calculation) -> ValueSource {
        let existing_calculation = self
            .calculations
            .iter()
            .find(|c| c.calculation == calculation);
        match existing_calculation {
            Some(existing_calculation) => ValueSource::Intermediate(existing_calculation.target),
            None => {
                let target = self.num_intermediates;
                self.calculations.push(CalculationInfo {
                    calculation,
                    target,
                });
                self.num_intermediates += 1;
                ValueSource::Intermediate(target)
            }
        }
    }

    /// Generates an optimized evaluation for the expression
    fn add_expression(&mut self, expr: &Expression<C::ScalarExt>) -> ValueSource {
        match expr {
            Expression::Constant(scalar) => self.add_constant(scalar),
            Expression::Selector(_selector) => unreachable!(),
            Expression::Fixed(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Fixed(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Advice(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Advice(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Instance(query) => {
                let rot_idx = self.add_rotation(&query.rotation);
                self.add_calculation(Calculation::Store(ValueSource::Instance(
                    query.column_index,
                    rot_idx,
                )))
            }
            Expression::Challenge(challenge) => self.add_calculation(Calculation::Store(
                ValueSource::Challenge(challenge.index()),
            )),
            Expression::Negated(a) => match **a {
                Expression::Constant(scalar) => self.add_constant(&-scalar),
                _ => {
                    let result_a = self.add_expression(a);
                    match result_a {
                        ValueSource::Constant(0) => result_a,
                        _ => self.add_calculation(Calculation::Negate(result_a)),
                    }
                }
            },
            Expression::Sum(a, b) => {
                // Undo subtraction stored as a + (-b) in expressions
                match &**b {
                    Expression::Negated(b_int) => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b_int);
                        if result_a == ValueSource::Constant(0) {
                            self.add_calculation(Calculation::Negate(result_b))
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else {
                            self.add_calculation(Calculation::Sub(result_a, result_b))
                        }
                    }
                    _ => {
                        let result_a = self.add_expression(a);
                        let result_b = self.add_expression(b);
                        if result_a == ValueSource::Constant(0) {
                            result_b
                        } else if result_b == ValueSource::Constant(0) {
                            result_a
                        } else if result_a <= result_b {
                            self.add_calculation(Calculation::Add(result_a, result_b))
                        } else {
                            self.add_calculation(Calculation::Add(result_b, result_a))
                        }
                    }
                }
            }
            Expression::Product(a, b) => {
                let result_a = self.add_expression(a);
                let result_b = self.add_expression(b);
                if result_a == ValueSource::Constant(0) || result_b == ValueSource::Constant(0) {
                    ValueSource::Constant(0)
                } else if result_a == ValueSource::Constant(1) {
                    result_b
                } else if result_b == ValueSource::Constant(1) {
                    result_a
                } else if result_a == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_b))
                } else if result_b == ValueSource::Constant(2) {
                    self.add_calculation(Calculation::Double(result_a))
                } else if result_a == result_b {
                    self.add_calculation(Calculation::Square(result_a))
                } else if result_a <= result_b {
                    self.add_calculation(Calculation::Mul(result_a, result_b))
                } else {
                    self.add_calculation(Calculation::Mul(result_b, result_a))
                }
            }
            Expression::Scaled(a, f) => {
                if *f == C::ScalarExt::ZERO {
                    ValueSource::Constant(0)
                } else if *f == C::ScalarExt::ONE {
                    self.add_expression(a)
                } else {
                    let cst = self.add_constant(f);
                    let result_a = self.add_expression(a);
                    self.add_calculation(Calculation::Mul(result_a, cst))
                }
            }
        }
    }

    /// Creates a new evaluation structure
    pub fn instance(&self) -> EvaluationData<C> {
        EvaluationData {
            intermediates: vec![C::ScalarExt::ZERO; self.num_intermediates],
            rotations: vec![0usize; self.rotations.len()],
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn evaluate<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,
        rot_scale: i32,
        isize: i32,
    ) -> C::ScalarExt {
        self.evaluate_with_part_stride(
            data,
            fixed,
            advice,
            instance,
            challenges,
            beta,
            gamma,
            theta,
            y,
            previous_value,
            idx,
            rot_scale,
            isize,
            1,
        )
    }

    /// Evaluate with full-domain rotations while reading row-strided parts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_part<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,
        rot_scale: i32,
        isize: i32,
        part_stride: usize,
    ) -> C::ScalarExt {
        self.evaluate_with_part_stride(
            data,
            fixed,
            advice,
            instance,
            challenges,
            beta,
            gamma,
            theta,
            y,
            previous_value,
            idx,
            rot_scale,
            isize,
            part_stride,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_with_part_stride<B: Basis>(
        &self,
        data: &mut EvaluationData<C>,
        fixed: &[Polynomial<C::ScalarExt, B>],
        advice: &[Polynomial<C::ScalarExt, B>],
        instance: &[Polynomial<C::ScalarExt, B>],
        challenges: &[C::ScalarExt],
        beta: &C::ScalarExt,
        gamma: &C::ScalarExt,
        theta: &C::ScalarExt,
        y: &C::ScalarExt,
        previous_value: &C::ScalarExt,
        idx: usize,
        rot_scale: i32,
        isize: i32,
        part_stride: usize,
    ) -> C::ScalarExt {
        assert!(part_stride > 0);
        // All rotation index values
        for (rot_idx, rot) in self.rotations.iter().enumerate() {
            data.rotations[rot_idx] = get_rotation_idx(idx, *rot, rot_scale, isize) / part_stride;
        }

        // All calculations, with cached intermediate results
        for calc in self.calculations.iter() {
            data.intermediates[calc.target] = calc.calculation.evaluate(
                &data.rotations,
                &self.constants,
                &data.intermediates,
                fixed,
                advice,
                instance,
                challenges,
                beta,
                gamma,
                theta,
                y,
                previous_value,
            );
        }

        // Return the result of the last calculation (if any)
        if let Some(calc) = self.calculations.last() {
            data.intermediates[calc.target]
        } else {
            C::ScalarExt::ZERO
        }
    }
}

/// Simple evaluation of an expression
pub fn evaluate<F: Field, B: Basis>(
    expression: &Expression<F>,
    size: usize,
    rot_scale: i32,
    fixed: &[FixedColumnSource<F>],
    advice: &[Polynomial<F, B>],
    instance: &[Polynomial<F, B>],
    challenges: &[F],
) -> Vec<F> {
    let mut values = vec![F::ZERO; size];
    let isize = size as i32;
    parallelize(&mut values, |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = start + i;
            *value = expression.evaluate(
                &|scalar| scalar,
                &|_| panic!("virtual selectors are removed during optimization"),
                &|query| {
                    fixed[query.column_index]
                        .value(get_rotation_idx(idx, query.rotation.0, rot_scale, isize))
                },
                &|query| {
                    advice[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|query| {
                    instance[query.column_index]
                        [get_rotation_idx(idx, query.rotation.0, rot_scale, isize)]
                },
                &|challenge| challenges[challenge.index()],
                &|a| -a,
                &|a, b| a + &b,
                &|a, b| a * b,
                &|a, scalar| a * scalar,
            );
        }
    });
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        halo2curves::bn256::G1Affine,
        plonk::ConstraintSystem,
        poly::Rotation,
    };

    #[test]
    fn native_resource_shape_counts_exact_group_dependency_unions() {
        let mut cs = ConstraintSystem::default();
        let advice = cs.advice_column();
        let fixed = cs.fixed_column();
        let instance = cs.instance_column();

        cs.create_gate("shape gate", |meta| {
            let advice_value = meta.query_advice(advice, Rotation::cur());
            let fixed_value = meta.query_fixed(fixed, Rotation::cur());
            vec![advice_value + fixed_value]
        });

        #[cfg(feature = "mv-lookup")]
        {
            for _ in 0..2 {
                cs.lookup_any("shape lookup", |meta| {
                    vec![(
                        meta.query_advice(advice, Rotation::cur()),
                        meta.query_fixed(fixed, Rotation::cur()),
                    )]
                });
            }
        }

        cs.shuffle("shape shuffle", |meta| {
            vec![(
                meta.query_advice(advice, Rotation::cur()),
                meta.query_instance(instance, Rotation::cur()),
            )]
        });

        #[cfg(feature = "mv-lookup")]
        let cs = cs.chunk_lookups();
        let shape = native_evaluation_resource_shape::<G1Affine>(&cs);

        assert_eq!(shape.custom_gate_dependency_columns, vec![2]);
        assert_eq!(shape.shuffle_dependency_columns, vec![2]);
        #[cfg(feature = "mv-lookup")]
        {
            assert_eq!(shape.lookup_dependency_columns, vec![2]);
            assert_eq!(shape.lookup_query_polynomial_counts, vec![2]);
        }
    }

    #[cfg(all(feature = "icicle_gpu", feature = "mv-lookup"))]
    #[test]
    #[ignore = "manual CUDA oracle for the resident permutation and multi-lookup H paths"]
    fn resident_h_permutation_and_mv_lookup_match_cpu_oracle_across_tiles_parts_sets_and_zeros() {
        use ff::BatchInvert;
        use halo2curves::bn256::Fr;

        fn assert_h_stage_matches(actual: &[Fr], expected: &[Fr], stage: &str, part: usize) {
            assert_eq!(actual.len(), expected.len(), "{stage} H length mismatch for part {part}");
            if let Some((row, (actual_value, expected_value))) = actual
                .iter()
                .zip(expected)
                .enumerate()
                .find(|(_, (actual_value, expected_value))| actual_value != expected_value)
            {
                panic!(
                    "resident H differs from independent CPU oracle after {stage}, part={part}, row={row}: GPU={actual_value:?}, CPU={expected_value:?}"
                );
            }
        }

        const ROWS: usize = (1 << 20) + 37;
        const PART_STRIDE: usize = 2;
        const LOOKUP_COUNT: usize = 2;
        const INPUTS_PER_LOOKUP: usize = 2;
        const LEGACY_ROW_CHUNK: usize = 4096;

        assert!(
            std::env::var_os("ENABLE_ICICLE_GPU").is_some(),
            "set ENABLE_ICICLE_GPU=1 to run the resident CUDA oracle"
        );
        crate::icicle::initialize_device(0).expect("initialize ICICLE CUDA device");
        crate::icicle::reset_execution_operation_stats();
        let _scope = crate::icicle::native_prove_ntt_scope();

        let full_size = ROWS * PART_STRIDE;
        let rot_scale = PART_STRIDE as i32;
        let isize = full_size as i32;
        let beta = Fr::from(29);
        let gamma = Fr::from(41);
        let theta = Fr::from(43);
        let y = Fr::from(47);
        let extended_omega = Fr::from(53);
        let last_rotation = -3;

        let mut advice_values = vec![vec![Fr::ZERO; ROWS]; 2];
        for (column, values) in advice_values.iter_mut().enumerate() {
            for (row, value) in values.iter_mut().enumerate() {
                *value = Fr::from(101 + row as u64 * 17 + column as u64 * 31);
            }
        }
        let lookup_rotations = [[1, -1], [-2, 2]];
        let zero_rows = [
            vec![0usize, 1, 2, ROWS - 1],
            vec![17usize, ROWS / 2, ROWS - 2],
        ];
        for lookup_index in 0..LOOKUP_COUNT {
            for input_index in 0..INPUTS_PER_LOOKUP {
                let rotation = lookup_rotations[lookup_index][input_index];
                for row in zero_rows[lookup_index].iter().copied() {
                    let source_row = (row as i64 + i64::from(rotation))
                        .rem_euclid(ROWS as i64) as usize;
                    advice_values[input_index][source_row] = -beta;
                }
            }
        }
        let fixed_values = (0..LOOKUP_COUNT)
            .map(|column| {
                (0..ROWS)
                    .map(|row| Fr::from(211 + row as u64 * 7 + column as u64 * 101))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let fixed = fixed_values
            .into_iter()
            .map(Polynomial::<Fr, LagrangeCoeff>::from_vec)
            .collect::<Vec<_>>();
        let advice = advice_values
            .into_iter()
            .map(Polynomial::<Fr, LagrangeCoeff>::from_vec)
            .collect::<Vec<_>>();
        for lookup_index in 0..LOOKUP_COUNT {
            for input_index in 0..INPUTS_PER_LOOKUP {
                for row in zero_rows[lookup_index].iter().copied() {
                    let source_row = (row as i64
                        + i64::from(lookup_rotations[lookup_index][input_index]))
                    .rem_euclid(ROWS as i64) as usize;
                    assert_eq!(
                        advice[input_index][source_row] + beta,
                        Fr::ZERO,
                        "oracle input must exercise the batch-invert zero convention"
                    );
                }
            }
        }

        let mut input_graphs = Vec::with_capacity(LOOKUP_COUNT);
        let mut table_graphs = Vec::with_capacity(LOOKUP_COUNT);
        for lookup_index in 0..LOOKUP_COUNT {
            let inputs = (0..INPUTS_PER_LOOKUP)
                .map(|input_index| {
                    let mut graph = GraphEvaluator::<G1Affine>::default();
                    let rotation = graph.add_rotation(&Rotation(
                        lookup_rotations[lookup_index][input_index],
                    ));
                    graph.add_calculation(Calculation::Add(
                        ValueSource::Advice(input_index, rotation),
                        ValueSource::Beta(),
                    ));
                    graph
                })
                .collect::<Vec<_>>();
            let mut table = GraphEvaluator::<G1Affine>::default();
            let rotation = table.add_rotation(&Rotation(if lookup_index == 0 { -1 } else { 1 }));
            table.add_calculation(Calculation::Add(
                ValueSource::Fixed(lookup_index, rotation),
                ValueSource::Beta(),
            ));
            input_graphs.push(inputs);
            table_graphs.push(table);
        }
        let programs = (0..LOOKUP_COUNT)
            .map(|lookup_index| {
                let lookup = [(input_graphs[lookup_index].as_slice(), &table_graphs[lookup_index])];
                compile_mv_lookup_program(
                    &lookup,
                    &[],
                    beta,
                    gamma,
                    theta,
                    y,
                    rot_scale,
                    isize,
                )
                .expect("compile merged CUDA lookup oracle")
                .expect("lookup graph must be admitted by the native compiler")
            })
            .collect::<Vec<_>>();
        for program in &programs {
            let lookup = program
                .mv_lookup
                .as_ref()
                .expect("compiled program contains the requested lookup fold");
            assert_eq!(
                lookup.input_counts.as_slice(),
                &[INPUTS_PER_LOOKUP],
                "each oracle argument must exercise multiple ordered lookup inputs"
            );
        }

        let mut l_values = (0..full_size)
            .map(|index| Fr::from(307 + index as u64 * 13))
            .collect::<Vec<_>>();
        l_values[0] = Fr::ZERO;
        l_values[1] = Fr::ONE;
        let z_sets = (0..2)
            .map(|set| {
                (0..ROWS)
                    .map(|row| Fr::from(401 + set as u64 * 10_007 + row as u64 * 19))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let source_sets = (0..2)
            .map(|set| {
                (0..ROWS)
                    .map(|row| Fr::from(503 + set as u64 * 20_011 + row as u64 * 23))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let sigma_sets = (0..2)
            .map(|set| {
                (0..ROWS)
                    .map(|row| Fr::from(601 + set as u64 * 30_013 + row as u64 * 29))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let phi = (0..ROWS)
            .map(|row| Fr::from(701 + row as u64 * 31))
            .collect::<Vec<_>>();
        let multiplicity = (0..ROWS)
            .map(|row| Fr::from(809 + row as u64 * 37))
            .collect::<Vec<_>>();

        let kernel_capacity = crate::icicle::resident_h_lookup_kernel_capacity()
            .expect("query resident CUDA kernel capacity")
            .expect("resident CUDA kernel must be available for this manual oracle");
        let chunks_before = crate::icicle::execution_stats().20;
        let syncs_before = crate::icicle::execution_stats().21;
        let calls_before = crate::icicle::execution_stats().17;
        let groups_before = crate::icicle::execution_stats().24;
        let fallback_before = crate::icicle::execution_stats().19;

        for part_index in 0..PART_STRIDE {
            let initial = (0..ROWS)
                .map(|row| Fr::from(907 + row as u64 * 41 + part_index as u64 * 43))
                .collect::<Vec<_>>();
            let mut expected = initial.clone();

            // Independent legacy-Horner oracle for the two permutation sets.
            for row in 0..ROWS {
                let full_index = row * PART_STRIDE + part_index;
                let previous = get_rotation_idx(
                    full_index,
                    last_rotation,
                    rot_scale,
                    isize,
                ) / PART_STRIDE;
                let boundary_first = (Fr::ONE - z_sets[0][row]) * l_values[full_index];
                let boundary_last =
                    (z_sets[1][row].square() - z_sets[1][row]) * l_values[full_index];
                let transition =
                    (z_sets[1][row] - z_sets[0][previous]) * l_values[full_index];
                let mut value = expected[row];
                for term in [boundary_first, boundary_last, transition] {
                    value = value * y + term;
                }
                let mut delta_power = Fr::ONE;
                for set_index in 0..2 {
                    let next = get_rotation_idx(full_index, 1, rot_scale, isize)
                        / PART_STRIDE;
                    let left = z_sets[set_index][next]
                        * (source_sets[set_index][row]
                        + beta * sigma_sets[set_index][row]
                        + gamma);
                    let beta_term = beta
                        * Fr::ZETA
                        * delta_power
                        * extended_omega.pow_vartime([full_index as u64]);
                    let right = z_sets[set_index][row]
                        * (source_sets[set_index][row] + beta_term + gamma);
                    value = value * y + (left - right) * l_values[full_index];
                    delta_power *= Fr::DELTA;
                }
                expected[row] = value;
            }
            let expected_after_permutation = expected.clone();

            crate::icicle::record_gpu_h_eval_group();
            crate::icicle::record_gpu_h_eval_group();
            let mut session = crate::icicle::try_begin_resident_h_session(
                &initial,
                kernel_capacity,
                true,
            )
            .expect("admit owner-bound resident H session")
            .expect("resident permutation H session must use CUDA");
            session
                .initialize_permutation_twiddles(extended_omega, PART_STRIDE, part_index)
                .expect("initialize permutation twiddles");
            session
                .scale_permutation_h(y, permutation_term_count(2).expect("permutation terms"))
                .expect("scale permutation accumulator");
            let mut delta_power = Fr::ONE;
            for set_index in 0..2 {
                session
                    .begin_permutation_set(
                        &z_sets[set_index],
                        &l_values,
                        &l_values,
                        y,
                        5,
                        set_index,
                        2,
                        last_rotation,
                        part_index,
                        PART_STRIDE,
                        rot_scale,
                        isize,
                    )
                    .expect("apply permutation boundaries and transition");
                session
                    .multiply_permutation_column(
                        &source_sets[set_index],
                        &sigma_sets[set_index],
                        beta,
                        gamma,
                        beta * Fr::ZETA * delta_power,
                    )
                    .expect("apply permutation copy-column products");
                session
                    .finish_permutation_set(
                        &l_values,
                        y,
                        5,
                        set_index,
                        2,
                        part_index,
                        PART_STRIDE,
                        isize,
                    )
                    .expect("fold permutation copy term");
                delta_power *= Fr::DELTA;
            }

            let permutation_snapshot = session
                .snapshot_for_test::<Fr>()
                .expect("snapshot resident H after permutation for stage diagnosis");
            assert_h_stage_matches(
                &permutation_snapshot,
                &expected_after_permutation,
                "permutation",
                part_index,
            );

            // Independent legacy batch-invert oracle for each lookup argument.
            for lookup_index in 0..LOOKUP_COUNT {
                crate::icicle::record_gpu_h_eval_group();
                let input_graphs = &input_graphs[lookup_index];
                let table_graph = &table_graphs[lookup_index];
                let mut input_data = input_graphs
                    .iter()
                    .map(GraphEvaluator::instance)
                    .collect::<Vec<_>>();
                let mut table_data = table_graph.instance();
                for start in (0..ROWS).step_by(LEGACY_ROW_CHUNK) {
                    let count = (ROWS - start).min(LEGACY_ROW_CHUNK);
                    let mut legacy_inputs = Vec::with_capacity(count * INPUTS_PER_LOOKUP);
                    for offset in 0..count {
                        let row = start + offset;
                        let full_index = row * PART_STRIDE + part_index;
                        for input_index in 0..INPUTS_PER_LOOKUP {
                            legacy_inputs.push(input_graphs[input_index].evaluate_part(
                                &mut input_data[input_index],
                                &fixed,
                                &advice,
                                &[],
                                &[],
                                &beta,
                                &gamma,
                                &theta,
                                &y,
                                &Fr::ZERO,
                                full_index,
                                rot_scale,
                                isize,
                                PART_STRIDE,
                            ));
                        }
                    }
                    let mut legacy_inverses = legacy_inputs.clone();
                    legacy_inverses.batch_invert();
                    for offset in 0..count {
                        let row = start + offset;
                        let full_index = row * PART_STRIDE + part_index;
                        let input_start = offset * INPUTS_PER_LOOKUP;
                        let inputs = &legacy_inputs[input_start..input_start + INPUTS_PER_LOOKUP];
                        let inverses =
                            &legacy_inverses[input_start..input_start + INPUTS_PER_LOOKUP];
                        let inputs_product =
                            inputs.iter().copied().fold(Fr::ONE, |acc, value| acc * value);
                        let inverse_sum =
                            inverses.iter().copied().fold(Fr::ZERO, |acc, value| acc + value);
                        let table_value = table_graph.evaluate_part(
                            &mut table_data,
                            &fixed,
                            &advice,
                            &[],
                            &[],
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &Fr::ZERO,
                            full_index,
                            rot_scale,
                            isize,
                            PART_STRIDE,
                        );
                        let next = get_rotation_idx(full_index, 1, rot_scale, isize)
                            / PART_STRIDE;
                        let lhs = table_value
                            * inputs_product
                            * (phi[next] - phi[row]);
                        // This is the old p * (table * sum(batch_inverse) - m)
                        // expression, not the new products-except recurrence.
                        let rhs = inputs_product * (table_value * inverse_sum - multiplicity[row]);
                        let mut value = expected[row];
                        value = value * y + l_values[full_index] * phi[row];
                        value = value * y + l_values[full_index] * phi[row];
                        value = value * y + (lhs - rhs) * l_values[full_index];
                        expected[row] = value;
                    }
                }
                session
                    .evaluate_mv_lookup_program(
                        &programs[lookup_index],
                        &fixed,
                        &advice,
                        &[],
                        &phi,
                        &multiplicity,
                        &l_values,
                        &l_values,
                        &l_values,
                        part_index,
                        PART_STRIDE,
                        rot_scale,
                        isize,
                        y,
                    )
                    .expect("execute resident MV lookup program on CUDA");
                let lookup_snapshot = session
                    .snapshot_for_test::<Fr>()
                    .expect("snapshot resident H after lookup for stage diagnosis");
                assert_h_stage_matches(
                    &lookup_snapshot,
                    &expected,
                    &format!("multi-lookup argument {lookup_index}"),
                    part_index,
                );
            }

            let mut gpu_values = initial;
            session
                .finish(&mut gpu_values)
                .expect("read back resident H after all permutation and lookup folds");
            assert_h_stage_matches(
                &gpu_values,
                &expected,
                "final readback after permutation and multi-lookup",
                part_index,
            );
        }

        let stats = crate::icicle::execution_stats();
        println!(
            "EZKL_RESIDENT_H_ORACLE parts={} gpu_calls={} gpu_chunks={} gpu_fallback_groups={} attempted_groups={}",
            PART_STRIDE,
            stats.17 - calls_before,
            stats.20 - chunks_before,
            stats.19 - fallback_before,
            stats.24 - groups_before,
        );
        let gpu_chunks = stats.20 - chunks_before;
        let gpu_syncs = stats.21 - syncs_before;
        assert!(stats.17 - calls_before >= (PART_STRIDE * (2 + LOOKUP_COUNT)) as u64);
        assert!(gpu_chunks > (PART_STRIDE * (2 + LOOKUP_COUNT)) as u64);
        assert!(
            gpu_syncs > 0 && gpu_syncs < gpu_chunks,
            "multiple permutation tiles must be covered by fewer actual H synchronization calls"
        );
        assert_eq!(stats.19 - fallback_before, 0, "oracle must not exercise a CPU fallback");
        assert_eq!(
            stats.24 - groups_before,
            stats.17 - calls_before,
            "every explicit resident H call must have a matching attempted GPU group"
        );
    }

    #[cfg(feature = "mv-lookup")]
    #[test]
    fn native_mv_lookup_fused_recurrence_matches_legacy_rows() {
        use halo2curves::bn256::Fr;

        const OLD_ROW_CHUNK: usize = 4096;
        const ROW_COUNT: usize = 4101;
        const PART_STRIDE: usize = 3;
        const LOOKUP_COUNT: usize = 2;
        let domain_size = ROW_COUNT * PART_STRIDE;
        let rot_scale = PART_STRIDE as i32;
        let isize = domain_size as i32;
        let rotations = [1, -2, 2];
        assert_eq!(lookup_rows_per_chunk(OLD_ROW_CHUNK, 0), None);

        for input_count in 1..=3 {
            let rows_per_chunk = lookup_rows_per_chunk(OLD_ROW_CHUNK, input_count)
                .expect("nonzero lookup input width has a bounded row chunk");
            assert!(
                rows_per_chunk * 2 <= OLD_ROW_CHUNK * input_count,
                "fused product scratch must fit the legacy inverse scratch bound"
            );
            assert_eq!(
                rows_per_chunk,
                match input_count {
                    1 => OLD_ROW_CHUNK / 2,
                    2 | 3 => OLD_ROW_CHUNK,
                    _ => unreachable!(),
                }
            );

            let mut graphs = Vec::with_capacity(input_count);
            for input_index in 0..input_count {
                let mut graph = GraphEvaluator::<G1Affine>::default();
                let rotation = graph.add_rotation(&Rotation(rotations[input_index]));
                graph.add_calculation(Calculation::Add(
                    ValueSource::Advice(input_index, rotation),
                    ValueSource::Challenge(0),
                ));
                graphs.push(graph);
            }

            for lookup_index in 0..LOOKUP_COUNT {
                let challenge = Fr::from(29 + lookup_index as u64 * 13);
                let challenges = [challenge];
                let is_zero_input = |row: usize, input_index: usize| {
                    (row == 0 && input_index == 0)
                        || (row == 1 && input_index < 2)
                        || (row == rows_per_chunk && input_index < 2)
                        || (row == OLD_ROW_CHUNK - 1 && input_index + 1 == input_count)
                        || (row == OLD_ROW_CHUNK && input_index == 0)
                };
                let advice = (0..input_count)
                    .map(|input_index| {
                        let mut values = vec![Fr::ZERO; ROW_COUNT];
                        for row in 0..ROW_COUNT {
                            let desired = if is_zero_input(row, input_index) {
                                Fr::ZERO
                            } else {
                                Fr::from(
                                    101 + lookup_index as u64 * 100_003
                                        + row as u64 * 37
                                        + input_index as u64 * 19,
                                )
                            };
                            let target_row = (row as i64 + i64::from(rotations[input_index]))
                                .rem_euclid(ROW_COUNT as i64)
                                as usize;
                            values[target_row] = desired - challenge;
                        }
                        Polynomial::<Fr, LagrangeCoeff>::from_vec(values)
                    })
                    .collect::<Vec<_>>();

                for y in [Fr::ZERO, Fr::ONE] {
                    for part_index in 0..PART_STRIDE {
                        let mut legacy_inverse_values = vec![Fr::ZERO; ROW_COUNT * input_count];
                        let mut legacy_graph_evaluations = 0usize;
                        let mut legacy_h = vec![Fr::ZERO; ROW_COUNT];

                        // Reproduce the former path: evaluate each input graph
                        // once to batch-invert, then reset graph scratch and
                        // evaluate every input graph again for its row product.
                        for start in (0..ROW_COUNT).step_by(OLD_ROW_CHUNK) {
                            let count = (ROW_COUNT - start).min(OLD_ROW_CHUNK);
                            let mut input_data =
                                graphs.iter().map(GraphEvaluator::instance).collect::<Vec<_>>();
                            let mut chunk_values = Vec::with_capacity(count * input_count);
                            for offset in 0..count {
                                let row = start + offset;
                                let idx = row * PART_STRIDE + part_index;
                                for input_index in 0..input_count {
                                    chunk_values.push(graphs[input_index].evaluate_part(
                                        &mut input_data[input_index],
                                        &[],
                                        &advice,
                                        &[],
                                        &challenges,
                                        &Fr::from(3),
                                        &Fr::from(5),
                                        &Fr::from(7),
                                        &y,
                                        &Fr::ZERO,
                                        idx,
                                        rot_scale,
                                        isize,
                                        PART_STRIDE,
                                    ));
                                    legacy_graph_evaluations += 1;
                                }
                            }
                            chunk_values.batch_invert();
                            let destination = start * input_count;
                            legacy_inverse_values[destination..destination + chunk_values.len()]
                                .copy_from_slice(&chunk_values);

                            let mut input_data =
                                graphs.iter().map(GraphEvaluator::instance).collect::<Vec<_>>();
                            for offset in 0..count {
                                let row = start + offset;
                                let idx = row * PART_STRIDE + part_index;
                                let mut inputs_product = Fr::ONE;
                                for input_index in 0..input_count {
                                    inputs_product *= graphs[input_index].evaluate_part(
                                        &mut input_data[input_index],
                                        &[],
                                        &advice,
                                        &[],
                                        &challenges,
                                        &Fr::from(3),
                                        &Fr::from(5),
                                        &Fr::from(7),
                                        &y,
                                        &Fr::ZERO,
                                        idx,
                                        rot_scale,
                                        isize,
                                        PART_STRIDE,
                                    );
                                    legacy_graph_evaluations += 1;
                                }
                                let inverse_start = row * input_count;
                                let inverse_sum = legacy_inverse_values
                                    [inverse_start..inverse_start + input_count]
                                    .iter()
                                    .fold(Fr::ZERO, |sum, inverse| sum + inverse);
                                let r_next = get_rotation_idx(idx, 1, rot_scale, isize);
                                let phi = Fr::from(17 + row as u64 + part_index as u64);
                                let phi_next = Fr::from(
                                    23 + (r_next / PART_STRIDE) as u64
                                        + lookup_index as u64 * 31,
                                );
                                let table_value = Fr::from(
                                    41 + row as u64 * 3 + lookup_index as u64 * 7,
                                );
                                let multiplicity = Fr::from(53 + row as u64 + part_index as u64);
                                let l0 = Fr::from(61 + row as u64);
                                let l_last = Fr::from(67 + part_index as u64);
                                let l_active_row = Fr::from(71 + lookup_index as u64);
                                let mut h_value = Fr::from(
                                    79 + row as u64 + part_index as u64 * 11,
                                );
                                let lhs = table_value * inputs_product * (phi_next - phi);
                                let rhs = inputs_product * (table_value * inverse_sum - multiplicity);
                                h_value = h_value * y + l0 * phi;
                                h_value = h_value * y + l_last * phi;
                                h_value = h_value * y + (lhs - rhs) * l_active_row;
                                legacy_h[row] = h_value;
                            }
                        }

                        let mut cached_h = vec![Fr::ZERO; ROW_COUNT];
                        let mut cached_graph_evaluations = 0usize;
                        for start in (0..ROW_COUNT).step_by(rows_per_chunk) {
                            let count = (ROW_COUNT - start).min(rows_per_chunk);
                            let mut input_data =
                                graphs.iter().map(GraphEvaluator::instance).collect::<Vec<_>>();
                            let cached_values = evaluate_lookup_input_values(
                                count,
                                input_count,
                                |offset, input_index| {
                                    let row = start + offset;
                                    let idx = row * PART_STRIDE + part_index;
                                    cached_graph_evaluations += 1;
                                    graphs[input_index].evaluate_part(
                                        &mut input_data[input_index],
                                        &[],
                                        &advice,
                                        &[],
                                        &challenges,
                                        &Fr::from(3),
                                        &Fr::from(5),
                                        &Fr::from(7),
                                        &y,
                                        &Fr::ZERO,
                                        idx,
                                        rot_scale,
                                        isize,
                                        PART_STRIDE,
                                    )
                                },
                            );
                            for offset in 0..count {
                                let row = start + offset;
                                let inverse_start = row * input_count;
                                let inputs_product = cached_values.inputs_product(offset);
                                let inverse_sum = legacy_inverse_values
                                    [inverse_start..inverse_start + input_count]
                                    .iter()
                                    .fold(Fr::ZERO, |sum, inverse| sum + inverse);
                                let legacy_weighted_inverse_sum = if bool::from(inputs_product.is_zero()) {
                                    Fr::ZERO
                                } else {
                                    inputs_product * inverse_sum
                                };
                                assert_eq!(
                                    cached_values.products_except_sum(offset),
                                    legacy_weighted_inverse_sum,
                                    "fused p/d recurrence differs from batch inversion at lookup={lookup_index}, inputs={input_count}, part={part_index}, row={row}"
                                );
                                for input_index in 0..input_count {
                                    if is_zero_input(row, input_index) {
                                        assert_eq!(
                                            legacy_inverse_values[inverse_start + input_index],
                                            Fr::ZERO,
                                            "zero lookup input must remain zero after batch inversion"
                                        );
                                    }
                                }
                                let r_next = get_rotation_idx(
                                    row * PART_STRIDE + part_index,
                                    1,
                                    rot_scale,
                                    isize,
                                );
                                let phi = Fr::from(17 + row as u64 + part_index as u64);
                                let phi_next = Fr::from(
                                    23 + (r_next / PART_STRIDE) as u64
                                        + lookup_index as u64 * 31,
                                );
                                let table_value = Fr::from(
                                    41 + row as u64 * 3 + lookup_index as u64 * 7,
                                );
                                let multiplicity = Fr::from(53 + row as u64 + part_index as u64);
                                let l0 = Fr::from(61 + row as u64);
                                let l_last = Fr::from(67 + part_index as u64);
                                let l_active_row = Fr::from(71 + lookup_index as u64);
                                let mut h_value = Fr::from(
                                    79 + row as u64 + part_index as u64 * 11,
                                );
                                fold_lookup_h_row(
                                    &mut h_value,
                                    y,
                                    l0,
                                    l_last,
                                    l_active_row,
                                    phi,
                                    phi_next,
                                    table_value,
                                    inputs_product,
                                    cached_values.products_except_sum(offset),
                                    multiplicity,
                                );
                                cached_h[row] = h_value;
                            }
                        }

                        assert_eq!(
                            cached_h, legacy_h,
                            "cached lookup H differs at lookup={lookup_index}, inputs={input_count}, y={y:?}, part={part_index}"
                        );
                        assert_eq!(legacy_graph_evaluations, ROW_COUNT * input_count * 2);
                        assert_eq!(cached_graph_evaluations, ROW_COUNT * input_count);
                    }
                }
            }
        }
    }

    #[cfg(feature = "mv-lookup")]
    #[test]
    fn native_mv_lookup_fused_recurrence_handles_empty_zero_and_large_inputs() {
        use halo2curves::bn256::Fr;

        let large = Option::<Fr>::from(Fr::from(2).invert())
            .expect("two has a nonzero field inverse");
        let rows = vec![
            Vec::new(),
            vec![large],
            vec![Fr::ZERO],
            vec![Fr::from(0xdead_beef_u64), Fr::from(0x1234_5678_u64)],
            vec![large, Fr::from(11)],
            vec![Fr::ZERO, large],
            vec![Fr::ZERO, Fr::from(13), Fr::ZERO],
            vec![Fr::from(0x5eed_u64), large, Fr::from(0xcafe_u64)],
        ];

        assert_eq!(lookup_rows_per_chunk(4096, 0), None);
        for row in rows {
            let fused = evaluate_lookup_input_values(1, row.len(), |_, index| row[index]);
            let mut inverses = row.clone();
            inverses.batch_invert();
            let inputs_product = row.iter().copied().fold(Fr::ONE, |product, value| product * value);
            let inverse_sum = inverses.iter().copied().fold(Fr::ZERO, |sum, value| sum + value);
            let expected_products_except_sum = if bool::from(inputs_product.is_zero()) {
                Fr::ZERO
            } else {
                inputs_product * inverse_sum
            };

            assert_eq!(fused.inputs_product(0), inputs_product);
            assert_eq!(
                fused.products_except_sum(0),
                expected_products_except_sum,
                "fused p/d recurrence differs from legacy batch inversion for input width {}",
                row.len()
            );
        }
    }

    #[test]
    fn native_shplonk_shape_counts_query_rotation_classes() {
        let mut cs = ConstraintSystem::default();
        let advice_cur_next_a = cs.advice_column();
        let advice_cur_next_b = cs.advice_column();
        let advice_cur = cs.advice_column();
        let fixed_cur = cs.fixed_column();
        let fixed_next = cs.fixed_column();
        let instance = cs.instance_column();

        cs.create_gate("shplonk resource shape", |meta| {
            let a0_cur = meta.query_advice(advice_cur_next_a, Rotation::cur());
            let a0_next = meta.query_advice(advice_cur_next_a, Rotation::next());
            let a1_cur = meta.query_advice(advice_cur_next_b, Rotation::cur());
            let a1_next = meta.query_advice(advice_cur_next_b, Rotation::next());
            let a2_cur = meta.query_advice(advice_cur, Rotation::cur());
            let fixed_cur = meta.query_fixed(fixed_cur, Rotation::cur());
            let fixed_next = meta.query_fixed(fixed_next, Rotation::next());
            vec![
                a0_cur * a0_next * a2_cur + a1_cur * a1_next,
                fixed_cur + fixed_next,
            ]
        });
        cs.shuffle("shplonk resource shuffle", |meta| {
            vec![(
                meta.query_advice(advice_cur_next_a, Rotation::cur()),
                meta.query_instance(instance, Rotation::cur()),
            )]
        });
        cs.enable_equality(advice_cur_next_a);
        cs.enable_equality(advice_cur_next_b);
        cs.enable_equality(fixed_cur);

        #[cfg(feature = "mv-lookup")]
        let cs = cs.chunk_lookups();
        let shape = native_shplonk_resource_shape::<halo2curves::bn256::Fr>(
            &cs,
            16,
            1,
            1,
            2,
        );

        // Advice columns, fixed columns, and all proof arguments are grouped
        // by their actual opened rotations, including the nonfinal
        // permutation-product set's prev/current/next openings.
        assert_eq!(shape.rotation_set_commitment_counts, vec![9, 5, 2, 1]);
        assert_eq!(shape.query_polynomial_count, 17);
        assert_eq!(shape.max_commitments_per_rotation_set, 9);
        assert_eq!(shape.max_rotation_count, 3);
        assert_eq!(shape.unique_opening_point_count, 3);
        assert_eq!(shape.permutation_product_set_count, 3);
        let _ = instance;
    }

    fn cache_fixture_gate(instance_column: usize, next_group: bool) -> GraphEvaluator<G1Affine> {
        let mut graph = GraphEvaluator::default();
        let (fixed_rotation, second_fixed_rotation, advice_rotation, instance_rotation) =
            if next_group {
                (Rotation::next(), Rotation::prev(), Rotation::cur(), Rotation::next())
            } else {
                (Rotation::cur(), Rotation::next(), Rotation::prev(), Rotation::cur())
            };
        let fixed = ValueSource::Fixed(0, graph.add_rotation(&fixed_rotation));
        let fixed_constant = ValueSource::Fixed(1, graph.add_rotation(&second_fixed_rotation));
        let advice = ValueSource::Advice(0, graph.add_rotation(&advice_rotation));
        let instance = ValueSource::Instance(instance_column, graph.add_rotation(&instance_rotation));

        let sum = graph.add_calculation(Calculation::Add(fixed, advice));
        let difference = graph.add_calculation(Calculation::Sub(fixed_constant, instance));
        let product = graph.add_calculation(Calculation::Mul(sum, difference));
        let horner = graph.add_calculation(Calculation::Horner(
            ValueSource::Constant(0),
            vec![product, fixed, advice],
            ValueSource::Y(),
        ));
        let negated = graph.add_calculation(Calculation::Negate(product));
        let combined = graph.add_calculation(Calculation::Add(horner, negated));
        let with_beta = graph.add_calculation(Calculation::Add(combined, ValueSource::Beta()));
        graph.add_calculation(Calculation::Store(with_beta));
        graph
    }

    fn cache_fixture_coefficients(seed: u64, size: usize) -> Polynomial<halo2curves::bn256::Fr, Coeff> {
        Polynomial::from_vec(
            (0..size)
                .map(|index| halo2curves::bn256::Fr::from(seed + (index as u64 + 1) * 17))
                .collect(),
        )
    }

    fn legacy_evaluate_one_gate(
        graph: &GraphEvaluator<G1Affine>,
        fixed: &[Polynomial<halo2curves::bn256::Fr, LagrangeCoeff>],
        advice: &[Polynomial<halo2curves::bn256::Fr, LagrangeCoeff>],
        instance: &[Polynomial<halo2curves::bn256::Fr, LagrangeCoeff>],
        values: &mut Polynomial<halo2curves::bn256::Fr, ExtendedLagrangeCoeff>,
        y: halo2curves::bn256::Fr,
        beta: halo2curves::bn256::Fr,
        rot_scale: i32,
        isize: i32,
        part_index: usize,
        part_stride: usize,
    ) {
        parallelize(&mut values[..], |h, start| {
            let mut data = graph.instance();
            for (offset, value) in h.iter_mut().enumerate() {
                let row = start + offset;
                let idx = row * part_stride + part_index;
                let contribution = graph.evaluate_part(
                    &mut data,
                    fixed,
                    advice,
                    instance,
                    &[],
                    &beta,
                    &halo2curves::bn256::Fr::ZERO,
                    &halo2curves::bn256::Fr::ZERO,
                    &y,
                    &halo2curves::bn256::Fr::ZERO,
                    idx,
                    rot_scale,
                    isize,
                    part_stride,
                );
                *value = *value * y + contribution;
            }
        });
    }

    #[cfg(feature = "icicle_gpu")]
    fn compiler_fixture_gate() -> GraphEvaluator<G1Affine> {
        use halo2curves::bn256::Fr;

        let mut graph = GraphEvaluator::default();
        let constant = graph.add_constant(&Fr::from(43));
        let next = graph.add_rotation(&Rotation::next());
        let previous = graph.add_rotation(&Rotation::prev());
        let current = graph.add_rotation(&Rotation::cur());

        let fixed_next = ValueSource::Fixed(0, next);
        let fixed_previous = ValueSource::Fixed(0, previous);
        let advice_current = ValueSource::Advice(0, current);
        let instance_next = ValueSource::Instance(0, next);
        let sum = graph.add_calculation(Calculation::Add(fixed_next, advice_current));
        let stored = graph.add_calculation(Calculation::Store(fixed_previous));
        let difference = graph.add_calculation(Calculation::Sub(instance_next, constant));
        let product = graph.add_calculation(Calculation::Mul(sum, difference));
        let square = graph.add_calculation(Calculation::Square(product));
        let doubled = graph.add_calculation(Calculation::Double(stored));
        let negated = graph.add_calculation(Calculation::Negate(advice_current));
        let horner = graph.add_calculation(Calculation::Horner(
            ValueSource::Challenge(0),
            vec![
                sum,
                difference,
                product,
                square,
                doubled,
                negated,
                ValueSource::PreviousValue(),
                ValueSource::Beta(),
                ValueSource::Gamma(),
                ValueSource::Theta(),
                ValueSource::Y(),
            ],
            ValueSource::Challenge(1),
        ));
        let with_square = graph.add_calculation(Calculation::Add(square, horner));
        let with_product = graph.add_calculation(Calculation::Add(with_square, product));
        graph.add_calculation(Calculation::Store(with_product));
        let repeated_operand = graph.add_calculation(Calculation::Mul(with_product, with_product));
        graph.add_calculation(Calculation::Store(repeated_operand));
        graph
    }

    #[cfg(feature = "icicle_gpu")]
    fn interpret_gate_program_row<F: PrimeField>(
        program: &GateProgram<F>,
        fixed: &[Polynomial<F, LagrangeCoeff>],
        advice: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        row_index: usize,
        part_index: usize,
        part_stride: usize,
        rot_scale: i32,
        isize: i32,
        y: F,
        initial: F,
    ) -> F {
        let mut scratch = vec![F::ZERO; program.scratch_slots];
        let mut accumulator = initial;
        let mut block_accumulator = F::ZERO;
        for instruction in &program.instructions {
            let dst = instruction[1] as usize;
            match instruction[0] {
                GATE_OP_LOAD_QUERY => {
                    let query = program.queries[instruction[2] as usize];
                    let idx = row_index * part_stride + part_index;
                    let row = get_rotation_idx(idx, query.rotation, rot_scale, isize) / part_stride;
                    scratch[dst] = match query.kind {
                        0 => fixed[query.column][row],
                        1 => advice[query.column][row],
                        2 => instance[query.column][row],
                        _ => unreachable!("compiler emitted an unsupported query kind"),
                    };
                }
                GATE_OP_LOAD_CONST => scratch[dst] = program.constants[instruction[2] as usize],
                GATE_OP_ADD => {
                    let lhs = scratch[instruction[2] as usize];
                    let rhs = scratch[instruction[3] as usize];
                    scratch[dst] = lhs + rhs;
                }
                GATE_OP_SUB => {
                    let lhs = scratch[instruction[2] as usize];
                    let rhs = scratch[instruction[3] as usize];
                    scratch[dst] = lhs - rhs;
                }
                GATE_OP_MUL => {
                    let lhs = scratch[instruction[2] as usize];
                    let rhs = scratch[instruction[3] as usize];
                    scratch[dst] = lhs * rhs;
                }
                GATE_OP_SQUARE => {
                    let value = scratch[instruction[2] as usize];
                    scratch[dst] = value.square();
                }
                GATE_OP_DOUBLE => {
                    let value = scratch[instruction[2] as usize];
                    scratch[dst] = value.double();
                }
                GATE_OP_NEGATE => {
                    let value = scratch[instruction[2] as usize];
                    scratch[dst] = -value;
                }
                GATE_OP_FOLD => accumulator = accumulator * y + scratch[dst],
                GATE_OP_SCALE_ACCUMULATOR => {
                    accumulator *= program.constants[instruction[1] as usize];
                }
                GATE_OP_FOLD_BLOCK => {
                    block_accumulator = block_accumulator * y + scratch[dst];
                }
                GATE_OP_COMMIT_BLOCK => {
                    accumulator += block_accumulator
                        * program.constants[instruction[1] as usize];
                }
                opcode => unreachable!("compiler emitted unknown opcode {opcode}"),
            }
        }
        accumulator
    }

    #[cfg(feature = "icicle_gpu")]
    #[test]
    fn gpu_gate_program_compiler_matches_graph_oracle_across_parts_and_instances() {
        use halo2curves::bn256::Fr;

        let first = compiler_fixture_gate();
        let repeated_root = first.clone();
        let mut later_root = first.clone();
        let first_root = later_root.calculations.last().unwrap().target;
        later_root.add_calculation(Calculation::Add(
            ValueSource::Intermediate(first_root),
            ValueSource::PreviousValue(),
        ));
        let graphs = vec![first, repeated_root, later_root];

        let part_stride = 8usize;
        let part_size = 4usize;
        let rot_scale = part_stride as i32;
        let isize = (part_stride * part_size) as i32;
        for proof_instance in 0..2u64 {
            let challenges = [Fr::from(101 + proof_instance), Fr::from(131 + proof_instance)];
            let y = Fr::from(151 + proof_instance);
            let beta = Fr::from(173 + proof_instance);
            let gamma = Fr::from(197 + proof_instance);
            let theta = Fr::from(211 + proof_instance);
            let program = compile_gate_program(
                &graphs,
                &challenges,
                beta,
                gamma,
                theta,
                y,
                Fr::ZERO,
                None,
                rot_scale,
                isize,
            )
            .expect("compile the bounded gate fixture")
            .expect("fixture fits the GPU H program bounds");
            assert_eq!(program.fold_count, graphs.len());
            assert!(program.queries.iter().any(|query| query.kind == 0));
            assert!(program.queries.iter().any(|query| query.kind == 1));
            assert!(program.queries.iter().any(|query| query.kind == 2));
            assert!(program.queries.iter().any(|query| query.rotation == (-1i32).rem_euclid(4)));
            assert!(program.queries.iter().any(|query| query.rotation == 1));
            let fold_slots = program
                .instructions
                .iter()
                .filter(|instruction| instruction[0] == GATE_OP_FOLD)
                .map(|instruction| instruction[1])
                .collect::<Vec<_>>();
            assert_eq!(fold_slots.len(), graphs.len());
            assert_eq!(fold_slots[0], fold_slots[1], "identical roots should share one live DAG node");
            let final_fold_index = program
                .instructions
                .iter()
                .rposition(|instruction| instruction[0] == GATE_OP_FOLD)
                .expect("fixture emits a final fold");
            let final_root_instruction = program.instructions[final_fold_index - 1];
            assert_eq!(final_root_instruction[0], GATE_OP_ADD);
            assert!(
                fold_slots[2] == final_root_instruction[2]
                    || fold_slots[2] == final_root_instruction[3],
                "the final root should reuse one of its last-use input slots"
            );
            assert!(program.instructions.iter().any(|instruction| {
                instruction[0] == GATE_OP_MUL
                    && instruction[1] == instruction[2]
                    && instruction[2] == instruction[3]
            }), "repeated lhs/rhs should be consumed safely in-place");
            let separate_instruction_count = graphs
                .iter()
                .map(|graph| {
                    compile_gate_program(
                        std::slice::from_ref(graph),
                        &challenges,
                        beta,
                        gamma,
                        theta,
                        y,
                        Fr::ZERO,
                        None,
                        rot_scale,
                        isize,
                    )
                    .expect("compile each bounded graph")
                    .expect("single graph fits program bounds")
                    .instructions
                    .len()
                })
                .sum::<usize>();
            assert!(program.instructions.len() < separate_instruction_count);
            for opcode in [
                GATE_OP_LOAD_QUERY,
                GATE_OP_LOAD_CONST,
                GATE_OP_ADD,
                GATE_OP_SUB,
                GATE_OP_MUL,
                GATE_OP_SQUARE,
                GATE_OP_DOUBLE,
                GATE_OP_NEGATE,
                GATE_OP_FOLD,
            ] {
                assert!(program.instructions.iter().any(|instruction| instruction[0] == opcode));
            }

            // Exercise the weighted block bytecode through the same CPU
            // interpreter used as the GPU compiler oracle. A native CUDA
            // execution is intentionally unnecessary for this unit oracle.
            let weighted_contribution = Fr::from(229 + proof_instance);
            let initial_scale = y.pow_vartime([graphs.len() as u64]);
            let weighted_program = compile_gate_program(
                &graphs,
                &challenges,
                beta,
                gamma,
                theta,
                y,
                Fr::ZERO,
                Some(WeightedGateGroup {
                    initial_scale,
                    contribution_scale: weighted_contribution,
                }),
                rot_scale,
                isize,
            )
            .expect("compile weighted bounded gate fixture")
            .expect("weighted fixture fits GPU H program bounds");
            assert_eq!(weighted_program.fold_count, graphs.len());
            assert_eq!(weighted_program.instructions.first().unwrap()[0], GATE_OP_SCALE_ACCUMULATOR);
            assert_eq!(weighted_program.instructions.last().unwrap()[0], GATE_OP_COMMIT_BLOCK);
            assert!(weighted_program
                .instructions
                .iter()
                .any(|instruction| instruction[0] == GATE_OP_FOLD_BLOCK));

            for part_index in 0..part_stride {
                let make_coset = |seed: u64| -> Polynomial<Fr, LagrangeCoeff> {
                    Polynomial::from_vec(
                        (0..part_size)
                            .map(|row| Fr::from(seed + (row as u64 + 1) * 19))
                            .collect(),
                    )
                };
                let fixed = vec![make_coset(17 + proof_instance * 101 + part_index as u64 * 7)];
                let advice = vec![make_coset(31 + proof_instance * 103 + part_index as u64 * 11)];
                let instance = vec![make_coset(47 + proof_instance * 107 + part_index as u64 * 13)];
                let initial = (0..part_size)
                    .map(|row| Fr::from(233 + proof_instance * 109 + part_index as u64 * 17 + row as u64))
                    .collect::<Vec<_>>();
                let mut compiled = initial.clone();
                let mut compiled_weighted = initial.clone();

                // Three-row chunks deliberately leave a one-row tail, matching
                // the evaluator's per-row behavior without invoking CUDA here.
                for chunk_start in (0..part_size).step_by(3) {
                    let chunk_end = (chunk_start + 3).min(part_size);
                    for row in chunk_start..chunk_end {
                        compiled[row] = interpret_gate_program_row(
                            &program,
                            &fixed,
                            &advice,
                            &instance,
                            row,
                            part_index,
                            part_stride,
                            rot_scale,
                            isize,
                            y,
                            initial[row],
                        );
                        compiled_weighted[row] = interpret_gate_program_row(
                            &weighted_program,
                            &fixed,
                            &advice,
                            &instance,
                            row,
                            part_index,
                            part_stride,
                            rot_scale,
                            isize,
                            y,
                            initial[row],
                        );
                    }
                }

                let mut legacy = initial;
                let mut legacy_weighted = legacy
                    .iter()
                    .map(|value| *value * initial_scale)
                    .collect::<Vec<_>>();
                for row in 0..part_size {
                    let idx = row * part_stride + part_index;
                    let mut accumulator = legacy[row];
                    let mut block_accumulator = Fr::ZERO;
                    for graph in &graphs {
                        let mut data = graph.instance();
                        let contribution = graph.evaluate_part(
                            &mut data,
                            &fixed,
                            &advice,
                            &instance,
                            &challenges,
                            &beta,
                            &gamma,
                            &theta,
                            &y,
                            &Fr::ZERO,
                            idx,
                            rot_scale,
                            isize,
                            part_stride,
                        );
                        accumulator = accumulator * y + contribution;
                        block_accumulator = block_accumulator * y + contribution;
                    }
                    legacy[row] = accumulator;
                    legacy_weighted[row] += block_accumulator * weighted_contribution;
                }
                assert_eq!(compiled, legacy, "part {part_index}, instance {proof_instance}");
                assert_eq!(
                    compiled_weighted,
                    legacy_weighted,
                    "weighted part {part_index}, instance {proof_instance}"
                );
            }
        }
    }

    #[test]
    fn native_gate_group_fusion_and_coset_cache_match_legacy() {
        use halo2curves::bn256::Fr;

        let domain = EvaluationDomain::<Fr>::new(3, 4);
        let n = domain.get_n() as usize;
        let extended_size = domain.extended_len();
        let part_stride = (extended_size / n) as usize;
        assert_eq!(part_stride, 2);
        let rot_scale = part_stride as i32;
        let isize = extended_size as i32;
        let fixed_sources = vec![
            FixedColumnSource::from_polynomial(Polynomial::<Fr, LagrangeCoeff>::from_vec(
                (0..n).map(|row| Fr::from((row * 3 + 1) as u64)).collect(),
            ))
            .expect("build nonconstant fixed source"),
            FixedColumnSource::from_polynomial(Polynomial::<Fr, LagrangeCoeff>::from_vec(
                vec![Fr::from(7); n],
            ))
            .expect("build compact constant fixed source"),
        ];
        assert!(fixed_sources[0].constant_value().is_none());
        assert!(fixed_sources[1].constant_value().is_some());

        let factor_source = cache_fixture_coefficients(89, n);
        let factor_buffer = domain
            .extended_part_factors_spilled(Fr::ONE)
            .expect("allocate shared factor output buffer");
        let factor_buffer_address = factor_buffer.as_ptr();
        let factor_buffer_result = domain
            .coeff_to_extended_part_in_factor_buffer(factor_buffer, &factor_source, Fr::ONE)
            .expect("reuse factor buffer as selected coset output");
        let factor_legacy_result = domain
            .coeff_to_extended_part_spilled(&factor_source, Fr::ONE)
            .expect("legacy selected coset conversion");
        assert_eq!(factor_buffer_address, factor_buffer_result.as_ptr());
        assert_eq!(&factor_buffer_result[..], &factor_legacy_result[..]);

        let gates = vec![
            cache_fixture_gate(0, false),
            cache_fixture_gate(1, true),
            cache_fixture_gate(1, true),
            cache_fixture_gate(0, false),
            cache_fixture_gate(0, false),
            cache_fixture_gate(1, true),
        ];
        let max_gate_dependencies = gates
            .iter()
            .map(|graph| graph.column_dependencies().column_count())
            .max()
            .expect("fixture has gates");
        assert_eq!(max_gate_dependencies, 4);

        let groups = custom_gate_groups(&gates);
        let group_ranges = groups
            .iter()
            .map(|group| (group.start, group.end))
            .collect::<Vec<_>>();
        assert_eq!(group_ranges, vec![(0, 1), (1, 3), (3, 5), (5, 6)]);
        assert_eq!(
            gates[0]
                .column_dependencies()
                .column_count()
                .saturating_add(
                    gates[1]
                        .column_dependencies()
                        .additional_column_count(&gates[0].column_dependencies()),
                ),
            max_gate_dependencies + 1,
        );

        let fixed_constant_columns = [1].into_iter().collect::<BTreeSet<_>>();
        let (group_order, original_fft_loads, scheduled_fft_loads, reordered) =
            schedule_custom_gate_groups(&gates, &groups, &fixed_constant_columns);
        assert!(reordered, "fixture must strictly reduce source FFT loads");
        assert_eq!(group_order, vec![0, 2, 1, 3]);
        assert_eq!(original_fft_loads, 6);
        assert_eq!(scheduled_fft_loads, 4);

        let mut previous_value_gate = gates.clone();
        let previous_root = previous_value_gate[0]
            .calculations
            .last()
            .expect("fixture gate root")
            .target;
        previous_value_gate[0].add_calculation(Calculation::Add(
            ValueSource::Intermediate(previous_root),
            ValueSource::PreviousValue(),
        ));
        let previous_value_groups = custom_gate_groups(&previous_value_gate);
        let (guarded_order, _, _, guarded_reorder) = schedule_custom_gate_groups(
            &previous_value_gate,
            &previous_value_groups,
            &fixed_constant_columns,
        );
        assert!(!guarded_reorder, "PreviousValue graphs retain native order");
        assert_eq!(guarded_order, (0..previous_value_groups.len()).collect::<Vec<_>>());

        let y = Fr::from(13);
        let beta = Fr::from(19);
        let mut total_groups = 0;
        let mut total_advice_loaded = 0;
        let mut total_advice_reused = 0;
        let mut total_instance_loaded = 0;
        let mut total_fixed_loaded = 0;
        let mut total_fixed_reused = 0;
        let mut total_constant_loaded = 0;
        let mut total_constant_reused = 0;
        let mut total_actual_fft_loads = 0;

        for part_index in 0..part_stride {
            let extended_omega_factor = domain
                .get_extended_omega()
                .pow_vartime([part_index as u64]);
            let mut fused_values = Polynomial::from_vec(vec![Fr::ZERO; n]);
            let mut legacy_values = Polynomial::from_vec(vec![Fr::ZERO; n]);

            for proof_instance_index in 0..2 {
                let advice = vec![cache_fixture_coefficients(23 + proof_instance_index as u64, n)];
                let instance = vec![
                    cache_fixture_coefficients(47 + proof_instance_index as u64, n),
                    cache_fixture_coefficients(71 + proof_instance_index as u64, n),
                ];
                let mut fixed_cosets = empty_coset_cache::<Fr>(fixed_sources.len());
                let mut advice_cosets = empty_coset_cache::<Fr>(advice.len());
                let mut instance_cosets = empty_coset_cache::<Fr>(instance.len());
                let mut previous_fixed_columns = BTreeSet::new();
                let mut previous_advice_columns = BTreeSet::new();
                let mut previous_instance_columns = BTreeSet::new();
                let mut instance_advice_loaded = 0;
                let mut instance_advice_reused = 0;
                let mut instance_columns_loaded = 0;
                let mut instance_fixed_loaded = 0;
                let mut instance_fixed_reused = 0;
                let mut instance_constant_loaded = 0;
                let mut instance_constant_reused = 0;

                for graph in &gates {
                    let graph_dependencies = graph.column_dependencies();
                    let graph_fixed = load_fixed_cosets(
                        &domain,
                        &fixed_sources,
                        &graph_dependencies.fixed,
                        extended_omega_factor,
                    )
                    .expect("legacy per-gate fixed conversion");
                    let graph_advice = load_cosets(
                        &domain,
                        &advice,
                        &graph_dependencies.advice,
                        extended_omega_factor,
                    )
                    .expect("legacy per-gate advice conversion");
                    let graph_instance = load_cosets(
                        &domain,
                        &instance,
                        &graph_dependencies.instance,
                        extended_omega_factor,
                    )
                    .expect("legacy per-gate instance conversion");
                    legacy_evaluate_one_gate(
                        graph,
                        &graph_fixed,
                        &graph_advice,
                        &graph_instance,
                        &mut legacy_values,
                        y,
                        beta,
                        rot_scale,
                        isize,
                        part_index,
                        part_stride,
                    );
                }

                for (group_position, group_index) in group_order.iter().copied().enumerate() {
                    let group = &groups[group_index];
                    let dependencies = &group.dependencies;

                    evict_inactive_cosets(
                        &mut fixed_cosets,
                        &previous_fixed_columns,
                        &dependencies.fixed,
                    )
                    .expect("evict inactive fixed cosets");
                    evict_inactive_cosets(
                        &mut advice_cosets,
                        &previous_advice_columns,
                        &dependencies.advice,
                    )
                    .expect("evict inactive advice cosets");
                    evict_inactive_cosets(
                        &mut instance_cosets,
                        &previous_instance_columns,
                        &dependencies.instance,
                    )
                    .expect("evict inactive instance cosets");

                    let (fixed_counts, advice_counts, instance_counts) = load_gate_group_cosets(
                        &domain,
                        &fixed_sources,
                        &advice,
                        &instance,
                        &dependencies.fixed,
                        &dependencies.advice,
                        &dependencies.instance,
                        extended_omega_factor,
                        &mut fixed_cosets,
                        &mut advice_cosets,
                        &mut instance_cosets,
                    )
                    .expect("load missing group cosets");

                    let live_fixed = fixed_cosets.iter().filter(|coset| !coset.is_empty()).count();
                    let live_advice = advice_cosets.iter().filter(|coset| !coset.is_empty()).count();
                    let live_instance = instance_cosets.iter().filter(|coset| !coset.is_empty()).count();
                    assert_eq!(
                        live_fixed + live_advice + live_instance,
                        dependencies.column_count(),
                    );
                    assert!(live_fixed + live_advice + live_instance <= max_gate_dependencies);
                    assert!(fixed_cosets.iter().enumerate().all(|(column, coset)| {
                        dependencies.fixed.contains(&column) || coset.is_empty()
                    }));
                    assert!(advice_cosets.iter().enumerate().all(|(column, coset)| {
                        dependencies.advice.contains(&column) || coset.is_empty()
                    }));
                    assert!(instance_cosets.iter().enumerate().all(|(column, coset)| {
                        dependencies.instance.contains(&column) || coset.is_empty()
                    }));

                    for &column in &dependencies.fixed {
                        let legacy_fixed = load_fixed_cosets(
                            &domain,
                            &fixed_sources,
                            &BTreeSet::from([column]),
                            extended_omega_factor,
                        )
                        .expect("legacy fixed conversion");
                        assert_eq!(&fixed_cosets[column][..], &legacy_fixed[column][..]);
                    }
                    for &column in &dependencies.advice {
                        let legacy_advice = load_cosets(
                            &domain,
                            &advice,
                            &BTreeSet::from([column]),
                            extended_omega_factor,
                        )
                        .expect("legacy advice conversion");
                        assert_eq!(&advice_cosets[column][..], &legacy_advice[column][..]);
                    }
                    for &column in &dependencies.instance {
                        let legacy_instance = load_cosets(
                            &domain,
                            &instance,
                            &BTreeSet::from([column]),
                            extended_omega_factor,
                        )
                        .expect("legacy instance conversion");
                        assert_eq!(&instance_cosets[column][..], &legacy_instance[column][..]);
                    }

                    let weighted = Some(WeightedGateGroup {
                        initial_scale: if group_position == 0 {
                            y.pow_vartime([gates.len() as u64])
                        } else {
                            Fr::ONE
                        },
                        contribution_scale: y.pow_vartime([
                            (gates.len() - group.end) as u64,
                        ]),
                    });

                    evaluate_graph_group(
                        &gates[group.start..group.end],
                        &fixed_cosets,
                        &advice_cosets,
                        &instance_cosets,
                        &[],
                        &mut fused_values,
                        y,
                        beta,
                        Fr::ZERO,
                        Fr::ZERO,
                        weighted,
                        rot_scale,
                        isize,
                        part_index,
                        part_stride,
                    ).expect("evaluate weighted native gate group");

                    instance_fixed_loaded += fixed_counts.transforms_loaded;
                    instance_fixed_reused += fixed_counts.transforms_reused;
                    instance_constant_loaded += fixed_counts.constant_fills_loaded;
                    instance_constant_reused += fixed_counts.constant_fills_reused;
                    instance_advice_loaded += advice_counts.transforms_loaded;
                    instance_advice_reused += advice_counts.transforms_reused;
                    instance_columns_loaded += instance_counts.transforms_loaded;
                    total_actual_fft_loads += fixed_counts.transforms_loaded
                        + advice_counts.transforms_loaded
                        + instance_counts.transforms_loaded;
                    total_groups += 1;
                    previous_fixed_columns = dependencies.fixed.clone();
                    previous_advice_columns = dependencies.advice.clone();
                    previous_instance_columns = dependencies.instance.clone();
                }

                // Production drops these caches before entering the next
                // permutation/lookup/shuffle phase.
                drop(fixed_cosets);
                drop(advice_cosets);
                drop(instance_cosets);

                assert_eq!(instance_fixed_loaded, 1);
                assert_eq!(instance_fixed_reused, 3);
                assert_eq!(instance_constant_loaded, 1);
                assert_eq!(instance_constant_reused, 3);
                assert_eq!(instance_advice_loaded, 1);
                assert_eq!(instance_advice_reused, 3);
                assert_eq!(instance_columns_loaded, 2);
                total_fixed_loaded += instance_fixed_loaded;
                total_fixed_reused += instance_fixed_reused;
                total_constant_loaded += instance_constant_loaded;
                total_constant_reused += instance_constant_reused;
                total_advice_loaded += instance_advice_loaded;
                total_advice_reused += instance_advice_reused;
                total_instance_loaded += instance_columns_loaded;
            }
            assert_eq!(&fused_values[..], &legacy_values[..]);
        }

        // Edge challenges exercise the exponent-zero term and confirm that
        // the weighted fold never divides by y. The values carry across two
        // proof instances just as they do in the production H evaluator.
        for edge_y in [Fr::ZERO, Fr::ONE] {
            let part_index = part_stride - 1;
            let extended_omega_factor = domain
                .get_extended_omega()
                .pow_vartime([part_index as u64]);
            let mut weighted_values = Polynomial::from_vec(vec![Fr::ZERO; n]);
            let mut legacy_values = Polynomial::from_vec(vec![Fr::ZERO; n]);
            for proof_instance_index in 0..2 {
                let advice = vec![cache_fixture_coefficients(
                    223 + proof_instance_index as u64,
                    n,
                )];
                let instance = vec![
                    cache_fixture_coefficients(247 + proof_instance_index as u64, n),
                    cache_fixture_coefficients(271 + proof_instance_index as u64, n),
                ];
                for graph in &gates {
                    let dependencies = graph.column_dependencies();
                    let fixed = load_fixed_cosets(
                        &domain,
                        &fixed_sources,
                        &dependencies.fixed,
                        extended_omega_factor,
                    )
                    .expect("edge-y legacy fixed conversion");
                    let advice_cosets = load_cosets(
                        &domain,
                        &advice,
                        &dependencies.advice,
                        extended_omega_factor,
                    )
                    .expect("edge-y legacy advice conversion");
                    let instance_cosets = load_cosets(
                        &domain,
                        &instance,
                        &dependencies.instance,
                        extended_omega_factor,
                    )
                    .expect("edge-y legacy instance conversion");
                    legacy_evaluate_one_gate(
                        graph,
                        &fixed,
                        &advice_cosets,
                        &instance_cosets,
                        &mut legacy_values,
                        edge_y,
                        beta,
                        rot_scale,
                        isize,
                        part_index,
                        part_stride,
                    );
                }

                let mut fixed_cosets = empty_coset_cache::<Fr>(fixed_sources.len());
                let mut advice_cosets = empty_coset_cache::<Fr>(advice.len());
                let mut instance_cosets = empty_coset_cache::<Fr>(instance.len());
                let mut previous_fixed = BTreeSet::new();
                let mut previous_advice = BTreeSet::new();
                let mut previous_instance = BTreeSet::new();
                for (group_position, group_index) in group_order.iter().copied().enumerate() {
                    let group = &groups[group_index];
                    let dependencies = &group.dependencies;
                    evict_inactive_cosets(
                        &mut fixed_cosets,
                        &previous_fixed,
                        &dependencies.fixed,
                    )
                    .expect("edge-y evict fixed cosets");
                    evict_inactive_cosets(
                        &mut advice_cosets,
                        &previous_advice,
                        &dependencies.advice,
                    )
                    .expect("edge-y evict advice cosets");
                    evict_inactive_cosets(
                        &mut instance_cosets,
                        &previous_instance,
                        &dependencies.instance,
                    )
                    .expect("edge-y evict instance cosets");
                    load_missing_fixed_cosets(
                        &domain,
                        &fixed_sources,
                        &dependencies.fixed,
                        extended_omega_factor,
                        &mut fixed_cosets,
                    )
                    .expect("edge-y load fixed cosets");
                    load_missing_cosets(
                        &domain,
                        &advice,
                        &dependencies.advice,
                        extended_omega_factor,
                        &mut advice_cosets,
                    )
                    .expect("edge-y load advice cosets");
                    load_missing_cosets(
                        &domain,
                        &instance,
                        &dependencies.instance,
                        extended_omega_factor,
                        &mut instance_cosets,
                    )
                    .expect("edge-y load instance cosets");
                    let weighted = Some(WeightedGateGroup {
                        initial_scale: if group_position == 0 {
                            edge_y.pow_vartime([gates.len() as u64])
                        } else {
                            Fr::ONE
                        },
                        contribution_scale: edge_y.pow_vartime([
                            (gates.len() - group.end) as u64,
                        ]),
                    });
                    evaluate_graph_group(
                        &gates[group.start..group.end],
                        &fixed_cosets,
                        &advice_cosets,
                        &instance_cosets,
                        &[],
                        &mut weighted_values,
                        edge_y,
                        beta,
                        Fr::ZERO,
                        Fr::ZERO,
                        weighted,
                        rot_scale,
                        isize,
                        part_index,
                        part_stride,
                    )
                    .expect("evaluate edge-y weighted gate group");
                    previous_fixed = dependencies.fixed.clone();
                    previous_advice = dependencies.advice.clone();
                    previous_instance = dependencies.instance.clone();
                }
            }
            assert_eq!(&weighted_values[..], &legacy_values[..], "edge challenge {edge_y:?}");
        }

        // Four groups per part/proof instance prove that the cache retains
        // only the current bounded dependency set while nearby groups reuse
        // their loaded columns.
        assert_eq!(total_groups, 16);
        assert_eq!(total_advice_loaded, 4);
        assert_eq!(total_advice_reused, 12);
        assert_eq!(total_instance_loaded, 8);
        assert_eq!(total_fixed_loaded, 4);
        assert_eq!(total_fixed_reused, 12);
        assert_eq!(total_constant_loaded, 4);
        assert_eq!(total_constant_reused, 12);
        assert_eq!(total_actual_fft_loads, 16);
    }

    #[test]
    fn native_lookup_argument_coset_cache_matches_legacy() {
        use halo2curves::bn256::Fr;

        fn lookup_argument_graph(
            fixed_columns: &[usize],
            advice_columns: &[usize],
            instance_columns: &[usize],
            rotation_offset: usize,
        ) -> GraphEvaluator<G1Affine> {
            let mut graph = GraphEvaluator::default();
            let rotations = [Rotation::cur(), Rotation::next(), Rotation::prev()];
            let mut queried_values = Vec::new();
            let mut query_index = 0usize;
            for &column in fixed_columns {
                let rotation = &rotations[(rotation_offset + query_index) % rotations.len()];
                queried_values.push(ValueSource::Fixed(column, graph.add_rotation(rotation)));
                query_index += 1;
            }
            for &column in advice_columns {
                let rotation = &rotations[(rotation_offset + query_index) % rotations.len()];
                queried_values.push(ValueSource::Advice(column, graph.add_rotation(rotation)));
                query_index += 1;
            }
            for &column in instance_columns {
                let rotation = &rotations[(rotation_offset + query_index) % rotations.len()];
                queried_values.push(ValueSource::Instance(column, graph.add_rotation(rotation)));
                query_index += 1;
            }

            let mut accumulated = queried_values[0];
            for value in queried_values.into_iter().skip(1) {
                accumulated = graph.add_calculation(Calculation::Add(accumulated, value));
            }
            let accumulated = graph.add_calculation(Calculation::Add(
                accumulated,
                ValueSource::Challenge(0),
            ));
            graph.add_calculation(Calculation::Store(accumulated));
            graph
        }

        let domain = EvaluationDomain::<Fr>::new(3, 4);
        let n = domain.get_n() as usize;
        let extended_size = domain.extended_len();
        let part_stride = (extended_size / n) as usize;
        assert_eq!(part_stride, 2);
        let part_size = extended_size / part_stride;
        let rot_scale = part_stride as i32;
        let isize = extended_size as i32;
        let fixed_sources = vec![
            FixedColumnSource::from_polynomial(Polynomial::<Fr, LagrangeCoeff>::from_vec(
                (0..n).map(|row| Fr::from((row * 5 + 3) as u64)).collect(),
            ))
            .expect("build first nonconstant fixed source"),
            FixedColumnSource::from_polynomial(Polynomial::<Fr, LagrangeCoeff>::from_vec(
                vec![Fr::from(17); n],
            ))
            .expect("build constant fixed source"),
            FixedColumnSource::from_polynomial(Polynomial::<Fr, LagrangeCoeff>::from_vec(
                (0..n).map(|row| Fr::from((row * 7 + 11) as u64)).collect(),
            ))
            .expect("build second nonconstant fixed source"),
        ];
        assert!(fixed_sources[0].constant_value().is_none());
        assert!(fixed_sources[1].constant_value().is_some());
        assert!(fixed_sources[2].constant_value().is_none());

        let argument_graphs = vec![
            lookup_argument_graph(&[0, 1], &[0, 1], &[0], 0),
            lookup_argument_graph(&[1, 2], &[1, 2], &[1], 1),
            lookup_argument_graph(&[0, 2], &[0, 2], &[0, 2], 2),
            lookup_argument_graph(&[0, 1], &[0, 1], &[0], 1),
        ];
        let y = Fr::from(23);
        let beta = Fr::from(31);
        let gamma = Fr::from(37);
        let theta = Fr::from(43);
        let challenges = [Fr::from(47)];

        for part_index in 0..part_stride {
            let extended_omega_factor = domain
                .get_extended_omega()
                .pow_vartime([part_index as u64]);
            for proof_instance_index in 0..2 {
                let advice = (0..3)
                    .map(|column| {
                        cache_fixture_coefficients(
                            101 + proof_instance_index as u64 * 13 + column as u64,
                            n,
                        )
                    })
                    .collect::<Vec<_>>();
                let instance = (0..3)
                    .map(|column| {
                        cache_fixture_coefficients(
                            211 + proof_instance_index as u64 * 17 + column as u64,
                            n,
                        )
                    })
                    .collect::<Vec<_>>();

                // These caches are freshly reconstructed at every part and
                // proof-instance boundary, as in the production lookup loop.
                let mut fixed_cache = empty_coset_cache::<Fr>(fixed_sources.len());
                let mut advice_cache = empty_coset_cache::<Fr>(advice.len());
                let mut instance_cache = empty_coset_cache::<Fr>(instance.len());
                let mut previous_dependencies = ColumnDependencies::default();
                let initial_h = vec![
                    Fr::from(53 + proof_instance_index as u64 * 19 + part_index as u64);
                    part_size
                ];
                let mut cached_h =
                    Polynomial::<Fr, ExtendedLagrangeCoeff>::from_vec(initial_h.clone());
                let mut legacy_h =
                    Polynomial::<Fr, ExtendedLagrangeCoeff>::from_vec(initial_h);

                for (argument_index, graph) in argument_graphs.iter().enumerate() {
                    let dependencies = graph.column_dependencies();
                    let prior_dependencies = previous_dependencies.clone();

                    // Independent legacy oracle: rebuild every argument's
                    // three source classes from scratch using the old loaders.
                    let legacy_fixed = load_fixed_cosets(
                        &domain,
                        &fixed_sources,
                        &dependencies.fixed,
                        extended_omega_factor,
                    )
                    .expect("load legacy fixed cosets");
                    let legacy_advice = load_cosets(
                        &domain,
                        &advice,
                        &dependencies.advice,
                        extended_omega_factor,
                    )
                    .expect("load legacy advice cosets");
                    let legacy_instance = load_cosets(
                        &domain,
                        &instance,
                        &dependencies.instance,
                        extended_omega_factor,
                    )
                    .expect("load legacy instance cosets");

                    let (fixed_counts, advice_counts, instance_counts) =
                        refresh_lookup_argument_cosets(
                            &domain,
                            &fixed_sources,
                            &advice,
                            &instance,
                            &dependencies,
                            extended_omega_factor,
                            &mut previous_dependencies,
                            &mut fixed_cache,
                            &mut advice_cache,
                            &mut instance_cache,
                        )
                        .expect("refresh production lookup coset cache");

                    let expected_fixed_transforms_loaded = dependencies
                        .fixed
                        .difference(&prior_dependencies.fixed)
                        .filter(|column| fixed_sources[**column].constant_value().is_none())
                        .count();
                    let expected_fixed_transforms_reused = dependencies
                        .fixed
                        .intersection(&prior_dependencies.fixed)
                        .filter(|column| fixed_sources[**column].constant_value().is_none())
                        .count();
                    let expected_constant_fills_loaded = dependencies
                        .fixed
                        .difference(&prior_dependencies.fixed)
                        .filter(|column| fixed_sources[**column].constant_value().is_some())
                        .count();
                    let expected_constant_fills_reused = dependencies
                        .fixed
                        .intersection(&prior_dependencies.fixed)
                        .filter(|column| fixed_sources[**column].constant_value().is_some())
                        .count();
                    assert_eq!(
                        fixed_counts.transforms_loaded,
                        expected_fixed_transforms_loaded
                    );
                    assert_eq!(
                        fixed_counts.transforms_reused,
                        expected_fixed_transforms_reused
                    );
                    assert_eq!(
                        fixed_counts.constant_fills_loaded,
                        expected_constant_fills_loaded
                    );
                    assert_eq!(
                        fixed_counts.constant_fills_reused,
                        expected_constant_fills_reused
                    );
                    assert_eq!(
                        advice_counts.transforms_loaded,
                        dependencies
                            .advice
                            .difference(&prior_dependencies.advice)
                            .count()
                    );
                    assert_eq!(
                        advice_counts.transforms_reused,
                        dependencies
                            .advice
                            .intersection(&prior_dependencies.advice)
                            .count()
                    );
                    assert_eq!(
                        instance_counts.transforms_loaded,
                        dependencies
                            .instance
                            .difference(&prior_dependencies.instance)
                            .count()
                    );
                    assert_eq!(
                        instance_counts.transforms_reused,
                        dependencies
                            .instance
                            .intersection(&prior_dependencies.instance)
                            .count()
                    );

                    for (column, coset) in fixed_cache.iter().enumerate() {
                        if dependencies.fixed.contains(&column) {
                            assert_eq!(&coset[..], &legacy_fixed[column][..]);
                        } else {
                            assert!(coset.is_empty(), "stale fixed column {column}");
                        }
                    }
                    for (column, coset) in advice_cache.iter().enumerate() {
                        if dependencies.advice.contains(&column) {
                            assert_eq!(&coset[..], &legacy_advice[column][..]);
                        } else {
                            assert!(coset.is_empty(), "stale advice column {column}");
                        }
                    }
                    for (column, coset) in instance_cache.iter().enumerate() {
                        if dependencies.instance.contains(&column) {
                            assert_eq!(&coset[..], &legacy_instance[column][..]);
                        } else {
                            assert!(coset.is_empty(), "stale instance column {column}");
                        }
                    }
                    let live_columns = fixed_cache.iter().filter(|coset| !coset.is_empty()).count()
                        + advice_cache.iter().filter(|coset| !coset.is_empty()).count()
                        + instance_cache.iter().filter(|coset| !coset.is_empty()).count();
                    assert_eq!(live_columns, dependencies.column_count());
                    assert!(live_columns <= dependencies.column_count());

                    evaluate_graph_group(
                        std::slice::from_ref(graph),
                        &legacy_fixed,
                        &legacy_advice,
                        &legacy_instance,
                        &challenges,
                        &mut legacy_h,
                        y,
                        beta,
                        gamma,
                        theta,
                        None,
                        rot_scale,
                        isize,
                        part_index,
                        part_stride,
                    )
                    .expect("evaluate legacy lookup graph");
                    evaluate_graph_group(
                        std::slice::from_ref(graph),
                        &fixed_cache,
                        &advice_cache,
                        &instance_cache,
                        &challenges,
                        &mut cached_h,
                        y,
                        beta,
                        gamma,
                        theta,
                        None,
                        rot_scale,
                        isize,
                        part_index,
                        part_stride,
                    )
                    .expect("evaluate lookup graph with incremental cache");
                    assert_eq!(
                        &cached_h[..],
                        &legacy_h[..],
                        "argument={argument_index}, part={part_index}, instance={proof_instance_index}"
                    );
                }
            }
        }
    }

    #[test]
    fn native_permutation_weighted_fold_stream_matches_legacy_horner() {
        use halo2curves::bn256::Fr;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;

        fn random_coefficients(
            rng: &mut ChaCha20Rng,
            size: usize,
        ) -> Polynomial<Fr, Coeff> {
            Polynomial::from_vec((0..size).map(|_| Fr::random(&mut *rng)).collect())
        }

        fn random_values(rng: &mut ChaCha20Rng, size: usize) -> Vec<Fr> {
            (0..size).map(|_| Fr::random(&mut *rng)).collect()
        }

        fn copy_term(
            z: &[Fr],
            source: &[Fr],
            sigma: &[Fr],
            active: &[Fr],
            beta: Fr,
            gamma: Fr,
            delta_power: Fr,
            extended_omega: Fr,
            domain_omega: Fr,
            part_index: usize,
            part_stride: usize,
            rot_scale: i32,
            isize: i32,
        ) -> Vec<Fr> {
            let delta_start = beta * Fr::ZETA * delta_power;
            let mut beta_term = delta_start * extended_omega.pow_vartime([part_index as u64]);
            let mut term = Vec::with_capacity(source.len());
            for row in 0..source.len() {
                let idx = row * part_stride + part_index;
                let next = get_rotation_idx(idx, 1, rot_scale, isize);
                let left = z[next / part_stride] * (source[row] + beta * sigma[row] + gamma);
                let right = z[row] * (source[row] + beta_term + gamma);
                term.push((left - right) * active[idx]);
                beta_term *= domain_omega;
            }
            term
        }

        let domain = EvaluationDomain::<Fr>::new(3, 4);
        let n = domain.get_n() as usize;
        let extended_size = domain.extended_len();
        let part_stride = (extended_size / n) as usize;
        assert!(part_stride > 0);
        let rot_scale = part_stride as i32;
        let isize = extended_size as i32;
        let part_size = extended_size / part_stride;
        let extended_omega = domain.get_extended_omega();
        let domain_omega = domain.get_omega();
        let last_rotation = Rotation(-2);
        let beta = Fr::from(29);
        let gamma = Fr::from(41);
        let mut rng = ChaCha20Rng::from_seed([0x5a; 32]);
        let mut saw_next_wrap = false;
        let mut saw_last_wrap = false;

        for set_count in [0usize, 1, 2, 3, 16, 130] {
            let z_coefficients = (0..set_count)
                .map(|_| random_coefficients(&mut rng, n))
                .collect::<Vec<_>>();

            for part_index in 0..part_stride {
                let extended_omega_factor = extended_omega.pow_vartime([part_index as u64]);
                let l0 = random_values(&mut rng, extended_size);
                let l_last = random_values(&mut rng, extended_size);
                let active = random_values(&mut rng, extended_size);
                let sources = (0..set_count)
                    .map(|_| random_values(&mut rng, part_size))
                    .collect::<Vec<_>>();
                let sigmas = (0..set_count)
                    .map(|_| random_values(&mut rng, part_size))
                    .collect::<Vec<_>>();
                let initial = (0..part_size)
                    .map(|_| {
                        let value = Fr::random(&mut rng);
                        if bool::from(value.is_zero()) { Fr::ONE } else { value }
                    })
                    .collect::<Vec<_>>();

                for row in 0..part_size {
                    let idx = row * part_stride + part_index;
                    let next = get_rotation_idx(idx, 1, rot_scale, isize);
                    let previous = get_rotation_idx(idx, last_rotation.0, rot_scale, isize);
                    saw_next_wrap |= next < idx;
                    saw_last_wrap |= previous > idx;
                }

                if set_count == 0 {
                    // The production evaluator skips the permutation fold
                    // altogether when there are no product sets.
                    continue;
                }

                let total_terms = permutation_term_count(set_count).expect("term count");
                for y in [Fr::ZERO, Fr::ONE, Fr::from(73)] {
                    let mut candidate = initial.clone();
                    let mut streamed_z_loads = 0usize;
                    if set_count == 1 {
                        // Keep the production one-set branch in its original
                        // Horner order and transform lifecycle.
                        let first_z = domain
                            .coeff_to_extended_part_spilled(
                                &z_coefficients[0],
                                extended_omega_factor,
                            )
                            .expect("load one-set first boundary Z");
                        streamed_z_loads += 1;
                        for row in 0..part_size {
                            let idx = row * part_stride + part_index;
                            candidate[row] = candidate[row] * y
                                + ((Fr::ONE - first_z[row]) * l0[idx]);
                        }
                        drop(first_z);
                        let last_z = domain
                            .coeff_to_extended_part_spilled(
                                &z_coefficients[0],
                                extended_omega_factor,
                            )
                            .expect("load one-set last boundary Z");
                        streamed_z_loads += 1;
                        for row in 0..part_size {
                            let idx = row * part_stride + part_index;
                            let z = last_z[row];
                            candidate[row] = candidate[row] * y
                                + ((z.square() - z) * l_last[idx]);
                        }
                        drop(last_z);
                        let copy_z = domain
                            .coeff_to_extended_part_spilled(
                                &z_coefficients[0],
                                extended_omega_factor,
                            )
                            .expect("load one-set copy Z");
                        streamed_z_loads += 1;
                        let term = copy_term(
                            &copy_z[..],
                            &sources[0],
                            &sigmas[0],
                            &active,
                            beta,
                            gamma,
                            Fr::ONE,
                            extended_omega,
                            domain_omega,
                            part_index,
                            part_stride,
                            rot_scale,
                            isize,
                        );
                        for row in 0..part_size {
                            candidate[row] = candidate[row] * y + term[row];
                        }
                        drop(copy_z);
                    } else {
                        scale_permutation_h(&mut candidate, y, total_terms)
                            .expect("scale weighted H");
                        let expected_split_contributions =
                            set_count.checked_mul(3).expect("split contribution count");
                        let mut logical_terms = 0usize;
                        let mut split_contributions = 0usize;
                        let mut current_z = Some(
                            domain
                                .coeff_to_extended_part_spilled(
                                    &z_coefficients[0],
                                    extended_omega_factor,
                                )
                                .expect("load first streamed Z coset"),
                        );
                        streamed_z_loads += 1;
                        let mut delta_power = Fr::ONE;

                        for set_index in 0..set_count {
                            let z = current_z.as_ref().expect("current streamed Z");
                            let (set_logical_terms, set_split_contributions) =
                                add_permutation_set_h_terms(
                                    &mut candidate,
                                    &z[..],
                                    &l0,
                                    &l_last,
                                    y,
                                    total_terms,
                                    set_index,
                                    set_count,
                                    last_rotation,
                                    part_index,
                                    part_stride,
                                    rot_scale,
                                    isize,
                                )
                                .expect("add row-combined boundary and transition halves");
                            logical_terms += set_logical_terms;
                            split_contributions += set_split_contributions;

                            let term = copy_term(
                                &z[..],
                                &sources[set_index],
                                &sigmas[set_index],
                                &active,
                                beta,
                                gamma,
                                delta_power,
                                extended_omega,
                                domain_omega,
                                part_index,
                                part_stride,
                                rot_scale,
                                isize,
                            );
                            let copy_term_index = set_count + 1 + set_index;
                            add_weighted_permutation_term(
                                &mut candidate,
                                y,
                                total_terms,
                                copy_term_index,
                                |row| term[row],
                            )
                            .expect("add weighted copy term");
                            logical_terms += 1;
                            split_contributions += 1;
                            delta_power *= Fr::DELTA;
                            drop(current_z.take());

                            if set_index + 1 < set_count {
                                current_z = Some(
                                    domain
                                        .coeff_to_extended_part_spilled(
                                            &z_coefficients[set_index + 1],
                                            extended_omega_factor,
                                        )
                                        .expect("load next streamed Z coset"),
                                );
                                streamed_z_loads += 1;
                            }
                        }
                        assert!(current_z.is_none());
                        assert_eq!(logical_terms, total_terms);
                        assert_eq!(split_contributions, expected_split_contributions);
                    }

                    let mut legacy = initial.clone();
                    let mut legacy_z_loads = 0usize;
                    let first_z = domain
                        .coeff_to_extended_part_spilled(
                            &z_coefficients[0],
                            extended_omega_factor,
                        )
                        .expect("load legacy first Z coset");
                    legacy_z_loads += 1;
                    for row in 0..part_size {
                        let idx = row * part_stride + part_index;
                        legacy[row] = legacy[row] * y + ((Fr::ONE - first_z[row]) * l0[idx]);
                    }
                    let last_z = domain
                        .coeff_to_extended_part_spilled(
                            &z_coefficients[set_count - 1],
                            extended_omega_factor,
                        )
                        .expect("load legacy final Z coset");
                    legacy_z_loads += 1;
                    for row in 0..part_size {
                        let idx = row * part_stride + part_index;
                        let z = last_z[row];
                        legacy[row] = legacy[row] * y + ((z.square() - z) * l_last[idx]);
                    }
                    drop(first_z);
                    drop(last_z);

                    if set_count > 1 {
                        let mut previous_z = domain
                            .coeff_to_extended_part_spilled(
                                &z_coefficients[0],
                                extended_omega_factor,
                            )
                            .expect("load legacy initial transition Z");
                        legacy_z_loads += 1;
                        for set_index in 1..set_count {
                            let current_z = domain
                                .coeff_to_extended_part_spilled(
                                    &z_coefficients[set_index],
                                    extended_omega_factor,
                                )
                                .expect("load legacy current transition Z");
                            legacy_z_loads += 1;
                            for row in 0..part_size {
                                let idx = row * part_stride + part_index;
                                let r_last = get_rotation_idx(
                                    idx,
                                    last_rotation.0,
                                    rot_scale,
                                    isize,
                                );
                                legacy[row] = legacy[row] * y
                                    + ((current_z[row]
                                        - previous_z[r_last / part_stride])
                                        * l0[idx]);
                            }
                            previous_z = current_z;
                        }
                    }

                    let mut delta_power = Fr::ONE;
                    for set_index in 0..set_count {
                        let z = domain
                            .coeff_to_extended_part_spilled(
                                &z_coefficients[set_index],
                                extended_omega_factor,
                            )
                            .expect("load legacy copy Z");
                        legacy_z_loads += 1;
                        let term = copy_term(
                            &z[..],
                            &sources[set_index],
                            &sigmas[set_index],
                            &active,
                            beta,
                            gamma,
                            delta_power,
                            extended_omega,
                            domain_omega,
                            part_index,
                            part_stride,
                            rot_scale,
                            isize,
                        );
                        for row in 0..part_size {
                            legacy[row] = legacy[row] * y + term[row];
                        }
                        delta_power *= Fr::DELTA;
                    }

                    let expected_streamed_loads = if set_count == 1 {
                        3
                    } else {
                        set_count
                    };
                    assert_eq!(streamed_z_loads, expected_streamed_loads);
                    let expected_legacy_loads = if set_count == 1 {
                        3
                    } else {
                        2 * set_count + 2
                    };
                    assert_eq!(legacy_z_loads, expected_legacy_loads);
                    assert_eq!(candidate, legacy, "sets={set_count}, part={part_index}, y={y:?}");
                }
            }
        }

        assert!(saw_next_wrap, "fixture must exercise next-rotation wrap");
        assert!(saw_last_wrap, "fixture must exercise last-rotation wrap");
    }
}
