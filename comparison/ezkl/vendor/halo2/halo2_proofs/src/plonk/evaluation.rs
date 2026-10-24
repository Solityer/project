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
use crate::h_eval::{GateProgram, GateQuery, GATE_OP_ADD, GATE_OP_DOUBLE, GATE_OP_FOLD, GATE_OP_LOAD_CONST, GATE_OP_LOAD_QUERY, GATE_OP_MUL, GATE_OP_NEGATE, GATE_OP_SQUARE, GATE_OP_SUB};
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

#[derive(Default)]
struct ColumnDependencies {
    fixed: BTreeSet<usize>,
    advice: BTreeSet<usize>,
    instance: BTreeSet<usize>,
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

    fn column_count(&self) -> usize {
        self.fixed.len() + self.advice.len() + self.instance.len()
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
            *coset = domain.coeff_to_extended_part_spilled(poly, extended_omega_factor)?;
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
            *coset = domain.coeff_to_extended_part(coeff, extended_omega_factor);
            counts.transforms_loaded += 1;
        } else {
            counts.transforms_reused += 1;
        }
    }
    Ok(counts)
}

#[allow(clippy::too_many_arguments)]
fn evaluate_graph_group<C: CurveAffine>(graphs: &[GraphEvaluator<C>], fixed: &[Polynomial<C::ScalarExt, LagrangeCoeff>], advice: &[Polynomial<C::ScalarExt, LagrangeCoeff>], instance: &[Polynomial<C::ScalarExt, LagrangeCoeff>], challenges: &[C::ScalarExt], values: &mut Polynomial<C::ScalarExt, ExtendedLagrangeCoeff>, y: C::ScalarExt, beta: C::ScalarExt, gamma: C::ScalarExt, theta: C::ScalarExt, rot_scale: i32, isize: i32, part_index: usize, part_stride: usize) -> Result<GateGroupEvalTiming, super::Error> {
    #[cfg(feature = "icicle_gpu")]
    let (program_compile_ms, gpu_h_call_ms, gpu_fallback) = {
        if crate::icicle::native_prove_ntt_scope_active()
            && std::env::var_os("ENABLE_ICICLE_GPU").is_some()
        {
            crate::icicle::record_gpu_h_eval_group();
            let compile_started = Instant::now();
            let previous_value = C::ScalarExt::ZERO;
            let program = compile_gate_program(graphs, challenges, beta, gamma, theta, y, previous_value, rot_scale, isize).map_err(super::Error::NativeStorage)?;
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
                accumulated = accumulated * y + contribution;
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
fn gate_program_bounds<C: CurveAffine>(graphs: &[GraphEvaluator<C>]) -> Option<(usize, usize, usize, usize, usize)> {
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
    constants = constants.checked_add(1)?;
    let nodes = queries.checked_add(constants)?.checked_add(operations)?;
    let instructions = nodes.checked_add(graphs.len())?;
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

#[cfg(feature = "icicle_gpu")]
fn gate_program_index(index: usize, what: &str) -> io::Result<u32> {
    u32::try_from(index).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("GPU H program {what} exceeds u32")))
}

#[cfg(feature = "icicle_gpu")]
fn compile_gate_program<C: CurveAffine>(graphs: &[GraphEvaluator<C>], challenges: &[C::ScalarExt], beta: C::ScalarExt, gamma: C::ScalarExt, theta: C::ScalarExt, y: C::ScalarExt, previous_value: C::ScalarExt, rot_scale: i32, isize: i32) -> io::Result<Option<GateProgram<C::ScalarExt>>> {
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
        fn constant(&mut self, value: F) -> io::Result<usize> {
            let index = match self.constants.iter().position(|candidate| *candidate == value) {
                Some(index) => index,
                None => { let index = self.constants.len(); self.constants.push(value); index }
            };
            Ok(self.intern(GateProgramNodeKey::Constant(gate_program_index(index, "constant table")?)))
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
    let Some((node_upper, instruction_upper, query_upper, constant_upper, max_intermediates)) = gate_program_bounds(graphs) else { return Ok(None); };
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
    let mut steps = Vec::with_capacity(instruction_upper);
    let mut node_start = 0;
    for (root, node_end) in roots.iter().copied().zip(graph_node_ends.iter().copied()) {
        steps.extend((node_start..node_end).map(GateProgramStep::Compute));
        steps.push(GateProgramStep::Fold(root));
        node_start = node_end;
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
            GateProgramStep::Fold(root) => { last_use[*root] = step_index; remaining_uses[*root] += 1; }
        }
    }
    let mut node_slots = vec![0usize; compiler.nodes.len()];
    let mut free_slots = Vec::with_capacity(compiler.nodes.len());
    let mut instructions = Vec::with_capacity(steps.len());
    let mut scratch_slots = 0usize;
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
                instructions.push([GATE_OP_FOLD, gate_program_index(slot, "scratch slot")?, 0, 0, 0, 0, 0, 0]);
                remaining_uses[*root] = remaining_uses[*root].checked_sub(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H program fold use count underflow"))?;
                if remaining_uses[*root] == 0 { free_slots.push(slot); }
            }
        }
    }
    if remaining_uses.iter().any(|uses| *uses != 0) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GPU H program liveness scan left unconsumed node uses",
        ));
    }
    if scratch_slots > MAX_GPU_H_PROGRAM_SCRATCH_SLOTS { return Ok(None); }
    Ok(Some(GateProgram { instructions, constants: compiler.constants, queries: compiler.queries, scratch_slots, fold_count: roots.len() }))
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
        let max_gate_dependencies = self.custom_gates.iter().map(|graph| graph.column_dependencies().column_count()).max().unwrap_or(0);

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
                let mut gate_start = 0;
                while gate_start < self.custom_gates.len() {
                    let mut dependencies = self.custom_gates[gate_start].column_dependencies();
                    let mut gate_end = gate_start + 1;
                    while gate_end < self.custom_gates.len() {
                        let next = self.custom_gates[gate_end].column_dependencies();
                        if dependencies.column_count().saturating_add(next.additional_column_count(&dependencies)) > max_gate_dependencies {
                            break;
                        }
                        dependencies.extend(&next);
                        gate_end += 1;
                    }
                    {
                        // Evict all source classes before loading any missing
                        // columns, keeping the transient live union within the
                        // existing dependency budget.
                        evict_inactive_cosets(&mut fixed_cosets, &previous_fixed_columns, &dependencies.fixed).map_err(super::Error::NativeStorage)?;
                        evict_inactive_cosets(&mut advice_cosets, &previous_advice_columns, &dependencies.advice).map_err(super::Error::NativeStorage)?;
                        evict_inactive_cosets(&mut instance_cosets, &previous_instance_columns, &dependencies.instance).map_err(super::Error::NativeStorage)?;

                        let source_conversion_started = Instant::now();
                        let fixed_counts = load_missing_fixed_cosets(domain, &pk.fixed_values, &dependencies.fixed, extended_omega_factor, &mut fixed_cosets).map_err(super::Error::NativeStorage)?;
                        let advice_counts = load_missing_cosets(domain, advice, &dependencies.advice, extended_omega_factor, &mut advice_cosets).map_err(super::Error::NativeStorage)?;
                        let instance_counts = load_missing_cosets(domain, instance, &dependencies.instance, extended_omega_factor, &mut instance_cosets).map_err(super::Error::NativeStorage)?;
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

                        let timing = evaluate_graph_group(&self.custom_gates[gate_start..gate_end], &fixed_cosets, &advice_cosets, &instance_cosets, challenges, &mut part_values, y, beta, gamma, theta, rot_scale, isize, part_index, part_stride)?;
                        gate_eval_summary.rows += part_values.len();
                        if timing.gpu_executed { gate_eval_summary.gpu_groups += 1; gate_eval_summary.gpu_rows += part_values.len(); }
                        else { gate_eval_summary.cpu_groups += 1; gate_eval_summary.cpu_rows += part_values.len(); }
                        if timing.gpu_fallback { gate_eval_summary.gpu_fallback_groups += 1; gate_eval_summary.gpu_fallback_rows += part_values.len(); }
                        gate_eval_summary.program_compile_ms += timing.program_compile_ms;
                        gate_eval_summary.gpu_or_cpu_eval_wall_ms += timing.gpu_h_call_ms + timing.cpu_eval_ms;
                        gate_eval_summary.gpu_h_call_ms += timing.gpu_h_call_ms;
                        gate_eval_summary.cpu_eval_ms += timing.cpu_eval_ms;
                    }
                    previous_fixed_columns = dependencies.fixed;
                    previous_advice_columns = dependencies.advice;
                    previous_instance_columns = dependencies.instance;
                    gate_start = gate_end;
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

                // Permutation boundary and copy constraints.
                let sets = &permutation.sets;
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

                    if sets.len() > 1 {
                        let mut previous_z = domain
                            .coeff_to_extended_part_spilled(
                                &sets[0].permutation_product_poly,
                                extended_omega_factor,
                            )
                            .map_err(super::Error::NativeStorage)?;
                        for set in sets.iter().skip(1) {
                            let current_z = domain
                                .coeff_to_extended_part_spilled(
                                    &set.permutation_product_poly,
                                    extended_omega_factor,
                                )
                                .map_err(super::Error::NativeStorage)?;
                            parallelize(&mut part_values[..], |h, start| {
                                for (offset, value) in h.iter_mut().enumerate() {
                                    let row = start + offset;
                                    let idx = row * part_stride + part_index;
                                    let r_last =
                                        get_rotation_idx(idx, last_rotation.0, rot_scale, isize);
                                    *value = *value * y
                                        + ((current_z[row]
                                            - previous_z[r_last / part_stride])
                                            * l0[idx]);
                                }
                            });
                            previous_z = current_z;
                        }
                    }

                    // One set at a time, and one source/sigma pair at a time. The
                    // running products are mapped arrays; no all-column coset bank
                    // or all-set product coset bank is retained.
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
                                    let coset = domain
                                        .coeff_to_extended_part(coeff, extended_omega_factor);
                                    coset
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
                            let sigma_coset = domain
                                .coeff_to_extended_part(sigma_poly, extended_omega_factor);

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
                }

                #[cfg(feature = "mv-lookup")]
                for (lookup_index, lookup) in lookup_sets.iter().enumerate() {
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
                    (&mut part_values[..])
                        .par_chunks_mut(LOOKUP_ROW_CHUNK)
                        .enumerate()
                        .for_each(|(chunk_index, h)| {
                            let start = chunk_index * LOOKUP_ROW_CHUNK;
                            let mut input_data = input_graphs
                                .iter()
                                .map(GraphEvaluator::instance)
                                .collect::<Vec<_>>();
                            let mut inverse_values =
                                Vec::with_capacity(h.len() * input_graphs.len());
                            for offset in 0..h.len() {
                                let row = start + offset;
                                let idx = row * part_stride + part_index;
                                for (graph, data) in input_graphs.iter().zip(input_data.iter_mut()) {
                                    inverse_values.push(graph.evaluate_part(
                                        data,
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
                                    ));
                                }
                            }
                            inverse_values.batch_invert();

                            let mut input_data = input_graphs
                                .iter()
                                .map(GraphEvaluator::instance)
                                .collect::<Vec<_>>();
                            let mut table_data = table_graph.instance();
                            for (offset, h_value) in h.iter_mut().enumerate() {
                                let row = start + offset;
                                let idx = row * part_stride + part_index;
                                let mut inputs_product = C::ScalarExt::ONE;
                                for (graph, data) in input_graphs.iter().zip(input_data.iter_mut()) {
                                    inputs_product *= graph.evaluate_part(
                                        data,
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
                                }
                                let inverse_sum = inverse_values[offset * input_graphs.len()
                                    ..(offset + 1) * input_graphs.len()]
                                    .iter()
                                    .fold(C::ScalarExt::ZERO, |sum, inverse| sum + inverse);
                                let table_value = table_graph.evaluate_part(
                                    &mut table_data,
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
                                let lhs = table_value
                                    * inputs_product
                                    * (phi_coset[r_next / part_stride] - phi_coset[row]);
                                let rhs = inputs_product * (table_value * inverse_sum - m_coset[row]);
                                *h_value = *h_value * y + l0[idx] * phi_coset[row];
                                *h_value = *h_value * y + l_last[idx] * phi_coset[row];
                                *h_value = *h_value * y + (lhs - rhs) * l_active_row[idx];
                            }
                        });
                    drop(m_coset);
                    drop(phi_coset);
                    drop(instance_cosets);
                    drop(advice_cosets);
                    drop(fixed);
                }

                #[cfg(not(feature = "mv-lookup"))]
                for (lookup_index, lookup) in lookup_sets.iter().enumerate() {
                    let graph = &self.lookups[lookup_index];
                    let dependencies = graph.column_dependencies();
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
                    }
                }

                let mut legacy = initial;
                for row in 0..part_size {
                    let idx = row * part_stride + part_index;
                    let mut accumulator = legacy[row];
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
                    }
                    legacy[row] = accumulator;
                }
                assert_eq!(compiled, legacy, "part {part_index}, instance {proof_instance}");
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

        let gates = vec![
            cache_fixture_gate(0, false),
            cache_fixture_gate(1, true),
            cache_fixture_gate(1, true),
        ];
        let max_gate_dependencies = gates
            .iter()
            .map(|graph| graph.column_dependencies().column_count())
            .max()
            .expect("fixture has gates");
        assert_eq!(max_gate_dependencies, 4);

        let mut group_ranges = Vec::new();
        let mut gate_start = 0;
        while gate_start < gates.len() {
            let mut dependencies = gates[gate_start].column_dependencies();
            let mut gate_end = gate_start + 1;
            while gate_end < gates.len() {
                let next = gates[gate_end].column_dependencies();
                if dependencies.column_count().saturating_add(next.additional_column_count(&dependencies))
                    > max_gate_dependencies
                {
                    break;
                }
                dependencies.extend(&next);
                gate_end += 1;
            }
            group_ranges.push((gate_start, gate_end));
            gate_start = gate_end;
        }
        assert_eq!(group_ranges, vec![(0, 1), (1, 3)]);
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

                for &(start, end) in &group_ranges {
                    let mut dependencies = gates[start].column_dependencies();
                    for graph in &gates[start + 1..end] {
                        dependencies.extend(&graph.column_dependencies());
                    }

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

                    let fixed_counts = load_missing_fixed_cosets(
                        &domain,
                        &fixed_sources,
                        &dependencies.fixed,
                        extended_omega_factor,
                        &mut fixed_cosets,
                    )
                    .expect("load missing fixed cosets");
                    let advice_counts = load_missing_cosets(
                        &domain,
                        &advice,
                        &dependencies.advice,
                        extended_omega_factor,
                        &mut advice_cosets,
                    )
                    .expect("load missing advice cosets");
                    let instance_counts = load_missing_cosets(
                        &domain,
                        &instance,
                        &dependencies.instance,
                        extended_omega_factor,
                        &mut instance_cosets,
                    )
                    .expect("load missing instance cosets");

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

                    let legacy_fixed = load_fixed_cosets(
                        &domain,
                        &fixed_sources,
                        &dependencies.fixed,
                        extended_omega_factor,
                    )
                    .expect("legacy fixed conversion");
                    let legacy_advice = load_cosets(
                        &domain,
                        &advice,
                        &dependencies.advice,
                        extended_omega_factor,
                    )
                    .expect("legacy advice conversion");
                    let legacy_instance = load_cosets(
                        &domain,
                        &instance,
                        &dependencies.instance,
                        extended_omega_factor,
                    )
                    .expect("legacy instance conversion");
                    for &column in &dependencies.fixed {
                        assert_eq!(&fixed_cosets[column][..], &legacy_fixed[column][..]);
                    }
                    for &column in &dependencies.advice {
                        assert_eq!(&advice_cosets[column][..], &legacy_advice[column][..]);
                    }
                    for &column in &dependencies.instance {
                        assert_eq!(&instance_cosets[column][..], &legacy_instance[column][..]);
                    }

                    evaluate_graph_group(
                        &gates[start..end],
                        &fixed_cosets,
                        &advice_cosets,
                        &instance_cosets,
                        &[],
                        &mut fused_values,
                        y,
                        beta,
                        Fr::ZERO,
                        Fr::ZERO,
                        rot_scale,
                        isize,
                        part_index,
                        part_stride,
                    ).expect("evaluate native gate group");
                    for graph in &gates[start..end] {
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

                    instance_fixed_loaded += fixed_counts.transforms_loaded;
                    instance_fixed_reused += fixed_counts.transforms_reused;
                    instance_constant_loaded += fixed_counts.constant_fills_loaded;
                    instance_constant_reused += fixed_counts.constant_fills_reused;
                    instance_advice_loaded += advice_counts.transforms_loaded;
                    instance_advice_reused += advice_counts.transforms_reused;
                    instance_columns_loaded += instance_counts.transforms_loaded;
                    total_groups += 1;
                    previous_fixed_columns = dependencies.fixed;
                    previous_advice_columns = dependencies.advice;
                    previous_instance_columns = dependencies.instance;
                }

                // Production drops these caches before entering the next
                // permutation/lookup/shuffle phase.
                drop(fixed_cosets);
                drop(advice_cosets);
                drop(instance_cosets);

                assert_eq!(instance_fixed_loaded, 1);
                assert_eq!(instance_fixed_reused, 1);
                assert_eq!(instance_constant_loaded, 1);
                assert_eq!(instance_constant_reused, 1);
                assert_eq!(instance_advice_loaded, 1);
                assert_eq!(instance_advice_reused, 1);
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

        // Two groups per part/proof instance prove that each cache is reset at
        // both lifecycle boundaries while cross-group columns are reused.
        assert_eq!(total_groups, 8);
        assert_eq!(total_advice_loaded, 4);
        assert_eq!(total_advice_reused, 4);
        assert_eq!(total_instance_loaded, 8);
        assert_eq!(total_fixed_loaded, 4);
        assert_eq!(total_fixed_reused, 4);
        assert_eq!(total_constant_loaded, 4);
        assert_eq!(total_constant_reused, 4);
    }
}
