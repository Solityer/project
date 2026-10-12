#![allow(clippy::int_plus_one)]

use std::ops::Range;

use ff::{Field, FromUniformBytes};
use group::Curve;

use super::{
    circuit::{
        Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner, Instance,
        PackedSelectorAssignment, Selector, SelectorActivationSet, SelectorBitmap,
    },
    evaluation::Evaluator,
    permutation, Assigned, Challenge, Error, LagrangeCoeff, Polynomial, ProvingKey, VerifyingKey,
};
use crate::{
    arithmetic::{parallelize, CurveAffine},
    circuit::Value,
    poly::{
        commitment::{Blind, Params},
        EvaluationDomain,
    },
};

pub(crate) fn create_domain<C, ConcreteCircuit>(
    k: u32,
    #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
) -> (
    EvaluationDomain<C::Scalar>,
    ConstraintSystem<C::Scalar>,
    ConcreteCircuit::Config,
)
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, params);
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);

    #[cfg(feature = "mv-lookup")]
    let cs = cs.chunk_lookups();

    let degree = cs.degree();

    log::debug!("Creating domain with degree {}", degree);

    let domain = EvaluationDomain::new(degree as u32, k);

    (domain, cs, config)
}

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<F: Field> {
    k: u32,
    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<SelectorBitmap>,
    // A range of available rows for assignment and copies.
    usable_rows: Range<usize>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> Assignment<F> for Assembly<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about regions in this context.
    }

    fn exit_region(&mut self) {
        // Do nothing; we don't care about regions in this context.
    }

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.selectors
            .get_mut(selector.0)
            .ok_or(Error::BoundsFailure)?
            .set_value(row, 1)
            .map_err(Error::NativeStorage)?;

        Ok(())
    }

    fn query_instance(&self, _: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        // There is no instance in this context.
        Ok(Value::unknown())
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // We only care about fixed columns here
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = to().into_field().assign()?;

        Ok(())
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        self.permutation
            .copy(left_column, left_row, right_column, right_row)
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }

        let col = self
            .fixed
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;

        let filler = to.assign()?;
        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = filler;
        }

        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn annotate_column<A, AR>(&mut self, _annotation: A, _column: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Do nothing
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self, _: Option<String>) {
        // Do nothing; we don't care about namespaces in this context.
    }
}

/// Exact selector and conservative compact-permutation census produced by a
/// shape-only synthesis. Copy calls are counted so the serialized sparse
/// successor mapping can use the source-derived `2 * copy_calls` bound rather
/// than the much looser full `permutation_columns * domain_size` bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeShapeCensus {
    /// Number of fixed columns emitted by native selector compression.
    pub selector_fixed_column_count: usize,
    /// High-water bytes used by activation row-set backing during this census.
    /// Includes the outer selector-set vector and transient old/new storage
    /// while a selector changes from runs to a packed bitmap; excludes allocator
    /// metadata and the separately tracked counters.
    pub selector_activation_storage_bytes: u64,
    /// Number of copy callbacks made by the real circuit floor planner.
    pub permutation_copy_call_count: u64,
    /// Exact enabled-row count for every virtual selector, in selector-index order.
    pub selector_activation_counts: Vec<u64>,
    /// Upper bounds on distinct fixed values assigned or filled per fixed column.
    pub fixed_value_assignment_counts: Vec<u64>,
}

/// Run shape-only synthesis without building fixed/permutation polynomials or
/// a proving/verifying key. This executes the circuit's normal floor planner
/// and records exact selector activations and copy-callback count.
pub fn census_native_shape<C, ConcreteCircuit>(
    k: u32,
    circuit: &ConcreteCircuit,
) -> Result<NativeShapeCensus, Error>
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let (_, cs, config) = create_domain::<C, ConcreteCircuit>(
        k,
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );
    let n = 1usize.checked_shl(k).ok_or(Error::BoundsFailure)?;
    if n < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(k));
    }
    let usable_end = n
        .checked_sub(cs.blinding_factors() + 1)
        .ok_or_else(|| Error::not_enough_rows_available(k))?;
    let mut selectors = Vec::new();
    selectors
        .try_reserve_exact(cs.num_selectors)
        .map_err(|error| Error::NativeStorage(std::io::Error::new(std::io::ErrorKind::OutOfMemory, error)))?;
    selectors.extend((0..cs.num_selectors).map(|_| SelectorActivationSet::zeroed(n)));
    let selector_activation_container_bytes = selectors
        .capacity()
        .checked_mul(std::mem::size_of::<SelectorActivationSet>())
        .ok_or(Error::BoundsFailure)?;
    let selector_activation_initial_bytes = selectors.iter().try_fold(
        selector_activation_container_bytes,
        |total, selector| total.checked_add(selector.storage_bytes()).ok_or(Error::BoundsFailure),
    )?;
    let mut assembly = SelectorOnlyAssembly {
        k,
        fixed_columns: cs.num_fixed_columns,
        permutation_columns: cs.permutation.columns.clone(),
        selectors,
        selector_activation_storage_current_bytes: selector_activation_initial_bytes,
        selector_activation_storage_peak_bytes: selector_activation_initial_bytes,
        selector_activation_counts: vec![0; cs.num_selectors],
        fixed_value_assignment_counts: vec![0; cs.num_fixed_columns],
        usable_rows: 0..usable_end,
        permutation_copy_call_count: 0,
        _marker: std::marker::PhantomData,
    };

    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    let selector_fixed_column_count =
        cs.count_selector_activation_combinations(&assembly.selectors);
    Ok(NativeShapeCensus {
        selector_fixed_column_count,
        selector_activation_storage_bytes: u64::try_from(
            assembly.selector_activation_storage_peak_bytes,
        )
        .map_err(|_| Error::BoundsFailure)?,
        permutation_copy_call_count: assembly.permutation_copy_call_count,
        selector_activation_counts: assembly.selector_activation_counts,
        fixed_value_assignment_counts: assembly.fixed_value_assignment_counts,
    })
}

/// Count only the fixed columns produced by selector compression.
pub fn count_compressed_selector_columns<C, ConcreteCircuit>(
    k: u32,
    circuit: &ConcreteCircuit,
) -> Result<usize, Error>
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    census_native_shape::<C, ConcreteCircuit>(k, circuit)
        .map(|census| census.selector_fixed_column_count)
}

/// Assignment backend used only to measure the actual compressed selector
/// count. It mirrors keygen's callback, selector, fixed-row, and permutation
/// bounds checks while avoiding the large polynomial and copy-cycle stores.
#[derive(Debug)]
struct SelectorOnlyAssembly<F: Field> {
    k: u32,
    fixed_columns: usize,
    permutation_columns: Vec<Column<Any>>,
    selectors: Vec<SelectorActivationSet>,
    selector_activation_storage_current_bytes: usize,
    selector_activation_storage_peak_bytes: usize,
    selector_activation_counts: Vec<u64>,
    fixed_value_assignment_counts: Vec<u64>,
    usable_rows: Range<usize>,
    permutation_copy_call_count: u64,
    _marker: std::marker::PhantomData<F>,
}

impl<F: Field> Assignment<F> for SelectorOnlyAssembly<F> {
    fn enter_region<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn exit_region(&mut self) {}

    fn enable_selector<A, AR>(&mut self, _: A, selector: &Selector, row: usize) -> Result<(), Error>
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }
        let selector_index = selector.0;
        let activations = self
            .selectors
            .get_mut(selector_index)
            .ok_or(Error::BoundsFailure)?;
        let old_storage_bytes = activations.storage_bytes();
        let inserted = activations.insert(row).map_err(Error::NativeStorage)?;
        let base_bytes = self
            .selector_activation_storage_current_bytes
            .checked_sub(old_storage_bytes)
            .ok_or(Error::BoundsFailure)?;
        self.selector_activation_storage_current_bytes = base_bytes
            .checked_add(inserted.storage_bytes)
            .ok_or(Error::BoundsFailure)?;
        let insertion_peak_bytes = base_bytes
            .checked_add(inserted.peak_storage_bytes)
            .ok_or(Error::BoundsFailure)?;
        self.selector_activation_storage_peak_bytes = self
            .selector_activation_storage_peak_bytes
            .max(insertion_peak_bytes)
            .max(self.selector_activation_storage_current_bytes);
        if inserted.activated {
            let count = self
                .selector_activation_counts
                .get_mut(selector_index)
                .ok_or(Error::BoundsFailure)?;
            *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
        }
        Ok(())
    }

    fn query_instance(&self, _: Column<Instance>, row: usize) -> Result<Value<F>, Error> {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }
        Ok(Value::unknown())
    }

    fn assign_advice<V, VR, A, AR>(
        &mut self,
        _: A,
        _: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        // Keygen does not evaluate advice callbacks during key generation.
        Ok(())
    }

    fn assign_fixed<V, VR, A, AR>(
        &mut self,
        _: A,
        column: Column<Fixed>,
        row: usize,
        to: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        if !self.usable_rows.contains(&row) {
            return Err(Error::not_enough_rows_available(self.k));
        }
        if column.index() >= self.fixed_columns {
            return Err(Error::BoundsFailure);
        }
        let _ = to().into_field().assign()?;
        let count = self
            .fixed_value_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
        Ok(())
    }

    fn copy(
        &mut self,
        left_column: Column<Any>,
        left_row: usize,
        right_column: Column<Any>,
        right_row: usize,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&left_row) || !self.usable_rows.contains(&right_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }
        if !self.permutation_columns.contains(&left_column) {
            return Err(Error::ColumnNotInPermutation(left_column));
        }
        if !self.permutation_columns.contains(&right_column) {
            return Err(Error::ColumnNotInPermutation(right_column));
        }
        self.permutation_copy_call_count = self
            .permutation_copy_call_count
            .checked_add(1)
            .ok_or(Error::BoundsFailure)?;
        Ok(())
    }

    fn fill_from_row(
        &mut self,
        column: Column<Fixed>,
        from_row: usize,
        to: Value<Assigned<F>>,
    ) -> Result<(), Error> {
        if !self.usable_rows.contains(&from_row) {
            return Err(Error::not_enough_rows_available(self.k));
        }
        if column.index() >= self.fixed_columns {
            return Err(Error::BoundsFailure);
        }
        let _ = to.assign()?;
        let count = self
            .fixed_value_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
        Ok(())
    }

    fn get_challenge(&self, _: Challenge) -> Value<F> {
        Value::unknown()
    }

    fn annotate_column<A, AR>(&mut self, _: A, _: Column<Any>)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
    }

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
    }

    fn pop_namespace(&mut self, _: Option<String>) {}
}

fn selector_combination_bitmap<F: Field>(
    selectors: &[SelectorBitmap],
    assignments: &[PackedSelectorAssignment<F>],
    combination_index: usize,
) -> Result<(SelectorBitmap, u64), Error> {
    let max_root = assignments
        .iter()
        .filter(|assignment| assignment.combination_index == combination_index)
        .map(|assignment| assignment.root_value)
        .max()
        .ok_or_else(|| {
            Error::NativeStorage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "compressed selector column has no selector assignments",
            ))
        })?;
    let bits_per_value = u64::BITS - max_root.leading_zeros();
    let row_count = selectors.first().map(SelectorBitmap::len).unwrap_or(0);
    let mut encoded =
        SelectorBitmap::zeroed(row_count, bits_per_value).map_err(Error::NativeStorage)?;
    for assignment in assignments
        .iter()
        .filter(|assignment| assignment.combination_index == combination_index)
    {
        let activation = selectors
            .get(assignment.selector)
            .ok_or(Error::BoundsFailure)?;
        if activation.len() != row_count {
            return Err(Error::NativeStorage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "selector activation columns have inconsistent domains",
            )));
        }
        let mut storage_error = None;
        activation.for_each_set_bit(|row| {
            if storage_error.is_none() {
                storage_error = encoded.set_value(row, assignment.root_value).err();
            }
        });
        if let Some(error) = storage_error {
            return Err(Error::NativeStorage(error));
        }
    }
    Ok((encoded, max_root))
}

fn selector_bitmap_to_field<F: Field>(encoded: &SelectorBitmap, max_root: u64) -> Vec<F> {
    let max_root = usize::try_from(max_root).expect("selector root fits host index");
    let mut roots = Vec::with_capacity(max_root + 1);
    roots.push(F::ZERO);
    for index in 1..=max_root {
        roots.push(roots[index - 1] + F::ONE);
    }
    (0..encoded.len())
        .map(|row| {
            roots[usize::try_from(encoded.value(row)).expect("selector root fits host index")]
        })
        .collect()
}

/// Generate a `VerifyingKey` from an instance of `Circuit`.
/// By default, selector compression is turned **off**.
pub fn keygen_vk<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
) -> Result<VerifyingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    keygen_vk_custom(params, circuit, true)
}

/// Generate a `VerifyingKey` from an instance of `Circuit`.
///
/// The selector compression optimization is turned on only if `compress_selectors` is `true`.
pub fn keygen_vk_custom<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
    compress_selectors: bool,
) -> Result<VerifyingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    let (domain, cs, config) = create_domain::<C, ConcreteCircuit>(
        params.k(),
        #[cfg(feature = "circuit-params")]
        circuit.params(),
    );

    if (params.n() as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k()));
    }

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed: (0..cs.num_fixed_columns)
            .map(|_| {
                Polynomial::mapped_filled(params.n() as usize, Assigned::Zero)
                    .map_err(Error::NativeStorage)
            })
            .collect::<Result<Vec<_>, _>>()?,
        permutation: permutation::keygen::Assembly::new(params.n() as usize, &cs.permutation),
        selectors: (0..cs.num_selectors)
            .map(|_| SelectorBitmap::zeroed(params.n() as usize, 1).map_err(Error::NativeStorage))
            .collect::<Result<Vec<_>, _>>()?,
        usable_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    let (cs, selector_column_count, selector_assignments) = if compress_selectors {
        cs.compress_selector_bitmaps(&assembly.selectors)
    } else {
        cs.directly_convert_selector_bitmaps(&assembly.selectors)
    };
    let permutation_vk = assembly
        .permutation
        .build_vk(params, &domain, &cs.permutation);

    let mut fixed_commitments = Vec::with_capacity(cs.num_fixed_columns + selector_column_count);
    for assigned in assembly.fixed {
        let fixed = assigned.invert_spilled().map_err(Error::NativeStorage)?;
        fixed_commitments.push(params.commit_lagrange(&fixed, Blind::default()).to_affine());
    }
    for combination_index in 0..selector_column_count {
        let (encoded, max_root) = selector_combination_bitmap(
            &assembly.selectors,
            &selector_assignments,
            combination_index,
        )?;
        let poly = selector_bitmap_to_field::<C::Scalar>(&encoded, max_root);
        let fixed = domain.lagrange_from_vec(poly);
        fixed_commitments.push(params.commit_lagrange(&fixed, Blind::default()).to_affine());
    }

    Ok(VerifyingKey::from_parts(
        domain,
        fixed_commitments,
        permutation_vk,
        cs,
        if compress_selectors {
            assembly.selectors
        } else {
            Vec::new()
        },
        compress_selectors,
    ))
}

/// Generate a `ProvingKey` from a `VerifyingKey` and an instance of `Circuit`.
pub fn keygen_pk<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: VerifyingKey<C>,
    circuit: &ConcreteCircuit,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, circuit.params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);
    #[cfg(feature = "mv-lookup")]
    let cs = cs.chunk_lookups();

    if (params.n() as usize) < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k()));
    }

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed: (0..cs.num_fixed_columns)
            .map(|_| {
                Polynomial::mapped_filled(params.n() as usize, Assigned::Zero)
                    .map_err(Error::NativeStorage)
            })
            .collect::<Result<Vec<_>, _>>()?,
        permutation: permutation::keygen::Assembly::new(params.n() as usize, &cs.permutation),
        selectors: (0..cs.num_selectors)
            .map(|_| SelectorBitmap::zeroed(params.n() as usize, 1).map_err(Error::NativeStorage))
            .collect::<Result<Vec<_>, _>>()?,
        usable_rows: 0..params.n() as usize - (cs.blinding_factors() + 1),
        _marker: std::marker::PhantomData,
    };

    // Synthesize the circuit to obtain URS
    ConcreteCircuit::FloorPlanner::synthesize(
        &mut assembly,
        circuit,
        config,
        cs.constants.clone(),
    )?;

    let selectors = std::mem::take(&mut assembly.selectors);
    let (cs, selector_column_count, selector_assignments) = if vk.compress_selectors {
        cs.compress_selector_bitmaps(&selectors)
    } else {
        cs.directly_convert_selector_bitmaps(&selectors)
    };
    let mut fixed = Vec::with_capacity(cs.num_fixed_columns + selector_column_count);
    for assigned in assembly.fixed {
        fixed.push(assigned.invert_spilled().map_err(Error::NativeStorage)?);
    }
    for combination_index in 0..selector_column_count {
        let (encoded, max_root) =
            selector_combination_bitmap(&selectors, &selector_assignments, combination_index)?;
        let poly = selector_bitmap_to_field::<C::Scalar>(&encoded, max_root);
        fixed.push(
            vk.domain
                .lagrange_from_vec(poly)
                .spill()
                .map_err(Error::NativeStorage)?,
        );
    }

    let fixed_polys: Vec<_> = fixed
        .iter()
        .map(|poly| {
            vk.domain
                .lagrange_to_coeff(poly.clone())
                .spill()
                .map_err(Error::NativeStorage)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let permutation_pk = assembly
        .permutation
        .build_pk(params, &vk.domain, &cs.permutation)?;

    // Compute l_0(X)
    // TODO: this can be done more efficiently
    let mut l0 = vk.domain.empty_lagrange();
    l0[0] = C::Scalar::ONE;
    let l0 = vk.domain.lagrange_to_coeff(l0);
    let l0 = vk
        .domain
        .coeff_to_extended_spilled(&l0)
        .map_err(Error::NativeStorage)?;

    // Compute l_blind(X) which evaluates to 1 for each blinding factor row
    // and 0 otherwise over the domain.
    let mut l_blind = vk.domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(cs.blinding_factors()) {
        *evaluation = C::Scalar::ONE;
    }
    let l_blind = vk.domain.lagrange_to_coeff(l_blind);
    let l_blind = vk
        .domain
        .coeff_to_extended_spilled(&l_blind)
        .map_err(Error::NativeStorage)?;

    // Compute l_last(X) which evaluates to 1 on the first inactive row (just
    // before the blinding factors) and 0 otherwise over the domain
    let mut l_last = vk.domain.empty_lagrange();
    l_last[params.n() as usize - cs.blinding_factors() - 1] = C::Scalar::ONE;
    let l_last = vk.domain.lagrange_to_coeff(l_last);
    let l_last = vk
        .domain
        .coeff_to_extended_spilled(&l_last)
        .map_err(Error::NativeStorage)?;

    // Compute l_active_row(X)
    let one = C::Scalar::ONE;
    let mut l_active_row = Polynomial::mapped_filled(vk.domain.extended_len(), C::Scalar::ZERO)
        .map_err(Error::NativeStorage)?;
    parallelize(&mut l_active_row[..], |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last[idx] + l_blind[idx]);
        }
    });

    // Compute the optimized evaluation data structure
    let ev = Evaluator::new(&vk.cs);

    Ok(ProvingKey {
        vk,
        l0,
        l_last,
        l_active_row,
        fixed_values: fixed,
        fixed_polys,
        permutation: permutation_pk,
        ev,
    })
}
