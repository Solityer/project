#![allow(clippy::int_plus_one)]

use std::io;
use std::ops::Range;

use ff::{Field, FromUniformBytes, PrimeField};
use group::Curve;

use super::{
    circuit::{
        Advice, Any, Assignment, Circuit, Column, ConstraintSystem, Fixed, FloorPlanner, Instance,
        PackedSelectorAssignment, Selector, SelectorActivationCache, SelectorActivationSet,
        SelectorBitmap,
    },
    evaluation::Evaluator,
    permutation, Assigned, Challenge, Error, FixedColumnSource, LagrangeCoeff, PackedIdTape,
    PackedIdWidth, Polynomial, ProvingKey, VerifyingKey,
};
use crate::{
    arithmetic::{parallelize, CurveAffine},
    circuit::Value,
    poly::{
        commitment::{Blind, Params},
        EvaluationDomain, NativePermutationStore,
    },
};

/// Rebuild the deterministic boundary polynomials stored in a proving key.
/// Keep this shared by key generation and PK restoration so the serialized
/// artifact can omit these domain-derived values without changing the native
/// prover inputs.
pub(super) fn compute_prover_boundary_polynomials<C: CurveAffine>(
    domain: &EvaluationDomain<C::Scalar>,
    blinding_factors: usize,
) -> io::Result<(
    Polynomial<C::Scalar, crate::poly::ExtendedLagrangeCoeff>,
    Polynomial<C::Scalar, crate::poly::ExtendedLagrangeCoeff>,
    Polynomial<C::Scalar, crate::poly::ExtendedLagrangeCoeff>,
)> {
    let n = usize::try_from(domain.get_n()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "native proving-key domain exceeds host index")
    })?;
    let last_active_row = blinding_factors
        .checked_add(1)
        .and_then(|reserved| n.checked_sub(reserved))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "native proving-key boundary row underflows"))?;

    // Compute l_0(X).
    let mut l0 = domain.empty_lagrange();
    l0[0] = C::Scalar::ONE;
    let l0 = domain.lagrange_to_coeff(l0);
    let l0 = domain.coeff_to_extended_spilled(&l0)?;

    // Compute l_blind(X), then l_last(X), preserving the original keygen order.
    let mut l_blind = domain.empty_lagrange();
    for evaluation in l_blind[..].iter_mut().rev().take(blinding_factors) {
        *evaluation = C::Scalar::ONE;
    }
    let l_blind = domain.lagrange_to_coeff(l_blind);
    let l_blind = domain.coeff_to_extended_spilled(&l_blind)?;

    let mut l_last = domain.empty_lagrange();
    l_last[last_active_row] = C::Scalar::ONE;
    let l_last = domain.lagrange_to_coeff(l_last);
    let l_last = domain.coeff_to_extended_spilled(&l_last)?;

    // Compute l_active_row(X) = 1 - l_last(X) - l_blind(X).
    let one = C::Scalar::ONE;
    let mut l_active_row = Polynomial::mapped_filled(domain.extended_len(), C::Scalar::ZERO)?;
    parallelize(&mut l_active_row[..], |values, start| {
        for (i, value) in values.iter_mut().enumerate() {
            let idx = i + start;
            *value = one - (l_last[idx] + l_blind[idx]);
        }
    });

    Ok((l0, l_last, l_active_row))
}

/// Source-derived geometry and allocation ceilings for one native keygen
/// synthesis. Byte counts are explicit backing/virtual-size estimates; they
/// are not presented as resident-memory measurements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeKeygenPreflight {
    /// Configured base-two logarithm of the evaluation domain size.
    pub logrows: u32,
    /// Number of rows in the evaluation domain.
    pub domain_size: u64,
    /// Configured base-two logarithm of the extended quotient domain.
    pub extended_logrows: u32,
    /// Maximum degree of the configured constraint system.
    pub constraint_degree: u32,
    /// Number of advice columns in the constraint system.
    pub advice_columns: usize,
    /// Number of fixed columns in the constraint system.
    pub fixed_columns: usize,
    /// Number of instance columns in the constraint system.
    pub instance_columns: usize,
    /// Number of columns in the permutation argument.
    pub permutation_columns: usize,
    /// Number of selectors before selector compression.
    pub selector_count: usize,
    /// Number of lookup arguments in the constraint system.
    pub lookup_count: usize,
    /// Number of shuffle arguments in the constraint system.
    pub shuffle_count: usize,
    /// Number of rows reserved for blinding and the final unusable row.
    pub blinding_factors: usize,
    /// Virtual backing bytes for fixed-column Lagrange values.
    pub fixed_lagrange_backing_bytes: u64,
    /// Dense bitmap byte ceiling for all selector activation sets.
    pub selector_dense_bitmap_ceiling_bytes: u64,
    /// Container and index bytes for selector activation storage.
    pub selector_activation_container_bytes: u64,
    /// Initial backing bytes for selector activation storage.
    pub selector_activation_initial_backing_bytes: u64,
    /// Bytes required to decode one selector into field values.
    pub one_selector_decode_bytes: u64,
    /// Encoded bytes for one selector activation set.
    pub one_selector_encoded_bytes: u64,
    /// Logical permutation cells across all permutation columns.
    pub permutation_logical_cells: u64,
    /// Virtual bytes in the sparse permutation data file.
    pub permutation_data_virtual_bytes: u64,
    /// Virtual bytes in the sparse permutation directory file.
    pub permutation_directory_virtual_bytes: u64,
    /// RAM bytes for sparse permutation allocation bitmaps.
    pub permutation_allocation_bitmap_bytes: u64,
    /// Bounded RAM cache bytes for sparse permutation state.
    pub permutation_bounded_cache_bytes: u64,
}

/// Exact counters captured from the same floor-planner synthesis that feeds
/// native key generation. The callback runs after selector grouping and before
/// selector field decoding, permutation VK construction, or FFT work.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeKeygenCensus {
    /// Pre-synthesis geometry and allocation estimates for this keygen pass.
    pub preflight: NativeKeygenPreflight,
    /// Number of fixed selector columns after compression.
    pub compressed_selector_columns: usize,
    /// Current bytes in selector activation storage after synthesis.
    pub selector_activation_storage_current_bytes: u64,
    /// Peak bytes observed in selector activation storage during synthesis.
    pub selector_activation_storage_peak_bytes: u64,
    /// Activation counts for each original selector.
    pub selector_activation_counts: Vec<u64>,
    /// Fixed assignment callback counts for each fixed column.
    pub fixed_assignment_callback_counts: Vec<u64>,
    /// Upper bounds on populated fixed rows for each fixed column. A
    /// `fill_from_row` callback contributes every row it fills.
    pub fixed_value_assignment_counts: Vec<u64>,
    /// Whether synthesis explicitly assigned a canonical zero in each fixed
    /// column.
    pub fixed_value_explicit_zero_seen: Vec<bool>,
    /// Advice assignment callback counts for each advice column.
    pub advice_assignment_counts: Vec<u64>,
    /// Number of copy calls recorded by the permutation argument.
    pub permutation_copy_calls: u64,
    /// Number of non-identity successor edges in the permutation argument.
    pub permutation_successor_edges: u64,
    /// Number of non-default sparse permutation records.
    pub permutation_nondefault_records: u64,
    /// Number of allocated permutation data pages.
    pub permutation_data_pages_allocated: u64,
    /// Number of allocated permutation directory pages.
    pub permutation_directory_pages_allocated: u64,
    /// Allocated bytes in permutation data pages.
    pub permutation_data_allocated_bytes: u64,
    /// Allocated bytes in permutation directory pages.
    pub permutation_directory_allocated_bytes: u64,
    /// RAM cache bytes used by sparse permutation storage.
    pub permutation_cache_bytes: u64,
    /// RAM bytes used by sparse permutation allocation bitmaps.
    pub permutation_allocation_bitmap_bytes: u64,
}

/// Dynamic selector-set memory allowance returned by the formal resource gate
/// after it accounts for the shared host/GPU physical pool and other live
/// setup allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeKeygenBudget {
    /// Maximum live bytes allowed for selector activation storage.
    pub max_selector_activation_bytes: u64,
    /// Maximum allocated backing bytes allowed for permutation state.
    pub max_permutation_backing_bytes: u64,
}

fn native_storage_error(message: &'static str) -> Error {
    Error::NativeStorage(io::Error::new(io::ErrorKind::InvalidInput, message))
}

/// Validate the quotient-domain requirement without calling
/// `EvaluationDomain::new`, whose field-capacity check is an assertion.
fn checked_extended_logrows<F: PrimeField>(k: u32, degree: usize) -> Result<(u64, u32), Error> {
    if k > F::S {
        return Err(native_storage_error(
            "base domain exceeds the scalar field two-adicity",
        ));
    }
    let quotient_degree = degree
        .checked_sub(1)
        .ok_or_else(|| native_storage_error("constraint-system degree is zero"))?;
    let quotient_degree = u64::try_from(quotient_degree)
        .map_err(|_| native_storage_error("quotient degree does not fit u64"))?;
    let quotient_bits = if quotient_degree <= 1 {
        0
    } else {
        let value = quotient_degree - 1;
        u64::BITS - value.leading_zeros()
    };
    let extended_k = k
        .checked_add(quotient_bits)
        .ok_or_else(|| native_storage_error("extended domain logrows overflow"))?;
    if extended_k > F::S {
        return Err(native_storage_error(
            "quotient domain exceeds the scalar field two-adicity",
        ));
    }
    let n = 1u64
        .checked_shl(k)
        .ok_or_else(|| native_storage_error("base domain size does not fit u64"))?;
    Ok((n, extended_k))
}

fn bytes_product(a: usize, b: usize, c: usize) -> Result<u64, Error> {
    let bytes = a
        .checked_mul(b)
        .and_then(|value| value.checked_mul(c))
        .ok_or(Error::BoundsFailure)?;
    u64::try_from(bytes).map_err(|_| Error::BoundsFailure)
}

fn native_keygen_preflight<F: Field + PrimeField>(
    k: u32,
    cs: &ConstraintSystem<F>,
) -> Result<NativeKeygenPreflight, Error> {
    let (domain_size, extended_logrows) = checked_extended_logrows::<F>(k, cs.degree())?;
    let domain_rows = usize::try_from(domain_size).map_err(|_| Error::BoundsFailure)?;
    let constraint_degree = u32::try_from(cs.degree()).map_err(|_| Error::BoundsFailure)?;
    let selector_bytes_per_column = domain_rows
        .checked_add(7)
        .ok_or(Error::BoundsFailure)?
        / 8;
    let selector_dense_bitmap_ceiling_bytes = bytes_product(
        cs.num_selectors,
        selector_bytes_per_column,
        1,
    )?;
    let selector_activation_container_bytes = cs
        .num_selectors
        .checked_mul(std::mem::size_of::<SelectorActivationSet>())
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(Error::BoundsFailure)?;
    let selector_root_bits = u64::BITS - (cs.degree() as u64).leading_zeros();
    let selector_encoded_bits = domain_size
        .checked_mul(selector_root_bits as u64)
        .ok_or(Error::BoundsFailure)?;
    let one_selector_encoded_bytes = selector_encoded_bits
        .checked_add(7)
        .ok_or(Error::BoundsFailure)?
        / 8;
    let permutation_columns = cs.permutation.columns.len();
    let permutation_logical_cells = domain_size
        .checked_mul(permutation_columns as u64)
        .ok_or(Error::BoundsFailure)?;
    let permutation_cells = usize::try_from(permutation_logical_cells)
        .map_err(|_| Error::BoundsFailure)?;
    let permutation_plan = NativePermutationStore::allocation_plan(permutation_cells)
        .map_err(Error::NativeStorage)?;
    Ok(NativeKeygenPreflight {
        logrows: k,
        domain_size,
        extended_logrows,
        constraint_degree,
        advice_columns: cs.num_advice_columns,
        fixed_columns: cs.num_fixed_columns,
        instance_columns: cs.num_instance_columns,
        permutation_columns,
        selector_count: cs.num_selectors,
        lookup_count: cs.lookups.len(),
        shuffle_count: cs.shuffles.len(),
        blinding_factors: cs.blinding_factors(),
        fixed_lagrange_backing_bytes: bytes_product(
            cs.num_fixed_columns,
            domain_rows,
            std::mem::size_of::<Assigned<F>>(),
        )?,
        selector_dense_bitmap_ceiling_bytes,
        selector_activation_container_bytes,
        selector_activation_initial_backing_bytes: selector_activation_container_bytes,
        one_selector_encoded_bytes,
        one_selector_decode_bytes: bytes_product(
            domain_rows,
            std::mem::size_of::<F>(),
            1,
        )?,
        permutation_logical_cells,
        permutation_data_virtual_bytes: permutation_plan.data_file_virtual_bytes,
        permutation_directory_virtual_bytes: permutation_plan.directory_file_virtual_bytes,
        permutation_allocation_bitmap_bytes: permutation_plan
            .data_allocation_bitmap_bytes
            .checked_add(permutation_plan.directory_allocation_bitmap_bytes)
            .ok_or(Error::BoundsFailure)?,
        permutation_bounded_cache_bytes: permutation_plan.bounded_cache_bytes,
    })
}

fn configured_constraint_system<C, ConcreteCircuit>(
    circuit: &ConcreteCircuit,
) -> (ConstraintSystem<C::Scalar>, ConcreteCircuit::Config)
where
    C: CurveAffine,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let mut cs = ConstraintSystem::default();
    #[cfg(feature = "circuit-params")]
    let config = ConcreteCircuit::configure_with_params(&mut cs, circuit.params());
    #[cfg(not(feature = "circuit-params"))]
    let config = ConcreteCircuit::configure(&mut cs);
    #[cfg(feature = "mv-lookup")]
    let cs = cs.chunk_lookups();
    (cs, config)
}

fn domain_for_configured_cs<F: PrimeField + ff::WithSmallOrderMulGroup<3>>(
    k: u32,
    cs: &ConstraintSystem<F>,
) -> Result<EvaluationDomain<F>, Error> {
    let (_, extended_k) = checked_extended_logrows::<F>(k, cs.degree())?;
    let degree = u32::try_from(cs.degree()).map_err(|_| Error::BoundsFailure)?;
    let domain = EvaluationDomain::new(degree, k);
    if domain.extended_k() != extended_k {
        return Err(native_storage_error(
            "native evaluation-domain geometry disagrees with checked quotient extent",
        ));
    }
    Ok(domain)
}

pub(crate) fn create_domain<C, ConcreteCircuit>(
    k: u32,
    #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
) -> Result<(
    EvaluationDomain<C::Scalar>,
    ConstraintSystem<C::Scalar>,
    ConcreteCircuit::Config,
), Error>
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

    let domain = domain_for_configured_cs(k, &cs)?;

    Ok((domain, cs, config))
}

fn new_selector_activation_sets(
    selector_count: usize,
    domain_rows: usize,
    maximum_bytes: usize,
) -> Result<(Vec<SelectorActivationSet>, usize), Error> {
    let minimum_container_bytes = selector_count
        .checked_mul(std::mem::size_of::<SelectorActivationSet>())
        .ok_or(Error::BoundsFailure)?;
    if minimum_container_bytes > maximum_bytes {
        return Err(native_storage_error(
            "selector activation container exceeds configured native budget",
        ));
    }
    let mut selectors = Vec::new();
    selectors
        .try_reserve_exact(selector_count)
        .map_err(|error| Error::NativeStorage(io::Error::new(io::ErrorKind::OutOfMemory, error)))?;
    selectors.extend((0..selector_count).map(|_| SelectorActivationSet::zeroed(domain_rows)));
    let container_bytes = selectors
        .capacity()
        .checked_mul(std::mem::size_of::<SelectorActivationSet>())
        .ok_or(Error::BoundsFailure)?;
    let initial_bytes = selectors.iter().try_fold(container_bytes, |total, selector| {
        total
            .checked_add(selector.storage_bytes())
            .ok_or(Error::BoundsFailure)
    })?;
    if initial_bytes > maximum_bytes {
        return Err(native_storage_error(
            "selector activation backing exceeds configured native budget",
        ));
    }
    Ok((selectors, initial_bytes))
}

/// Assembly to be used in circuit synthesis.
#[derive(Debug)]
struct Assembly<F: Field> {
    k: u32,
    fixed: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
    permutation: permutation::keygen::Assembly,
    selectors: Vec<SelectorActivationSet>,
    selector_activation_storage_current_bytes: usize,
    selector_activation_storage_peak_bytes: usize,
    selector_activation_budget_bytes: usize,
    selector_activation_counts: Vec<u64>,
    fixed_assignment_callback_counts: Vec<u64>,
    fixed_value_assignment_counts: Vec<u64>,
    fixed_value_explicit_zero_seen: Vec<bool>,
    advice_assignment_counts: Vec<u64>,
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

        let current_storage_bytes = self.selector_activation_storage_current_bytes;
        let activation_budget_bytes = self.selector_activation_budget_bytes;
        let selector_activations = self
            .selectors
            .get_mut(selector.0)
            .ok_or(Error::BoundsFailure)?;
        let old_storage_bytes = selector_activations.storage_bytes();
        let inserted = selector_activations
            .insert_bounded(row, current_storage_bytes, activation_budget_bytes)
            .map_err(Error::NativeStorage)?;
        let base_bytes = current_storage_bytes
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
                .get_mut(selector.0)
                .ok_or(Error::BoundsFailure)?;
            *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
        }

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
        column: Column<Advice>,
        _: usize,
        _: V,
    ) -> Result<(), Error>
    where
        V: FnOnce() -> Value<VR>,
        VR: Into<Assigned<F>>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let count = self
            .advice_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
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

        let assigned = to().into_field().assign()?;
        *self
            .fixed
            .get_mut(column.index())
            .and_then(|v| v.get_mut(row))
            .ok_or(Error::BoundsFailure)? = assigned;
        let count = self
            .fixed_assignment_callback_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
        let assigned_rows = self
            .fixed_value_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *assigned_rows = assigned_rows
            .saturating_add(1)
            .min(self.usable_rows.end as u64);
        if assigned_value_is_explicit_zero(assigned) {
            *self
                .fixed_value_explicit_zero_seen
                .get_mut(column.index())
                .ok_or(Error::BoundsFailure)? = true;
        }

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
        let filled_rows = self.usable_rows.end.saturating_sub(from_row);
        for row in self.usable_rows.clone().skip(from_row) {
            col[row] = filler;
        }
        let count = self
            .fixed_assignment_callback_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
        let assigned_rows = self
            .fixed_value_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *assigned_rows = assigned_rows
            .saturating_add(filled_rows as u64)
            .min(self.usable_rows.end as u64);
        if assigned_value_is_explicit_zero(filler) {
            *self
                .fixed_value_explicit_zero_seen
                .get_mut(column.index())
                .ok_or(Error::BoundsFailure)? = true;
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

/// Exact selector/advice-assignment and conservative compact-permutation
/// census produced by a shape-only synthesis. Copy calls are counted so the
/// serialized sparse successor mapping can use the source-derived
/// `2 * copy_calls` bound rather than the much looser full
/// `permutation_columns * domain_size` bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeShapeCensus {
    /// Number of fixed columns emitted by native selector compression.
    pub selector_fixed_column_count: usize,
    /// Exact serialized bytes for the dictionary-backed VK selector cache.
    pub selector_cache_encoded_bytes: u64,
    /// High-water bytes used by activation row-set backing during this census.
    /// Includes the outer selector-set vector and transient old/new storage
    /// while a selector changes from runs to a packed bitmap; excludes allocator
    /// metadata and the separately tracked counters.
    pub selector_activation_storage_bytes: u64,
    /// Number of copy callbacks made by the real circuit floor planner.
    pub permutation_copy_call_count: u64,
    /// Exact enabled-row count for every virtual selector, in selector-index order.
    pub selector_activation_counts: Vec<u64>,
    /// Upper bounds on distinct fixed values per fixed column, based on the
    /// number of assign/fill callbacks (one `fill_from_row` repeats one value).
    pub fixed_assignment_callback_counts: Vec<u64>,
    /// Upper bounds on populated fixed rows per fixed column. A fill callback
    /// contributes each row in its filled suffix.
    pub fixed_value_assignment_counts: Vec<u64>,
    /// Whether synthesis explicitly assigned a canonical zero representation
    /// (Zero or Trivial(0)) in each fixed column. Other encodings do not prove
    /// that the implicit unassigned-row zero is already represented.
    pub fixed_value_explicit_zero_seen: Vec<bool>,
    /// Exact number of advice assignment callbacks made per configured advice
    /// column by this circuit's normal synthesis.
    pub advice_assignment_counts: Vec<u64>,
}

/// Run shape-only synthesis without building fixed/permutation polynomials or
/// a proving/verifying key. This executes the circuit's normal floor planner
/// and records exact selector activations, advice assignment callback counts,
/// and copy-callback count.
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
    )?;
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
        fixed_assignment_callback_counts: vec![0; cs.num_fixed_columns],
        fixed_value_assignment_counts: vec![0; cs.num_fixed_columns],
        fixed_value_explicit_zero_seen: vec![false; cs.num_fixed_columns],
        advice_assignment_counts: vec![0; cs.num_advice_columns],
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
    let selector_cache = SelectorActivationCache::from_selectors(assembly.selectors)
        .map_err(Error::NativeStorage)?;
    let selector_cache_encoded_bytes = u64::try_from(selector_cache.encoded_byte_len())
        .map_err(|_| Error::BoundsFailure)?;
    Ok(NativeShapeCensus {
        selector_fixed_column_count,
        selector_cache_encoded_bytes,
        selector_activation_storage_bytes: u64::try_from(
            assembly.selector_activation_storage_peak_bytes,
        )
        .map_err(|_| Error::BoundsFailure)?,
        permutation_copy_call_count: assembly.permutation_copy_call_count,
        selector_activation_counts: assembly.selector_activation_counts,
        fixed_assignment_callback_counts: assembly.fixed_assignment_callback_counts,
        fixed_value_assignment_counts: assembly.fixed_value_assignment_counts,
        fixed_value_explicit_zero_seen: assembly.fixed_value_explicit_zero_seen,
        advice_assignment_counts: assembly.advice_assignment_counts,
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
    fixed_assignment_callback_counts: Vec<u64>,
    fixed_value_assignment_counts: Vec<u64>,
    fixed_value_explicit_zero_seen: Vec<bool>,
    advice_assignment_counts: Vec<u64>,
    usable_rows: Range<usize>,
    permutation_copy_call_count: u64,
    _marker: std::marker::PhantomData<F>,
}

fn assigned_value_is_explicit_zero<F: Field>(value: Assigned<F>) -> bool {
    match value {
        Assigned::Zero => true,
        Assigned::Trivial(value) => value.is_zero_vartime(),
        Assigned::Rational(_, _) => false,
    }
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
        column: Column<Advice>,
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
        let count = self
            .advice_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
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
        let value = to().into_field().assign()?;
        let callback_count = self
            .fixed_assignment_callback_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *callback_count = callback_count.checked_add(1).ok_or(Error::BoundsFailure)?;
        let count = self
            .fixed_value_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *count = count.checked_add(1).ok_or(Error::BoundsFailure)?;
        if assigned_value_is_explicit_zero(value) {
            *self
                .fixed_value_explicit_zero_seen
                .get_mut(column.index())
                .ok_or(Error::BoundsFailure)? = true;
        }
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
        let value = to.assign()?;
        let callback_count = self
            .fixed_assignment_callback_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        *callback_count = callback_count.checked_add(1).ok_or(Error::BoundsFailure)?;
        let count = self
            .fixed_value_assignment_counts
            .get_mut(column.index())
            .ok_or(Error::BoundsFailure)?;
        let filled_rows = self.usable_rows.end.saturating_sub(from_row);
        *count = count
            .saturating_add(filled_rows as u64)
            .min(self.usable_rows.end as u64);
        if assigned_value_is_explicit_zero(value) {
            *self
                .fixed_value_explicit_zero_seen
                .get_mut(column.index())
                .ok_or(Error::BoundsFailure)? = true;
        }
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
    selectors: &[SelectorActivationSet],
    assignments: &[PackedSelectorAssignment<F>],
    combination_index: usize,
) -> Result<(SelectorBitmap, u64), Error> {
    let max_root = assignments
        .iter()
        .filter(|assignment| {
            assignment.combination_index == combination_index
                && assignment.is_activation_representative
        })
        .map(|assignment| assignment.root_value)
        .max()
        .ok_or_else(|| {
            Error::NativeStorage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "compressed selector column has no selector assignments",
            ))
        })?;
    let bits_per_value = u64::BITS - max_root.leading_zeros();
    let row_count = selectors
        .first()
        .map(SelectorActivationSet::len)
        .unwrap_or(0);
    let mut encoded =
        SelectorBitmap::zeroed(row_count, bits_per_value).map_err(Error::NativeStorage)?;
    for assignment in assignments
        .iter()
        .filter(|assignment| {
            assignment.combination_index == combination_index
                && assignment.is_activation_representative
        })
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

fn native_keygen_census(
    preflight: &NativeKeygenPreflight,
    compressed_selector_columns: usize,
    selector_activation_storage_current_bytes: usize,
    selector_activation_storage_peak_bytes: usize,
    selector_activation_counts: Vec<u64>,
    fixed_assignment_callback_counts: Vec<u64>,
    fixed_value_assignment_counts: Vec<u64>,
    fixed_value_explicit_zero_seen: Vec<bool>,
    advice_assignment_counts: Vec<u64>,
    permutation: crate::poly::NativePermutationStats,
) -> Result<NativeKeygenCensus, Error> {
    let domain_size = preflight.domain_size;
    let permutation_cell_bound = u64::try_from(preflight.permutation_columns)
        .ok()
        .and_then(|columns| columns.checked_mul(domain_size))
        .ok_or(Error::BoundsFailure)?;
    if selector_activation_counts.len() != preflight.selector_count
        || selector_activation_counts
            .iter()
            .any(|count| *count > domain_size)
        || fixed_assignment_callback_counts.len() != preflight.fixed_columns
        || fixed_assignment_callback_counts
            .iter()
            .any(|count| *count > domain_size)
        || fixed_value_assignment_counts.len() != preflight.fixed_columns
        || fixed_value_assignment_counts
            .iter()
            .any(|count| *count > domain_size)
        || fixed_value_explicit_zero_seen.len() != preflight.fixed_columns
        || advice_assignment_counts.len() != preflight.advice_columns
        || advice_assignment_counts
            .iter()
            .any(|count| *count > domain_size)
        || permutation.copy_calls > permutation_cell_bound
    {
        return Err(native_storage_error(
            "native keygen synthesis census exceeds configured per-column or permutation bounds",
        ));
    }
    Ok(NativeKeygenCensus {
        preflight: preflight.clone(),
        compressed_selector_columns,
        selector_activation_storage_current_bytes: u64::try_from(
            selector_activation_storage_current_bytes,
        )
        .map_err(|_| Error::BoundsFailure)?,
        selector_activation_storage_peak_bytes: u64::try_from(
            selector_activation_storage_peak_bytes,
        )
        .map_err(|_| Error::BoundsFailure)?,
        selector_activation_counts,
        fixed_assignment_callback_counts,
        fixed_value_assignment_counts,
        fixed_value_explicit_zero_seen,
        advice_assignment_counts,
        permutation_copy_calls: permutation.copy_calls,
        permutation_successor_edges: permutation.successor_edges,
        permutation_nondefault_records: permutation.nondefault_records,
        permutation_data_pages_allocated: permutation.data_pages_allocated,
        permutation_directory_pages_allocated: permutation.directory_pages_allocated,
        permutation_data_allocated_bytes: permutation.data_page_bytes,
        permutation_directory_allocated_bytes: permutation.directory_page_bytes,
        permutation_cache_bytes: permutation.cache_bytes,
        permutation_allocation_bitmap_bytes: permutation.allocation_bitmap_bytes,
    })
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
    keygen_vk_custom_with_native_gate(
        params,
        circuit,
        compress_selectors,
        |_| {
            Ok(NativeKeygenBudget {
                max_selector_activation_bytes: u64::MAX,
                max_permutation_backing_bytes: u64::MAX,
            })
        },
        |_| Ok(()),
    )
}

/// Generate a VK while exposing pre-allocation and exact synthesis counters to
/// a same-process resource gate. The hooks observe the native synthesis and do
/// not cause a second circuit pass.
pub fn keygen_vk_custom_with_native_gate<'params, C, P, ConcreteCircuit>(
    params: &P,
    circuit: &ConcreteCircuit,
    compress_selectors: bool,
    before_assembly: impl FnOnce(&NativeKeygenPreflight) -> io::Result<NativeKeygenBudget>,
    after_synthesis: impl FnOnce(&NativeKeygenCensus) -> io::Result<()>,
) -> Result<VerifyingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
    C::Scalar: FromUniformBytes<64>,
{
    let (cs, config) = configured_constraint_system::<C, ConcreteCircuit>(circuit);
    let preflight = native_keygen_preflight::<C::Scalar>(params.k(), &cs)?;
    let n = usize::try_from(params.n()).map_err(|_| Error::BoundsFailure)?;
    if u64::try_from(n).map_err(|_| Error::BoundsFailure)? != preflight.domain_size {
        return Err(native_storage_error(
            "SRS base domain does not match configured keygen geometry",
        ));
    }

    if n < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k()));
    }
    let budget = before_assembly(&preflight).map_err(Error::NativeStorage)?;
    let selector_activation_budget_bytes = usize::try_from(budget.max_selector_activation_bytes)
        .map_err(|_| native_storage_error("selector activation budget exceeds host index"))?;
    let (selectors, selector_activation_storage_current_bytes) =
        new_selector_activation_sets(cs.num_selectors, n, selector_activation_budget_bytes)?;
    let domain = domain_for_configured_cs(params.k(), &cs)?;
    let usable_end = n
        .checked_sub(cs.blinding_factors().checked_add(1).ok_or(Error::BoundsFailure)?)
        .ok_or_else(|| Error::not_enough_rows_available(params.k()))?;

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed: (0..cs.num_fixed_columns)
            .map(|_| {
                Polynomial::mapped_filled(n, Assigned::Zero)
                    .map_err(Error::NativeStorage)
            })
            .collect::<Result<Vec<_>, _>>()?,
        permutation: permutation::keygen::Assembly::new_with_budget(
            n,
            &cs.permutation,
            budget.max_permutation_backing_bytes,
        )?,
        selectors,
        selector_activation_storage_current_bytes,
        selector_activation_storage_peak_bytes: selector_activation_storage_current_bytes,
        selector_activation_budget_bytes,
        selector_activation_counts: vec![0; cs.num_selectors],
        fixed_assignment_callback_counts: vec![0; cs.num_fixed_columns],
        fixed_value_assignment_counts: vec![0; cs.num_fixed_columns],
        fixed_value_explicit_zero_seen: vec![false; cs.num_fixed_columns],
        advice_assignment_counts: vec![0; cs.num_advice_columns],
        usable_rows: 0..usable_end,
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
        cs.compress_selector_activation_sets(&assembly.selectors)
    } else {
        cs.directly_convert_selector_activation_sets(&assembly.selectors)
    };
    let permutation_stats = assembly.permutation.native_stats()?;
    let native_census = native_keygen_census(
        &preflight,
        selector_column_count,
        assembly.selector_activation_storage_current_bytes,
        assembly.selector_activation_storage_peak_bytes,
        assembly.selector_activation_counts.clone(),
        assembly.fixed_assignment_callback_counts.clone(),
        assembly.fixed_value_assignment_counts.clone(),
        assembly.fixed_value_explicit_zero_seen.clone(),
        assembly.advice_assignment_counts.clone(),
        permutation_stats,
    )?;
    after_synthesis(&native_census).map_err(Error::NativeStorage)?;
    let permutation_vk = assembly
        .permutation
        .build_vk(params, &domain, &cs.permutation)?;

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

    let selector_cache = if compress_selectors {
        SelectorActivationCache::from_selectors(assembly.selectors)
            .map_err(Error::NativeStorage)?
    } else {
        SelectorActivationCache::default()
    };

    Ok(VerifyingKey::from_parts(
        domain,
        fixed_commitments,
        permutation_vk,
        cs,
        selector_cache,
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
    keygen_pk_with_native_gate(
        params,
        vk,
        circuit,
        |_| {
            Ok(NativeKeygenBudget {
                max_selector_activation_bytes: u64::MAX,
                max_permutation_backing_bytes: u64::MAX,
            })
        },
        |_| Ok(()),
    )
}

/// Generate a PK while exposing the same native pre-allocation and exact
/// synthesis counters used by the VK resource gate.
pub fn keygen_pk_with_native_gate<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: VerifyingKey<C>,
    circuit: &ConcreteCircuit,
    before_assembly: impl FnOnce(&NativeKeygenPreflight) -> io::Result<NativeKeygenBudget>,
    after_synthesis: impl FnOnce(&NativeKeygenCensus) -> io::Result<()>,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    keygen_pk_with_native_gate_mode(params, vk, circuit, before_assembly, after_synthesis, false)
}

/// Generate a mapping-only proving key for the formal setup artifact. The
/// returned value can be serialized, but must be reloaded through the native
/// bounded reader before it reaches proof creation.
#[cfg(feature = "formal-permutation-spill")]
pub fn keygen_pk_with_native_gate_compact_artifact<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: VerifyingKey<C>,
    circuit: &ConcreteCircuit,
    before_assembly: impl FnOnce(&NativeKeygenPreflight) -> io::Result<NativeKeygenBudget>,
    after_synthesis: impl FnOnce(&NativeKeygenCensus) -> io::Result<()>,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    keygen_pk_with_native_gate_mode(params, vk, circuit, before_assembly, after_synthesis, true)
}

fn keygen_pk_with_native_gate_mode<'params, C, P, ConcreteCircuit>(
    params: &P,
    vk: VerifyingKey<C>,
    circuit: &ConcreteCircuit,
    before_assembly: impl FnOnce(&NativeKeygenPreflight) -> io::Result<NativeKeygenBudget>,
    after_synthesis: impl FnOnce(&NativeKeygenCensus) -> io::Result<()>,
    compact_artifact: bool,
) -> Result<ProvingKey<C>, Error>
where
    C: CurveAffine,
    P: Params<'params, C>,
    ConcreteCircuit: Circuit<C::Scalar>,
{
    let (cs, config) = configured_constraint_system::<C, ConcreteCircuit>(circuit);
    let preflight = native_keygen_preflight::<C::Scalar>(params.k(), &cs)?;
    let n = usize::try_from(params.n()).map_err(|_| Error::BoundsFailure)?;
    if u64::try_from(n).map_err(|_| Error::BoundsFailure)? != preflight.domain_size
        || vk.domain.k() != params.k()
        || vk.domain.get_n() != preflight.domain_size
        || vk.domain.extended_k() != preflight.extended_logrows
    {
        return Err(native_storage_error(
            "PK SRS/VK domains do not match configured keygen geometry",
        ));
    }

    if n < cs.minimum_rows() {
        return Err(Error::not_enough_rows_available(params.k()));
    }
    let budget = before_assembly(&preflight).map_err(Error::NativeStorage)?;
    let selector_activation_budget_bytes = usize::try_from(budget.max_selector_activation_bytes)
        .map_err(|_| native_storage_error("selector activation budget exceeds host index"))?;
    let (selectors, selector_activation_storage_current_bytes) =
        new_selector_activation_sets(cs.num_selectors, n, selector_activation_budget_bytes)?;
    let usable_end = n
        .checked_sub(cs.blinding_factors().checked_add(1).ok_or(Error::BoundsFailure)?)
        .ok_or_else(|| Error::not_enough_rows_available(params.k()))?;

    let mut assembly: Assembly<C::Scalar> = Assembly {
        k: params.k(),
        fixed: (0..cs.num_fixed_columns)
            .map(|_| {
                Polynomial::mapped_filled(n, Assigned::Zero)
                    .map_err(Error::NativeStorage)
            })
            .collect::<Result<Vec<_>, _>>()?,
        permutation: permutation::keygen::Assembly::new_with_budget(
            n,
            &cs.permutation,
            budget.max_permutation_backing_bytes,
        )?,
        selectors,
        selector_activation_storage_current_bytes,
        selector_activation_storage_peak_bytes: selector_activation_storage_current_bytes,
        selector_activation_budget_bytes,
        selector_activation_counts: vec![0; cs.num_selectors],
        fixed_assignment_callback_counts: vec![0; cs.num_fixed_columns],
        fixed_value_assignment_counts: vec![0; cs.num_fixed_columns],
        fixed_value_explicit_zero_seen: vec![false; cs.num_fixed_columns],
        advice_assignment_counts: vec![0; cs.num_advice_columns],
        usable_rows: 0..usable_end,
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
        cs.compress_selector_activation_sets(&selectors)
    } else {
        cs.directly_convert_selector_activation_sets(&selectors)
    };
    let permutation_stats = assembly.permutation.native_stats()?;
    let native_census = native_keygen_census(
        &preflight,
        selector_column_count,
        assembly.selector_activation_storage_current_bytes,
        assembly.selector_activation_storage_peak_bytes,
        assembly.selector_activation_counts.clone(),
        assembly.fixed_assignment_callback_counts.clone(),
        assembly.fixed_value_assignment_counts.clone(),
        assembly.fixed_value_explicit_zero_seen.clone(),
        assembly.advice_assignment_counts.clone(),
        permutation_stats,
    )?;
    after_synthesis(&native_census).map_err(Error::NativeStorage)?;
    let mut fixed = Vec::with_capacity(cs.num_fixed_columns + selector_column_count);
    for assigned in assembly.fixed {
        let polynomial = assigned.invert_spilled().map_err(Error::NativeStorage)?;
        fixed.push(
            FixedColumnSource::from_polynomial(polynomial).map_err(Error::NativeStorage)?,
        );
    }
    for combination_index in 0..selector_column_count {
        let (encoded, max_root) =
            selector_combination_bitmap(&selectors, &selector_assignments, combination_index)?;
        let cardinality = usize::try_from(max_root)
            .ok()
            .and_then(|root| root.checked_add(1))
            .ok_or(Error::BoundsFailure)?;
        if cardinality <= 256 {
            let mut codebook = Vec::with_capacity(cardinality);
            codebook.push(C::Scalar::ZERO);
            for _ in 1..cardinality {
                let next = codebook.last().copied().unwrap_or(C::Scalar::ZERO) + C::Scalar::ONE;
                codebook.push(next);
            }
            let width = PackedIdWidth::for_cardinality(cardinality)
                .map_err(Error::NativeStorage)?;
            let mut ids = PackedIdTape::zeroed(n, width).map_err(Error::NativeStorage)?;
            for row in 0..n {
                let id = usize::try_from(encoded.value(row)).map_err(|_| Error::BoundsFailure)?;
                ids.write(row, id).map_err(Error::NativeStorage)?;
            }
            fixed.push(FixedColumnSource::Compact { codebook, ids });
        } else {
            let poly = selector_bitmap_to_field::<C::Scalar>(&encoded, max_root);
            let polynomial = vk
                .domain
                .lagrange_from_vec(poly)
                .spill()
                .map_err(Error::NativeStorage)?;
            fixed.push(
                FixedColumnSource::from_polynomial(polynomial)
                    .map_err(Error::NativeStorage)?,
            );
        }
    }

    let permutation_pk = if compact_artifact {
        #[cfg(feature = "formal-permutation-spill")]
        {
            assembly
                .permutation
                .build_pk_compact_artifact(params, &vk.domain, &cs.permutation)?
        }
        #[cfg(not(feature = "formal-permutation-spill"))]
        {
            return Err(native_storage_error(
                "compact native PK artifacts require formal permutation spill support",
            ));
        }
    } else {
        assembly
            .permutation
            .build_pk(params, &vk.domain, &cs.permutation)?
    };

    let (l0, l_last, l_active_row) = compute_prover_boundary_polynomials::<C>(
        &vk.domain,
        cs.blinding_factors(),
    )
    .map_err(Error::NativeStorage)?;

    // Compute the optimized evaluation data structure
    let ev = Evaluator::new(&vk.cs);

    Ok(ProvingKey {
        vk,
        l0,
        l_last,
        l_active_row,
        fixed_values: fixed,
        permutation: permutation_pk,
        ev,
    })
}

#[cfg(test)]
mod native_domain_capacity_tests {
    use super::checked_extended_logrows;
    use crate::halo2curves::bn256::Fr;

    #[test]
    fn quotient_domain_capacity_is_checked_before_domain_construction() {
        assert_eq!(checked_extended_logrows::<Fr>(25, 6).unwrap(), (1 << 25, 28));
        assert_eq!(checked_extended_logrows::<Fr>(26, 5).unwrap(), (1 << 26, 28));
        assert!(checked_extended_logrows::<Fr>(26, 6).is_err());
    }
}
