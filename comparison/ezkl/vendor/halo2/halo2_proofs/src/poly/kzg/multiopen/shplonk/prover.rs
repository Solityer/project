use super::{
    construct_intermediate_sets, native_shplonk_fixed_aggregate_cache_limits, ChallengeU,
    ChallengeV, ChallengeY, Commitment, NativeShplonkFixedAggregateCacheLimits, RotationSet,
};
use crate::arithmetic::{
    eval_polynomial, evaluate_vanishing_polynomial, kate_division, lagrange_interpolate,
    parallelize, CurveAffine,
};
use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::{Blind, ParamsProver, Prover};
use crate::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use crate::poly::query::{PolynomialPointer, ProverPolynomial, ProverQuery};
use crate::poly::{Basis, Coeff, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use ff::Field;
use group::Curve;
use halo2curves::pairing::Engine;
use halo2curves::CurveExt;
use maybe_rayon::iter::{IndexedParallelIterator, ParallelIterator};
use maybe_rayon::prelude::IntoParallelIterator;
use rand_core::RngCore;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use group::ff::WithSmallOrderMulGroup;
use std::ops::MulAssign;

fn div_by_vanishing<F: Field>(poly: Polynomial<F, Coeff>, roots: &[F]) -> Vec<F> {
    let poly = roots
        .iter()
        .fold(poly.into_values(), |poly, point| kate_division(&poly, *point));

    poly
}

fn accumulate_scaled<F: Field, B: Basis>(
    accumulator: &mut Option<Polynomial<F, B>>,
    polynomial: Polynomial<F, B>,
    power: F,
) {
    let polynomial = polynomial * power;
    *accumulator = Some(match accumulator.take() {
        Some(accumulated) => accumulated + &polynomial,
        None => polynomial,
    });
}

fn shplonk_batch_budget<E: Engine>(
    params: &ParamsKZG<E>,
    requested_width: usize,
) -> io::Result<(usize, u64, u64, u64)> {
    let field_bytes = std::mem::size_of::<E::Fr>() as u128;
    let polynomial_header = std::mem::size_of::<Polynomial<E::Fr, Coeff>>() as u128;
    let domain = u128::from(params.n);
    // A live task touches the opened source polynomial and one spilled clone.
    let per_task = domain
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_mul(field_bytes))
        .and_then(|bytes| bytes.checked_add(2 * polynomial_header))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "SHPLONK task bound overflow"))?;
    // At quotient/linearisation division, one per-set accumulator and one
    // proof accumulator coexist with `into_values()` and one kate-division
    // output vector. Bound those four N-field windows even between batches.
    let fixed_workspace = domain
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_mul(field_bytes))
        .and_then(|bytes| bytes.checked_add(4 * polynomial_header))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "SHPLONK workspace bound overflow"))?;
    let per_task = u64::try_from(per_task)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "SHPLONK task bound exceeds u64"))?;
    let fixed_workspace = u64::try_from(fixed_workspace).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "SHPLONK workspace bound exceeds u64")
    })?;
    let (batch_width, available) = crate::plonk::native_shplonk_batch_width(
        requested_width,
        per_task,
        fixed_workspace,
    )
    .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error.to_string()))?;
    Ok((batch_width, available, per_task, fixed_workspace))
}

struct CommitmentExtension<'a, C: CurveAffine> {
    commitment: Commitment<C::Scalar, PolynomialPointer<'a, C>>,
    low_degree_equivalent: Polynomial<C::Scalar, Coeff>,
}

impl<'a, C: CurveAffine> Commitment<C::Scalar, PolynomialPointer<'a, C>> {
    fn extend(&self, points: &[C::Scalar]) -> CommitmentExtension<'a, C> {
        let poly = lagrange_interpolate(points, &self.evals()[..]);

        let low_degree_equivalent = Polynomial {
            values: poly,
            mapped: None,
            _marker: PhantomData,
        };

        CommitmentExtension {
            commitment: self.clone(),
            low_degree_equivalent,
        }
    }
}

impl<'a, C: CurveAffine> CommitmentExtension<'a, C> {
    fn linearisation_contribution(
        &self,
        u: C::Scalar,
    ) -> io::Result<Polynomial<C::Scalar, Coeff>> {
        let r_eval = eval_polynomial(&self.low_degree_equivalent[..], u);
        let mut contribution = self.commitment.get().materialize()?;
        contribution[0] -= r_eval;
        Ok(contribution)
    }

    fn quotient_contribution(&self) -> io::Result<Polynomial<C::Scalar, Coeff>> {
        let len = self.low_degree_equivalent.len();
        let mut p_x = self.commitment.get().materialize()?;
        parallelize(&mut p_x[0..len], |lhs, start| {
            for (lhs, rhs) in lhs
                .iter_mut()
                .zip(self.low_degree_equivalent[start..].iter())
            {
                *lhs -= *rhs;
            }
        });
        Ok(p_x)
    }
}

struct RotationSetExtension<'a, C: CurveAffine> {
    commitments: Vec<CommitmentExtension<'a, C>>,
    points: Vec<C::Scalar>,
}

struct FixedAggregate<F: Field> {
    rotation_set_index: usize,
    fixed_commitment_indices: Vec<usize>,
    polynomial: Polynomial<F, Coeff>,
    low_degree_equivalent: Vec<F>,
}

pub(super) fn native_shplonk_resource_layout<C: CurveAffine>() -> (usize, usize, usize) {
    (
        std::mem::size_of::<CommitmentExtension<'static, C>>(),
        std::mem::size_of::<RotationSetExtension<'static, C>>(),
        std::mem::size_of::<FixedAggregate<C::Scalar>>(),
    )
}

fn fixed_source_status<C: CurveAffine>(commitment: &CommitmentExtension<'_, C>) -> Option<bool> {
    match commitment.commitment.get().poly {
        ProverPolynomial::FixedColumn { source, .. } => Some(source.constant_value().is_some()),
        _ => None,
    }
}

fn fixed_aggregate_contains<F: Field>(aggregate: &FixedAggregate<F>, commitment_index: usize) -> bool {
    aggregate.fixed_commitment_indices.binary_search(&commitment_index).is_ok()
}

fn cache_group_disk_bytes<F: Field>(domain_size: usize) -> Option<u64> {
    let payload = (domain_size as u128).checked_mul(std::mem::size_of::<F>() as u128)?;
    let reserved = payload.checked_add(4095)?.checked_div(4096)?.checked_mul(4096)?.checked_add(4096)?;
    u64::try_from(reserved).ok()
}

fn cache_group_actual_host_bytes<F: Field>(aggregate: &FixedAggregate<F>) -> Option<u64> {
    let mapped = (aggregate.polynomial.len() as u128).checked_mul(std::mem::size_of::<F>() as u128)?;
    let indices = (aggregate.fixed_commitment_indices.capacity() as u128).checked_mul(std::mem::size_of::<usize>() as u128)?;
    let points = (aggregate.low_degree_equivalent.capacity() as u128).checked_mul(std::mem::size_of::<F>() as u128)?;
    let bytes = (std::mem::size_of::<FixedAggregate<F>>() as u128)
        .checked_add(mapped)?
        .checked_add(indices)?
        .checked_add(points)?
        .checked_add(3 * 4096)?;
    u64::try_from(bytes).ok()
}

fn prepare_fixed_aggregate_cache<E: Engine>(
    params: &ParamsKZG<E>,
    rotation_sets: &[RotationSetExtension<E::G1Affine>],
    y: E::Fr,
    limits: NativeShplonkFixedAggregateCacheLimits,
) -> io::Result<Vec<FixedAggregate<E::Fr>>>
where
    E::Fr: WithSmallOrderMulGroup<3>,
    E::G1Affine: SerdeCurveAffine<ScalarExt = E::Fr, CurveExt = E::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
{
    let disabled = || Ok(Vec::new());
    if limits.max_groups == 0 {
        return disabled();
    }
    if limits.max_groups > 2 || limits.domain_size != u64::from(params.n) || limits.scalar_bytes != std::mem::size_of::<E::Fr>() as u64 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "native SHPLONK fixed-cache limits do not match the proof domain"));
    }
    let mut candidates = Vec::new();
    if candidates.try_reserve_exact(rotation_sets.len()).is_err() {
        return disabled();
    }
    for (rotation_set_index, rotation_set) in rotation_sets.iter().enumerate() {
        let mut fixed_count = 0usize;
        let mut nonconstant_fixed_count = 0usize;
        let mut nonfixed_count = 0usize;
        for commitment in &rotation_set.commitments {
            match fixed_source_status(commitment) {
                Some(is_constant) => {
                    fixed_count += 1;
                    if !is_constant {
                        nonconstant_fixed_count += 1;
                    }
                }
                None => nonfixed_count += 1,
            }
        }
        if fixed_count > 0 && nonconstant_fixed_count > 0 && nonfixed_count > 0 {
            candidates.push((rotation_set_index, fixed_count, nonconstant_fixed_count, rotation_set.commitments.len()));
        }
    }
    candidates.sort_by(|left, right| right.2.cmp(&left.2).then_with(|| left.0.cmp(&right.0)));
    candidates.truncate(limits.max_groups);
    if candidates.is_empty() {
        return disabled();
    }
    // Cache groups remain live through both SHPLONK passes. Account for the
    // largest batch in the whole proof, including unselected rotation sets.
    let requested_width = rotation_sets
        .iter()
        .map(|rotation_set| rotation_set.commitments.len())
        .max()
        .unwrap_or(1);
    let (batch_width, available, per_task_bytes, fixed_workspace_bytes) = match shplonk_batch_budget(params, requested_width) {
        Ok(budget) => budget,
        Err(_) => return disabled(),
    };
    let base_workspace = u128::from(fixed_workspace_bytes)
        .checked_add(u128::from(per_task_bytes).checked_mul(batch_width as u128).unwrap_or(u128::MAX))
        .unwrap_or(u128::MAX);
    let live_host_cache_headroom = u128::from(available).saturating_sub(base_workspace);
    let n = usize::try_from(params.n).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "SHPLONK domain exceeds host size"))?;
    let mut selected_count = candidates.len();
    while selected_count > 0 {
        let mut host_required = std::mem::size_of::<Vec<FixedAggregate<E::Fr>>>() as u128 + 4096;
        let mut disk_required = 0u128;
        let mut fits = true;
        for (rotation_set_index, fixed_count, _, _) in candidates.iter().take(selected_count) {
            let rotation_set = &rotation_sets[*rotation_set_index];
            let group_host = (std::mem::size_of::<FixedAggregate<E::Fr>>() as u128)
                .checked_add((n as u128).checked_mul(std::mem::size_of::<E::Fr>() as u128).unwrap_or(u128::MAX))
                .and_then(|bytes| bytes.checked_add((*fixed_count as u128).checked_mul(std::mem::size_of::<usize>() as u128)?))
                .and_then(|bytes| bytes.checked_add((rotation_set.points.len() as u128).checked_mul(std::mem::size_of::<E::Fr>() as u128)?))
                .and_then(|bytes| bytes.checked_add(3 * 4096));
            let Some(group_host) = group_host else {
                fits = false;
                break;
            };
            let group_disk = match cache_group_disk_bytes::<E::Fr>(n) {
                Some(bytes) => u128::from(bytes),
                None => {
                    fits = false;
                    break;
                }
            };
            if group_host > u128::from(limits.group_host_bytes)
                || group_disk > u128::from(limits.group_disk_bytes)
            {
                fits = false;
                break;
            }
            host_required = match host_required.checked_add(group_host) {
                Some(bytes) => bytes,
                None => {
                    fits = false;
                    break;
                }
            };
            disk_required = match disk_required.checked_add(group_disk) {
                Some(bytes) => bytes,
                None => {
                    fits = false;
                    break;
                }
            };
        }
        if fits
            && host_required <= u128::from(limits.host_bytes)
            && host_required <= live_host_cache_headroom
            && disk_required <= u128::from(limits.disk_bytes)
        {
            break;
        }
        selected_count -= 1;
    }
    if selected_count == 0 {
        return disabled();
    }
    let mut aggregates = Vec::new();
    if aggregates.try_reserve_exact(selected_count).is_err() {
        return disabled();
    }
    for (rotation_set_index, fixed_count, _, _) in candidates.iter().take(selected_count) {
        let rotation_set = &rotation_sets[*rotation_set_index];
        let polynomial = match Polynomial::<E::Fr, Coeff>::mapped_filled(n, E::Fr::ZERO) {
            Ok(polynomial) => polynomial,
            Err(_) => return disabled(),
        };
        let mut fixed_commitment_indices = Vec::new();
        if fixed_commitment_indices.try_reserve_exact(*fixed_count).is_err() {
            return disabled();
        }
        let mut low_degree_equivalent = Vec::new();
        if low_degree_equivalent.try_reserve_exact(rotation_set.points.len()).is_err() {
            return disabled();
        }
        low_degree_equivalent.resize(rotation_set.points.len(), E::Fr::ZERO);
        aggregates.push(FixedAggregate {
            rotation_set_index: *rotation_set_index,
            fixed_commitment_indices,
            polynomial,
            low_degree_equivalent,
        });
    }
    // Check actual Vec capacities and mapped lengths before decoding any
    // fixed source. Vec may reserve slightly more than requested on some
    // allocators, so the authenticated optional quota must fit the real bank.
    let actual_host_total = (std::mem::size_of::<Vec<FixedAggregate<E::Fr>>>() as u128 + 4096)
        .checked_add(
            aggregates
                .iter()
                .try_fold(0u128, |total, aggregate| {
                    total.checked_add(u128::from(cache_group_actual_host_bytes(aggregate)?))
                })
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "native SHPLONK fixed-cache total host bound overflow"))?,
        )
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "native SHPLONK fixed-cache total host bound overflow"))?;
    let actual_disk_total = aggregates
        .iter()
        .try_fold(0u128, |total, _| {
            total.checked_add(u128::from(cache_group_disk_bytes::<E::Fr>(n)?))
        })
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "native SHPLONK fixed-cache total disk bound overflow"))?;
    let actual_bank_header = std::mem::size_of::<Vec<FixedAggregate<E::Fr>>>() as u64 + 4096;
    let actual_groups_fit = aggregates.iter().all(|aggregate| {
        cache_group_actual_host_bytes(aggregate)
            .map(|bytes| bytes <= limits.group_host_bytes)
            .unwrap_or(false)
            && cache_group_disk_bytes::<E::Fr>(n)
                .map(|bytes| bytes <= limits.group_disk_bytes)
                .unwrap_or(false)
    });
    if !actual_groups_fit
        || actual_bank_header > limits.bank_header_bytes
        || actual_host_total > u128::from(limits.host_bytes)
        || actual_disk_total > u128::from(limits.disk_bytes)
    {
        return disabled();
    }
    for aggregate in &mut aggregates {
        let rotation_set = &rotation_sets[aggregate.rotation_set_index];
        let mut power_of_y = E::Fr::ONE;
        for (commitment_index, commitment) in rotation_set.commitments.iter().enumerate() {
            if fixed_source_status(commitment).is_some() {
                let polynomial = commitment.commitment.get().materialize()?;
                if polynomial.len() != n || commitment.low_degree_equivalent.len() != aggregate.low_degree_equivalent.len() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "native SHPLONK fixed source has inconsistent dimensions"));
                }
                for (destination, source) in aggregate.polynomial.iter_mut().zip(polynomial.iter()) {
                    *destination += power_of_y * *source;
                }
                for (destination, source) in aggregate.low_degree_equivalent.iter_mut().zip(commitment.low_degree_equivalent.iter()) {
                    *destination += power_of_y * *source;
                }
                aggregate.fixed_commitment_indices.push(commitment_index);
            }
            power_of_y *= y;
        }
    }
    #[cfg(test)]
    super::record_native_shplonk_test_selected_cache_groups(
        aggregates
            .iter()
            .map(|aggregate| {
                let rotation_set = &rotation_sets[aggregate.rotation_set_index];
                let nonconstant_fixed = rotation_set
                    .commitments
                    .iter()
                    .filter(|commitment| fixed_source_status(commitment) == Some(false))
                    .count();
                (
                    aggregate.rotation_set_index,
                    rotation_set.commitments.len(),
                    nonconstant_fixed,
                    aggregate.fixed_commitment_indices.clone(),
                )
            })
            .collect(),
    );
    Ok(aggregates)
}

impl<'a, C: CurveAffine> RotationSet<C::Scalar, PolynomialPointer<'a, C>> {
    fn extend(self, commitments: Vec<CommitmentExtension<'a, C>>) -> RotationSetExtension<'a, C> {
        RotationSetExtension {
            commitments,
            points: self.points,
        }
    }
}

/// Concrete KZG prover with SHPLONK variant
#[derive(Debug)]
pub struct ProverSHPLONK<'a, E: Engine> {
    params: &'a ParamsKZG<E>,
}

impl<'a, E: Engine> ProverSHPLONK<'a, E> {
    /// Given parameters creates new prover instance
    pub fn new(params: &'a ParamsKZG<E>) -> Self {
        Self { params }
    }
}

/// Create a multi-opening proof
impl<'params, E: Engine + Debug> Prover<'params, KZGCommitmentScheme<E>>
    for ProverSHPLONK<'params, E>
where
    E::Fr: WithSmallOrderMulGroup<3>,
    E::Fr: Ord,
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
{
    const QUERY_INSTANCE: bool = false;

    fn new(params: &'params ParamsKZG<E>) -> Self {
        Self { params }
    }

    /// Create a multi-opening proof
    fn create_proof<
        'com,
        Ch: EncodedChallenge<E::G1Affine>,
        T: TranscriptWrite<E::G1Affine, Ch>,
        R,
        I,
    >(
        &self,
        _: R,
        transcript: &mut T,
        queries: I,
    ) -> io::Result<()>
    where
        I: IntoIterator<Item = ProverQuery<'com, E::G1Affine>> + Clone,
        R: RngCore,
    {
        // TODO: explore if it is safe to use same challenge
        // for different sets that are already combined with another challenge
        let y: ChallengeY<_> = transcript.squeeze_challenge_scalar();

        let quotient_contribution = |rotation_set: &RotationSetExtension<E::G1Affine>,
                                     set_index: usize,
                                     batch_width: usize,
                                     available_budget: u64,
                                     per_task_bytes: u64,
                                     fixed_aggregate: Option<&FixedAggregate<E::Fr>>|
         -> io::Result<Polynomial<E::Fr, Coeff>> {
            // Preserve the original commitment order and complete R_i
            // accumulator, but retain at most one budgeted indexed batch of
            // spilled per-polynomial contributions at a time.
            let mut n_x = None;
            let mut power_of_y = E::Fr::ONE;
            let total = rotation_set.commitments.len();
            for batch_start in (0..total).step_by(batch_width) {
                let batch_end = batch_start.saturating_add(batch_width).min(total);
                let batch_started = std::time::Instant::now();
                println!(
                    "EZKL_NATIVE_PHASE START stage=shplonk_quotient_batch rotation_set={} first_commitment={} end_commitment={} total_commitments={} batch_width={} per_task_bytes={} available_budget_bytes={}",
                    set_index,
                    batch_start,
                    batch_end,
                    total,
                    batch_end - batch_start,
                    per_task_bytes,
                    available_budget,
                );
                let numerators = rotation_set.commitments[batch_start..batch_end]
                    .as_ref()
                    .into_par_iter()
                    .enumerate()
                    .map(|(offset, commitment)| {
                        if fixed_aggregate.map_or(false, |aggregate| fixed_aggregate_contains(aggregate, batch_start + offset)) {
                            Ok(None)
                        } else {
                            commitment.quotient_contribution().map(Some)
                        }
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                for numerator in numerators {
                    if let Some(numerator) = numerator {
                        accumulate_scaled(&mut n_x, numerator, power_of_y);
                    }
                    power_of_y *= *y;
                }
                println!(
                    "EZKL_NATIVE_PHASE DONE stage=shplonk_quotient_batch rotation_set={} first_commitment={} end_commitment={} elapsed_ms={:.3}",
                    set_index,
                    batch_start,
                    batch_end,
                    batch_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let mut n_x = n_x.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native SHPLONK rotation set has no quotient contributions",
                )
            })?;
            if let Some(aggregate) = fixed_aggregate {
                for (destination, source) in n_x.iter_mut().zip(aggregate.polynomial.iter()) {
                    *destination += *source;
                }
                for (destination, source) in n_x.iter_mut().zip(aggregate.low_degree_equivalent.iter()) {
                    *destination -= *source;
                }
            }

            let points = &rotation_set.points[..];

            // quotient contribution of this evaluation set is
            // Q_i(X) = N_i(X) / Z_i(X) where
            // Z_i(X) = (x - r_i_0) * (x - r_i_1) * ...
            let mut poly = div_by_vanishing(n_x, points);
            poly.resize(self.params.n as usize, E::Fr::ZERO);

            Ok(Polynomial {
                values: poly,
                mapped: None,
                _marker: PhantomData,
            })
        };

        let intermediate_sets = construct_intermediate_sets(queries);
        let (rotation_sets, super_point_set) = (
            intermediate_sets.rotation_sets,
            intermediate_sets.super_point_set,
        );

        let rotation_sets: Vec<RotationSetExtension<E::G1Affine>> = rotation_sets
            .into_par_iter()
            .map(|rotation_set| {
                let commitments: Vec<CommitmentExtension<E::G1Affine>> = rotation_set
                    .commitments
                    .as_slice()
                    .into_par_iter()
                    .map(|commitment_data| commitment_data.extend(&rotation_set.points))
                    .collect();
                rotation_set.extend(commitments)
            })
            .collect();

        let fixed_aggregate_cache = match native_shplonk_fixed_aggregate_cache_limits() {
            Some(limits) => prepare_fixed_aggregate_cache(self.params, &rotation_sets, *y, limits)?,
            None => Vec::new(),
        };
        if !fixed_aggregate_cache.is_empty() {
            println!("EZKL_NATIVE_PHASE DONE stage=shplonk_fixed_aggregate_cache groups={} domain={}", fixed_aggregate_cache.len(), self.params.n);
        }

        let v: ChallengeV<_> = transcript.squeeze_challenge_scalar();

        let quotient_started = std::time::Instant::now();
        let quotient_polynomial_count = rotation_sets
            .iter()
            .map(|rotation_set| rotation_set.commitments.len())
            .sum::<usize>();
        println!(
            "EZKL_NATIVE_PHASE START stage=shplonk_quotient sets={} query_polynomials={} domain={}",
            rotation_sets.len(),
            quotient_polynomial_count,
            self.params.n,
        );
        let mut h_x = None;
        let mut power_of_v = E::Fr::ONE;
        for (set_index, rotation_set) in rotation_sets.iter().enumerate() {
            let (batch_width, available_budget, per_task_bytes, fixed_workspace_bytes) =
                shplonk_batch_budget(self.params, rotation_set.commitments.len())?;
            println!(
                "EZKL_NATIVE_PHASE START stage=shplonk_quotient_rotation_set index={} commitments={} points={} batch_width={} per_task_bytes={} fixed_workspace_bytes={} available_budget_bytes={}",
                set_index,
                rotation_set.commitments.len(),
                rotation_set.points.len(),
                batch_width,
                per_task_bytes,
                fixed_workspace_bytes,
                available_budget,
            );
            let quotient = quotient_contribution(
                rotation_set,
                set_index,
                batch_width,
                available_budget,
                per_task_bytes,
                fixed_aggregate_cache.iter().find(|aggregate| aggregate.rotation_set_index == set_index),
            )?;
            accumulate_scaled(&mut h_x, quotient, power_of_v);
            power_of_v *= *v;
        }
        let h_x = h_x.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "native SHPLONK proof has no quotient rotation sets",
            )
        })?;
        println!(
            "EZKL_NATIVE_PHASE DONE stage=shplonk_quotient sets={} query_polynomials={} elapsed_ms={:.3}",
            rotation_sets.len(),
            quotient_polynomial_count,
            quotient_started.elapsed().as_secs_f64() * 1000.0,
        );

        let h = self.params.commit(&h_x, Blind::default()).to_affine();
        transcript.write_point(h)?;
        let u: ChallengeU<_> = transcript.squeeze_challenge_scalar();

        let linearisation_contribution = |rotation_set: &RotationSetExtension<E::G1Affine>,
                                          set_index: usize,
                                          batch_width: usize,
                                          available_budget: u64,
                                          per_task_bytes: u64,
                                          fixed_aggregate: Option<&FixedAggregate<E::Fr>>|
         -> io::Result<(Polynomial<E::Fr, Coeff>, E::Fr)> {
            let mut diffs = super_point_set.clone();
            for point in rotation_set.points.iter() {
                diffs.remove(point);
            }
            let diffs = diffs.into_iter().collect::<Vec<_>>();

            // calculate difference vanishing polynomial evaluation
            let z_i = evaluate_vanishing_polynomial(&diffs[..], *u);

            // inner linearisation contributions are
            // [P_i_0(X) - r_i_0, P_i_1(X) - r_i_1, ... ] where
            // r_i_j = R_i_j(u) is the evaluation of low degree equivalent polynomial
            // where u is random evaluation point
            // define inner contributor polynomial as
            // L_i_j(X) = (P_i_j(X) - r_i_j)
            // and combine polynomials with same evaluation point set
            // L_i(X) = linear_combination(y, L_i_j(X))
            // where y is random scalar to combine inner contributors
            let mut l_x = None;
            let mut power_of_y = E::Fr::ONE;
            let total = rotation_set.commitments.len();
            for batch_start in (0..total).step_by(batch_width) {
                let batch_end = batch_start.saturating_add(batch_width).min(total);
                let batch_started = std::time::Instant::now();
                println!(
                    "EZKL_NATIVE_PHASE START stage=shplonk_linearisation_batch rotation_set={} first_commitment={} end_commitment={} total_commitments={} batch_width={} per_task_bytes={} available_budget_bytes={}",
                    set_index,
                    batch_start,
                    batch_end,
                    total,
                    batch_end - batch_start,
                    per_task_bytes,
                    available_budget,
                );
                let inner_contributions = rotation_set.commitments[batch_start..batch_end]
                    .as_ref()
                    .into_par_iter()
                    .enumerate()
                    .map(|(offset, commitment)| {
                        if fixed_aggregate.map_or(false, |aggregate| fixed_aggregate_contains(aggregate, batch_start + offset)) {
                            Ok(None)
                        } else {
                            commitment.linearisation_contribution(*u).map(Some)
                        }
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                for contribution in inner_contributions {
                    if let Some(contribution) = contribution {
                        accumulate_scaled(&mut l_x, contribution, power_of_y);
                    }
                    power_of_y *= *y;
                }
                println!(
                    "EZKL_NATIVE_PHASE DONE stage=shplonk_linearisation_batch rotation_set={} first_commitment={} end_commitment={} elapsed_ms={:.3}",
                    set_index,
                    batch_start,
                    batch_end,
                    batch_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let mut l_x = l_x.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native SHPLONK rotation set has no linearisation contributions",
                )
            })?;
            if let Some(aggregate) = fixed_aggregate {
                for (destination, source) in l_x.iter_mut().zip(aggregate.polynomial.iter()) {
                    *destination += *source;
                }
                l_x[0] -= eval_polynomial(&aggregate.low_degree_equivalent, *u);
            }

            // finally scale l_x by difference vanishing polynomial evaluation z_i
            Ok((l_x * z_i, z_i))
        };

        let linearisation_started = std::time::Instant::now();
        println!(
            "EZKL_NATIVE_PHASE START stage=shplonk_linearisation sets={} query_polynomials={} domain={}",
            rotation_sets.len(),
            quotient_polynomial_count,
            self.params.n,
        );
        let mut linearisation_accumulator = None;
        let mut z_diffs = Vec::with_capacity(rotation_sets.len());
        let mut power_of_v = E::Fr::ONE;
        for (set_index, rotation_set) in rotation_sets.iter().enumerate() {
            let (batch_width, available_budget, per_task_bytes, fixed_workspace_bytes) =
                shplonk_batch_budget(self.params, rotation_set.commitments.len())?;
            println!(
                "EZKL_NATIVE_PHASE START stage=shplonk_linearisation_rotation_set index={} commitments={} points={} batch_width={} per_task_bytes={} fixed_workspace_bytes={} available_budget_bytes={}",
                set_index,
                rotation_set.commitments.len(),
                rotation_set.points.len(),
                batch_width,
                per_task_bytes,
                fixed_workspace_bytes,
                available_budget,
            );
            let (linearisation, z_i) = linearisation_contribution(
                rotation_set,
                set_index,
                batch_width,
                available_budget,
                per_task_bytes,
                fixed_aggregate_cache.iter().find(|aggregate| aggregate.rotation_set_index == set_index),
            )?;
            accumulate_scaled(&mut linearisation_accumulator, linearisation, power_of_v);
            z_diffs.push(z_i);
            power_of_v *= *v;
        }
        let l_x = linearisation_accumulator.ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "native SHPLONK proof has no linearisation rotation sets",
            )
        })?;
        println!(
            "EZKL_NATIVE_PHASE DONE stage=shplonk_linearisation sets={} query_polynomials={} elapsed_ms={:.3}",
            rotation_sets.len(),
            quotient_polynomial_count,
            linearisation_started.elapsed().as_secs_f64() * 1000.0,
        );

        let super_point_set = super_point_set.into_iter().collect::<Vec<_>>();
        let zt_eval = evaluate_vanishing_polynomial(&super_point_set[..], *u);
        let l_x = l_x - &(h_x * zt_eval);

        // sanity check
        #[cfg(debug_assertions)]
        {
            let must_be_zero = eval_polynomial(&l_x[..], *u);
            assert_eq!(must_be_zero, E::Fr::ZERO);
        }

        let mut h_x = div_by_vanishing(l_x, &[*u]);

        // normalize coefficients by the coefficient of the first polynomial
        let z_0_diff_inv = z_diffs[0].invert().unwrap();
        for h_i in h_x.iter_mut() {
            h_i.mul_assign(z_0_diff_inv)
        }

        let h_x = Polynomial {
            values: h_x,
            mapped: None,
            _marker: PhantomData,
        };

        let h = self.params.commit(&h_x, Blind::default()).to_affine();
        transcript.write_point(h)?;

        Ok(())
    }
}
