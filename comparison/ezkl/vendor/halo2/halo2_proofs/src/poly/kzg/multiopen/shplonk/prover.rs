use super::{
    construct_intermediate_sets, ChallengeU, ChallengeV, ChallengeY, Commitment, RotationSet,
};
use crate::arithmetic::{
    eval_polynomial, evaluate_vanishing_polynomial, kate_division, lagrange_interpolate,
    parallelize, CurveAffine,
};
use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::{Blind, ParamsProver, Prover};
use crate::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use crate::poly::query::{PolynomialPointer, ProverQuery};
use crate::poly::{Basis, Coeff, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use ff::Field;
use group::Curve;
use halo2curves::pairing::Engine;
use halo2curves::CurveExt;
use maybe_rayon::iter::ParallelIterator;
use maybe_rayon::prelude::IntoParallelIterator;
use rand_core::RngCore;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
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
        let p_x = self.commitment.get().poly;
        let r_eval = eval_polynomial(&self.low_degree_equivalent[..], u);
        let mut contribution = p_x.try_clone()?;
        contribution[0] -= r_eval;
        Ok(contribution)
    }

    fn quotient_contribution(&self) -> io::Result<Polynomial<C::Scalar, Coeff>> {
        let len = self.low_degree_equivalent.len();
        let mut p_x = self.commitment.get().poly.try_clone()?;
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

pub(super) fn native_shplonk_resource_layout<C: CurveAffine>() -> (usize, usize) {
    (
        std::mem::size_of::<CommitmentExtension<'static, C>>(),
        std::mem::size_of::<RotationSetExtension<'static, C>>(),
    )
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
                                     per_task_bytes: u64|
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
                    .map(|commitment| commitment.quotient_contribution())
                    .collect::<io::Result<Vec<_>>>()?;
                for numerator in numerators {
                    accumulate_scaled(&mut n_x, numerator, power_of_y);
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
            let n_x = n_x.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native SHPLONK rotation set has no quotient contributions",
                )
            })?;

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
                                          per_task_bytes: u64|
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
                    .map(|commitment| commitment.linearisation_contribution(*u))
                    .collect::<io::Result<Vec<_>>>()?;
                for contribution in inner_contributions {
                    accumulate_scaled(&mut l_x, contribution, power_of_y);
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
            let l_x = l_x.ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native SHPLONK rotation set has no linearisation contributions",
                )
            })?;

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
