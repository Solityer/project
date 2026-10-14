use super::{construct_intermediate_sets, ChallengeV, Query};
use crate::arithmetic::{kate_division, powers};
use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::ParamsProver;
use crate::poly::commitment::Prover;
use crate::poly::kzg::commitment::{KZGCommitmentScheme, ParamsKZG};
use crate::poly::query::ProverQuery;
use crate::poly::{commitment::Blind, Coeff, Polynomial};
use crate::transcript::{EncodedChallenge, TranscriptWrite};

use group::Curve;
use halo2curves::pairing::Engine;
use halo2curves::CurveExt;
use group::ff::{Field, WithSmallOrderMulGroup};
use rand_core::RngCore;
use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;

/// Concrete KZG prover with GWC variant
#[derive(Debug)]
pub struct ProverGWC<'params, E: Engine> {
    params: &'params ParamsKZG<E>,
}

/// Create a multi-opening proof
impl<'params, E: Engine + Debug> Prover<'params, KZGCommitmentScheme<E>> for ProverGWC<'params, E>
where
    E::Fr: WithSmallOrderMulGroup<3>,
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
        let v: ChallengeV<_> = transcript.squeeze_challenge_scalar();
        let commitment_data = construct_intermediate_sets(queries);

        for commitment_at_a_point in commitment_data.iter() {
            let z = commitment_at_a_point.point;
            let mut poly_batch: Option<Polynomial<E::Fr, Coeff>> = None;
            let mut eval_batch = E::Fr::ZERO;
            for (query, power_of_v) in commitment_at_a_point.queries.iter().zip(powers(*v)) {
                assert_eq!(query.get_point(), z);

                let poly = query.get_commitment().materialize()?;
                let eval = query.get_eval() * power_of_v;
                let poly = poly * power_of_v;
                if let Some(accumulator) = poly_batch.as_mut() {
                    if accumulator.len() != poly.len() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "GWC query polynomials have inconsistent coefficient lengths",
                        ));
                    }
                    for (accumulator, coefficient) in accumulator.iter_mut().zip(poly.iter()) {
                        *accumulator += coefficient;
                    }
                } else {
                    poly_batch = Some(poly);
                }
                eval_batch += eval;
            }
            let poly_batch = poly_batch.ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "GWC point set has no queries")
            })?;

            let poly_batch = &poly_batch - eval_batch;
            let witness_poly = Polynomial {
                values: kate_division(&poly_batch[..], z),
                mapped: None,
                _marker: PhantomData,
            };
            let w = self
                .params
                .commit(&witness_poly, Blind::default())
                .to_affine();

            transcript.write_point(w)?;
        }
        Ok(())
    }
}
