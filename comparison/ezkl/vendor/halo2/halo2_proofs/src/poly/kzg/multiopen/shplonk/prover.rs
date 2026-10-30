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

enum QueryPolynomialSource<'a, F> {
    Borrowed(&'a Polynomial<F, Coeff>),
    Owned(Polynomial<F, Coeff>),
}

struct PreparedQueryContribution<'a, F> {
    source: Option<QueryPolynomialSource<'a, F>>,
    low_degree_equivalent: &'a [F],
    constant_correction: Option<F>,
    power: F,
}

impl<F> QueryPolynomialSource<'_, F> {
    fn as_slice(&self) -> &[F] {
        match self {
            Self::Borrowed(polynomial) => polynomial,
            Self::Owned(polynomial) => polynomial,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        accumulate_scaled, accumulate_scaled_query, accumulate_scaled_query_batch,
        Polynomial, PreparedQueryContribution, QueryPolynomialSource,
    };
    use ff::Field;
    use halo2curves::bn256::Fr;

    fn legacy_accumulate_query(
        accumulator: &mut Option<Polynomial<Fr, super::Coeff>>,
        mut polynomial: Polynomial<Fr, super::Coeff>,
        low_degree_equivalent: &[Fr],
        constant_correction: Option<Fr>,
        power: Fr,
    ) {
        for (destination, source) in polynomial.as_mut_slice()[..low_degree_equivalent.len()]
            .iter_mut()
            .zip(low_degree_equivalent.iter())
        {
            *destination -= *source;
        }
        if let Some(correction) = constant_correction {
            polynomial[0] -= correction;
        }
        accumulate_scaled(accumulator, polynomial, power);
    }

    #[test]
    fn fused_query_accumulation_matches_materialize_scale_and_add() {
        let values = vec![
            vec![Fr::from(3), Fr::from(5), Fr::from(7), Fr::from(11), Fr::from(13)],
            vec![Fr::from(17), Fr::from(19), Fr::from(23), Fr::from(29), Fr::from(31), Fr::from(37), Fr::from(41), Fr::from(43)],
            vec![Fr::from(47), Fr::from(53), Fr::from(59), Fr::from(61), Fr::from(67), Fr::from(71), Fr::from(73), Fr::from(79)],
            vec![Fr::from(83), Fr::from(89), Fr::from(97), Fr::from(101), Fr::from(103), Fr::from(107), Fr::from(109), Fr::from(113)],
            vec![Fr::from(127), Fr::from(131), Fr::from(137), Fr::from(139), Fr::from(149), Fr::from(151), Fr::from(157), Fr::from(163)],
            vec![Fr::from(167), Fr::from(173), Fr::from(179), Fr::from(181), Fr::from(191), Fr::from(193), Fr::from(197), Fr::from(199)],
            vec![Fr::from(211), Fr::from(223), Fr::from(227), Fr::from(229), Fr::from(233), Fr::from(239), Fr::from(241), Fr::from(251)],
            vec![Fr::from(257), Fr::from(263), Fr::from(269), Fr::from(271), Fr::from(277), Fr::from(281), Fr::from(283), Fr::from(293)],
        ];
        let polynomials = values
            .into_iter()
            .map(Polynomial::<Fr, super::Coeff>::from_vec)
            .collect::<Vec<_>>();
        let low_degree_equivalents = vec![
            vec![Fr::from(2), Fr::from(4)],
            vec![Fr::from(6)],
            vec![],
            vec![Fr::from(8), Fr::from(10), Fr::from(12)],
            vec![Fr::from(14)],
            vec![Fr::from(16), Fr::from(18)],
            vec![],
            vec![Fr::from(20), Fr::from(22), Fr::from(24)],
        ];
        let powers = [Fr::ONE, Fr::from(5), Fr::from(25), Fr::from(125)];
        let constants = [
            None,
            Some(Fr::from(3)),
            Some(Fr::from(9)),
            None,
            Some(Fr::from(15)),
            Some(Fr::from(21)),
            None,
            Some(Fr::from(27)),
        ];
        // Borrowed Dense and owned deferred materializations (fixed, sigma,
        // and lookup sources) share one coefficient accumulator. Splitting
        // each set into batches exercises the original challenge ranks.
        let owned_sources = [false, true, true, true, true, true, true, true];
        let rotation_sets = [&[0usize, 1, 2, 3][..], &[4usize, 5, 6, 7][..]];

        for rotation_set in rotation_sets {
            let mut legacy = None;
            let mut fused = None;
            let mut tiled = None;
            for batch in rotation_set.chunks(2) {
                let mut prepared = Vec::with_capacity(batch.len());
                for &index in batch {
                    let power = powers[index % 4];
                    let source = polynomials[index].try_clone().unwrap();
                    legacy_accumulate_query(
                        &mut legacy,
                        source,
                        &low_degree_equivalents[index],
                        constants[index],
                        power,
                    );

                    let source = if owned_sources[index] {
                        QueryPolynomialSource::Owned(polynomials[index].try_clone().unwrap())
                    } else {
                        QueryPolynomialSource::Borrowed(&polynomials[index])
                    };
                    accumulate_scaled_query(
                        &mut fused,
                        source,
                        &low_degree_equivalents[index],
                        constants[index],
                        power,
                    )
                    .unwrap();

                    // Leave cached positions empty while retaining their
                    // original challenge rank in the next contribution.
                    let cached = index % 4 == 2;
                    let source = if cached {
                        None
                    } else if owned_sources[index] {
                        Some(QueryPolynomialSource::Owned(
                            polynomials[index].try_clone().unwrap(),
                        ))
                    } else {
                        Some(QueryPolynomialSource::Borrowed(&polynomials[index]))
                    };
                    prepared.push(PreparedQueryContribution {
                        source,
                        low_degree_equivalent: &low_degree_equivalents[index],
                        constant_correction: constants[index],
                        power,
                    });
                }
                accumulate_scaled_query_batch(&mut tiled, prepared).unwrap();
            }
            assert_eq!(&legacy.unwrap()[..], &fused.unwrap()[..]);
            let mut legacy_with_cached = None;
            for &index in rotation_set {
                if index % 4 != 2 {
                    legacy_accumulate_query(
                        &mut legacy_with_cached,
                        polynomials[index].try_clone().unwrap(),
                        &low_degree_equivalents[index],
                        constants[index],
                        powers[index % 4],
                    );
                }
            }
            assert_eq!(&legacy_with_cached.unwrap()[..], &tiled.unwrap()[..]);
        }

        // A zero challenge keeps the first term and zeros every later rank.
        let mut legacy_zero_challenge = None;
        let mut fused_zero_challenge = None;
        for index in 0..4 {
            let power = if index == 0 { Fr::ONE } else { Fr::ZERO };
            legacy_accumulate_query(
                &mut legacy_zero_challenge,
                polynomials[index].try_clone().unwrap(),
                &low_degree_equivalents[index],
                constants[index],
                power,
            );
            let source = if owned_sources[index] {
                QueryPolynomialSource::Owned(polynomials[index].try_clone().unwrap())
            } else {
                QueryPolynomialSource::Borrowed(&polynomials[index])
            };
            accumulate_scaled_query(
                &mut fused_zero_challenge,
                source,
                &low_degree_equivalents[index],
                constants[index],
                power,
            )
            .unwrap();
        }
        assert_eq!(
            &legacy_zero_challenge.unwrap()[..],
            &fused_zero_challenge.unwrap()[..]
        );

        let constant = Polynomial::<Fr, super::Coeff>::from_vec(vec![Fr::from(17)]);
        let zero = Polynomial::<Fr, super::Coeff>::from_vec(vec![Fr::ZERO, Fr::ZERO, Fr::ZERO]);
        let constant_equivalent = [Fr::from(5)];
        let zero_equivalent = [Fr::ZERO, Fr::ZERO];
        let mut legacy_short = None;
        let mut fused_short = None;
        legacy_accumulate_query(
            &mut legacy_short,
            constant.try_clone().unwrap(),
            &constant_equivalent,
            None,
            Fr::ONE,
        );
        legacy_accumulate_query(
            &mut legacy_short,
            zero.try_clone().unwrap(),
            &zero_equivalent,
            None,
            Fr::from(5),
        );
        accumulate_scaled_query(
            &mut fused_short,
            QueryPolynomialSource::Borrowed(&constant),
            &constant_equivalent,
            None,
            Fr::ONE,
        )
        .unwrap();
        accumulate_scaled_query(
            &mut fused_short,
            QueryPolynomialSource::Owned(zero.try_clone().unwrap()),
            &zero_equivalent,
            None,
            Fr::from(5),
        )
        .unwrap();
        assert_eq!(&legacy_short.unwrap()[..], &fused_short.unwrap()[..]);

        // The low-degree correction keeps the former checked slice boundary.
        let invalid_source = Polynomial::<Fr, super::Coeff>::from_vec(vec![Fr::ONE, Fr::ONE]);
        let invalid_equivalent = [Fr::ONE, Fr::ONE, Fr::ONE];
        let legacy_panics = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut accumulator = None;
            legacy_accumulate_query(
                &mut accumulator,
                invalid_source.try_clone().unwrap(),
                &invalid_equivalent,
                None,
                Fr::ONE,
            );
        }));
        let fused_panics = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut accumulator = None;
            accumulate_scaled_query(
                &mut accumulator,
                QueryPolynomialSource::Borrowed(&invalid_source),
                &invalid_equivalent,
                None,
                Fr::ONE,
            )
            .unwrap();
        }));
        assert!(legacy_panics.is_err());
        assert!(fused_panics.is_err());
    }

    #[test]
    fn batch_accumulation_preserves_lengths_empty_sources_and_validation() {
        let long = Polynomial::<Fr, super::Coeff>::from_vec(
            (1..=9).map(Fr::from).collect::<Vec<_>>(),
        );
        let short = Polynomial::<Fr, super::Coeff>::from_vec(
            vec![Fr::from(21), Fr::from(22), Fr::from(23)],
        );
        let empty = Polynomial::<Fr, super::Coeff>::from_vec(Vec::new());
        let long_equivalent = [Fr::from(2), Fr::from(3)];
        let short_equivalent = [Fr::from(4)];
        let challenge = Fr::from(5);

        // The first contribution fixes output length. Later sources may be
        // shorter or longer; coefficient-wise zip behavior clips each source.
        let mut expected = long.try_clone().unwrap();
        for (index, coefficient) in expected.as_mut_slice().iter_mut().enumerate() {
            if index < long_equivalent.len() {
                *coefficient -= long_equivalent[index];
            }
            if index == 0 {
                *coefficient -= Fr::from(7);
            }
        }
        for (index, (destination, source)) in expected
            .as_mut_slice()
            .iter_mut()
            .zip(short.as_slice().iter())
            .enumerate()
        {
            let mut contribution = *source;
            if index < short_equivalent.len() {
                contribution -= short_equivalent[index];
            }
            if index == 0 {
                contribution -= Fr::from(11);
            }
            contribution *= challenge;
            *destination += contribution;
        }
        let mut tiled = None;
        accumulate_scaled_query_batch(
            &mut tiled,
            vec![
                PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Borrowed(&long)),
                    low_degree_equivalent: &long_equivalent,
                    constant_correction: Some(Fr::from(7)),
                    power: Fr::ONE,
                },
                PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Owned(short.try_clone().unwrap())),
                    low_degree_equivalent: &short_equivalent,
                    constant_correction: Some(Fr::from(11)),
                    power: challenge,
                },
            ],
        )
        .unwrap();
        assert_eq!(&expected[..], &tiled.unwrap()[..]);

        // The first term keeps rank zero; a zero challenge zeros every later
        // query rank without changing the batch's source order.
        let mut legacy_zero = None;
        legacy_accumulate_query(
            &mut legacy_zero,
            long.try_clone().unwrap(),
            &long_equivalent,
            Some(Fr::from(7)),
            Fr::ONE,
        );
        legacy_accumulate_query(
            &mut legacy_zero,
            long.try_clone().unwrap(),
            &long_equivalent,
            Some(Fr::from(11)),
            Fr::ZERO,
        );
        let mut tiled_zero = None;
        accumulate_scaled_query_batch(
            &mut tiled_zero,
            vec![
                PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Borrowed(&long)),
                    low_degree_equivalent: &long_equivalent,
                    constant_correction: Some(Fr::from(7)),
                    power: Fr::ONE,
                },
                PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Borrowed(&long)),
                    low_degree_equivalent: &long_equivalent,
                    constant_correction: Some(Fr::from(11)),
                    power: Fr::ZERO,
                },
            ],
        )
        .unwrap();
        assert_eq!(&legacy_zero.unwrap()[..], &tiled_zero.unwrap()[..]);

        // An empty first source still fixes the accumulator length at zero.
        let mut legacy_empty = None;
        legacy_accumulate_query(
            &mut legacy_empty,
            empty.try_clone().unwrap(),
            &[],
            None,
            Fr::ONE,
        );
        legacy_accumulate_query(
            &mut legacy_empty,
            long.try_clone().unwrap(),
            &[],
            None,
            challenge,
        );
        let mut tiled_empty = None;
        accumulate_scaled_query_batch(
            &mut tiled_empty,
            vec![
                PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Borrowed(&empty)),
                    low_degree_equivalent: &[],
                    constant_correction: None,
                    power: Fr::ONE,
                },
                PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Borrowed(&long)),
                    low_degree_equivalent: &[],
                    constant_correction: None,
                    power: challenge,
                },
            ],
        )
        .unwrap();
        assert_eq!(
            &legacy_empty.as_ref().unwrap()[..],
            &tiled_empty.as_ref().unwrap()[..]
        );

        // An all-cached batch must leave an existing accumulator untouched.
        let before = tiled_empty.as_ref().unwrap().try_clone().unwrap();
        accumulate_scaled_query_batch(
            &mut tiled_empty,
            vec![
                PreparedQueryContribution {
                    source: None,
                    low_degree_equivalent: &[],
                    constant_correction: None,
                    power: Fr::ONE,
                },
                PreparedQueryContribution {
                    source: None,
                    low_degree_equivalent: &[],
                    constant_correction: None,
                    power: challenge,
                },
            ],
        )
        .unwrap();
        assert_eq!(&before[..], &tiled_empty.as_ref().unwrap()[..]);

        // Validate every source boundary before changing an existing output.
        let mut unchanged = Some(long.try_clone().unwrap());
        let before = unchanged.as_ref().unwrap().try_clone().unwrap();
        let invalid_correction = [Fr::ONE, Fr::ONE, Fr::ONE, Fr::ONE];
        let panics = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = accumulate_scaled_query_batch(
                &mut unchanged,
                vec![PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Borrowed(&short)),
                    low_degree_equivalent: &invalid_correction,
                    constant_correction: None,
                    power: Fr::ONE,
                }],
            );
        }));
        assert!(panics.is_err());
        assert_eq!(&before[..], &unchanged.unwrap()[..]);

        let mut unchanged = Some(long.try_clone().unwrap());
        let before = unchanged.as_ref().unwrap().try_clone().unwrap();
        let empty_source = Polynomial::<Fr, super::Coeff>::from_vec(Vec::new());
        let panics = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = accumulate_scaled_query_batch(
                &mut unchanged,
                vec![PreparedQueryContribution {
                    source: Some(QueryPolynomialSource::Borrowed(&empty_source)),
                    low_degree_equivalent: &[],
                    constant_correction: Some(Fr::ONE),
                    power: Fr::ONE,
                }],
            );
        }));
        assert!(panics.is_err());
        assert_eq!(&before[..], &unchanged.unwrap()[..]);
    }
}

fn query_polynomial_source<'com, C: CurveAffine>(
    commitment: &CommitmentExtension<'com, C>,
    own_dense: bool,
) -> io::Result<QueryPolynomialSource<'com, C::Scalar>>
where
    C::Scalar: WithSmallOrderMulGroup<3>,
{
    let polynomial = commitment.commitment.get();
    if own_dense {
        return polynomial
            .materialize()
            .map(QueryPolynomialSource::Owned);
    }

    match polynomial.poly {
        ProverPolynomial::Dense(polynomial) => Ok(QueryPolynomialSource::Borrowed(polynomial)),
        _ => polynomial
            .materialize()
            .map(QueryPolynomialSource::Owned),
    }
}

#[cfg(test)]
fn apply_scaled_correction<F: Field>(
    polynomial: &mut Polynomial<F, Coeff>,
    low_degree_equivalent: &[F],
    constant_correction: Option<F>,
    power: F,
) {
    // Keep the same range contract as the former low-degree subtraction.
    let _ = &polynomial.as_slice()[..low_degree_equivalent.len()];
    if constant_correction.is_some() {
        let _ = polynomial[0];
    }

    if power == F::ZERO {
        parallelize(polynomial.as_mut_slice(), |chunk, _| chunk.fill(F::ZERO));
        return;
    }

    parallelize(polynomial.as_mut_slice(), |chunk, start| {
        for (offset, coefficient) in chunk.iter_mut().enumerate() {
            let index = start + offset;
            if index < low_degree_equivalent.len() {
                *coefficient -= low_degree_equivalent[index];
            }
            if index == 0 {
                if let Some(correction) = constant_correction {
                    *coefficient -= correction;
                }
            }
            if power != F::ONE {
                *coefficient *= power;
            }
        }
    });
}

#[cfg(test)]
fn accumulate_scaled_query<F: Field>(
    accumulator: &mut Option<Polynomial<F, Coeff>>,
    source: QueryPolynomialSource<'_, F>,
    low_degree_equivalent: &[F],
    constant_correction: Option<F>,
    power: F,
) -> io::Result<()> {
    let Some(accumulated) = accumulator.as_mut() else {
        let mut polynomial = match source {
            QueryPolynomialSource::Borrowed(polynomial) => polynomial.try_clone()?,
            QueryPolynomialSource::Owned(polynomial) => polynomial,
        };
        apply_scaled_correction(
            &mut polynomial,
            low_degree_equivalent,
            constant_correction,
            power,
        );
        *accumulator = Some(polynomial);
        return Ok(());
    };

    let source = source.as_slice();
    // Validate the same low-degree range before touching the accumulator.
    let _ = &source[..low_degree_equivalent.len()];
    if constant_correction.is_some() {
        let _ = source[0];
    }

    parallelize(accumulated.as_mut_slice(), |chunk, start| {
        // Matching the former Polynomial addition's start-indexed slice and
        // zip behavior preserves the established coefficient-length contract.
        let source = &source[start..];
        for (offset, (destination, source_coefficient)) in
            chunk.iter_mut().zip(source.iter()).enumerate()
        {
            if power == F::ZERO {
                continue;
            }
            let index = start + offset;
            let mut contribution = *source_coefficient;
            if index < low_degree_equivalent.len() {
                contribution -= low_degree_equivalent[index];
            }
            if index == 0 {
                if let Some(correction) = constant_correction {
                    contribution -= correction;
                }
            }
            if power != F::ONE {
                contribution *= power;
            }
            *destination += contribution;
        }
    });
    Ok(())
}

fn accumulate_scaled_query_batch<F: Field>(
    accumulator: &mut Option<Polynomial<F, Coeff>>,
    mut contributions: Vec<PreparedQueryContribution<'_, F>>,
) -> io::Result<()> {
    let mut first_contribution = None;
    // Validate all prepared source boundaries in original query order before
    // modifying the destination. Cached entries still occupy their rank.
    for (index, contribution) in contributions.iter().enumerate() {
        if let Some(source) = contribution.source.as_ref() {
            let source = source.as_slice();
            let _ = &source[..contribution.low_degree_equivalent.len()];
            if contribution.constant_correction.is_some() {
                let _ = source[0];
            }
            if first_contribution.is_none() {
                first_contribution = Some(index);
            }
        }
    }

    let Some(first_contribution) = first_contribution else {
        return Ok(());
    };

    let had_accumulator = accumulator.is_some();
    let mut output = if let Some(output) = accumulator.take() {
        output
    } else {
        let first = &mut contributions[first_contribution];
        match first
            .source
            .take()
            .expect("validated first SHPLONK contribution must have a source")
        {
            QueryPolynomialSource::Borrowed(polynomial) => polynomial.try_clone()?,
            QueryPolynomialSource::Owned(polynomial) => polynomial,
        }
    };

    parallelize(output.as_mut_slice(), |chunk, start| {
        for (offset, destination) in chunk.iter_mut().enumerate() {
            let coefficient_index = start + offset;
            if !had_accumulator {
                // The first source is already in `output`; initialize its
                // corrected, weighted value before folding later sources.
                let first = &contributions[first_contribution];
                if first.power == F::ZERO {
                    *destination = F::ZERO;
                } else {
                    if coefficient_index < first.low_degree_equivalent.len() {
                        *destination -= first.low_degree_equivalent[coefficient_index];
                    }
                    if coefficient_index == 0 {
                        if let Some(correction) = first.constant_correction {
                            *destination -= correction;
                        }
                    }
                    if first.power != F::ONE {
                        *destination *= first.power;
                    }
                }
            }

            for (index, contribution) in contributions.iter().enumerate() {
                if !had_accumulator && index == first_contribution {
                    continue;
                }
                if contribution.power == F::ZERO {
                    continue;
                }
                let Some(source) = contribution.source.as_ref() else {
                    continue;
                };
                let Some(mut value) = source.as_slice().get(coefficient_index).copied() else {
                    continue;
                };
                if coefficient_index < contribution.low_degree_equivalent.len() {
                    value -= contribution.low_degree_equivalent[coefficient_index];
                }
                if coefficient_index == 0 {
                    if let Some(correction) = contribution.constant_correction {
                        value -= correction;
                    }
                }
                if contribution.power != F::ONE {
                    value *= contribution.power;
                }
                *destination += value;
            }
        }
    });
    *accumulator = Some(output);
    Ok(())
}

fn shplonk_batch_budget<E: Engine>(
    params: &ParamsKZG<E>,
    requested_width: usize,
) -> io::Result<(usize, u64, u64, u64)> {
    let field_bytes = std::mem::size_of::<E::Fr>() as u128;
    let polynomial_header = std::mem::size_of::<Polynomial<E::Fr, Coeff>>() as u128;
    let domain = u128::from(params.n);
    // Conservatively bound one opened source and one owned polynomial per
    // task. Dense query sources are borrowed after initializing the output
    // accumulator, so this preserves the existing cap with extra headroom.
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
            // source views and owned deferred materializations at a time.
            let mut n_x = None;
            let mut power_of_y = E::Fr::ONE;
            let total = rotation_set.commitments.len();
            // The first non-cached contribution becomes the output accumulator;
            // only that dense input needs an owned copy.
            let first_contribution = rotation_set.commitments.iter().enumerate().find_map(|(index, _)| {
                if fixed_aggregate.map_or(false, |aggregate| fixed_aggregate_contains(aggregate, index)) {
                    None
                } else {
                    Some(index)
                }
            });
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
                        let index = batch_start + offset;
                        if fixed_aggregate.map_or(false, |aggregate| fixed_aggregate_contains(aggregate, index)) {
                            Ok(None)
                        } else {
                            let source = query_polynomial_source(
                                commitment,
                                first_contribution == Some(index),
                            )?;
                            let _ = &source.as_slice()[..commitment.low_degree_equivalent.len()];
                            Ok(Some(source))
                        }
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut prepared = Vec::with_capacity(numerators.len());
                for (offset, numerator) in numerators.into_iter().enumerate() {
                    let commitment = &rotation_set.commitments[batch_start + offset];
                    prepared.push(PreparedQueryContribution {
                        source: numerator,
                        low_degree_equivalent: commitment.low_degree_equivalent.as_slice(),
                        constant_correction: None,
                        power: power_of_y,
                    });
                    power_of_y *= *y;
                }
                accumulate_scaled_query_batch(&mut n_x, prepared)?;
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
            // As for the quotient sum, only the first non-cached dense source
            // is copied into the output accumulator.
            let first_contribution = rotation_set.commitments.iter().enumerate().find_map(|(index, _)| {
                if fixed_aggregate.map_or(false, |aggregate| fixed_aggregate_contains(aggregate, index)) {
                    None
                } else {
                    Some(index)
                }
            });
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
                        let index = batch_start + offset;
                        if fixed_aggregate.map_or(false, |aggregate| fixed_aggregate_contains(aggregate, index)) {
                            Ok(None)
                        } else {
                            let source = query_polynomial_source(
                                commitment,
                                first_contribution == Some(index),
                            )?;
                            let _ = source.as_slice()[0];
                            Ok(Some(source))
                        }
                    })
                    .collect::<io::Result<Vec<_>>>()?;
                let mut prepared = Vec::with_capacity(inner_contributions.len());
                for (offset, contribution) in inner_contributions.into_iter().enumerate() {
                    let commitment = &rotation_set.commitments[batch_start + offset];
                    let constant_correction = contribution.as_ref().map(|_| {
                        eval_polynomial(&commitment.low_degree_equivalent[..], *u)
                    });
                    prepared.push(PreparedQueryContribution {
                        source: contribution,
                        low_degree_equivalent: &[],
                        constant_correction,
                        power: power_of_y,
                    });
                    power_of_y *= *y;
                }
                accumulate_scaled_query_batch(&mut l_x, prepared)?;
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
