use std::fmt::Debug;

use super::commitment::{Blind, MSM};
use crate::{
    arithmetic::eval_polynomial,
    plonk::FixedColumnSource,
    poly::{Coeff, EvaluationDomain, Polynomial},
};
use halo2curves::CurveAffine;
use group::ff::Field;
use std::convert::TryFrom;
use std::io;

/// A polynomial source used by the native prover. Most queries borrow an
/// already materialized coefficient polynomial. Native permutation queries
/// instead retain the authenticated sparse copy mapping and materialize one
/// sigma polynomial only when the opening scheme consumes it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ProverPolynomial<'com, C: CurveAffine> {
    Dense(&'com Polynomial<C::Scalar, Coeff>),
    FixedColumn {
        source: &'com FixedColumnSource<C::Scalar>,
        domain: &'com EvaluationDomain<C::Scalar>,
    },
    PermutationSigma {
        key: &'com crate::plonk::permutation::ProvingKey<C>,
        domain: &'com EvaluationDomain<C::Scalar>,
        omega_powers: &'com [C::Scalar],
        column: usize,
    },
    MvLookup {
        source: &'com crate::plonk::mv_lookup::prover::LookupPolynomialSource<C>,
        kind: crate::plonk::mv_lookup::prover::LookupPolynomialKind,
    },
}

impl<'com, C: CurveAffine> ProverPolynomial<'com, C> {
    pub(crate) fn materialize(&self) -> io::Result<Polynomial<C::Scalar, Coeff>>
    where
        C::Scalar: group::ff::WithSmallOrderMulGroup<3>,
    {
        match self {
            Self::Dense(poly) => poly.try_clone(),
            Self::FixedColumn { source, domain } => {
                if let Some(value) = source.constant_value() {
                    let domain_len = usize::try_from(domain.get_n())
                        .expect("evaluation domain length exceeds host size");
                    assert_eq!(source.len(), domain_len);
                    let mut coeff = Polynomial::<C::Scalar, Coeff>::mapped_filled(
                        domain_len,
                        C::Scalar::ZERO,
                    )?;
                    coeff[0] = value;
                    return Ok(coeff);
                }
                let lagrange = source.materialize_lagrange()?;
                domain.lagrange_to_coeff(lagrange).spill()
            }
            Self::PermutationSigma {
                key,
                domain,
                omega_powers,
                column,
            } => key.sigma_coeff(domain, *column, omega_powers),
            Self::MvLookup { source, kind } => source.materialize_coeff(*kind),
        }
    }

    fn same_source(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Dense(left), Self::Dense(right)) => std::ptr::eq(*left, *right),
            (
                Self::FixedColumn { source: left, .. },
                Self::FixedColumn { source: right, .. },
            ) => std::ptr::eq(*left, *right),
            (
                Self::PermutationSigma {
                    key: left_key,
                    column: left_column,
                    ..
                },
                Self::PermutationSigma {
                    key: right_key,
                    column: right_column,
                    ..
                },
            ) => std::ptr::eq(*left_key, *right_key) && left_column == right_column,
            (
                Self::MvLookup { source: left_source, kind: left_kind },
                Self::MvLookup { source: right_source, kind: right_kind },
            ) => std::ptr::eq(*left_source, *right_source) && left_kind == right_kind,
            _ => false,
        }
    }
}

pub trait Query<F>: Sized + Clone + Send + Sync {
    type Commitment: PartialEq + Copy + Send + Sync;
    type Eval: Clone + Default + Debug;

    fn get_point(&self) -> F;
    fn get_eval(&self) -> Self::Eval;
    fn get_commitment(&self) -> Self::Commitment;
}

/// A polynomial query at a point
#[derive(Debug, Clone, Copy)]
pub struct ProverQuery<'com, C: CurveAffine> {
    /// Point at which polynomial is queried
    pub(crate) point: C::Scalar,
    /// Coefficients of polynomial
    pub(crate) poly: ProverPolynomial<'com, C>,
    /// Deferred sources carry the evaluation already absorbed into the
    /// transcript. Dense sources compute it from their borrowed coefficients.
    pub(crate) eval: Option<C::Scalar>,
    /// Blinding factor of polynomial
    pub(crate) blind: Blind<C::Scalar>,
}

impl<'com, C> ProverQuery<'com, C>
where
    C: CurveAffine,
{
    /// Create a new prover query based on a polynomial
    pub fn new(
        point: C::Scalar,
        poly: &'com Polynomial<C::Scalar, Coeff>,
        blind: Blind<C::Scalar>,
    ) -> Self {
        ProverQuery {
            point,
            poly: ProverPolynomial::Dense(poly),
            eval: None,
            blind,
        }
    }

    pub(crate) fn new_with_eval(
        point: C::Scalar,
        poly: &'com Polynomial<C::Scalar, Coeff>,
        eval: C::Scalar,
        blind: Blind<C::Scalar>,
    ) -> Self {
        ProverQuery {
            point,
            poly: ProverPolynomial::Dense(poly),
            eval: Some(eval),
            blind,
        }
    }

    pub(crate) fn new_permutation_sigma(
        point: C::Scalar,
        key: &'com crate::plonk::permutation::ProvingKey<C>,
        domain: &'com EvaluationDomain<C::Scalar>,
        omega_powers: &'com [C::Scalar],
        column: usize,
        eval: C::Scalar,
    ) -> Self {
        ProverQuery {
            point,
            poly: ProverPolynomial::PermutationSigma {
                key,
                domain,
                omega_powers,
                column,
            },
            eval: Some(eval),
            blind: Blind::default(),
        }
    }

    pub(crate) fn new_mv_lookup(
        point: C::Scalar,
        source: &'com crate::plonk::mv_lookup::prover::LookupPolynomialSource<C>,
        kind: crate::plonk::mv_lookup::prover::LookupPolynomialKind,
        eval: C::Scalar,
        blind: Blind<C::Scalar>,
    ) -> Self {
        ProverQuery {
            point,
            poly: ProverPolynomial::MvLookup { source, kind },
            eval: Some(eval),
            blind,
        }
    }

    pub(crate) fn new_fixed_column(
        point: C::Scalar,
        source: &'com FixedColumnSource<C::Scalar>,
        domain: &'com EvaluationDomain<C::Scalar>,
        eval: C::Scalar,
    ) -> Self {
        ProverQuery {
            point,
            poly: ProverPolynomial::FixedColumn { source, domain },
            eval: Some(eval),
            blind: Blind::default(),
        }
    }
}

#[doc(hidden)]
#[derive(Copy, Clone)]
pub struct PolynomialPointer<'com, C: CurveAffine> {
    pub(crate) poly: ProverPolynomial<'com, C>,
    pub(crate) blind: Blind<C::Scalar>,
}

impl<'com, C: CurveAffine> PartialEq for PolynomialPointer<'com, C> {
    fn eq(&self, other: &Self) -> bool {
        self.poly.same_source(&other.poly)
    }
}

impl<'com, C: CurveAffine> PolynomialPointer<'com, C> {
    pub(crate) fn materialize(&self) -> io::Result<Polynomial<C::Scalar, Coeff>>
    where
        C::Scalar: group::ff::WithSmallOrderMulGroup<3>,
    {
        self.poly.materialize()
    }
}

impl<'com, C: CurveAffine> Query<C::Scalar> for ProverQuery<'com, C> {
    type Commitment = PolynomialPointer<'com, C>;
    type Eval = C::Scalar;

    fn get_point(&self) -> C::Scalar {
        self.point
    }
    fn get_eval(&self) -> Self::Eval {
        self.eval.unwrap_or_else(|| match self.poly {
            ProverPolynomial::Dense(poly) => eval_polynomial(&poly[..], self.get_point()),
            ProverPolynomial::PermutationSigma { .. } => {
                panic!("deferred permutation query is missing its transcript evaluation")
            }
            ProverPolynomial::MvLookup { .. } => {
                panic!("deferred mv-lookup query is missing its transcript evaluation")
            }
            ProverPolynomial::FixedColumn { .. } => {
                panic!("deferred fixed-column query is missing its transcript evaluation")
            }
        })
    }
    fn get_commitment(&self) -> Self::Commitment {
        PolynomialPointer {
            poly: self.poly,
            blind: self.blind,
        }
    }
}

impl<'com, C: CurveAffine, M: MSM<C>> VerifierQuery<'com, C, M> {
    /// Create a new verifier query based on a commitment
    pub fn new_commitment(commitment: &'com C, point: C::Scalar, eval: C::Scalar) -> Self {
        VerifierQuery {
            point,
            eval,
            commitment: CommitmentReference::Commitment(commitment),
        }
    }

    /// Create a new verifier query based on a linear combination of commitments
    pub fn new_msm(msm: &'com M, point: C::Scalar, eval: C::Scalar) -> VerifierQuery<'com, C, M> {
        VerifierQuery {
            point,
            eval,
            commitment: CommitmentReference::MSM(msm),
        }
    }
}

/// A polynomial query at a point
#[derive(Debug, Clone, Copy)]
pub struct VerifierQuery<'com, C: CurveAffine, M: MSM<C>> {
    /// Point at which polynomial is queried
    pub(crate) point: C::Scalar,
    /// Commitment to polynomial
    pub(crate) commitment: CommitmentReference<'com, C, M>,
    /// Evaluation of polynomial at query point
    pub(crate) eval: C::Scalar,
}

impl<'com, C, M> VerifierQuery<'com, C, M>
where
    C: CurveAffine,
    M: MSM<C>,
{
    /// Create a new verifier query based on a commitment
    pub fn new(
        point: C::Scalar,
        commitment: CommitmentReference<'com, C, M>,
        eval: C::Scalar,
    ) -> Self {
        VerifierQuery {
            point,
            commitment,
            eval,
        }
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub enum CommitmentReference<'r, C: CurveAffine, M: MSM<C>> {
    Commitment(&'r C),
    MSM(&'r M),
}

impl<'r, C: CurveAffine, M: MSM<C>> Copy for CommitmentReference<'r, C, M> {}

impl<'r, C: CurveAffine, M: MSM<C>> PartialEq for CommitmentReference<'r, C, M> {
    #![allow(ambiguous_wide_pointer_comparisons)]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (&CommitmentReference::Commitment(a), &CommitmentReference::Commitment(b)) => {
                std::ptr::eq(a, b)
            }
            (&CommitmentReference::MSM(a), &CommitmentReference::MSM(b)) => std::ptr::eq(a, b),
            _ => false,
        }
    }
}

impl<'com, C: CurveAffine, M: MSM<C>> Query<C::Scalar> for VerifierQuery<'com, C, M> {
    type Eval = C::Scalar;
    type Commitment = CommitmentReference<'com, C, M>;

    fn get_point(&self) -> C::Scalar {
        self.point
    }
    fn get_eval(&self) -> C::Scalar {
        self.eval
    }
    fn get_commitment(&self) -> Self::Commitment {
        self.commitment
    }
}
