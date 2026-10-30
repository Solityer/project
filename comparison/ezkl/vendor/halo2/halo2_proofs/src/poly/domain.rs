//! Contains utilities for performing polynomial arithmetic over an evaluation
//! domain that is of a suitable size for the application.

use crate::{
    arithmetic::{best_fft, best_fft_cpu, gpu_coset_fft_is_eligible, parallelize, try_gpu_coset_fft_if_eligible, try_gpu_coset_fft_if_eligible_with_factors, try_gpu_fft_if_eligible},
    fft::recursive::FFTData,
    plonk::Assigned,
};

use super::{Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation};

use group::ff::{BatchInvert, Field, WithSmallOrderMulGroup};
use std::{
    collections::HashMap,
    io,
    marker::PhantomData,
    sync::{Arc, OnceLock},
};

/// This structure contains precomputed constants and other details needed for
/// performing operations on an evaluation domain of size $2^k$ and an extended
/// domain of size $2^{k} * j$ with $j \neq 0$.
#[derive(Clone, Debug)]
pub struct EvaluationDomain<F: Field> {
    n: u64,
    k: u32,
    extended_k: u32,
    omega: F,
    omega_inv: F,
    extended_omega: F,
    extended_omega_inv: F,
    g_coset: F,
    g_coset_inv: F,
    quotient_poly_degree: u64,
    ifft_divisor: F,
    extended_ifft_divisor: F,
    t_evaluations: Vec<F>,
    barycentric_weight: F,

    // Recursive twiddles are initialized only for lengths actually requested.
    // Cloned domains share the immutable cached arrays.
    fft_data: HashMap<usize, Arc<OnceLock<FFTData<F>>>>,
}

impl<F: WithSmallOrderMulGroup<3>> EvaluationDomain<F> {
    /// This constructs a new evaluation domain object based on the provided
    /// values $j, k$.
    pub fn new(j: u32, k: u32) -> Self {
        // quotient_poly_degree * params.n - 1 is the degree of the quotient polynomial
        let quotient_poly_degree = (j - 1) as u64;

        // n = 2^k
        let n = 1u64 << k;

        // We need to work within an extended domain, not params.k but params.k + i
        // for some integer i such that 2^(params.k + i) is sufficiently large to
        // describe the quotient polynomial.
        let mut extended_k = k;
        while (1 << extended_k) < (n * quotient_poly_degree) {
            extended_k += 1;
        }

        // ensure extended_k <= S
        assert!(
            extended_k <= F::S,
            "extended_k ({}, k={}, j={}) with degree ({}) must be <= S ({}), the size of the field",
            extended_k,
            k,
            j,
            n * quotient_poly_degree,
            F::S
        );

        let mut extended_omega = F::ROOT_OF_UNITY;

        // Get extended_omega, the 2^{extended_k}'th root of unity
        // The loop computes extended_omega = omega^{2 ^ (S - extended_k)}
        // Notice that extended_omega ^ {2 ^ extended_k} = omega ^ {2^S} = 1.
        for _ in extended_k..F::S {
            extended_omega = extended_omega.square();
        }
        let extended_omega = extended_omega;

        // Get omega, the 2^{k}'th root of unity (i.e. n'th root of unity)
        // The loop computes omega = extended_omega ^ {2 ^ (extended_k - k)}
        //           = (omega^{2 ^ (S - extended_k)})  ^ {2 ^ (extended_k - k)}
        //           = omega ^ {2 ^ (S - k)}.
        // Notice that omega ^ {2^k} = omega ^ {2^S} = 1.
        let mut omegas = Vec::with_capacity((extended_k - k + 1) as usize);
        let mut omega = extended_omega;
        omegas.push(omega);
        for _ in k..extended_k {
            omega = omega.square();
            omegas.push(omega);
        }
        let omega = omega;
        omegas.reverse();
        let mut omegas_inv = omegas.clone(); // Inversion computed later

        // We use zeta here because we know it generates a coset, and it's available
        // already.
        // The coset evaluation domain is:
        // zeta {1, extended_omega, extended_omega^2, ..., extended_omega^{(2^extended_k) - 1}}
        let g_coset = F::ZETA;
        let g_coset_inv = g_coset.square();

        let mut t_evaluations = Vec::with_capacity(1 << (extended_k - k));
        {
            // Compute the evaluations of t(X) = X^n - 1 in the coset evaluation domain.
            // We don't have to compute all of them, because it will repeat.
            let orig = F::ZETA.pow_vartime([n]);
            let step = extended_omega.pow_vartime([n]);
            let mut cur = orig;
            loop {
                t_evaluations.push(cur);
                cur *= &step;
                if cur == orig {
                    break;
                }
            }
            assert_eq!(t_evaluations.len(), 1 << (extended_k - k));

            // Subtract 1 from each to give us t_evaluations[i] = t(zeta * extended_omega^i)
            for coeff in &mut t_evaluations {
                *coeff -= &F::ONE;
            }

            // Invert, because we're dividing by this polynomial.
            // We invert in a batch, below.
        }

        let mut ifft_divisor = F::from(1 << k); // Inversion computed later
        let mut extended_ifft_divisor = F::from(1 << extended_k); // Inversion computed later

        // The barycentric weight of 1 over the evaluation domain
        // 1 / \prod_{i != 0} (1 - omega^i)
        let mut barycentric_weight = F::from(n); // Inversion computed later

        // Compute batch inversion
        t_evaluations
            .iter_mut()
            .chain(Some(&mut ifft_divisor))
            .chain(Some(&mut extended_ifft_divisor))
            .chain(Some(&mut barycentric_weight))
            .chain(&mut omegas_inv)
            .batch_invert();

        let omega_inv = omegas_inv[0];
        let extended_omega_inv = *omegas_inv.last().unwrap();
        let fft_data = (k..=extended_k)
            .map(|log_n| (1usize << log_n, Arc::new(OnceLock::new())))
            .collect::<HashMap<usize, Arc<OnceLock<FFTData<F>>>>>();

        EvaluationDomain {
            n,
            k,
            extended_k,
            omega,
            omega_inv,
            extended_omega,
            extended_omega_inv,
            g_coset,
            g_coset_inv,
            quotient_poly_degree,
            ifft_divisor,
            extended_ifft_divisor,
            t_evaluations,
            barycentric_weight,
            fft_data,
        }
    }

    /// Obtains a polynomial in Lagrange form when given a vector of Lagrange
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn lagrange_from_vec(&self, values: Vec<F>) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial {
            values,
            mapped: None,
            _marker: PhantomData,
        }
    }

    ///
    pub fn lagrange_assigned_from_vec(
        &self,
        values: Vec<Assigned<F>>,
    ) -> Polynomial<Assigned<F>, LagrangeCoeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial {
            values,
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Obtains a polynomial in coefficient form when given a vector of
    /// coefficients of size `n`; panics if the provided vector is the wrong
    /// length.
    pub fn coeff_from_vec(&self, values: Vec<F>) -> Polynomial<F, Coeff> {
        assert_eq!(values.len(), self.n as usize);

        Polynomial {
            values,
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Obtains a polynomial in ExtendedLagrange form when given a vector of
    /// Lagrange polynomials with total size `extended_n`; panics if the
    /// provided vector is the wrong length.
    pub fn lagrange_vec_to_extended(
        &self,
        values: Vec<Polynomial<F, LagrangeCoeff>>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(values.len(), self.extended_len() >> self.k);
        assert_eq!(values[0].len(), self.n as usize);

        // transpose the values in parallel
        let mut transposed = vec![vec![F::ZERO; values.len()]; self.n as usize];
        values.into_iter().enumerate().for_each(|(i, p)| {
            parallelize(&mut transposed, |transposed, start| {
                for (transposed, p) in transposed.iter_mut().zip(p[start..].iter()) {
                    transposed[i] = *p;
                }
            });
        });

        Polynomial {
            values: transposed.into_iter().flatten().collect(),
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Obtains a polynomial in ExtendedLagrange form when given a vector of
    /// Lagrange polynomials with total size `extended_n`; panics if the
    /// provided vector is the wrong length.
    pub fn extended_from_lagrange_vec(
        &self,
        values: Vec<Polynomial<F, LagrangeCoeff>>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(values.len(), self.extended_len() >> self.k);
        assert_eq!(values[0].len(), self.n as usize);

        // transpose the values in parallel
        let mut transposed = vec![vec![F::ZERO; values.len()]; self.n as usize];
        values.into_iter().enumerate().for_each(|(i, p)| {
            parallelize(&mut transposed, |transposed, start| {
                for (transposed, p) in transposed.iter_mut().zip(p[start..].iter()) {
                    transposed[i] = *p;
                }
            });
        });

        Polynomial {
            values: transposed.into_iter().flatten().collect(),
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the coefficient basis
    pub fn empty_coeff(&self) -> Polynomial<F, Coeff> {
        Polynomial {
            values: vec![F::ZERO; self.n as usize],
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient basis
    pub fn empty_lagrange(&self) -> Polynomial<F, LagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO; self.n as usize],
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the Lagrange coefficient basis, with
    /// deferred inversions.
    pub(crate) fn empty_lagrange_assigned(&self) -> Polynomial<Assigned<F>, LagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO.into(); self.n as usize],
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Returns a constant polynomial in the Lagrange coefficient basis
    pub fn constant_lagrange(&self, scalar: F) -> Polynomial<F, LagrangeCoeff> {
        Polynomial {
            values: vec![scalar; self.n as usize],
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Returns an empty (zero) polynomial in the extended Lagrange coefficient
    /// basis
    pub fn empty_extended(&self) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial {
            values: vec![F::ZERO; self.extended_len()],
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Returns a constant polynomial in the extended Lagrange coefficient
    /// basis
    pub fn constant_extended(&self, scalar: F) -> Polynomial<F, ExtendedLagrangeCoeff> {
        Polynomial {
            values: vec![scalar; self.extended_len()],
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// This takes us from an n-length vector into the coefficient form.
    ///
    /// This function will panic if the provided vector is not the correct
    /// length.
    pub fn lagrange_to_coeff(&self, mut a: Polynomial<F, LagrangeCoeff>) -> Polynomial<F, Coeff> {
        assert_eq!(a.len(), 1 << self.k);

        // Perform inverse FFT to obtain the polynomial in coefficient form
        self.ifft(a.as_mut_slice(), self.omega_inv, self.k, self.ifft_divisor);

        a.into_basis()
    }

    /// This takes us from an n-length coefficient vector into a coset of the extended
    /// evaluation domain, rotating by `rotation` if desired.
    pub fn coeff_to_extended(
        &self,
        p: &Polynomial<F, Coeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(p.len(), 1 << self.k);

        let mut a = Vec::with_capacity(self.extended_len());
        a.extend(&p[..]);

        self.distribute_powers_zeta(&mut a, true);
        a.resize(self.extended_len(), F::ZERO);
        self.fft_inner(&mut a, self.extended_omega, self.extended_k, false);

        Polynomial {
            values: a,
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Converts one coefficient polynomial to the native extended domain in
    /// session-scoped mapped storage, preserving the same FFT and field order.
    pub(crate) fn coeff_to_extended_spilled(
        &self,
        p: &Polynomial<F, Coeff>,
    ) -> io::Result<Polynomial<F, ExtendedLagrangeCoeff>> {
        assert_eq!(p.len(), 1 << self.k);

        let mut a: Polynomial<F, ExtendedLagrangeCoeff> =
            Polynomial::mapped_filled(self.extended_len(), F::ZERO)?;
        a[0..p.len()].copy_from_slice(&p[..]);
        self.distribute_powers_zeta(a.as_mut_slice(), true);
        self.fft_inner(a.as_mut_slice(), self.extended_omega, self.extended_k, false);
        Ok(a.into_basis())
    }

    /// This takes us from an n-length coefficient vector into parts of the
    /// extended evaluation domain. For example, for a polynomial with size n,
    /// and an extended domain of size mn, we can compute all parts
    /// independently, which are
    ///     `FFT(f(zeta * X), n)`
    ///     `FFT(f(zeta * extended_omega * X), n)`
    ///     ...
    ///     `FFT(f(zeta * extended_omega^{m-1} * X), n)`
    pub fn coeff_to_extended_parts(
        &self,
        a: &Polynomial<F, Coeff>,
    ) -> Vec<Polynomial<F, LagrangeCoeff>> {
        assert_eq!(a.len(), 1 << self.k);

        let num_parts = self.extended_len() >> self.k;
        let mut extended_omega_factor = F::ONE;
        (0..num_parts)
            .map(|_| {
                let part = self.coeff_to_extended_part(a.clone(), extended_omega_factor);
                extended_omega_factor *= self.extended_omega;
                part
            })
            .collect()
    }

    /// This takes us from several n-length coefficient vectors each into parts
    /// of the extended evaluation domain. For example, for a polynomial with
    /// size n, and an extended domain of size mn, we can compute all parts
    /// independently, which are
    ///     `FFT(f(zeta * X), n)`
    ///     `FFT(f(zeta * extended_omega * X), n)`
    ///     ...
    ///     `FFT(f(zeta * extended_omega^{m-1} * X), n)`
    pub fn batched_coeff_to_extended_parts(
        &self,
        a: &[Polynomial<F, Coeff>],
    ) -> Vec<Vec<Polynomial<F, LagrangeCoeff>>> {
        assert_eq!(a[0].len(), 1 << self.k);

        let mut extended_omega_factor = F::ONE;
        let num_parts = self.extended_len() >> self.k;
        (0..num_parts)
            .map(|_| {
                let a_lagrange = a
                    .iter()
                    .map(|poly| self.coeff_to_extended_part(poly.clone(), extended_omega_factor))
                    .collect();
                extended_omega_factor *= self.extended_omega;
                a_lagrange
            })
            .collect()
    }

    /// This takes us from an n-length coefficient vector into a part of the
    /// extended evaluation domain. For example, for a polynomial with size n,
    /// and an extended domain of size mn, we can compute one of the m parts
    /// separately, which is
    ///     `FFT(f(zeta * extended_omega_factor * X), n)`
    /// where `extended_omega_factor` is `extended_omega^i` with `i` in `[0, m)`.
    pub fn coeff_to_extended_part(
        &self,
        mut a: Polynomial<F, Coeff>,
        extended_omega_factor: F,
    ) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(a.len(), 1 << self.k);

        let coset = self.g_coset * extended_omega_factor;
        if !try_gpu_coset_fft_if_eligible(a.as_mut_slice(), self.omega, self.k, coset) {
            self.distribute_powers(a.as_mut_slice(), coset);
            let data = self.get_fft_data(a.len());
            best_fft(a.as_mut_slice(), self.omega, self.k, data, false);
        }

        a.into_basis()
    }

    /// Allocate one mapped buffer containing the coefficient powers for a
    /// strided extended-domain part. A group of polynomials at the same part
    /// can share these powers instead of rebuilding the recurrence for every
    /// polynomial.
    pub(crate) fn extended_part_factors_spilled(
        &self,
        extended_omega_factor: F,
    ) -> io::Result<Polynomial<F, Coeff>> {
        let n = self.extended_part_base_len()?;
        let mut factors = Polynomial::<F, Coeff>::mapped_filled(n, F::ZERO)?;
        let factor = self.g_coset * extended_omega_factor;
        parallelize(&mut factors[..], |chunk, start| {
            let mut power = factor.pow_vartime([start as u64]);
            for value in chunk {
                *value = power;
                power *= factor;
            }
        });
        Ok(factors)
    }

    /// Whether this coefficient vector can use the fused native GPU coset
    /// transform for the requested extended-domain part. Callers can use this
    /// before constructing shared host powers, which the GPU path can generate
    /// and retain in its per-device NTT workspace.
    pub(crate) fn extended_part_gpu_coset_is_eligible(
        &self,
        coefficients: &[F],
        extended_omega_factor: F,
    ) -> bool {
        gpu_coset_fft_is_eligible(
            coefficients,
            self.omega,
            self.k,
            self.g_coset * extended_omega_factor,
        )
    }

    /// Apply precomputed coefficient powers and perform the same native FFT
    /// used by `coeff_to_extended_part`. Both inputs must describe exactly
    /// one full base-domain part.
    pub(crate) fn coeff_to_extended_part_with_factors(
        &self,
        mut coefficients: Polynomial<F, Coeff>,
        factors: &[F],
        extended_omega_factor: F,
    ) -> io::Result<Polynomial<F, LagrangeCoeff>> {
        let coset = self.g_coset * extended_omega_factor;
        if try_gpu_coset_fft_if_eligible_with_factors(
            coefficients.as_mut_slice(),
            self.omega,
            self.k,
            coset,
            factors,
        ) {
            return Ok(coefficients.into_basis());
        }
        self.apply_extended_part_factors(coefficients.as_mut_slice(), factors)?;
        let data = self.get_fft_data(coefficients.len());
        best_fft(coefficients.as_mut_slice(), self.omega, self.k, data, false);
        Ok(coefficients.into_basis())
    }

    /// Copy a coefficient polynomial into mapped storage, apply shared powers,
    /// and perform the native FFT without retaining an additional N-element
    /// coefficient vector.
    pub(crate) fn coeff_to_extended_part_spilled_with_factors(
        &self,
        coefficients: &Polynomial<F, Coeff>,
        factors: &[F],
        extended_omega_factor: F,
    ) -> io::Result<Polynomial<F, LagrangeCoeff>> {
        let n = self.extended_part_base_len()?;
        if coefficients.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "coefficient polynomial has inconsistent dimensions",
            ));
        }
        if factors.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "extended-part factor buffer has inconsistent dimensions",
            ));
        }
        let mut output = Polynomial::<F, Coeff>::mapped_filled(n, F::ZERO)?;
        output.copy_from_slice(coefficients);
        self.coeff_to_extended_part_with_factors(output, factors, extended_omega_factor)
    }

    /// Turn the reserved factor/output buffer into one selected coset in
    /// place. The selected source is intentionally consumed last by the group
    /// loader, so this buffer becomes the final output allocation.
    pub(crate) fn coeff_to_extended_part_in_factor_buffer(
        &self,
        mut factors: Polynomial<F, Coeff>,
        coefficients: &Polynomial<F, Coeff>,
        extended_omega_factor: F,
    ) -> io::Result<Polynomial<F, LagrangeCoeff>> {
        let n = self.extended_part_base_len()?;
        if factors.len() != n || coefficients.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "factor buffer or coefficient polynomial has inconsistent dimensions",
            ));
        }
        let coset = self.g_coset * extended_omega_factor;
        if gpu_coset_fft_is_eligible(&factors[..], self.omega, self.k, coset) {
            factors.copy_from_slice(coefficients);
            if try_gpu_coset_fft_if_eligible(factors.as_mut_slice(), self.omega, self.k, coset) {
                return Ok(factors.into_basis());
            }
            // Eligibility cannot normally change between the query and
            // dispatch. If it does, rebuild the exact CPU-scaled input before
            // using the unchanged native FFT fallback.
            parallelize(&mut factors[..], |chunk, start| {
                let mut power = coset.pow_vartime([start as u64]);
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value = coefficients[start + offset] * power;
                    power *= coset;
                }
            });
            let data = self.get_fft_data(n);
            best_fft(factors.as_mut_slice(), self.omega, self.k, data, false);
            return Ok(factors.into_basis());
        }
        parallelize(&mut factors[..], |chunk, start| {
            for (offset, factor) in chunk.iter_mut().enumerate() {
                *factor *= coefficients[start + offset];
            }
        });
        let data = self.get_fft_data(n);
        best_fft(factors.as_mut_slice(), self.omega, self.k, data, false);
        Ok(factors.into_basis())
    }

    fn extended_part_base_len(&self) -> io::Result<usize> {
        let n = usize::try_from(self.n).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "base domain length does not fit in memory index space",
            )
        })?;
        let expected = 1usize.checked_shl(self.k).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "base domain length overflows memory index space",
            )
        })?;
        if n != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "base domain length has inconsistent dimensions",
            ));
        }
        Ok(n)
    }

    fn apply_extended_part_factors(&self, coefficients: &mut [F], factors: &[F]) -> io::Result<()> {
        let n = self.extended_part_base_len()?;
        if coefficients.len() != n || factors.len() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "factor buffer or coefficient polynomial has inconsistent dimensions",
            ));
        }
        parallelize(coefficients, |chunk, start| {
            for (offset, coefficient) in chunk.iter_mut().enumerate() {
                *coefficient *= factors[start + offset];
            }
        });
        Ok(())
    }

    /// Converts one coefficient polynomial to one native extended-domain
    /// strided part in mapped storage, preserving the native forward FFT.
    pub(crate) fn coeff_to_extended_part_spilled(
        &self,
        coefficients: &Polynomial<F, Coeff>,
        extended_omega_factor: F,
    ) -> io::Result<Polynomial<F, LagrangeCoeff>> {
        assert_eq!(coefficients.len(), 1 << self.k);
        let mut part_input = Polynomial::mapped_filled(coefficients.len(), F::ZERO)?;
        part_input.copy_from_slice(coefficients);
        Ok(self.coeff_to_extended_part(part_input, extended_omega_factor))
    }

    /// Rotate the extended domain polynomial over the original domain.
    pub fn rotate_extended(
        &self,
        poly: &Polynomial<F, ExtendedLagrangeCoeff>,
        rotation: Rotation,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let new_rotation = ((1 << (self.extended_k - self.k)) * rotation.0.abs()) as usize;

        let mut poly = poly.clone();

        if rotation.0 >= 0 {
            poly.as_mut_slice().rotate_left(new_rotation);
        } else {
            poly.as_mut_slice().rotate_right(new_rotation);
        }

        poly
    }

    /// This takes us from the extended evaluation domain and gets us the
    /// quotient polynomial coefficients.
    ///
    /// This function will panic if the provided vector is not the correct
    /// length.
    // TODO/FIXME: caller should be responsible for truncating
    pub fn extended_to_coeff(&self, mut a: Polynomial<F, ExtendedLagrangeCoeff>) -> Vec<F> {
        assert_eq!(a.len(), self.extended_len());

        // Inverse FFT
        self.ifft(
            a.as_mut_slice(),
            self.extended_omega_inv,
            self.extended_k,
            self.extended_ifft_divisor,
        );

        // Distribute powers to move from coset; opposite from the
        // transformation we performed earlier.
        self.distribute_powers_zeta(a.as_mut_slice(), false);

        // Truncate it to match the size of the quotient polynomial; the
        // evaluation domain might be slightly larger than necessary because
        // it always lies on a power-of-two boundary.
        let mut values = a.into_values();
        values
            .truncate((&self.n * self.quotient_poly_degree) as usize);

        values
    }

    /// Converts an extended-domain polynomial to coefficient form while
    /// retaining its mapped backing and truncating only its visible length.
    pub(crate) fn extended_to_coeff_spilled(&self, mut a: Polynomial<F, ExtendedLagrangeCoeff>) -> io::Result<Polynomial<F, Coeff>> {
        if a.len() != self.extended_len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "extended quotient length does not match the domain"));
        }
        let coefficient_len = self.n.checked_mul(self.quotient_poly_degree).and_then(|len| usize::try_from(len).ok()).filter(|len| *len <= a.len()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "quotient coefficient length overflows or exceeds the extended domain"))?;
        self.ifft(
            a.as_mut_slice(),
            self.extended_omega_inv,
            self.extended_k,
            self.extended_ifft_divisor,
        );
        self.distribute_powers_zeta(a.as_mut_slice(), false);
        a.truncate(coefficient_len);
        Ok(a.into_basis())
    }

    /// This takes us from the a list of lagrange-based polynomials with
    /// different degrees and gets their extended lagrange-based summation.
    pub fn lagrange_vecs_to_extended(
        &self,
        mut a: Vec<Vec<Polynomial<F, LagrangeCoeff>>>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        let mut result_poly = if a[a.len() - 1].len() == 1 << (self.extended_k - self.k) {
            self.lagrange_vec_to_extended(a.pop().unwrap())
        } else {
            self.empty_extended()
        };

        // Transform from each cluster of lagrange representations to coeff representations.
        let mut ifft_divisor = self.extended_ifft_divisor;
        let mut omega_inv = self.extended_omega_inv;
        {
            let mut i = a.last().unwrap().len() << self.k;
            while i < (1 << self.extended_k) {
                ifft_divisor = ifft_divisor + ifft_divisor;
                omega_inv = omega_inv * omega_inv;
                i <<= 1;
            }
        }

        let mut result = vec![F::ZERO; 1 << self.extended_k as usize];
        for (i, a_parts) in a.into_iter().enumerate().rev() {
            // transpose the values in parallel
            assert_eq!(1 << i, a_parts.len());
            let mut a_poly: Vec<F> = {
                let mut transposed = vec![vec![F::ZERO; a_parts.len()]; self.n as usize];
                a_parts.into_iter().enumerate().for_each(|(j, p)| {
                    parallelize(&mut transposed, |transposed, start| {
                        for (transposed, p) in transposed.iter_mut().zip(p[start..].iter()) {
                            transposed[j] = *p;
                        }
                    });
                });
                transposed.into_iter().flatten().collect()
            };

            self.ifft(&mut a_poly, omega_inv, self.k + i as u32, ifft_divisor);
            ifft_divisor = ifft_divisor + ifft_divisor;
            omega_inv = omega_inv * omega_inv;

            parallelize(&mut result[0..(self.n << i) as usize], |result, start| {
                for (other, current) in result.iter_mut().zip(a_poly[start..].iter()) {
                    *other += current;
                }
            });
        }
        let data = self.get_fft_data(result.len());
        best_fft(
            &mut result,
            self.extended_omega,
            self.extended_k,
            data,
            false,
        );
        parallelize(result_poly.as_mut_slice(), |values, start| {
            for (value, other) in values.iter_mut().zip(result[start..].iter()) {
                *value += other;
            }
        });
        result_poly
    }

    /// This divides the polynomial (in the extended domain) by the vanishing
    /// polynomial of the $2^k$ size domain.
    pub fn divide_by_vanishing_poly(
        &self,
        mut a: Polynomial<F, ExtendedLagrangeCoeff>,
    ) -> Polynomial<F, ExtendedLagrangeCoeff> {
        assert_eq!(a.len(), self.extended_len());

        // Divide to obtain the quotient polynomial in the coset evaluation
        // domain.
        parallelize(a.as_mut_slice(), |h, mut index| {
            for h in h {
                *h *= &self.t_evaluations[index % self.t_evaluations.len()];
                index += 1;
            }
        });

        a.into_basis()
    }

    /// Given a slice of group elements `[a_0, a_1, a_2, ...]`, this returns
    /// `[a_0, [zeta]a_1, [zeta^2]a_2, a_3, [zeta]a_4, [zeta^2]a_5, a_6, ...]`,
    /// where zeta is a cube root of unity in the multiplicative subgroup with
    /// order (p - 1), i.e. zeta^3 = 1.
    ///
    /// `into_coset` should be set to `true` when moving into the coset,
    /// and `false` when moving out. This toggles the choice of `zeta`.
    fn distribute_powers_zeta(&self, a: &mut [F], into_coset: bool) {
        let coset_powers = if into_coset {
            [self.g_coset, self.g_coset_inv]
        } else {
            [self.g_coset_inv, self.g_coset]
        };
        parallelize(a, |a, mut index| {
            for a in a {
                // Distribute powers to move into/from coset
                let i = index % (coset_powers.len() + 1);
                if i != 0 {
                    *a *= &coset_powers[i - 1];
                }
                index += 1;
            }
        });
    }

    /// Given a slice of group elements `[a_0, a_1, a_2, ...]`, this returns
    /// `[a_0, [c]a_1, [c^2]a_2, [c^3]a_3, [c^4]a_4, ...]`,
    ///
    fn distribute_powers(&self, a: &mut [F], c: F) {
        parallelize(a, |a, index| {
            let mut c_power = c.pow_vartime([index as u64]);
            for a in a {
                *a *= c_power;
                c_power *= c;
            }
        });
    }

    fn ifft(&self, a: &mut [F], omega_inv: F, log_n: u32, divisor: F) {
        if !try_gpu_fft_if_eligible(a, omega_inv, log_n, true) {
            #[cfg(target_arch = "x86_64")]
            best_fft_cpu(a, omega_inv, log_n, &FFTData::default(), true);
            #[cfg(not(target_arch = "x86_64"))]
            self.fft_inner(a, omega_inv, log_n, true);
        }
        parallelize(a, |a, _| {
            for a in a {
                // Finish iFFT
                *a *= &divisor;
            }
        });
    }

    fn fft_inner(&self, a: &mut [F], omega: F, log_n: u32, inverse: bool) {
        if try_gpu_fft_if_eligible(a, omega, log_n, inverse) {
            return;
        }
        let fft_data = self.get_fft_data(a.len());
        best_fft(a, omega, log_n, fft_data, inverse)
    }

    /// Get the size of the domain
    pub fn k(&self) -> u32 {
        self.k
    }

    /// Get the size of the extended domain
    pub fn extended_k(&self) -> u32 {
        self.extended_k
    }

    /// Get the size of the extended domain
    pub fn extended_len(&self) -> usize {
        1 << self.extended_k
    }

    /// Get $\omega$, the generator of the $2^k$ order multiplicative subgroup.
    pub fn get_omega(&self) -> F {
        self.omega
    }

    /// Get $\omega^{-1}$, the inverse of the generator of the $2^k$ order
    /// multiplicative subgroup.
    pub fn get_omega_inv(&self) -> F {
        self.omega_inv
    }

    /// Get the generator of the extended domain's multiplicative subgroup.
    pub fn get_extended_omega(&self) -> F {
        self.extended_omega
    }

    /// Multiplies a value by some power of $\omega$, essentially rotating over
    /// the domain.
    pub fn rotate_omega(&self, value: F, rotation: Rotation) -> F {
        let mut point = value;
        if rotation.0 >= 0 {
            point *= &self.get_omega().pow_vartime([rotation.0 as u64]);
        } else {
            point *= &self
                .get_omega_inv()
                .pow_vartime([(rotation.0 as i64).unsigned_abs()]);
        }
        point
    }

    /// Computes evaluations (at the point `x`, where `xn = x^n`) of Lagrange
    /// basis polynomials `l_i(X)` defined such that `l_i(omega^i) = 1` and
    /// `l_i(omega^j) = 0` for all `j != i` at each provided rotation `i`.
    ///
    /// # Implementation
    ///
    /// The polynomial
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)$$
    /// has a root at all points in the domain except $\omega^i$, where it evaluates to
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)$$
    /// and so we divide that polynomial by this value to obtain $l_i(X)$. Since
    ///     $$\prod_{j=0,j \neq i}^{n - 1} (X - \omega^j)
    ///       = \frac{X^n - 1}{X - \omega^i}$$
    /// then $l_i(x)$ for some $x$ is evaluated as
    ///     $$\left(\frac{x^n - 1}{x - \omega^i}\right)
    ///       \cdot \left(\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)}\right).$$
    /// We refer to
    ///     $$1 \over \prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)$$
    /// as the barycentric weight of $\omega^i$.
    ///
    /// We know that for $i = 0$
    ///     $$\frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)} = \frac{1}{n}.$$
    ///
    /// If we multiply $(1 / n)$ by $\omega^i$ then we obtain
    ///     $$\frac{1}{\prod_{j=0,j \neq 0}^{n - 1} (\omega^i - \omega^j)}
    ///       = \frac{1}{\prod_{j=0,j \neq i}^{n - 1} (\omega^i - \omega^j)}$$
    /// which is the barycentric weight of $\omega^i$.
    pub fn l_i_range<I: IntoIterator<Item = i32> + Clone>(
        &self,
        x: F,
        xn: F,
        rotations: I,
    ) -> Vec<F> {
        let mut results;
        {
            let rotations = rotations.clone().into_iter();
            results = Vec::with_capacity(rotations.size_hint().1.unwrap_or(0));
            for rotation in rotations {
                let rotation = Rotation(rotation);
                let result = x - self.rotate_omega(F::ONE, rotation);
                results.push(result);
            }
            results.iter_mut().batch_invert();
        }

        let common = (xn - F::ONE) * self.barycentric_weight;
        for (rotation, result) in rotations.into_iter().zip(results.iter_mut()) {
            let rotation = Rotation(rotation);
            *result = self.rotate_omega(*result * common, rotation);
        }

        results
    }

    /// Gets the quotient polynomial's degree (as a multiple of n)
    pub fn get_quotient_poly_degree(&self) -> usize {
        self.quotient_poly_degree as usize
    }

    /// Obtain a pinned version of this evaluation domain; a structure with the
    /// minimal parameters needed to determine the rest of the evaluation
    /// domain.
    pub fn pinned(&self) -> PinnedEvaluationDomain<'_, F> {
        PinnedEvaluationDomain {
            k: &self.k,
            extended_k: &self.extended_k,
            omega: &self.omega,
        }
    }

    /// Get the private field `n`
    pub fn get_n(&self) -> u64 {
        self.n
    }

    /// Number of supported recursive FFT lengths retained by the domain cache.
    pub fn fft_cache_entry_count(&self) -> usize {
        self.fft_data.len()
    }

    /// Allocated entry capacity of the recursive FFT cache map.
    pub fn fft_cache_entry_capacity(&self) -> usize {
        self.fft_data.capacity()
    }

    /// Get the private `fft_data`
    pub fn get_fft_data(&self, l: usize) -> &FFTData<F> {
        self.fft_data
            .get(&l)
            .expect("log_2(l) must be in k..=extended_k")
            .get_or_init(|| {
                let log_n = l.trailing_zeros();
                let exponent = 1u64 << (self.extended_k - log_n);
                let omega = self.extended_omega.pow_vartime([exponent]);
                let omega_inv = self.extended_omega_inv.pow_vartime([exponent]);
                FFTData::<F>::new(l, omega, omega_inv)
            })
    }
}

/// Represents the minimal parameters that determine an `EvaluationDomain`.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedEvaluationDomain<'a, F: Field> {
    k: &'a u32,
    extended_k: &'a u32,
    omega: &'a F,
}

#[test]
fn test_rotate() {
    use rand_core::OsRng;

    use crate::arithmetic::eval_polynomial;
    use halo2curves::pasta::pallas::Scalar;

    let domain = EvaluationDomain::<Scalar>::new(1, 3);
    let rng = OsRng;

    let mut poly = domain.empty_lagrange();
    assert_eq!(poly.len(), 8);
    for value in poly.iter_mut() {
        *value = Scalar::random(rng);
    }

    let poly_rotated_cur = poly.rotate(Rotation::cur());
    let poly_rotated_next = poly.rotate(Rotation::next());
    let poly_rotated_prev = poly.rotate(Rotation::prev());

    let poly = domain.lagrange_to_coeff(poly);
    let poly_rotated_cur = domain.lagrange_to_coeff(poly_rotated_cur);
    let poly_rotated_next = domain.lagrange_to_coeff(poly_rotated_next);
    let poly_rotated_prev = domain.lagrange_to_coeff(poly_rotated_prev);

    let x = Scalar::random(rng);

    assert_eq!(
        eval_polynomial(&poly[..], x),
        eval_polynomial(&poly_rotated_cur[..], x)
    );
    assert_eq!(
        eval_polynomial(&poly[..], x * domain.omega),
        eval_polynomial(&poly_rotated_next[..], x)
    );
    assert_eq!(
        eval_polynomial(&poly[..], x * domain.omega_inv),
        eval_polynomial(&poly_rotated_prev[..], x)
    );
}

#[test]
fn test_l_i() {
    use rand_core::OsRng;

    use crate::arithmetic::{eval_polynomial, lagrange_interpolate};
    use halo2curves::pasta::pallas::Scalar;
    let domain = EvaluationDomain::<Scalar>::new(1, 3);

    let mut l = vec![];
    let mut points = vec![];
    for i in 0..8 {
        points.push(domain.omega.pow([i]));
    }
    for i in 0..8 {
        let mut l_i = vec![Scalar::zero(); 8];
        l_i[i] = Scalar::ONE;
        let l_i = lagrange_interpolate(&points[..], &l_i[..]);
        l.push(l_i);
    }

    let x = Scalar::random(OsRng);
    let xn = x.pow([8]);

    let evaluations = domain.l_i_range(x, xn, -7..=7);
    for i in 0..8 {
        assert_eq!(eval_polynomial(&l[i][..], x), evaluations[7 + i]);
        assert_eq!(eval_polynomial(&l[(8 - i) % 8][..], x), evaluations[7 - i]);
    }
}

#[test]
fn test_coeff_to_extended_part() {
    use halo2curves::pasta::pallas::Scalar;
    use rand_core::OsRng;

    let domain = EvaluationDomain::<Scalar>::new(1, 3);
    let rng = OsRng;
    let mut poly = domain.empty_coeff();
    assert_eq!(poly.len(), 8);
    for value in poly.iter_mut() {
        *value = Scalar::random(rng);
    }

    let want = domain.coeff_to_extended(&poly);
    let got = {
        let parts = domain.coeff_to_extended_parts(&poly);
        domain.lagrange_vec_to_extended(parts)
    };
    assert_eq!(want.values, got.values);
}

#[test]
fn test_extended_parts_preserve_stride_rotations_and_quotient() {
    use halo2curves::pasta::pallas::Scalar;
    use rand_core::OsRng;

    // j=5 makes the extended domain four times the base domain. Each
    // coeff_to_extended_part result is one strided lane of the full vector.
    let domain = EvaluationDomain::<Scalar>::new(5, 4);
    let mut poly = domain.empty_coeff();
    for value in poly.iter_mut() {
        *value = Scalar::random(OsRng);
    }

    let full = domain.coeff_to_extended(&poly);
    let parts = domain.coeff_to_extended_parts(&poly);
    let stride = domain.extended_len() / poly.len();
    assert_eq!(parts.len(), stride);

    let mut reconstructed = domain.empty_extended();
    for (part_idx, part) in parts.into_iter().enumerate() {
        for (row, value) in part.iter().enumerate() {
            reconstructed[row * stride + part_idx] = *value;
        }
    }
    assert_eq!(full.values, reconstructed.values);

    for rotation in [-9, -1, 1, 9] {
        let rotated = domain.rotate_extended(&full, Rotation(rotation));
        let shift = (rotation.unsigned_abs() as usize * stride) % domain.extended_len();
        let right_shift = if rotation < 0 {
            domain.extended_len() - shift
        } else {
            shift
        };
        for index in 0..domain.extended_len() {
            let source = (index + right_shift) % domain.extended_len();
            assert_eq!(rotated[index], reconstructed[source]);
        }
    }

    let full_quotient = domain.divide_by_vanishing_poly(full.clone());
    let parts_quotient = domain.divide_by_vanishing_poly(reconstructed.clone());
    assert_eq!(full_quotient.values, parts_quotient.values);
    assert_eq!(
        domain.extended_to_coeff(full_quotient.clone()),
        domain.extended_to_coeff(parts_quotient),
    );
}

#[test]
fn test_extended_to_coeff_spilled_matches_native_transform() {
    use crate::poly::native_spill_scope;
    use halo2curves::pasta::pallas::Scalar;
    use rand_core::OsRng;

    let base = std::env::temp_dir().join(format!(
        "halo2-domain-spill-test-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&base).unwrap();
    let spill = native_spill_scope(&base).unwrap();
    let domain = EvaluationDomain::<Scalar>::new(5, 4);
    let mut poly = domain.empty_coeff();
    for value in poly.iter_mut() {
        *value = Scalar::random(OsRng);
    }
    let expected = domain.extended_to_coeff(domain.divide_by_vanishing_poly(domain.coeff_to_extended(&poly)));
    let mapped_quotient = domain.divide_by_vanishing_poly(domain.coeff_to_extended_spilled(&poly).unwrap());
    let actual = domain.extended_to_coeff_spilled(mapped_quotient).unwrap();
    assert!(actual.is_mapped());
    assert_eq!(actual.len(), expected.len());
    assert_eq!(&actual[..], expected.as_slice());
    let piece_len = 1usize << domain.k();
    let mut reassembled = Vec::with_capacity(actual.len());
    for index in 0..domain.get_quotient_poly_degree() {
        let start = index * piece_len;
        let mut piece = Polynomial::<Scalar, Coeff>::mapped_filled(piece_len, Scalar::ZERO).unwrap();
        piece.copy_from_slice(&actual[start..start + piece_len]);
        reassembled.extend_from_slice(&piece[..]);
    }
    assert_eq!(reassembled, expected);
    drop(actual);
    drop(spill);
    assert_eq!(std::fs::read_dir(&base).unwrap().count(), 0);
    std::fs::remove_dir(base).unwrap();
}

#[test]
fn bench_coeff_to_extended_parts() {
    use halo2curves::pasta::pallas::Scalar;
    use instant::Instant;
    use rand_core::OsRng;

    let k = 20;
    let domain = EvaluationDomain::<Scalar>::new(3, k);
    let rng = OsRng;
    let mut poly1 = domain.empty_coeff();
    assert_eq!(poly1.len(), 1 << k);

    for value in poly1.iter_mut() {
        *value = Scalar::random(rng);
    }

    let poly2 = poly1.clone();

    let coeff_to_extended_timer = Instant::now();
    let _ = domain.coeff_to_extended(&poly1);
    println!(
        "domain.coeff_to_extended time: {}s",
        coeff_to_extended_timer.elapsed().as_secs_f64()
    );

    let coeff_to_extended_parts_timer = Instant::now();
    let _ = domain.coeff_to_extended_parts(&poly2);
    println!(
        "domain.coeff_to_extended_parts time: {}s",
        coeff_to_extended_parts_timer.elapsed().as_secs_f64()
    );
}

#[test]
fn test_lagrange_vecs_to_extended() {
    use halo2curves::pasta::pallas::Scalar;
    use rand_core::OsRng;

    let rng = OsRng;
    let domain = EvaluationDomain::<Scalar>::new(8, 10);
    let mut poly_vec = vec![];
    let mut poly_lagrange_vecs = vec![];
    let mut want = domain.empty_extended();
    let mut omega = domain.extended_omega;
    for i in (0..(domain.extended_k - domain.k + 1)).rev() {
        let mut poly = vec![Scalar::zero(); (1 << i) * domain.n as usize];
        for value in poly.iter_mut() {
            *value = Scalar::random(rng);
        }
        // poly under coeff representation.
        poly_vec.push(poly.clone());
        // poly under lagrange vector representation.
        let mut poly2 = poly.clone();
        let data = domain.get_fft_data(poly2.len());
        best_fft(&mut poly2, omega, i + domain.k, data, false);
        let transposed_poly: Vec<Polynomial<Scalar, LagrangeCoeff>> = (0..(1 << i))
            .map(|j| {
                let mut p = domain.empty_lagrange();
                for k in 0..domain.n {
                    p[k as usize] = poly2[j + (k as usize) * (1 << i)];
                }
                p
            })
            .collect();
        poly_lagrange_vecs.push(transposed_poly);
        // poly under extended representation.
        poly.resize(domain.extended_len(), Scalar::zero());
        let data = domain.get_fft_data(poly.len());
        best_fft(
            &mut poly,
            domain.extended_omega,
            domain.extended_k,
            data,
            false,
        );
        let poly = {
            let mut p = domain.empty_extended();
            p.values = poly;
            p
        };
        want = want + &poly;
        omega = omega * omega;
    }
    poly_lagrange_vecs.reverse();
    let got = domain.lagrange_vecs_to_extended(poly_lagrange_vecs);
    assert_eq!(want.values, got.values);
}

#[test]
fn bench_lagrange_vecs_to_extended() {
    use halo2curves::pasta::pallas::Scalar;
    use instant::Instant;
    use rand_core::OsRng;

    let rng = OsRng;
    let domain = EvaluationDomain::<Scalar>::new(8, 10);
    let mut poly_vec = vec![];
    let mut poly_lagrange_vecs = vec![];
    let mut poly_extended_vecs = vec![];
    let mut omega = domain.extended_omega;

    for i in (0..(domain.extended_k - domain.k + 1)).rev() {
        let mut poly = vec![Scalar::zero(); (1 << i) * domain.n as usize];
        for value in poly.iter_mut() {
            *value = Scalar::random(rng);
        }
        // poly under coeff representation.
        poly_vec.push(poly.clone());
        // poly under lagrange vector representation.
        let mut poly2 = poly.clone();
        let data = domain.get_fft_data(poly2.len());
        best_fft(&mut poly2, omega, i + domain.k, data, false);
        let transposed_poly: Vec<Polynomial<Scalar, LagrangeCoeff>> = (0..(1 << i))
            .map(|j| {
                let mut p = domain.empty_lagrange();
                for k in 0..domain.n {
                    p[k as usize] = poly2[j + (k as usize) * (1 << i)];
                }
                p
            })
            .collect();
        poly_lagrange_vecs.push(transposed_poly);
        // poly under extended representation.
        poly.resize(domain.extended_len(), Scalar::zero());
        let data = domain.get_fft_data(poly.len());
        best_fft(
            &mut poly,
            domain.extended_omega,
            domain.extended_k,
            data,
            false,
        );
        let poly = {
            let mut p = domain.empty_extended();
            p.values = poly;
            p
        };
        poly_extended_vecs.push(poly);
        omega = omega * omega;
    }

    let want_timer = Instant::now();
    let _ = poly_extended_vecs
        .iter()
        .fold(domain.empty_extended(), |acc, p| acc + p);
    println!("want time: {}s", want_timer.elapsed().as_secs_f64());
    poly_lagrange_vecs.reverse();
    let got_timer = Instant::now();
    let _ = domain.lagrange_vecs_to_extended(poly_lagrange_vecs);
    println!("got time: {}s", got_timer.elapsed().as_secs_f64());
}

#[test]
fn recursive_fft_cache_is_lazy_and_shared_by_domain_clones() {
    use halo2curves::pasta::pallas::Scalar;

    let domain = EvaluationDomain::<Scalar>::new(4, 3);
    let clone = domain.clone();
    assert!(domain.fft_data.values().all(|cache| cache.get().is_none()));
    for (length, cache) in &domain.fft_data {
        assert!(Arc::ptr_eq(
            cache,
            clone.fft_data.get(length).expect("clone has same FFT lengths")
        ));
    }

    let n = domain.get_n() as usize;
    let input = (0..n)
        .map(|value| Scalar::from(value as u64))
        .collect::<Vec<_>>();
    let mut transformed = input.clone();
    domain.fft_inner(&mut transformed, domain.omega, domain.k, false);
    assert!(std::ptr::eq(
        domain.get_fft_data(n),
        clone.get_fft_data(n)
    ));
    assert_eq!(
        domain.fft_data.values().filter(|cache| cache.get().is_some()).count(),
        1
    );

    domain.ifft(
        &mut transformed,
        domain.omega_inv,
        domain.k,
        domain.ifft_divisor,
    );
    assert_eq!(transformed, input);

    let e = domain.extended_len();
    let extended_input = (0..e)
        .map(|value| Scalar::from(value as u64))
        .collect::<Vec<_>>();
    let mut extended_transformed = extended_input.clone();
    domain.fft_inner(
        &mut extended_transformed,
        domain.extended_omega,
        domain.extended_k,
        false,
    );
    assert!(std::ptr::eq(
        domain.get_fft_data(e),
        clone.get_fft_data(e)
    ));
    assert_eq!(
        domain.fft_data.values().filter(|cache| cache.get().is_some()).count(),
        2
    );
    domain.ifft(
        &mut extended_transformed,
        domain.extended_omega_inv,
        domain.extended_k,
        domain.extended_ifft_divisor,
    );
    assert_eq!(extended_transformed, extended_input);

    let ifft_only_domain = EvaluationDomain::<Scalar>::new(4, 3);
    let mut zeroes = vec![Scalar::ZERO; ifft_only_domain.get_n() as usize];
    ifft_only_domain.ifft(
        &mut zeroes,
        ifft_only_domain.omega_inv,
        ifft_only_domain.k,
        ifft_only_domain.ifft_divisor,
    );
    assert!(ifft_only_domain
        .fft_data
        .values()
        .all(|cache| cache.get().is_none()));
}
