//! Contains utilities for performing arithmetic over univariate polynomials in
//! various forms, including computing commitments to them and provably opening
//! the committed polynomials at arbitrary points.

use crate::arithmetic::parallelize;
use crate::helpers::SerdePrimeField;
use crate::plonk::Assigned;
use crate::SerdeFormat;
use group::ff::{BatchInvert, Field};
#[cfg(feature = "parallel-poly-read")]
use maybe_rayon::{iter::ParallelIterator, prelude::ParallelSlice};
use rand_core::RngCore;

use std::fmt::Debug;
use std::io;
use std::marker::PhantomData;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Range, RangeFrom, RangeFull, Sub};

/// Generic commitment scheme structures
pub mod commitment;
mod domain;
mod query;
mod spill;
mod strategy;

/// Inner product argument commitment scheme
pub mod ipa;

/// KZG commitment scheme
pub mod kzg;

#[cfg(test)]
mod multiopen_test;

pub use domain::*;
pub use query::{ProverQuery, VerifierQuery};
pub use spill::NativeSpillScope;
pub use spill::{
    native_target_bitmap_allocation_bitmap_bytes, NATIVE_SUCCESSOR_READ_CACHE_BYTES,
    NATIVE_TARGET_BITMAP_FIXED_CACHE_BYTES,
};
pub(crate) use spill::SpillLog;
pub(crate) use spill::{
    NativePermutationAllocationPlan, NativePermutationStats, NativePermutationStore,
    NativeSuccessorStore, NativeSuccessorValidationStats,
};
pub use strategy::{Guard, VerificationStrategy};

/// Creates a private, session-local backing directory for native polynomial
/// scratch files under `base`.
pub fn native_spill_scope(base: impl AsRef<std::path::Path>) -> io::Result<NativeSpillScope> {
    NativeSpillScope::new(base.as_ref())
}

pub(crate) fn native_spill_log() -> io::Result<SpillLog> {
    spill::spill_log()
}

/// This is an error that could occur during proving or circuit synthesis.
// TODO: these errors need to be cleaned up
#[derive(Debug)]
pub enum Error {
    /// OpeningProof is not well-formed
    OpeningError,
    /// Caller needs to re-sample a point
    SamplingError,
}

/// The basis over which a polynomial is described.
pub trait Basis: Copy + Debug + Send + Sync {}

/// The polynomial is defined as coefficients
#[derive(Clone, Copy, Debug)]
pub struct Coeff;
impl Basis for Coeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials
#[derive(Clone, Copy, Debug)]
pub struct LagrangeCoeff;
impl Basis for LagrangeCoeff {}

/// The polynomial is defined as coefficients of Lagrange basis polynomials in
/// an extended size domain which supports multiplication
#[derive(Clone, Copy, Debug)]
pub struct ExtendedLagrangeCoeff;
impl Basis for ExtendedLagrangeCoeff {}

/// Represents a univariate polynomial defined over a field and a particular
/// basis.
#[derive(Debug)]
pub struct Polynomial<F, B> {
    values: Vec<F>,
    mapped: Option<spill::MappedValues<F>>,
    _marker: PhantomData<B>,
}

impl<F: Copy, B: Clone> Clone for Polynomial<F, B> {
    fn clone(&self) -> Self {
        self.try_clone()
            .expect("cloning a spilled polynomial requires private disk backing")
    }
}

impl<F: Copy, B> Polynomial<F, B> {
    pub(crate) fn try_clone(&self) -> io::Result<Self> {
        let mapped = match self.mapped.as_ref() {
            Some(_) => Some(spill::map_values(self.as_slice())?),
            None => None,
        };
        Ok(Self {
            values: if self.mapped.is_some() {
                Vec::new()
            } else {
                self.values.clone()
            },
            mapped,
            _marker: PhantomData,
        })
    }
}

impl<F, B> Index<usize> for Polynomial<F, B> {
    type Output = F;

    fn index(&self, index: usize) -> &F {
        self.as_slice().index(index)
    }
}

impl<F, B> IndexMut<usize> for Polynomial<F, B> {
    fn index_mut(&mut self, index: usize) -> &mut F {
        self.as_mut_slice().index_mut(index)
    }
}

impl<F, B> Index<RangeFrom<usize>> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFrom<usize>) -> &[F] {
        self.as_slice().index(index)
    }
}

impl<F, B> IndexMut<RangeFrom<usize>> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFrom<usize>) -> &mut [F] {
        self.as_mut_slice().index_mut(index)
    }
}

impl<F, B> Index<RangeFull> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: RangeFull) -> &[F] {
        self.as_slice().index(index)
    }
}

impl<F, B> IndexMut<RangeFull> for Polynomial<F, B> {
    fn index_mut(&mut self, index: RangeFull) -> &mut [F] {
        self.as_mut_slice().index_mut(index)
    }
}

impl<F, B> Index<Range<usize>> for Polynomial<F, B> {
    type Output = [F];

    fn index(&self, index: Range<usize>) -> &[F] {
        self.as_slice().index(index)
    }
}

impl<F, B> IndexMut<Range<usize>> for Polynomial<F, B> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut [F] {
        self.as_mut_slice().index_mut(index)
    }
}

impl<F, B> Deref for Polynomial<F, B> {
    type Target = [F];

    fn deref(&self) -> &[F] {
        self.as_slice()
    }
}

impl<F: Clone, B> DerefMut for Polynomial<F, B> {
    fn deref_mut(&mut self) -> &mut [F] {
        self.as_mut_slice()
    }
}

impl<F, B> Polynomial<F, B> {
    pub(crate) fn from_vec(values: Vec<F>) -> Self {
        Polynomial {
            values,
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Iterate over the values, which are either in coefficient or evaluation
    /// form depending on the basis `B`.
    pub fn iter(&self) -> impl Iterator<Item = &F> {
        self.as_slice().iter()
    }

    /// Iterate over the values mutably, which are either in coefficient or
    /// evaluation form depending on the basis `B`.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut F>
    where
        F: Clone,
    {
        self.as_mut_slice().iter_mut()
    }

    /// Gets the size of this polynomial in terms of the number of
    /// coefficients used to describe it.
    pub fn num_coeffs(&self) -> usize {
        self.as_slice().len()
    }

    pub(crate) fn truncate(&mut self, len: usize) {
        assert!(len <= self.len());
        if let Some(mapped) = self.mapped.as_mut() {
            mapped.truncate(len);
        } else {
            self.values.truncate(len);
        }
    }

    fn as_slice(&self) -> &[F] {
        self.mapped
            .as_ref()
            .map(|mapped| mapped.as_slice())
            .unwrap_or(&self.values)
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [F] {
        if let Some(mapped) = self.mapped.as_mut() {
            mapped.as_mut_slice()
        } else {
            &mut self.values
        }
    }

    pub(crate) fn into_values(self) -> Vec<F>
    where
        F: Clone,
    {
        if let Some(mapped) = self.mapped {
            mapped.as_slice().to_vec()
        } else {
            self.values
        }
    }

    pub(crate) fn into_basis<T>(self) -> Polynomial<F, T> {
        Polynomial {
            values: self.values,
            mapped: self.mapped,
            _marker: PhantomData,
        }
    }

    #[cfg(test)]
    fn is_mapped(&self) -> bool {
        self.mapped.is_some()
    }

    pub(crate) fn spill(self) -> io::Result<Self>
    where
        F: Copy,
    {
        if self.mapped.is_some() || self.values.is_empty() {
            return Ok(self);
        }
        let mapped = spill::map_values(&self.values)?;
        Ok(Self {
            values: Vec::new(),
            mapped: Some(mapped),
            _marker: self._marker,
        })
    }

    pub(crate) fn mapped_filled(len: usize, value: F) -> io::Result<Self>
    where
        F: Copy,
    {
        let mut builder = spill::spill_builder::<F>(len)?;
        let chunk = vec![value; len.min(4096)];
        let mut remaining = len;
        while remaining > 0 {
            let take = remaining.min(chunk.len());
            builder.write_chunk(&chunk[..take])?;
            remaining -= take;
        }
        Ok(Self {
            values: Vec::new(),
            mapped: Some(builder.finish()?),
            _marker: PhantomData,
        })
    }
}

impl<F: SerdePrimeField, B> Polynomial<F, B> {
    /// create a random polynomial with `num_coeffs` coefficients
    pub fn random<R: RngCore>(num_coeffs: usize, rng: &mut R) -> Self {
        Polynomial {
            values: (0..num_coeffs).map(|_| F::random(&mut *rng)).collect(),
            mapped: None,
            _marker: PhantomData,
        }
    }

    /// Reads polynomial from buffer using `SerdePrimeField::read`.  
    #[cfg(feature = "parallel-poly-read")]
    pub(crate) fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let poly_len = u32::from_be_bytes({
            let mut buf = [0u8; 4];
            reader.read_exact(&mut buf)?;
            buf
        });

        Self::read_with_len(reader, format, poly_len)
    }

    #[cfg(feature = "parallel-poly-read")]
    pub(crate) fn read_with_len<R: io::Read>(
        reader: &mut R,
        format: SerdeFormat,
        poly_len: u32,
    ) -> io::Result<Self> {
        let poly_len = poly_len as usize;
        let repr_len = F::default().to_repr().as_ref().len();

        if poly_len == 0 {
            return Ok(Self::from_vec(Vec::new()));
        }
        let mut builder = spill::spill_builder::<F>(poly_len as usize)?;
        const CHUNK_FIELDS: usize = 16 * 1024;
        let mut remaining = poly_len as usize;
        let chunk_fields = remaining.min(CHUNK_FIELDS);
        let mut buffer = vec![0u8; chunk_fields.checked_mul(repr_len).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "polynomial chunk size overflow")
        })?];
        while remaining > 0 {
            let fields = remaining.min(CHUNK_FIELDS);
            let bytes = fields.checked_mul(repr_len).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidInput, "polynomial chunk size overflow")
            })?;
            reader.read_exact(&mut buffer[..bytes])?;
            let values = buffer[..bytes]
                .par_chunks_exact(repr_len)
                .map(|chunk| F::read(&mut io::Cursor::new(chunk), format))
                .collect::<io::Result<Vec<_>>>()?;
            builder.write_chunk(&values)?;
            remaining -= fields;
        }
        Ok(Self {
            values: Vec::new(),
            mapped: Some(builder.finish()?),
            _marker: PhantomData,
        })
    }

    /// Reads polynomial from buffer using `SerdePrimeField::read`.  
    #[cfg(not(feature = "parallel-poly-read"))]
    pub fn read<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        Self::read_serial(reader, format)
    }

    #[cfg(not(feature = "parallel-poly-read"))]
    pub(crate) fn read_with_len<R: io::Read>(
        reader: &mut R,
        format: SerdeFormat,
        poly_len: u32,
    ) -> io::Result<Self> {
        if poly_len == 0 {
            return Ok(Self::from_vec(Vec::new()));
        }
        let mut builder = spill::spill_builder::<F>(poly_len as usize)?;
        for _ in 0..poly_len {
            builder.write_value(F::read(reader, format)?)?;
        }
        Ok(Self {
            values: Vec::new(),
            mapped: Some(builder.finish()?),
            _marker: PhantomData,
        })
    }

    /// Reads polynomial from buffer using `SerdePrimeField::read`.  
    pub fn read_serial<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self> {
        let mut poly_len = [0u8; 4];
        reader.read_exact(&mut poly_len)?;
        let poly_len = u32::from_be_bytes(poly_len);
        Self::read_with_len(reader, format, poly_len)
    }

    /// Writes polynomial to buffer using `SerdePrimeField::write`.  
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        writer.write_all(&(self.as_slice().len() as u32).to_be_bytes())?;
        for value in self.as_slice().iter() {
            value.write(writer, format)?;
        }
        Ok(())
    }
}

pub(crate) fn batch_invert_assigned<F: Field>(
    assigned: Vec<Polynomial<Assigned<F>, LagrangeCoeff>>,
) -> Vec<Polynomial<F, LagrangeCoeff>> {
    let mut assigned_denominators: Vec<_> = assigned
        .iter()
        .map(|f| {
            f.iter()
                .map(|value| value.denominator())
                .collect::<Vec<_>>()
        })
        .collect();

    assigned_denominators
        .iter_mut()
        .flat_map(|f| {
            f.iter_mut()
                // If the denominator is trivial, we can skip it, reducing the
                // size of the batch inversion.
                .filter_map(|d| d.as_mut())
        })
        .batch_invert();

    assigned
        .iter()
        .zip(assigned_denominators)
        .map(|(poly, inv_denoms)| poly.invert(inv_denoms.into_iter().map(|d| d.unwrap_or(F::ONE))))
        .collect()
}

impl<F: Field> Polynomial<Assigned<F>, LagrangeCoeff> {
    /// Batch-inverts one assigned column into mapped field storage. Processing
    /// one column at a time avoids retaining denominator and output vectors for
    /// every advice column during proof generation.
    pub(crate) fn invert_spilled(self) -> io::Result<Polynomial<F, LagrangeCoeff>> {
        let mut inv_denoms = self.iter().map(|value| value.denominator()).collect::<Vec<_>>();
        inv_denoms
            .iter_mut()
            .map(|denominator| denominator.get_or_insert(F::ONE))
            .batch_invert();
        let mut values = Polynomial::mapped_filled(self.len(), F::ZERO)?;
        for (idx, (assigned, inv_denom)) in self.iter().zip(inv_denoms).enumerate() {
            values[idx] = assigned.numerator() * inv_denom.unwrap_or(F::ONE);
        }
        Ok(values)
    }

    pub(crate) fn invert(
        &self,
        inv_denoms: impl ExactSizeIterator<Item = F>,
    ) -> Polynomial<F, LagrangeCoeff> {
        assert_eq!(inv_denoms.len(), self.as_slice().len());
        Polynomial {
            values: self
                .as_slice()
                .iter()
                .zip(inv_denoms)
                .map(|(a, inv_den)| a.numerator() * inv_den)
                .collect(),
            mapped: None,
            _marker: self._marker,
        }
    }
}

impl<'a, F: Field, B: Basis> Add<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn add(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        parallelize(self.as_mut_slice(), |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.as_slice()[start..].iter()) {
                *lhs += *rhs;
            }
        });

        self
    }
}

impl<'a, F: Field, B: Basis> Sub<&'a Polynomial<F, B>> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(mut self, rhs: &'a Polynomial<F, B>) -> Polynomial<F, B> {
        parallelize(self.as_mut_slice(), |lhs, start| {
            for (lhs, rhs) in lhs.iter_mut().zip(rhs.as_slice()[start..].iter()) {
                *lhs -= *rhs;
            }
        });

        self
    }
}

impl<F: Field> Polynomial<F, LagrangeCoeff> {
    /// Rotates the values in a Lagrange basis polynomial by `Rotation`
    pub fn rotate(&self, rotation: Rotation) -> Polynomial<F, LagrangeCoeff> {
        let mut values = self.as_slice().to_vec();
        if rotation.0 < 0 {
            values.rotate_right((-rotation.0) as usize);
        } else {
            values.rotate_left(rotation.0 as usize);
        }
        Polynomial {
            values,
            mapped: None,
            _marker: PhantomData,
        }
    }
}

impl<F: Field, B: Basis> Mul<F> for Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn mul(mut self, rhs: F) -> Polynomial<F, B> {
        if rhs == F::ZERO {
            return Polynomial {
                values: vec![F::ZERO; self.len()],
                mapped: None,
                _marker: PhantomData,
            };
        }
        if rhs == F::ONE {
            return self;
        }

        parallelize(self.as_mut_slice(), |lhs, _| {
            for lhs in lhs.iter_mut() {
                *lhs *= rhs;
            }
        });

        self
    }
}

impl<'a, F: Field, B: Basis> Sub<F> for &'a Polynomial<F, B> {
    type Output = Polynomial<F, B>;

    fn sub(self, rhs: F) -> Polynomial<F, B> {
        let mut res = self.clone();
        res.as_mut_slice()[0] -= rhs;
        res
    }
}

/// Describes the relative rotation of a vector. Negative numbers represent
/// reverse (leftmost) rotations and positive numbers represent forward (rightmost)
/// rotations. Zero represents no rotation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Rotation(pub i32);

impl Rotation {
    /// The current location in the evaluation domain
    pub fn cur() -> Rotation {
        Rotation(0)
    }

    /// The previous location in the evaluation domain
    pub fn prev() -> Rotation {
        Rotation(-1)
    }

    /// The next location in the evaluation domain
    pub fn next() -> Rotation {
        Rotation(1)
    }
}

#[cfg(test)]
mod native_spill_tests {
    use super::{batch_invert_assigned, native_spill_scope, Assigned, Coeff, LagrangeCoeff, Polynomial};
    use halo2curves::bn256::Fr;

    #[test]
    fn mapped_clone_has_independent_writable_backing() {
        let base = std::env::temp_dir().join(format!(
            "halo2-spill-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&base).unwrap();
        {
            let _scope = native_spill_scope(&base).unwrap();
            let original = Polynomial::<Fr, Coeff>::from_vec(vec![Fr::from(3), Fr::from(5)])
                .spill()
                .unwrap();
            let mut clone = original.clone();
            assert!(original.is_mapped());
            assert!(clone.is_mapped());
            clone[0] = Fr::from(11);
            assert_eq!(original[0], Fr::from(3));
            assert_eq!(clone[0], Fr::from(11));
        }
        assert_eq!(std::fs::read_dir(&base).unwrap().count(), 0);
        std::fs::remove_dir(base).unwrap();
    }

    #[test]
    fn spilled_assigned_inversion_matches_materialized_batch_inversion() {
        let base = std::env::temp_dir().join(format!(
            "halo2-spill-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&base).unwrap();
        {
            let _scope = native_spill_scope(&base).unwrap();
            let assigned = vec![
                Assigned::Zero,
                Assigned::Trivial(Fr::from(9)),
                Assigned::Rational(Fr::from(0), Fr::from(7)),
                Assigned::Rational(Fr::from(11), Fr::from(0)),
                Assigned::Rational(Fr::from(8), Fr::from(2)),
                Assigned::Rational(Fr::from(5), Fr::from(3)),
            ];
            let materialized = batch_invert_assigned(vec![
                Polynomial::<Assigned<Fr>, LagrangeCoeff>::from_vec(assigned.clone()),
            ])
            .remove(0);
            let spilled = Polynomial::<Assigned<Fr>, LagrangeCoeff>::from_vec(assigned)
                .invert_spilled()
                .unwrap();
            assert_eq!(materialized.as_slice(), spilled.as_slice());
        }
        assert_eq!(std::fs::read_dir(&base).unwrap().count(), 0);
        std::fs::remove_dir(base).unwrap();
    }
}
