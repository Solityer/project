//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;
use group::ff::{Field, FromUniformBytes, PrimeField};

use crate::arithmetic::CurveAffine;
use crate::helpers::{
    SerdeCurveAffine, SerdePrimeField,
};
use crate::poly::{
    Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, PinnedEvaluationDomain,
    Polynomial,
};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};
use crate::SerdeFormat;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

mod assigned;
mod circuit;
pub mod commitment_timing;
mod error;
mod evaluation;
mod keygen;
#[cfg(not(feature = "mv-lookup"))]
mod lookup;
#[cfg(feature = "mv-lookup")]
pub(crate) mod mv_lookup;
pub mod permutation;
mod shuffle;
mod vanishing;

use circuit::SelectorActivationCache;

mod prover;
mod verifier;

pub use assigned::*;
pub use circuit::*;
pub use error::*;
pub use keygen::*;
pub use prover::*;
pub use verifier::*;

pub use evaluation::{
    native_evaluation_resource_shape, native_shplonk_resource_shape, NativeEvaluationShape,
    NativeShplonkShape,
};
pub(crate) use prover::native_shplonk_batch_width;
use evaluation::Evaluator;
use std::io::{self, Read};

const NATIVE_PK_MAGIC: &[u8; 8] = b"EZKLPK\0\x04";
const FIXED_VALUES_RAW: u8 = 0;
const FIXED_VALUES_CODEBOOK: u8 = 1;
const FIXED_VALUES_CODEBOOK_MAX: usize = 256;
const FIXED_VALUES_IO_CHUNK: usize = 64 * 1024;

/// The smallest number of bits that can encode every ID in the declared
/// cardinality. A one-element codebook has a zero-bit tape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PackedIdWidth(u8);

impl PackedIdWidth {
    pub(crate) fn for_cardinality(cardinality: usize) -> io::Result<Self> {
        if cardinality == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID codebook is empty"));
        }
        let bits = if cardinality <= 1 {
            0
        } else {
            usize::BITS - (cardinality - 1).leading_zeros()
        };
        let bits = u8::try_from(bits)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "packed ID width exceeds wire representation"))?;
        Ok(Self(bits))
    }

    pub(crate) fn bits(self) -> usize {
        self.0 as usize
    }
}

/// A little-endian, bit-packed ID sequence backed by the current native spill
/// scope. The same representation is used by compact lookup tapes and fixed
/// columns; wire readers validate IDs and the unused tail bits before use.
#[derive(Clone, Debug)]
pub(crate) struct PackedIdTape {
    bytes: Polynomial<u8, Coeff>,
    width: PackedIdWidth,
    ids: usize,
}

impl PackedIdTape {
    pub(crate) fn byte_len(ids: usize, width: PackedIdWidth) -> io::Result<usize> {
        ids.checked_mul(width.bits())
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed ID tape byte length overflows host size"))
    }

    pub(crate) fn zeroed(ids: usize, width: PackedIdWidth) -> io::Result<Self> {
        let byte_len = Self::byte_len(ids, width)?;
        let bytes = if byte_len == 0 {
            Polynomial::from_vec(Vec::new())
        } else {
            Polynomial::<u8, Coeff>::mapped_filled(byte_len, 0)?
        };
        Ok(Self { bytes, width, ids })
    }

    pub(crate) fn ids(&self) -> usize {
        self.ids
    }

    pub(crate) fn width(&self) -> PackedIdWidth {
        self.width
    }

    pub(crate) fn byte_len_for_tape(&self) -> usize {
        self.bytes.len()
    }

    pub(crate) fn write(&mut self, index: usize, code: usize) -> io::Result<()> {
        if index >= self.ids {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "packed ID write exceeds declared tape length"));
        }
        let bits = self.width.bits();
        if bits == 0 {
            if code != 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "nonzero ID does not fit a zero-bit tape"));
            }
            return Ok(());
        }
        if bits < usize::BITS as usize && code >= (1usize << bits) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID does not fit its selected width"));
        }
        let bit_start = index.checked_mul(bits)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed ID bit offset overflows host size"))?;
        let first_byte = bit_start / 8;
        let shift = bit_start % 8;
        let byte_count = (shift + bits + 7) / 8;
        let bit_mask = ((1u128 << bits) - 1) << shift;
        let mut word = 0u128;
        for byte in 0..byte_count {
            word |= (self.bytes[first_byte + byte] as u128) << (byte * 8);
        }
        word = (word & !bit_mask) | ((code as u128) << shift);
        for byte in 0..byte_count {
            self.bytes[first_byte + byte] = (word >> (byte * 8)) as u8;
        }
        Ok(())
    }

    pub(crate) fn write_codes(&mut self, start_index: usize, codes: &[usize]) -> io::Result<()> {
        let end = start_index.checked_add(codes.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed ID sequence range overflows host size"))?;
        if end > self.ids {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "packed ID sequence exceeds declared tape length"));
        }
        let bits = self.width.bits();
        if bits == 0 {
            if codes.iter().any(|&code| code != 0) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "nonzero ID does not fit a zero-bit tape"));
            }
            return Ok(());
        }
        let start_bit = start_index.checked_mul(bits)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed ID sequence bit offset overflows host size"))?;
        let bit_len = codes.len().checked_mul(bits)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed ID sequence bit length overflows host size"))?;
        let end_bit = start_bit.checked_add(bit_len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed ID sequence end overflows host size"))?;
        let first_byte = start_bit / 8;
        let shift = start_bit % 8;
        let packed_len = shift.checked_add(bit_len)
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "packed ID sequence byte length overflows host size"))?;
        if packed_len == 0 {
            return Ok(());
        }
        let mut packed = Vec::with_capacity(packed_len);
        let mut accumulator = if shift == 0 {
            0u128
        } else {
            (self.bytes[first_byte] as u128) & ((1u128 << shift) - 1)
        };
        let mut accumulator_bits = shift;
        for &code in codes {
            if bits < usize::BITS as usize && code >= (1usize << bits) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID does not fit its selected width"));
            }
            accumulator |= (code as u128) << accumulator_bits;
            accumulator_bits += bits;
            while accumulator_bits >= 8 {
                packed.push(accumulator as u8);
                accumulator >>= 8;
                accumulator_bits -= 8;
            }
        }
        if accumulator_bits > 0 {
            packed.push(accumulator as u8);
        }
        if packed.len() != packed_len {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID sequence encoder produced an unexpected length"));
        }
        let last_byte = first_byte + packed_len - 1;
        let tail_bits = end_bit % 8;
        if tail_bits != 0 {
            let used_mask = (1u8 << tail_bits) - 1;
            packed[packed_len - 1] = (packed[packed_len - 1] & used_mask)
                | (self.bytes[last_byte] & !used_mask);
        }
        self.bytes[first_byte..first_byte + packed_len].copy_from_slice(&packed);
        Ok(())
    }

    pub(crate) fn read(&self, index: usize) -> io::Result<usize> {
        if index >= self.ids {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID read exceeds declared tape length"));
        }
        let bits = self.width.bits();
        if bits == 0 {
            return Ok(0);
        }
        let bit_start = index.checked_mul(bits)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "packed ID bit offset overflows host size"))?;
        let first_byte = bit_start / 8;
        let shift = bit_start % 8;
        let byte_count = (shift + bits + 7) / 8;
        let mut word = 0u128;
        for byte in 0..byte_count {
            word |= (self.bytes[first_byte + byte] as u128) << (byte * 8);
        }
        let mask = (1u128 << bits) - 1;
        Ok(((word >> shift) & mask) as usize)
    }

    pub(crate) fn write_to<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        for chunk in self.bytes.chunks(FIXED_VALUES_IO_CHUNK) {
            writer.write_all(chunk)?;
        }
        Ok(())
    }

    pub(crate) fn read_from<R: io::Read>(
        reader: &mut R,
        ids: usize,
        width: PackedIdWidth,
        cardinality: usize,
    ) -> io::Result<Self> {
        let mut tape = Self::zeroed(ids, width)?;
        for chunk in tape.bytes.chunks_mut(FIXED_VALUES_IO_CHUNK) {
            reader.read_exact(chunk)?;
        }
        tape.validate(cardinality)?;
        Ok(tape)
    }

    pub(crate) fn validate(&self, cardinality: usize) -> io::Result<()> {
        if self.width != PackedIdWidth::for_cardinality(cardinality)? {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID width is not canonical for codebook cardinality"));
        }
        for index in 0..self.ids {
            if self.read(index)? >= cardinality {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID exceeds codebook cardinality"));
            }
        }
        let used_bits = self.ids.checked_mul(self.width.bits())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "packed ID bit length overflows host size"))?;
        let tail_bits = used_bits % 8;
        if tail_bits != 0 && self.bytes.len() > 0 {
            let unused_mask = !((1u8 << tail_bits) - 1);
            if self.bytes[self.bytes.len() - 1] & unused_mask != 0 {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "packed ID tape has nonzero unused tail bits"));
            }
        }
        Ok(())
    }
}

/// Read-only fixed column storage. Dense values are retained only when their
/// cardinality cannot be represented by the bounded codebook; selector
/// columns are created directly from their activation tape.
#[derive(Clone, Debug)]
pub(crate) enum FixedColumnSource<F: Copy> {
    Dense(Polynomial<F, LagrangeCoeff>),
    Compact { codebook: Vec<F>, ids: PackedIdTape },
}

/// Returns the byte size of Halo2's internal read-only fixed-column source for `F`.
///
/// This is intended for conservative resource accounting; it does not expose
/// or permit construction of the internal fixed-column representation.
pub fn fixed_column_source_size_bytes<F: Copy>() -> usize {
    std::mem::size_of::<FixedColumnSource<F>>()
}

impl<F: Copy> FixedColumnSource<F> {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Dense(values) => values.len(),
            Self::Compact { ids, .. } => ids.ids(),
        }
    }

    pub(crate) fn constant_value(&self) -> Option<F> {
        match self {
            Self::Compact { codebook, ids }
                if codebook.len() == 1
                    && ids.width().bits() == 0
                    && ids.byte_len_for_tape() == 0 =>
            {
                Some(codebook[0])
            }
            _ => None,
        }
    }

    pub(crate) fn value(&self, row: usize) -> F {
        match self {
            Self::Dense(values) => values[row],
            Self::Compact { codebook, ids } => codebook[ids.read(row).expect("validated fixed ID row")],
        }
    }

    pub(crate) fn materialize_lagrange(&self) -> io::Result<Polynomial<F, LagrangeCoeff>> {
        match self {
            Self::Dense(values) => values.try_clone(),
            Self::Compact { codebook, ids } => {
                if codebook.is_empty() || codebook.len() > FIXED_VALUES_CODEBOOK_MAX || ids.ids() != self.len() {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "compact fixed source has inconsistent dimensions"));
                }
                let mut values = Polynomial::mapped_filled(ids.ids(), codebook[0])?;
                for (row, value) in values.iter_mut().enumerate() {
                    *value = codebook[ids.read(row)?];
                }
                Ok(values)
            }
        }
    }
}

/// Non-overlapping timing breakdown for loading the compact native proving
/// key. Artifact validation, deterministic PK materialization, and sparse
/// source preparation are reported separately.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativePkReadTiming {
    /// Wall time for artifact reading and validation, excluding PK materialization and sigma preparation.
    pub artifact_read_validate: Duration,
    /// Wall time for compact fixed-value decoding and boundary polynomial rebuild.
    pub pk_materialize: Duration,
    /// Wall time for constructing the deferred sparse sigma source descriptor.
    pub sigma_deferred_source_prepare: Duration,
    /// Deprecated compatibility field. Dense sigma reconstruction no longer occurs on restore.
    pub sigma_rebuild: Duration,
}

struct FixedValueCodebook<F> {
    values: Vec<F>,
    by_fingerprint: HashMap<u64, Vec<u8>>,
}

fn fixed_value_fingerprint<F: PrimeField>(value: &F) -> u64 {
    let mut hasher = DefaultHasher::new();
    value.to_repr().as_ref().hash(&mut hasher);
    hasher.finish()
}

fn fixed_value_codebook_index<F: PrimeField + Copy>(
    value: &F,
    fingerprint: u64,
    codebook: &[F],
    by_fingerprint: &HashMap<u64, Vec<u8>>,
) -> Option<u8> {
    by_fingerprint
        .get(&fingerprint)?
        .iter()
        .copied()
        .find(|index| codebook.get(*index as usize).map_or(false, |entry| entry == value))
}

fn fixed_value_codebook<F: PrimeField + Copy, B>(
    polynomial: &Polynomial<F, B>,
) -> Option<FixedValueCodebook<F>> {
    let mut values = Vec::with_capacity(FIXED_VALUES_CODEBOOK_MAX);
    let mut by_fingerprint = HashMap::with_capacity(FIXED_VALUES_CODEBOOK_MAX + 1);
    for value in polynomial.iter() {
        let fingerprint = fixed_value_fingerprint(value);
        if fixed_value_codebook_index(value, fingerprint, &values, &by_fingerprint).is_some() {
            continue;
        }
        if values.len() == FIXED_VALUES_CODEBOOK_MAX {
            return None;
        }
        let index = values.len() as u8;
        values.push(*value);
        by_fingerprint.entry(fingerprint).or_insert_with(Vec::new).push(index);
    }
    Some(FixedValueCodebook { values, by_fingerprint })
}

impl<F: PrimeField + Copy> FixedColumnSource<F> {
    pub(crate) fn from_polynomial(polynomial: Polynomial<F, LagrangeCoeff>) -> io::Result<Self> {
        let Some(codebook) = fixed_value_codebook(&polynomial) else {
            return Ok(Self::Dense(polynomial));
        };
        let width = PackedIdWidth::for_cardinality(codebook.values.len())?;
        let mut ids = PackedIdTape::zeroed(polynomial.len(), width)?;
        for (row, value) in polynomial.iter().enumerate() {
            let fingerprint = fixed_value_fingerprint(value);
            let index = fixed_value_codebook_index(
                value,
                fingerprint,
                &codebook.values,
                &codebook.by_fingerprint,
            )
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "fixed value disappeared from its codebook"))?;
            ids.write(row, index as usize)?;
        }
        Ok(Self::Compact { codebook: codebook.values, ids })
    }

    pub(crate) fn from_compact(codebook: Vec<F>, ids: PackedIdTape) -> io::Result<Self> {
        if codebook.is_empty() || codebook.len() > FIXED_VALUES_CODEBOOK_MAX {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "fixed source codebook size is outside 1..=256"));
        }
        ids.validate(codebook.len())?;
        Ok(Self::Compact { codebook, ids })
    }
}

struct CountingReader<'a, R> {
    inner: &'a mut R,
    bytes_read: u64,
}

impl<'a, R> CountingReader<'a, R> {
    fn new(inner: &'a mut R) -> Self {
        Self { inner, bytes_read: 0 }
    }
}

impl<R: io::Read> io::Read for CountingReader<'_, R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let read = self.inner.read(buffer)?;
        self.bytes_read = self.bytes_read.checked_add(read as u64).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "proving-key reader offset overflows u64")
        })?;
        Ok(read)
    }
}

fn fixed_values_serialized_length<F: SerdePrimeField>(
    fixed_values: &[FixedColumnSource<F>],
    expected_columns: usize,
    expected_rows: usize,
) -> io::Result<usize> {
    if fixed_values.len() != expected_columns || expected_rows == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed-value shape does not match its VK domain"));
    }
    let _rows_u32 = u32::try_from(expected_rows)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "proving-key domain exceeds fixed-column wire length"))?;
    let scalar_len = F::default().to_repr().as_ref().len();
    let mut total = std::mem::size_of::<u32>();
    for source in fixed_values {
        if source.len() != expected_rows {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed column has an unexpected row count"));
        }
        let encoded_len = match source {
            FixedColumnSource::Compact { codebook, ids } => {
                if codebook.is_empty() || codebook.len() > FIXED_VALUES_CODEBOOK_MAX
                    || ids.width() != PackedIdWidth::for_cardinality(codebook.len())?
                    || ids.ids() != expected_rows
                {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "compact fixed column has inconsistent dimensions"));
                }
                1usize
                    .checked_add(std::mem::size_of::<u16>())
                    .and_then(|length| length.checked_add(std::mem::size_of::<u8>()))
                    .and_then(|length| length.checked_add(codebook.len().checked_mul(scalar_len)?))
                    .and_then(|length| length.checked_add(ids.byte_len_for_tape()))
            }
            FixedColumnSource::Dense(polynomial) => {
                if polynomial.len() != expected_rows {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed column has an unexpected row count"));
                }
            1usize
                .checked_add(std::mem::size_of::<u32>())
                .and_then(|length| length.checked_add(expected_rows.checked_mul(scalar_len)?))
            }
        }
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed-value byte length overflows host size"))?;
        total = total
            .checked_add(encoded_len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed-value byte length overflows host size"))?;
    }
    Ok(total)
}

fn write_fixed_values<F: SerdePrimeField, W: io::Write>(
    fixed_values: &[FixedColumnSource<F>],
    expected_columns: usize,
    expected_rows: usize,
    writer: &mut W,
    format: SerdeFormat,
) -> io::Result<()> {
    if fixed_values.len() != expected_columns || expected_rows == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed-value shape does not match its VK domain"));
    }
    let columns = u32::try_from(expected_columns)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "too many proving-key fixed columns"))?;
    let _rows = u32::try_from(expected_rows)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "proving-key domain exceeds fixed-column wire length"))?;
    writer.write_all(&columns.to_be_bytes())?;
    for source in fixed_values {
        if source.len() != expected_rows {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed column has an unexpected row count"));
        }
        match source {
            FixedColumnSource::Compact { codebook, ids } => {
                if codebook.is_empty() || codebook.len() > FIXED_VALUES_CODEBOOK_MAX
                    || ids.width() != PackedIdWidth::for_cardinality(codebook.len())?
                    || ids.ids() != expected_rows
                {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "compact fixed column has inconsistent dimensions"));
                }
                writer.write_all(&[FIXED_VALUES_CODEBOOK])?;
                writer.write_all(&(codebook.len() as u16).to_be_bytes())?;
                writer.write_all(&[ids.width().bits() as u8])?;
                for value in codebook {
                    value.write(writer, format)?;
                }
                ids.write_to(writer)?;
            }
            FixedColumnSource::Dense(polynomial) => {
                if polynomial.len() != expected_rows {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "proving-key fixed column has an unexpected row count"));
                }
                writer.write_all(&[FIXED_VALUES_RAW])?;
                polynomial.write(writer, format)?;
            }
        }
    }
    Ok(())
}

fn read_fixed_values<F: SerdePrimeField, R: io::Read>(
    reader: &mut R,
    format: SerdeFormat,
    expected_columns: usize,
    expected_rows: usize,
) -> io::Result<Vec<FixedColumnSource<F>>> {
    let invalid = |message| io::Error::new(io::ErrorKind::InvalidData, message);
    let mut column_count = [0u8; 4];
    reader.read_exact(&mut column_count)?;
    let column_count = usize::try_from(u32::from_be_bytes(column_count))
        .map_err(|_| invalid("proving-key fixed-value column count exceeds host index"))?;
    if column_count != expected_columns || expected_rows == 0 {
        return Err(invalid("proving-key fixed-value count or domain length does not match its VK"));
    }
    let _rows_u32 = u32::try_from(expected_rows)
        .map_err(|_| invalid("proving-key domain exceeds fixed-column wire length"))?;
    let scalar_len = F::default().to_repr().as_ref().len();
    let _field_bytes = expected_rows
        .checked_mul(scalar_len)
        .ok_or_else(|| invalid("proving-key fixed-column byte length overflows host size"))?;

    let mut fixed_values = Vec::with_capacity(expected_columns);
    for _ in 0..expected_columns {
        let mut encoding = [0u8; 1];
        reader.read_exact(&mut encoding)?;
        match encoding[0] {
            FIXED_VALUES_RAW => {
                let mut length = [0u8; 4];
                reader.read_exact(&mut length)?;
                let length = u32::from_be_bytes(length);
                let length_usize = usize::try_from(length)
                    .map_err(|_| invalid("raw proving-key fixed-column row count exceeds host index"))?;
                if length_usize != expected_rows {
                    return Err(invalid("raw proving-key fixed column has an unexpected row count"));
                }
                let mut polynomial = Polynomial::mapped_filled(expected_rows, F::ZERO)?;
                for value in polynomial.iter_mut() {
                    *value = F::read(reader, format)?;
                }
                fixed_values.push(FixedColumnSource::Dense(polynomial));
            }
            FIXED_VALUES_CODEBOOK => {
                let mut count = [0u8; 2];
                reader.read_exact(&mut count)?;
                let count = usize::from(u16::from_be_bytes(count));
                if count == 0 || count > FIXED_VALUES_CODEBOOK_MAX {
                    return Err(invalid("proving-key fixed-value codebook size is outside 1..=256"));
                }
                let mut width = [0u8; 1];
                reader.read_exact(&mut width)?;
                let width = PackedIdWidth(width[0]);
                if width != PackedIdWidth::for_cardinality(count)? {
                    return Err(invalid("proving-key fixed-value ID width is not canonical"));
                }
                count
                    .checked_mul(scalar_len)
                    .ok_or_else(|| invalid("proving-key fixed-value codebook byte length overflows host size"))?;
                let mut codebook = Vec::with_capacity(count);
                let mut by_fingerprint = HashMap::with_capacity(count);
                for _ in 0..count {
                    let value = F::read(reader, format)?;
                    let fingerprint = fixed_value_fingerprint(&value);
                    if fixed_value_codebook_index(&value, fingerprint, &codebook, &by_fingerprint).is_some() {
                        return Err(invalid("proving-key fixed-value codebook contains a duplicate"));
                    }
                    let index = codebook.len() as u8;
                    codebook.push(value);
                    by_fingerprint.entry(fingerprint).or_insert_with(Vec::new).push(index);
                }

                let ids = PackedIdTape::read_from(reader, expected_rows, width, count)?;
                fixed_values.push(FixedColumnSource::Compact { codebook, ids });
            }
            _ => return Err(invalid("unsupported proving-key fixed-value encoding")),
        }
    }
    Ok(fixed_values)
}

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    fixed_commitments: Vec<C>,
    permutation: permutation::VerifyingKey<C>,
    cs: ConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: C::Scalar,
    selectors: SelectorActivationCache,
    /// Whether selector compression is turned on or not.
    compress_selectors: bool,
}

// Current version of the VK
const VERSION: u8 = 0x04;

impl<C: SerdeCurveAffine> VerifyingKey<C>
where
    C::Scalar: SerdePrimeField + FromUniformBytes<64>,
{
    /// Writes a verifying key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in standard form.
    /// Writes a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    /// Writes a field element into raw bytes in its internal Montgomery representation,
    /// WITHOUT performing the expensive Montgomery reduction.
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        // Version byte that will be checked on read.
        writer.write_all(&[VERSION])?;
        let k = &self.domain.k();
        assert!(*k <= C::Scalar::S);
        // k value fits in 1 byte
        writer.write_all(&[*k as u8])?;
        writer.write_all(&[self.compress_selectors as u8])?;
        writer.write_all(&(self.fixed_commitments.len() as u32).to_le_bytes())?;
        for commitment in &self.fixed_commitments {
            commitment.write(writer, format)?;
        }
        self.permutation.write(writer, format)?;

        if !self.compress_selectors {
            assert!(self.selectors.patterns.is_empty());
            assert!(self.selectors.selector_pattern_indices.is_empty());
        } else {
            self.selectors.write(writer)?;
        }
        Ok(())
    }

    /// Reads a verification key from a buffer.
    ///
    /// Reads a curve element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    /// Reads a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    /// Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    /// does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        let mut version_byte = [0u8; 1];
        reader.read_exact(&mut version_byte)?;
        if VERSION != version_byte[0] {
            let message = if version_byte[0] == 0x03 {
                "native VK v3 selector cache is incompatible; regenerate native VK/PK artifacts"
            } else {
                "unexpected native VK version byte"
            };
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                message,
            ));
        }

        let mut k = [0u8; 1];
        reader.read_exact(&mut k)?;
        let k = u8::from_le_bytes(k);
        if k as u32 > C::Scalar::S {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "circuit size value (k): {} exceeds maxium: {}",
                    k,
                    C::Scalar::S
                ),
            ));
        }
        let mut compress_selectors = [0u8; 1];
        reader.read_exact(&mut compress_selectors)?;
        if compress_selectors[0] != 0 && compress_selectors[0] != 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected compress_selectors not boolean",
            ));
        }
        let compress_selectors = compress_selectors[0] == 1;
        let (domain, cs, _) = keygen::create_domain::<C, ConcreteCircuit>(
            k as u32,
            #[cfg(feature = "circuit-params")]
            params,
        )
        .map_err(|error| match error {
            Error::NativeStorage(error) | Error::Transcript(error) => error,
            error => io::Error::new(io::ErrorKind::InvalidData, error.to_string()),
        })?;
        let mut num_fixed_columns = [0u8; 4];
        reader.read_exact(&mut num_fixed_columns)?;
        let num_fixed_columns = u32::from_le_bytes(num_fixed_columns);

        let fixed_commitments: Vec<_> = (0..num_fixed_columns)
            .map(|_| C::read(reader, format))
            .collect::<Result<_, _>>()?;

        let permutation = permutation::VerifyingKey::read(reader, &cs.permutation, format)?;

        let (cs, selectors) = if compress_selectors {
            let selectors = SelectorActivationCache::read(
                reader,
                1usize << k,
                cs.num_selectors,
            )?;
            let (cs, _, _) = cs.compress_selector_activation_cache(&selectors);
            (cs, selectors)
        } else {
            // we still need to replace selectors with fixed Expressions in `cs`
            let fake_selectors = vec![vec![]; cs.num_selectors];
            let (cs, _) = cs.directly_convert_selectors_to_fixed(fake_selectors, false);
            (cs, SelectorActivationCache::default())
        };

        Ok(Self::from_parts(
            domain,
            fixed_commitments,
            permutation,
            cs,
            selectors,
            compress_selectors,
        ))
    }

    /// Writes a verifying key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length(format));
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        bytes
    }

    /// Reads a verification key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<C::Scalar>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    fn bytes_length(&self, format: SerdeFormat) -> usize
    where
        C: SerdeCurveAffine,
    {
        7 + (self.fixed_commitments.len() * C::byte_length(format))
            + self.permutation.bytes_length(format)
            + if self.compress_selectors {
                self.selectors.encoded_byte_len()
            } else {
                0
            }
    }

    fn from_parts(
        domain: EvaluationDomain<C::Scalar>,
        fixed_commitments: Vec<C>,
        permutation: permutation::VerifyingKey<C>,
        cs: ConstraintSystem<C::Scalar>,
        selectors: SelectorActivationCache,
        compress_selectors: bool,
    ) -> Self
    where
        C::ScalarExt: FromUniformBytes<64>,
    {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: C::Scalar::ZERO,
            selectors,
            compress_selectors,
        };

        let mut hasher = Blake2bParams::new()
            .hash_length(64)
            .personal(b"Halo2-Verify-Key")
            .to_state();

        let s = format!("{:?}", vk.pinned());

        hasher.update(&(s.len() as u64).to_le_bytes());
        hasher.update(s.as_bytes());

        // Hash in final Blake2bState
        vk.transcript_repr = C::Scalar::from_uniform_bytes(hasher.finalize().as_array());

        vk
    }

    /// Hashes a verification key into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, C> {
        PinnedVerificationKey {
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }

    /// Returns commitments of fixed polynomials
    pub fn fixed_commitments(&self) -> &Vec<C> {
        &self.fixed_commitments
    }

    /// Returns `VerifyingKey` of permutation
    pub fn permutation(&self) -> &permutation::VerifyingKey<C> {
        &self.permutation
    }

    /// Returns `ConstraintSystem`
    pub fn cs(&self) -> &ConstraintSystem<C::Scalar> {
        &self.cs
    }

    /// Returns representative of this `VerifyingKey` in transcripts
    pub fn transcript_repr(&self) -> C::Scalar {
        self.transcript_repr
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, C: CurveAffine> {
    base_modulus: &'static str,
    scalar_modulus: &'static str,
    domain: PinnedEvaluationDomain<'a, C::Scalar>,
    cs: PinnedConstraintSystem<'a, C::Scalar>,
    fixed_commitments: &'a Vec<C>,
    permutation: &'a permutation::VerifyingKey<C>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<C: CurveAffine> {
    vk: VerifyingKey<C>,
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_active_row: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    pub(crate) fixed_values: Vec<FixedColumnSource<C::Scalar>>,
    permutation: permutation::ProvingKey<C>,
    ev: Evaluator<C>,
}

impl<C: CurveAffine> ProvingKey<C>
where
    C::Scalar: FromUniformBytes<64>,
{
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        &self.vk
    }

    /// Gets the total number of bytes in the serialization of `self`
    fn bytes_length(&self, format: SerdeFormat) -> io::Result<usize>
    where
        C: SerdeCurveAffine,
        C::Scalar: SerdePrimeField,
    {
        let rows = usize::try_from(self.vk.domain.get_n())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "proving-key domain exceeds host index"))?;
        let fixed_values_len = fixed_values_serialized_length(
            &self.fixed_values,
            self.vk.fixed_commitments.len(),
            rows,
        )?;
        self.vk
            .bytes_length(format)
            .checked_add(NATIVE_PK_MAGIC.len())
            .and_then(|length| length.checked_add(fixed_values_len))
            .and_then(|length| length.checked_add(self.permutation.bytes_length()))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "proving-key serialized length overflows host size"))
    }
}

impl<C: SerdeCurveAffine> ProvingKey<C>
where
    C::Scalar: SerdePrimeField + FromUniformBytes<64>,
{
    /// Writes a proving key to a buffer.
    ///
    /// Writes a curve element according to `format`:
    /// - `Processed`: Writes a compressed curve element with coordinates in standard form.
    /// Writes a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation.
    /// - Otherwise: Writes an uncompressed curve element with coordinates in Montgomery form
    /// Writes a field element into raw bytes in its internal Montgomery representation,
    /// WITHOUT performing the expensive Montgomery reduction.
    /// Writes the verifying key, compact fixed-value columns, and permutation key.
    /// Fixed coefficient polynomials and boundary polynomials are reconstructed on read.
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        let rows = usize::try_from(self.vk.domain.get_n())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "proving-key domain exceeds host index"))?;
        writer.write_all(NATIVE_PK_MAGIC)?;
        self.vk.write(writer, format)?;
        write_fixed_values(
            &self.fixed_values,
            self.vk.fixed_commitments.len(),
            rows,
            writer,
            format,
        )?;
        self.permutation.write(writer, format)?;
        Ok(())
    }

    /// Reads a proving key from a buffer.
    /// Does so by reading verification key first, and then deserializing the rest of the file into the remaining proving key data.
    ///
    /// Reads a curve element from the buffer and parses it according to the `format`:
    /// - `Processed`: Reads a compressed curve element and decompresses it.
    /// Reads a field element in standard form, with endianness specified by the
    /// `PrimeField` implementation, and checks that the element is less than the modulus.
    /// - `RawBytes`: Reads an uncompressed curve element with coordinates in Montgomery form.
    /// Checks that field elements are less than modulus, and then checks that the point is on the curve.
    /// - `RawBytesUnchecked`: Reads an uncompressed curve element with coordinates in Montgomery form;
    /// does not perform any checks
    pub fn read<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read_with_native_backing_budget::<R, ConcreteCircuit>(reader, format, u64::MAX, #[cfg(feature = "circuit-params")] params)
    }

    /// Reads a proving key with a checked combined cap for the native sparse
    /// permutation mapping and the temporary target-membership bitmap.
    pub fn read_with_native_backing_budget<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(reader: &mut R, format: SerdeFormat, max_total_backing_bytes: u64, #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params) -> io::Result<Self> {
        Self::read_with_native_backing_budget_and_timing::<R, ConcreteCircuit>(
            reader,
            format,
            max_total_backing_bytes,
            #[cfg(feature = "circuit-params")]
            params,
        )
        .map(|(key, _)| key)
    }

    /// Reads a compact proving key and reports artifact validation, deterministic
    /// PK materialization, and deferred sparse sigma-source preparation separately.
    pub fn read_with_native_backing_budget_and_timing<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        max_total_backing_bytes: u64,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<(Self, NativePkReadTiming)> {
        Self::read_with_native_backing_budget_and_timing_inner::<R, ConcreteCircuit>(
            reader,
            format,
            max_total_backing_bytes,
            None,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }

    /// Reads a proving key from the authenticated file that owns its sparse
    /// permutation payload. The file-backed source is read-only and never
    /// assumes ownership of the PK artifact. `reader` and `backing` must refer
    /// to the same file, with `reader` positioned at byte zero on entry, so the
    /// checked logical byte count identifies the permutation section offset.
    pub fn read_with_native_backing_budget_and_timing_from_file<
        R: io::Read,
        ConcreteCircuit: Circuit<C::Scalar>,
    >(
        reader: &mut R,
        format: SerdeFormat,
        max_total_backing_bytes: u64,
        backing: Arc<File>,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<(Self, NativePkReadTiming)> {
        Self::read_with_native_backing_budget_and_timing_inner::<R, ConcreteCircuit>(
            reader,
            format,
            max_total_backing_bytes,
            Some(backing),
            #[cfg(feature = "circuit-params")]
            params,
        )
    }

    fn read_with_native_backing_budget_and_timing_inner<
        R: io::Read,
        ConcreteCircuit: Circuit<C::Scalar>,
    >(
        reader: &mut R,
        format: SerdeFormat,
        max_total_backing_bytes: u64,
        backing: Option<Arc<File>>,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<(Self, NativePkReadTiming)> {
        let load_started = Instant::now();
        let mut reader = CountingReader::new(reader);
        let mut magic = [0u8; NATIVE_PK_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != NATIVE_PK_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported or legacy native proving key format",
            ));
        }
        let vk = VerifyingKey::<C>::read::<_, ConcreteCircuit>(
            &mut reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        let rows = usize::try_from(vk.domain.get_n())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "proving-key domain exceeds host index"))?;
        let materialize_started = Instant::now();
        let fixed_values = read_fixed_values::<C::Scalar, _>(
            &mut reader,
            format,
            vk.fixed_commitments.len(),
            rows,
        )?;
        let (l0, l_last, l_active_row) = keygen::compute_prover_boundary_polynomials::<C>(
            &vk.domain,
            vk.cs.blinding_factors(),
        )?;
        let pk_materialize = materialize_started.elapsed();
        let sigma_section_offset = reader.bytes_read;
        let (permutation, sigma_deferred_source_prepare) = match backing {
            Some(backing) => permutation::ProvingKey::read_with_backing(
                &mut reader,
                format,
                &vk.domain,
                &vk.cs.permutation().get_columns(),
                max_total_backing_bytes,
                backing,
                sigma_section_offset,
            )?,
            None => permutation::ProvingKey::read_with_native_backing_budget_and_timing(
                &mut reader,
                format,
                &vk.domain,
                &vk.cs.permutation().get_columns(),
                max_total_backing_bytes,
            )?,
        };
        let artifact_read_validate = load_started
            .elapsed()
            .saturating_sub(sigma_deferred_source_prepare)
            .saturating_sub(pk_materialize);
        let ev = Evaluator::new(vk.cs());
        Ok((
            Self {
                vk,
                l0,
                l_last,
                l_active_row,
                fixed_values,
                permutation,
                ev,
            },
            NativePkReadTiming {
                artifact_read_validate,
                pk_materialize,
                sigma_deferred_source_prepare,
                sigma_rebuild: Duration::ZERO,
            },
        ))
    }

    /// Writes a proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let capacity = self.bytes_length(format).expect("proving-key serialized length must fit host size");
        let mut bytes = Vec::<u8>::with_capacity(capacity);
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
        debug_assert_eq!(bytes.len(), capacity);
        bytes
    }

    /// Reads a proving key from a slice of bytes using [`Self::read`].
    pub fn from_bytes<ConcreteCircuit: Circuit<C::Scalar>>(
        mut bytes: &[u8],
        format: SerdeFormat,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<Self> {
        Self::read::<_, ConcreteCircuit>(
            &mut bytes,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )
    }
}

#[cfg(test)]
mod native_pk_format_tests {
    use super::{Circuit, ConstraintSystem, ProvingKey, SerdeFormat};
    use crate::{
        circuit::{Layouter, SimpleFloorPlanner},
        halo2curves::bn256::{Fr, G1Affine},
    };
    use std::io::Cursor;

    struct LegacyFormatCircuit;

    impl Circuit<Fr> for LegacyFormatCircuit {
        type Config = ();
        type FloorPlanner = SimpleFloorPlanner;
        #[cfg(feature = "circuit-params")]
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self
        }

        fn configure(_meta: &mut ConstraintSystem<Fr>) -> Self::Config {}

        fn synthesize(
            &self,
            _config: Self::Config,
            _layouter: impl Layouter<Fr>,
        ) -> Result<(), super::Error> {
            Ok(())
        }
    }

    #[test]
    fn legacy_proving_key_version_is_rejected_before_payload_decode() {
        let mut reader = Cursor::new(b"EZKLPK\0\x01".to_vec());
        #[cfg(feature = "circuit-params")]
        let result = ProvingKey::<G1Affine>::read::<_, LegacyFormatCircuit>(
            &mut reader,
            SerdeFormat::RawBytes,
            (),
        );
        #[cfg(not(feature = "circuit-params"))]
        let result = ProvingKey::<G1Affine>::read::<_, LegacyFormatCircuit>(
            &mut reader,
            SerdeFormat::RawBytes,
        );
        let error = result.expect_err("legacy native proving key version must fail closed");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
        assert!(error
            .to_string()
            .contains("unsupported or legacy native proving key format"));
    }

}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<C::Scalar> {
        &self.domain
    }
}

#[derive(Clone, Copy, Debug)]
struct Theta;
type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

#[derive(Clone, Copy, Debug)]
struct Beta;
type ChallengeBeta<F> = ChallengeScalar<F, Beta>;

#[derive(Clone, Copy, Debug)]
struct Gamma;
type ChallengeGamma<F> = ChallengeScalar<F, Gamma>;

#[derive(Clone, Copy, Debug)]
struct Y;
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Clone, Copy, Debug)]
struct X;
type ChallengeX<F> = ChallengeScalar<F, X>;
