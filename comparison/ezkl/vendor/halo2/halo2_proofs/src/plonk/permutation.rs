//! Implementation of permutation argument.

use super::circuit::{Any, Column};
use crate::{
    arithmetic::{parallelize, CurveAffine},
    helpers::SerdeCurveAffine,
    poly::{Coeff, EvaluationDomain, LagrangeCoeff, Polynomial},
    SerdeFormat,
};
use ff::{Field, PrimeField};
use group::ff::WithSmallOrderMulGroup;

pub(crate) mod keygen;
pub(crate) mod prover;
pub(crate) mod verifier;

pub use keygen::Assembly;

use std::io;
use std::marker::PhantomData;
use std::fs::File;
use std::sync::Arc;
#[cfg(feature = "formal-permutation-spill")]
use crate::poly::NativeSuccessorStore;

#[derive(Clone, Debug)]
enum SuccessorEntries {
    Memory(Vec<(u64, u64)>),
    #[cfg(feature = "formal-permutation-spill")]
    Spilled(crate::poly::NativeSuccessorStore),
}

impl SuccessorEntries {
    fn new_for_read(len: u64, max_total_backing_bytes: u64) -> io::Result<Self> {
        #[cfg(feature = "formal-permutation-spill")]
        {
            Ok(Self::Spilled(crate::poly::NativeSuccessorStore::new_with_budget(len, max_total_backing_bytes)?))
        }
        #[cfg(not(feature = "formal-permutation-spill"))]
        {
            let capacity = usize::try_from(len)
                .map_err(|_| invalid_mapping("successor count does not fit host index"))?;
            let mut entries = Vec::new();
            entries.try_reserve_exact(capacity).map_err(|error| {
                io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve native successor entries: {error}"))
            })?;
            Ok(Self::Memory(entries))
        }
    }

    fn len(&self) -> u64 {
        match self {
            Self::Memory(entries) => entries.len() as u64,
            #[cfg(feature = "formal-permutation-spill")]
            Self::Spilled(entries) => entries.len(),
        }
    }

    fn append_raw(&mut self, start: u64, raw: &[u8]) -> io::Result<()> {
        if raw.len() % 16 != 0 {
            return Err(invalid_mapping("successor bytes are not pair-aligned"));
        }
        match self {
            Self::Memory(entries) => {
                let first = usize::try_from(start)
                    .map_err(|_| invalid_mapping("successor offset does not fit host index"))?;
                if first != entries.len() {
                    return Err(invalid_mapping("successor entries were appended out of order"));
                }
                for pair in raw.chunks_exact(16) {
                    entries.push((
                        u64::from_le_bytes(pair[..8].try_into().unwrap()),
                        u64::from_le_bytes(pair[8..].try_into().unwrap()),
                    ));
                }
                Ok(())
            }
            #[cfg(feature = "formal-permutation-spill")]
            Self::Spilled(entries) => entries.write_bytes(start, raw),
        }
    }

    fn read_pairs(&self, start: u64, out: &mut [(u64, u64)]) -> io::Result<()> {
        match self {
            Self::Memory(entries) => {
                let start = usize::try_from(start)
                    .map_err(|_| invalid_mapping("successor offset does not fit host index"))?;
                let end = start.checked_add(out.len()).ok_or_else(|| invalid_mapping("successor read range overflow"))?;
                let source = entries.get(start..end).ok_or_else(|| invalid_mapping("successor read exceeds declared pair count"))?;
                out.copy_from_slice(source);
                Ok(())
            }
            #[cfg(feature = "formal-permutation-spill")]
            Self::Spilled(entries) => entries.read_pairs(start, out),
        }
    }

    fn read_pairs_inplace(
        &self,
        start: u64,
        out: &mut [(u64, u64)],
        raw: &mut Vec<u8>,
    ) -> io::Result<()> {
        match self {
            Self::Memory(_) => self.read_pairs(start, out),
            #[cfg(feature = "formal-permutation-spill")]
            Self::Spilled(entries) => entries.read_pairs_inplace(start, out, raw),
        }
    }

    fn pair_at(&self, index: u64) -> io::Result<(u64, u64)> {
        let mut pair = [(0u64, 0u64)];
        self.read_pairs(index, &mut pair)?;
        Ok(pair[0])
    }

    fn lower_bound_source(&self, source: u64) -> io::Result<u64> {
        match self {
            Self::Memory(entries) => Ok(entries.partition_point(|entry| entry.0 < source) as u64),
            #[cfg(feature = "formal-permutation-spill")]
            Self::Spilled(entries) => entries.lower_bound_source(source),
        }
    }

    fn validate(&self, max_cells: u64, max_total_backing_bytes: u64) -> io::Result<()> {
        #[cfg(feature = "formal-permutation-spill")]
        if let Self::Spilled(entries) = self {
            return entries.validate_bijection(max_cells, max_total_backing_bytes).map(|_| ());
        }
        if let Self::Memory(entries) = self {
            validate_successor_permutation(entries)?;
            if entries.iter().any(|&(source, target)| {
                source >= max_cells || target >= max_cells || source == target
            }) {
                return Err(invalid_mapping("successor entry is outside the permutation domain or is identity"));
            }
        }
        Ok(())
    }
}

const MAPPING_PERSONAL: &[u8; 16] = b"EZKL-PermMap-v1!";

pub(crate) fn omega_powers<F: WithSmallOrderMulGroup<3>>(
    domain: &EvaluationDomain<F>,
) -> io::Result<Polynomial<F, LagrangeCoeff>> {
    let n = usize::try_from(domain.get_n()).map_err(|_| invalid_mapping("native domain size does not fit usize"))?;
    let mut powers = Polynomial::mapped_filled(n, F::ONE)?;
    let omega = domain.get_omega();
    for row in 1..n {
        powers[row] = powers[row - 1] * omega;
    }
    Ok(powers)
}

/// A permutation argument.
#[derive(Debug, Clone)]
pub struct Argument {
    /// A sequence of columns involved in the argument.
    pub(super) columns: Vec<Column<Any>>,
}

impl Argument {
    pub(crate) fn new() -> Self {
        Argument { columns: vec![] }
    }

    /// Returns the minimum circuit degree required by the permutation argument.
    /// The argument may use larger degree gates depending on the actual
    /// circuit's degree and how many columns are involved in the permutation.
    pub(crate) fn required_degree(&self) -> usize {
        // degree 2:
        // l_0(X) * (1 - z(X)) = 0
        //
        // We will fit as many polynomials p_i(X) as possible
        // into the required degree of the circuit, so the
        // following will not affect the required degree of
        // this middleware.
        //
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z(\omega X) \prod (p(X) + \beta s_i(X) + \gamma)
        // - z(X) \prod (p(X) + \delta^i \beta X + \gamma)
        // )
        //
        // On the first sets of columns, except the first
        // set, we will do
        //
        // l_0(X) * (z(X) - z'(\omega^(last) X)) = 0
        //
        // where z'(X) is the permutation for the previous set
        // of columns.
        //
        // On the final set of columns, we will do
        //
        // degree 3:
        // l_last(X) * (z'(X)^2 - z'(X)) = 0
        //
        // which will allow the last value to be zero to
        // ensure the argument is perfectly complete.

        // There are constraints of degree 3 regardless of the
        // number of columns involved.
        3
    }

    pub(crate) fn add_column(&mut self, column: Column<Any>) {
        if !self.columns.contains(&column) {
            self.columns.push(column);
        }
    }

    /// Returns columns that participate on the permutation argument.
    pub fn get_columns(&self) -> Vec<Column<Any>> {
        self.columns.clone()
    }
}

/// The verifying key for a single permutation argument.
#[derive(Clone, Debug)]
pub struct VerifyingKey<C: CurveAffine> {
    commitments: Vec<C>,
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Returns commitments of sigma polynomials
    pub fn commitments(&self) -> &Vec<C> {
        &self.commitments
    }

    pub(crate) fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        C: SerdeCurveAffine,
    {
        for commitment in &self.commitments {
            commitment.write(writer, format)?;
        }
        Ok(())
    }

    pub(crate) fn read<R: io::Read>(
        reader: &mut R,
        argument: &Argument,
        format: SerdeFormat,
    ) -> io::Result<Self>
    where
        C: SerdeCurveAffine,
    {
        let commitments = (0..argument.columns.len())
            .map(|_| C::read(reader, format))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(VerifyingKey { commitments })
    }

    pub(crate) fn bytes_length(&self, format: SerdeFormat) -> usize
    where
        C: SerdeCurveAffine,
    {
        self.commitments.len() * C::byte_length(format)
    }
}

/// The proving key for a single permutation argument.
#[derive(Clone, Debug)]
pub(crate) struct ProvingKey<C: CurveAffine> {
    domain_size: u64,
    columns: Vec<Column<Any>>,
    /// Sorted nonidentity source -> successor flat-cell mappings. Unlisted
    /// source cells map to themselves, exactly as native Assembly does.
    successors: SuccessorEntries,
    _marker: PhantomData<C>,
}

impl<C: SerdeCurveAffine> ProvingKey<C>
where
    C::Scalar: Field,
{
    /// Reads the versioned sparse native copy mapping and checks it against
    /// the circuit's ordered permutation columns and domain.
    pub(super) fn read<R: io::Read>(
        reader: &mut R,
        _format: SerdeFormat,
        domain: &EvaluationDomain<C::Scalar>,
        expected_columns: &[Column<Any>],
    ) -> io::Result<Self> {
        Self::read_with_native_backing_budget(reader, _format, domain, expected_columns, u64::MAX)
    }

    pub(super) fn read_with_native_backing_budget<R: io::Read>(reader: &mut R, _format: SerdeFormat, domain: &EvaluationDomain<C::Scalar>, expected_columns: &[Column<Any>], max_total_backing_bytes: u64) -> io::Result<Self> {
        Self::read_with_native_backing_budget_and_timing(
            reader,
            _format,
            domain,
            expected_columns,
            max_total_backing_bytes,
        )
        .map(|(key, _)| key)
    }

    pub(super) fn read_with_native_backing_budget_and_timing<R: io::Read>(
        reader: &mut R,
        _format: SerdeFormat,
        domain: &EvaluationDomain<C::Scalar>,
        expected_columns: &[Column<Any>],
        max_total_backing_bytes: u64,
    ) -> io::Result<(Self, std::time::Duration)> {
        Self::read_internal(
            reader,
            _format,
            domain,
            expected_columns,
            max_total_backing_bytes,
            None,
        )
    }

    /// Reads a compact mapping whose payload is held in the authenticated PK
    /// file. The section offset points at the first permutation-domain word.
    pub(super) fn read_with_backing<R: io::Read>(
        reader: &mut R,
        format: SerdeFormat,
        domain: &EvaluationDomain<C::Scalar>,
        expected_columns: &[Column<Any>],
        max_total_backing_bytes: u64,
        backing: Arc<File>,
        section_offset: u64,
    ) -> io::Result<(Self, std::time::Duration)> {
        Self::read_internal(
            reader,
            format,
            domain,
            expected_columns,
            max_total_backing_bytes,
            Some((backing, section_offset)),
        )
    }

    fn read_internal<R: io::Read>(
        reader: &mut R,
        _format: SerdeFormat,
        domain: &EvaluationDomain<C::Scalar>,
        expected_columns: &[Column<Any>],
        max_total_backing_bytes: u64,
        file_backing: Option<(Arc<File>, u64)>,
    ) -> io::Result<(Self, std::time::Duration)> {
        let mut state = mapping_hasher();
        let domain_size = read_hashed_u64(reader, &mut state)?;
        let column_count = read_hashed_u64(reader, &mut state)?;
        if domain_size != domain.get_n() || column_count != expected_columns.len() as u64 {
            return Err(invalid_mapping(
                "domain size or permutation column count mismatch",
            ));
        }
        for expected in expected_columns {
            let (kind, phase, index) = read_column_identity(reader, &mut state)?;
            if expected.native_storage_identity() != (kind, phase, index) {
                return Err(invalid_mapping("ordered permutation columns mismatch"));
            }
        }
        let edge_count = read_hashed_u64(reader, &mut state)?;
        let max_cells = domain_size
            .checked_mul(column_count)
            .ok_or_else(|| invalid_mapping("permutation domain size overflow"))?;
        if edge_count > max_cells || edge_count > usize::MAX as u64 {
            return Err(invalid_mapping(
                "successor entry count exceeds permutation domain",
            ));
        }
        let bits_per_index = packed_index_width(max_cells);
        if edge_count > 0 && bits_per_index == 0 {
            return Err(invalid_mapping("nonempty successor mapping has a zero-width index"));
        }
        let section_header_bytes = 24u64
            .checked_add(column_count.checked_mul(10).ok_or_else(|| invalid_mapping("permutation section header size overflow"))?)
            .ok_or_else(|| invalid_mapping("permutation section header size overflow"))?;
        let packed_payload_offset = match &file_backing {
            Some((_, section_offset)) => Some(
                section_offset
                    .checked_add(section_header_bytes)
                    .ok_or_else(|| invalid_mapping("packed successor file offset overflow"))?,
            ),
            None => None,
        };
        let mut fallback_successors = if packed_payload_offset.is_none() {
            Some(SuccessorEntries::new_for_read(edge_count, max_total_backing_bytes)?)
        } else {
            None
        };
        let mut previous_source = None;
        const READ_CHUNK_PAIRS: usize = 4096;
        let max_chunk_bytes = READ_CHUNK_PAIRS
            .checked_mul(2)
            .and_then(|count| count.checked_mul(usize::from(bits_per_index)))
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| invalid_mapping("packed successor chunk size overflow"))?;
        let mut packed = Vec::new();
        packed.try_reserve_exact(max_chunk_bytes).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded successor input buffer: {error}"))
        })?;
        let mut decoded_pairs = Vec::new();
        decoded_pairs.try_reserve_exact(READ_CHUNK_PAIRS * 16).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded decoded successor buffer: {error}"))
        })?;
        let mut start = 0u64;
        while start < edge_count {
            let pairs = usize::try_from((edge_count - start).min(READ_CHUNK_PAIRS as u64))
                .map_err(|_| invalid_mapping("successor read chunk length overflow"))?;
            let chunk_bits = pairs
                .checked_mul(2)
                .and_then(|count| count.checked_mul(usize::from(bits_per_index)))
                .ok_or_else(|| invalid_mapping("packed successor read chunk bit length overflow"))?;
            let byte_len = chunk_bits
                .checked_add(7)
                .ok_or_else(|| invalid_mapping("packed successor read chunk size overflow"))?
                / 8;
            packed.resize(byte_len, 0);
            reader.read_exact(&mut packed)?;
            state.update(&packed);
            if start + pairs as u64 == edge_count && chunk_bits % 8 != 0 {
                let used = (chunk_bits % 8) as u8;
                let unused_mask = !((1u8 << used) - 1);
                if packed.last().copied().unwrap_or(0) & unused_mask != 0 {
                    return Err(invalid_mapping("packed successor payload has nonzero tail bits"));
                }
            }
            decoded_pairs.clear();
            for pair_index in 0..pairs {
                let bit_offset = pair_index
                    .checked_mul(2)
                    .and_then(|value| value.checked_mul(usize::from(bits_per_index)))
                    .ok_or_else(|| invalid_mapping("packed successor pair offset overflow"))?;
                let source = decode_packed_index(&packed, bit_offset, bits_per_index)?;
                let target = decode_packed_index(
                    &packed,
                    bit_offset + usize::from(bits_per_index),
                    bits_per_index,
                )?;
                if source >= max_cells || target >= max_cells || source == target
                    || previous_source.map(|previous| previous >= source).unwrap_or(false)
                {
                    return Err(invalid_mapping("successor entry is outside the domain or sources are not strictly ordered"));
                }
                previous_source = Some(source);
                decoded_pairs.extend_from_slice(&source.to_le_bytes());
                decoded_pairs.extend_from_slice(&target.to_le_bytes());
            }
            if let Some(successors) = &mut fallback_successors {
                successors.append_raw(start, &decoded_pairs)?;
            }
            start = start.checked_add(pairs as u64).ok_or_else(|| invalid_mapping("successor read offset overflow"))?;
        }
        let mut successors = match (file_backing, packed_payload_offset, fallback_successors) {
            (Some((file, _)), Some(offset), None) => {
                #[cfg(feature = "formal-permutation-spill")]
                {
                    SuccessorEntries::Spilled(NativeSuccessorStore::packed_file_view(
                        file,
                        offset,
                        edge_count,
                        bits_per_index,
                    )?)
                }
                #[cfg(not(feature = "formal-permutation-spill"))]
                {
                    let _ = (file, offset);
                    return Err(invalid_mapping("file-backed packed successor storage requires formal permutation spill"));
                }
            }
            (_, _, Some(successors)) => successors,
            _ => return Err(invalid_mapping("packed successor storage state is inconsistent")),
        };
        let mut expected_digest = [0u8; 32];
        reader.read_exact(&mut expected_digest)?;
        let actual_digest = state.finalize();
        if actual_digest.as_bytes() != &expected_digest {
            return Err(invalid_mapping("successor mapping payload digest mismatch"));
        }
        // Serialized proving-key inputs are untrusted: authenticate target
        // membership and uniqueness before retaining the lazy source mapping.
        successors.validate(max_cells, max_total_backing_bytes)?;
        let sigma_deferred_source_prepare_started = std::time::Instant::now();
        let key = Self {
            domain_size,
            columns: expected_columns.to_vec(),
            successors,
            _marker: PhantomData,
        };
        Ok((key, sigma_deferred_source_prepare_started.elapsed()))
    }

    /// Writes proving key for a single permutation argument to buffer using `Polynomial::write`.  
    pub(super) fn write<W: io::Write>(
        &self,
        writer: &mut W,
        _format: SerdeFormat,
    ) -> io::Result<()> {
        let mut state = mapping_hasher();
        write_hashed_u64(writer, &mut state, self.domain_size)?;
        write_hashed_u64(writer, &mut state, self.columns.len() as u64)?;
        for column in &self.columns {
            let (kind, phase, index) = column.native_storage_identity();
            write_hashed(writer, &mut state, &[kind, phase])?;
            write_hashed(writer, &mut state, &index.to_le_bytes())?;
        }
        write_hashed_u64(writer, &mut state, self.successors.len())?;
        let max_cells = self
            .domain_size
            .checked_mul(self.columns.len() as u64)
            .ok_or_else(|| invalid_mapping("permutation domain size overflow"))?;
        let bits_per_index = packed_index_width(max_cells);
        if self.successors.len() > 0 && bits_per_index == 0 {
            return Err(invalid_mapping("nonempty successor mapping has a zero-width index"));
        }
        const WRITE_CHUNK_PAIRS: usize = 4096;
        let mut pairs = Vec::new();
        pairs.try_reserve_exact(WRITE_CHUNK_PAIRS).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded successor output buffer: {error}"))
        })?;
        pairs.resize(WRITE_CHUNK_PAIRS, (0u64, 0u64));
        let max_chunk_bytes = WRITE_CHUNK_PAIRS
            .checked_mul(2)
            .and_then(|count| count.checked_mul(usize::from(bits_per_index)))
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(|| invalid_mapping("packed successor serialization chunk size overflow"))?;
        let mut packed = Vec::new();
        packed.try_reserve_exact(max_chunk_bytes).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded packed successor output buffer: {error}"))
        })?;
        let mut unused_raw = Vec::new();
        let mut start = 0u64;
        while start < self.successors.len() {
            let count = usize::try_from((self.successors.len() - start).min(WRITE_CHUNK_PAIRS as u64))
                .map_err(|_| invalid_mapping("successor serialization chunk length overflow"))?;
            self.successors.read_pairs_inplace(start, &mut pairs[..count], &mut unused_raw)?;
            let chunk_bits = count
                .checked_mul(2)
                .and_then(|value| value.checked_mul(usize::from(bits_per_index)))
                .ok_or_else(|| invalid_mapping("packed successor output bit length overflow"))?;
            packed.clear();
            packed.resize(
                chunk_bits
                    .checked_add(7)
                    .ok_or_else(|| invalid_mapping("packed successor output size overflow"))?
                    / 8,
                0,
            );
            let mut bit_offset = 0usize;
            for &(source, target) in &pairs[..count] {
                encode_packed_index(&mut packed, bit_offset, bits_per_index, source)?;
                bit_offset += usize::from(bits_per_index);
                encode_packed_index(&mut packed, bit_offset, bits_per_index, target)?;
                bit_offset += usize::from(bits_per_index);
            }
            write_hashed(writer, &mut state, &packed)?;
            start = start
                .checked_add(count as u64)
                .ok_or_else(|| invalid_mapping("successor serialization offset overflow"))?;
        }
        writer.write_all(state.finalize().as_bytes())?;
        Ok(())
    }
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Gets the total number of bytes in the serialization of `self`
    pub(super) fn bytes_length(&self) -> usize {
        let max_cells = self.domain_size.saturating_mul(self.columns.len() as u64);
        let payload_bits = (self.successors.len() as u128)
            .saturating_mul(2)
            .saturating_mul(u128::from(packed_index_width(max_cells)));
        let payload_bytes = payload_bits.saturating_add(7) / 8;
        8usize
            .saturating_add(8)
            .saturating_add(self.columns.len().saturating_mul(10))
            .saturating_add(8)
            .saturating_add(usize::try_from(payload_bytes).unwrap_or(usize::MAX))
            .saturating_add(32)
    }

    pub(super) fn from_successors(
        domain_size: u64,
        columns: Vec<Column<Any>>,
        successors: Vec<(u64, u64)>,
    ) -> io::Result<Self> {
        validate_successor_permutation(&successors)?;
        let max_cells = domain_size
            .checked_mul(columns.len() as u64)
            .ok_or_else(|| invalid_mapping("permutation domain size overflow"))?;
        if successors
            .iter()
            .any(|&(source, target)| source >= max_cells || target >= max_cells || source == target)
        {
            return Err(invalid_mapping(
                "successor mapping is outside the permutation domain",
            ));
        }
        Ok(Self {
            domain_size,
            columns,
            successors: SuccessorEntries::Memory(successors),
            _marker: PhantomData,
        })
    }

    #[cfg(feature = "formal-permutation-spill")]
    pub(super) fn from_spilled_successors(
        domain: &EvaluationDomain<C::Scalar>,
        columns: Vec<Column<Any>>,
        successors: NativeSuccessorStore,
    ) -> io::Result<Self> {
        let domain_size = domain.get_n();
        // This constructor is used only for mappings produced in this process
        // by the native Assembly DSU. Serialized PK inputs use `read`, which
        // performs strict membership and uniqueness validation first.
        Ok(Self {
            domain_size,
            columns,
            successors: SuccessorEntries::Spilled(successors),
            _marker: PhantomData,
        })
    }

    #[cfg(feature = "formal-permutation-spill")]
    pub(super) fn from_spilled_successors_compact_artifact(
        domain: &EvaluationDomain<C::Scalar>,
        columns: Vec<Column<Any>>,
        successors: NativeSuccessorStore,
    ) -> io::Result<Self> {
        Ok(Self {
            domain_size: domain.get_n(),
            columns,
            successors: SuccessorEntries::Spilled(successors),
            _marker: PhantomData,
        })
    }

    #[cfg(test)]
    pub(super) fn memory_successors(&self) -> Option<&[(u64, u64)]> {
        match &self.successors {
            SuccessorEntries::Memory(entries) => Some(entries),
            #[cfg(feature = "formal-permutation-spill")]
            SuccessorEntries::Spilled(_) => None,
        }
    }

    pub(super) fn domain_size(&self) -> u64 {
        self.domain_size
    }

    pub(crate) fn sigma_lagrange(
        &self,
        domain: &EvaluationDomain<C::Scalar>,
        column: usize,
        omega_powers: &[C::Scalar],
    ) -> io::Result<Polynomial<C::Scalar, LagrangeCoeff>> {
        if domain.get_n() != self.domain_size
            || column >= self.columns.len()
            || omega_powers.len() != self.domain_size as usize
        {
            return Err(invalid_mapping(
                "native sigma reconstruction inputs mismatch",
            ));
        }
        let n = self.domain_size;
        let start = (column as u64)
            .checked_mul(n)
            .ok_or_else(|| invalid_mapping("permutation column offset overflow"))?;
        let max_cells = n
            .checked_mul(self.columns.len() as u64)
            .ok_or_else(|| invalid_mapping("permutation domain size overflow"))?;
        let delta_powers = (0..self.columns.len())
            .map(|index| C::Scalar::DELTA.pow_vartime([index as u64]))
            .collect::<Vec<_>>();
        let mut sigma = Polynomial::mapped_filled(n as usize, C::Scalar::ZERO)?;
        let identity_scale = delta_powers[column];
        parallelize(&mut sigma[..], |values, row_start| {
            for (offset, value) in values.iter_mut().enumerate() {
                *value = identity_scale * omega_powers[row_start + offset];
            }
        });
        const SIGMA_EDGE_CHUNK_PAIRS: usize = 4096;
        let mut edge_buffer = Vec::new();
        edge_buffer.try_reserve_exact(SIGMA_EDGE_CHUNK_PAIRS).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot reserve bounded sigma mapping buffer: {error}"))
        })?;
        edge_buffer.resize(SIGMA_EDGE_CHUNK_PAIRS, (0u64, 0u64));
        let mut edge_bytes = Vec::new();
        edge_bytes
            .try_reserve_exact(SIGMA_EDGE_CHUNK_PAIRS * 16)
            .map_err(|error| {
                io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    format!("cannot reserve bounded sigma mapping byte buffer: {error}"),
                )
            })?;
        let first_edge = self.successors.lower_bound_source(start)?;
        let end = start
            .checked_add(n)
            .ok_or_else(|| invalid_mapping("permutation column end overflow"))?;
        let mut edge = first_edge;
        while edge < self.successors.len() {
            let count = usize::try_from((self.successors.len() - edge).min(SIGMA_EDGE_CHUNK_PAIRS as u64))
                .map_err(|_| invalid_mapping("sigma mapping chunk length overflow"))?;
            self.successors.read_pairs_inplace(
                edge,
                &mut edge_buffer[..count],
                &mut edge_bytes,
            )?;
            let mut consumed = 0usize;
            while consumed < count {
                let (source, target) = edge_buffer[consumed];
                if source >= end {
                    break;
                }
                if source < start || target >= max_cells {
                    return Err(invalid_mapping("sigma mapping entry is outside its permutation column"));
                }
                let row = usize::try_from(source - start)
                    .map_err(|_| invalid_mapping("sigma row offset does not fit usize"))?;
                let target_column = usize::try_from(target / n)
                    .map_err(|_| invalid_mapping("target column does not fit usize"))?;
                let target_row = usize::try_from(target % n)
                    .map_err(|_| invalid_mapping("target row does not fit usize"))?;
                let scale = delta_powers
                    .get(target_column)
                    .ok_or_else(|| invalid_mapping("target column is outside the permutation argument"))?;
                let value = sigma.get_mut(row).ok_or_else(|| invalid_mapping("sigma row is outside the declared domain"))?;
                *value = *scale * omega_powers[target_row];
                consumed += 1;
            }
            if consumed < count && edge_buffer[consumed].0 >= end {
                break;
            }
            edge += count as u64;
        }
        Ok(sigma)
    }

    pub(crate) fn sigma_coeff(
        &self,
        domain: &EvaluationDomain<C::Scalar>,
        column: usize,
        omega_powers: &[C::Scalar],
    ) -> io::Result<Polynomial<C::Scalar, Coeff>>
    where
        C::Scalar: WithSmallOrderMulGroup<3>,
    {
        Ok(domain.lagrange_to_coeff(self.sigma_lagrange(domain, column, omega_powers)?))
    }
}

fn invalid_mapping(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn mapping_hasher() -> blake2b_simd::State {
    blake2b_simd::Params::new()
        .hash_length(32)
        .personal(MAPPING_PERSONAL)
        .to_state()
}

fn write_hashed<W: io::Write>(
    writer: &mut W,
    state: &mut blake2b_simd::State,
    bytes: &[u8],
) -> io::Result<()> {
    writer.write_all(bytes)?;
    state.update(bytes);
    Ok(())
}

fn write_hashed_u64<W: io::Write>(
    writer: &mut W,
    state: &mut blake2b_simd::State,
    value: u64,
) -> io::Result<()> {
    write_hashed(writer, state, &value.to_le_bytes())
}

fn read_hashed_u64<R: io::Read>(
    reader: &mut R,
    state: &mut blake2b_simd::State,
) -> io::Result<u64> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    state.update(&bytes);
    Ok(u64::from_le_bytes(bytes))
}

/// Width of the smallest unsigned representation for every cell in
/// `[0, max_cells)`. A one-cell domain has only the value zero and uses no
/// payload bits; a nonempty nonidentity mapping cannot exist in that domain.
fn packed_index_width(max_cells: u64) -> u8 {
    if max_cells <= 1 {
        0
    } else {
        (64 - (max_cells - 1).leading_zeros()) as u8
    }
}

fn decode_packed_index(bytes: &[u8], bit_offset: usize, width: u8) -> io::Result<u64> {
    let bits = usize::from(width);
    if bits == 0 || bits > 64 {
        return Err(invalid_mapping("packed successor index width is invalid"));
    }
    let shift = bit_offset % 8;
    let byte_offset = bit_offset / 8;
    let byte_count = (shift + bits + 7) / 8;
    let encoded = bytes
        .get(byte_offset..byte_offset.checked_add(byte_count).ok_or_else(|| invalid_mapping("packed successor byte range overflow"))?)
        .ok_or_else(|| invalid_mapping("packed successor index exceeds its chunk"))?;
    let mut value = 0u128;
    for (byte_index, byte) in encoded.iter().enumerate() {
        value |= u128::from(*byte) << (byte_index * 8);
    }
    let mask = if bits == 64 { u64::MAX as u128 } else { (1u128 << bits) - 1 };
    Ok(((value >> shift) & mask) as u64)
}

fn encode_packed_index(bytes: &mut [u8], bit_offset: usize, width: u8, value: u64) -> io::Result<()> {
    let bits = usize::from(width);
    if bits == 0 || bits > 64 || (bits < 64 && value >= (1u64 << bits)) {
        return Err(invalid_mapping("successor index does not fit its packed width"));
    }
    let shift = bit_offset % 8;
    let byte_offset = bit_offset / 8;
    let byte_count = (shift + bits + 7) / 8;
    let end = byte_offset
        .checked_add(byte_count)
        .ok_or_else(|| invalid_mapping("packed successor byte range overflow"))?;
    let encoded = bytes
        .get_mut(byte_offset..end)
        .ok_or_else(|| invalid_mapping("packed successor output exceeds its chunk"))?;
    let shifted = u128::from(value) << shift;
    for (byte_index, byte) in encoded.iter_mut().enumerate() {
        *byte |= (shifted >> (byte_index * 8)) as u8;
    }
    Ok(())
}

fn read_column_identity<R: io::Read>(
    reader: &mut R,
    state: &mut blake2b_simd::State,
) -> io::Result<(u8, u8, u64)> {
    let mut kinds = [0u8; 2];
    reader.read_exact(&mut kinds)?;
    state.update(&kinds);
    let index = read_hashed_u64(reader, state)?;
    if kinds[0] > 2 || (kinds[0] != 1 && kinds[1] != 0) {
        return Err(invalid_mapping("invalid permutation column identity"));
    }
    Ok((kinds[0], kinds[1], index))
}

fn validate_successor_permutation(successors: &[(u64, u64)]) -> io::Result<()> {
    for pair in successors.windows(2) {
        if pair[0].0 >= pair[1].0 {
            return Err(invalid_mapping(
                "successor sources are not strictly ordered",
            ));
        }
    }
        let mut targets = successors.iter().map(|entry| entry.1).collect::<Vec<_>>();
    targets.sort_unstable();
    if targets.windows(2).any(|pair| pair[0] == pair[1])
        || targets
            .iter()
            .copied()
            .ne(successors.iter().map(|entry| entry.0))
    {
        return Err(invalid_mapping(
            "successor entries do not form disjoint native cycles",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod native_mapping_tests {
    use super::ProvingKey;
    use crate::{
        halo2curves::bn256::{Fr, G1Affine},
        plonk::{Any, Column},
        poly::{native_spill_scope, EvaluationDomain},
        SerdeFormat,
    };
    use ff::{Field, PrimeField};
    use std::io::Cursor;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn compact_mapping_roundtrip_rebuilds_dense_sigma_and_rejects_bad_digest() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock before Unix epoch")
            .as_nanos();
        let base =
            std::env::temp_dir().join(format!("halo2-compact-pk-{}-{unique}", std::process::id()));
        std::fs::create_dir(&base).expect("create compact PK test scratch parent");
        let _spill = native_spill_scope(&base).expect("create compact PK scoped scratch");
        let domain = EvaluationDomain::<Fr>::new(3, 3);
        let columns = (0..2)
            .map(|index| Column::new(index, Any::advice()))
            .collect::<Vec<_>>();
        let successors = vec![(0, 8), (8, 0), (11, 14), (14, 11)];
        let original = ProvingKey::<G1Affine>::from_successors(
            domain.get_n(),
            columns.clone(),
            successors.clone(),
        )
        .expect("construct compact native mapping");
        let mut encoded = Vec::new();
        original
            .write(&mut encoded, SerdeFormat::RawBytes)
            .expect("serialize compact native mapping");

        #[cfg(feature = "formal-permutation-spill")]
        {
            let mut reader = Cursor::new(encoded.clone());
            let error = ProvingKey::<G1Affine>::read_with_native_backing_budget(&mut reader, SerdeFormat::RawBytes, &domain, &columns, 4096)
                .expect_err("validation pages must share the successor backing budget");
            assert!(error.to_string().contains("native sparse backing budget exceeded"));
            let mut reader = Cursor::new(encoded.clone());
            ProvingKey::<G1Affine>::read_with_native_backing_budget(&mut reader, SerdeFormat::RawBytes, &domain, &columns, 2 * 4096)
                .expect("combined successor and validation backing budget");
        }

        let mut reader = Cursor::new(encoded.clone());
        let decoded =
            ProvingKey::<G1Affine>::read(&mut reader, SerdeFormat::RawBytes, &domain, &columns)
                .expect("roundtrip compact native mapping");
        let mut decoded_pairs = vec![(0u64, 0u64); successors.len()];
        decoded
            .successors
            .read_pairs(0, &mut decoded_pairs)
            .expect("read bounded mapping pairs");
        assert_eq!(decoded_pairs, successors);
        assert_eq!(decoded.columns, columns);

        let mut omega_powers = vec![Fr::ONE; domain.get_n() as usize];
        for row in 1..omega_powers.len() {
            omega_powers[row] = omega_powers[row - 1] * domain.get_omega();
        }
        for column in 0..columns.len() {
            let sigma = decoded
                .sigma_lagrange(&domain, column, &omega_powers)
                .expect("rebuild sparse sigma");
            for row in 0..domain.get_n() {
                let source = column as u64 * domain.get_n() + row;
                let target = successors
                    .iter()
                    .find_map(|&(from, to)| (from == source).then_some(to))
                    .unwrap_or(source);
                let expected = Fr::DELTA.pow_vartime([target / domain.get_n()])
                    * omega_powers[(target % domain.get_n()) as usize];
                assert_eq!(
                    sigma[row as usize], expected,
                    "dense sigma at ({column}, {row})"
                );
            }
        }

        let mut malformed = encoded;
        *malformed.last_mut().expect("mapping digest bytes") ^= 1;
        let mut reader = Cursor::new(malformed);
        let error =
            ProvingKey::<G1Affine>::read(&mut reader, SerdeFormat::RawBytes, &domain, &columns)
                .expect_err("reject corrupted mapping digest");
        assert!(error
            .to_string()
            .contains("successor mapping payload digest mismatch"));
        drop(_spill);
        std::fs::remove_dir_all(&base).expect("remove compact PK test scratch parent");
    }
}
