use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeTheta, ChallengeX, Error, FixedColumnSource,
    PackedIdTape, PackedIdWidth, ProvingKey, VerifyingKey,
};
use super::Argument;
use crate::helpers::SerdeCurveAffine;
use crate::plonk::evaluation::evaluate;
use crate::SerdeFormat;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    poly::{
        commitment::{validate_sparse_lagrange_indices, Blind, Params},
        Coeff, EvaluationDomain, LagrangeCoeff, Polynomial, ProverQuery, Rotation,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};
use ff::WithSmallOrderMulGroup;
use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use rustc_hash::FxHashMap as HashMap;

use maybe_rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
    ParallelIterator, ParallelSliceMut,
};
use std::{
    io,
    iter,
    ops::{Mul, MulAssign},
    sync::{Arc, Mutex},
};

fn compress_expressions<C: CurveAffine>(
    expressions: &[Expression<C::Scalar>],
    vk: &VerifyingKey<C>,
    n: usize,
    domain: &EvaluationDomain<C::Scalar>,
    theta: ChallengeTheta<C>,
    advice_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
    fixed_values: &[FixedColumnSource<C::Scalar>],
    instance_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
    challenges: &[C::Scalar],
) -> Result<Polynomial<C::Scalar, LagrangeCoeff>, Error> {
    expressions
        .iter()
        .map(|expression| {
            vk.domain.lagrange_from_vec(evaluate(
                expression,
                n,
                1,
                fixed_values,
                advice_values,
                instance_values,
                challenges,
            ))
        })
        .fold(domain.empty_lagrange(), |acc, expression| {
            acc * *theta + &expression
        })
        .spill()
        .map_err(Error::NativeStorage)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::plonk) struct TableMembership {
    code: usize,
    winner_row: usize,
}

#[derive(Debug)]
struct CompactTable<F> {
    codebook: Vec<F>,
    winner_rows: Vec<usize>,
    row_codes: PackedIdTape,
    active_rows: usize,
    beta_inverses: Mutex<Option<Arc<Vec<F>>>>,
}

impl<F: Copy + Field> CompactTable<F> {
    fn beta_inverses(&self, beta: F) -> io::Result<Arc<Vec<F>>> {
        let mut cached = self.beta_inverses.lock().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "lookup inverse-codebook cache is poisoned")
        })?;
        if let Some(inverses) = cached.as_ref() {
            return Ok(Arc::clone(inverses));
        }
        let mut inverses = Vec::new();
        inverses.try_reserve_exact(self.codebook.len()).map_err(|error| {
            io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot allocate lookup inverse codebook: {error}"))
        })?;
        inverses.extend(self.codebook.iter().map(|value| beta + value));
        inverses.iter_mut().batch_invert();
        let inverses = Arc::new(inverses);
        *cached = Some(Arc::clone(&inverses));
        Ok(inverses)
    }
}

pub(in crate::plonk) struct PreparedTable<C: CurveAffine> {
    compact: Arc<CompactTable<C::Scalar>>,
    membership: HashMap<Vec<u8>, TableMembership>,
}

impl<C: CurveAffine> PreparedTable<C> {
    pub(in crate::plonk) fn membership_map_shape(&self) -> (usize, usize) {
        (self.membership.len(), self.membership.capacity())
    }

    pub(in crate::plonk) fn compact_storage_shape(&self) -> (usize, usize, usize, usize, usize) {
        (
            self.compact.active_rows,
            PackedIdTape::byte_len(self.compact.row_codes.ids(), self.compact.row_codes.width())
                .unwrap_or(usize::MAX),
            self.compact.codebook.len(),
            self.compact.codebook.capacity(),
            self.compact.winner_rows.capacity(),
        )
    }
}

fn insert_table_membership(
    index: &mut HashMap<Vec<u8>, TableMembership>,
    representation: &[u8],
    code: usize,
    row: usize,
    cardinality_upper: usize,
) -> Result<bool, Error> {
    if let Some(previous) = index.get_mut(representation) {
        previous.winner_row = row;
        Ok(false)
    } else {
        if index.len() >= cardinality_upper {
            return Err(Error::NativeStorage(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("native mv-lookup table cardinality exceeded admitted bound {}", cardinality_upper),
            )));
        }
        index.insert(representation.to_owned(), TableMembership { code, winner_row: row });
        Ok(true)
    }
}

#[derive(Debug)]
pub(in crate::plonk) struct Prepared<C: CurveAffine> {
    argument_index: usize,
    theta: ChallengeTheta<C>,
    table: Arc<CompactTable<C::Scalar>>,
    input_codes: Arc<PackedIdTape>,
    input_count: usize,
    counts_by_code: Vec<u64>,
    pub(in crate::plonk) commitment: C,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LookupPolynomialKind {
    Multiplicity,
    GrandSum,
}

#[derive(Debug)]
struct CompactLookupPolynomials<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    n: usize,
    blinding_factors: usize,
    table: Arc<CompactTable<C::Scalar>>,
    input_codes: Arc<PackedIdTape>,
    input_count: usize,
    counts_by_code: Vec<u64>,
    inverse_by_code: Arc<Vec<C::Scalar>>,
    phi_blinds: Vec<C::Scalar>,
}

#[derive(Debug)]
enum LookupPolynomialBacking<C: CurveAffine> {
    Compact(CompactLookupPolynomials<C>),
    Dense {
        m_poly: Polynomial<C::Scalar, Coeff>,
        phi_poly: Polynomial<C::Scalar, Coeff>,
    },
}

#[derive(Debug)]
pub(crate) struct LookupPolynomialSource<C: CurveAffine> {
    backing: LookupPolynomialBacking<C>,
}

impl<C: CurveAffine> LookupPolynomialSource<C> {
    fn compact(backing: CompactLookupPolynomials<C>) -> Self {
        Self { backing: LookupPolynomialBacking::Compact(backing) }
    }

    fn dense(m_poly: Polynomial<C::Scalar, Coeff>, phi_poly: Polynomial<C::Scalar, Coeff>) -> Self {
        Self { backing: LookupPolynomialBacking::Dense { m_poly, phi_poly } }
    }

    pub(crate) fn materialize_coeff(
        &self,
        kind: LookupPolynomialKind,
    ) -> io::Result<Polynomial<C::Scalar, Coeff>>
    where
        C::Scalar: ff::WithSmallOrderMulGroup<3>,
    {
        match &self.backing {
            LookupPolynomialBacking::Compact(compact) => {
                let lagrange = compact.materialize_lagrange(kind)?;
                Ok(compact.domain.lagrange_to_coeff(lagrange))
            }
            LookupPolynomialBacking::Dense { m_poly, phi_poly } => match kind {
                LookupPolynomialKind::Multiplicity => m_poly.try_clone(),
                LookupPolynomialKind::GrandSum => phi_poly.try_clone(),
            },
        }
    }

    pub(crate) fn evaluate_sparse_m(
        &self,
        domain: &EvaluationDomain<C::Scalar>,
        x: C::Scalar,
    ) -> io::Result<Option<C::Scalar>>
    where
        C::Scalar: WithSmallOrderMulGroup<3>,
    {
        let compact = match &self.backing {
            LookupPolynomialBacking::Compact(compact) => compact,
            LookupPolynomialBacking::Dense { .. } => return Ok(None),
        };

        // Preserve the prior materializer's metadata contract before the
        // sparse path can return early, including winner rows for zero counts.
        for code in 0..compact.counts_by_code.len() {
            let row = *compact.table.winner_rows.get(code).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "lookup winner row is missing")
            })?;
            if row >= compact.table.active_rows || row >= compact.n {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "lookup winner row is outside active rows",
                ));
            }
        }

        let logical_len = domain.get_n();
        let logical_len_usize = usize::try_from(logical_len).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "sparse mv-lookup domain length does not fit the host index type",
            )
        })?;
        if u64::try_from(compact.n).ok() != Some(logical_len)
            || compact.domain.get_n() != logical_len
            || compact.domain.get_omega() != domain.get_omega()
            || compact.table.active_rows
                .checked_add(compact.blinding_factors)
                .and_then(|rows| rows.checked_add(1))
                != Some(logical_len_usize)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "compact mv-lookup multiplicity dimensions do not match the evaluation domain",
            ));
        }

        let used_winner_rows = &compact.table.winner_rows[..compact.counts_by_code.len()];
        let entries = sparse_multiset_entries::<C::Scalar>(
            &compact.counts_by_code,
            used_winner_rows,
            compact.table.active_rows,
            logical_len,
            logical_len,
        )?;
        let entries = match entries {
            Some(entries) => entries,
            None => return Ok(None),
        };
        validate_sparse_lagrange_indices(logical_len, logical_len, &entries)?;
        if entries.is_empty() {
            return Ok(Some(C::Scalar::ZERO));
        }

        // The old dense materializer assigns zero rows too, so a zero row
        // overlapping a supported row can overwrite that multiplicity.
        // Keep that exact behavior through the unchanged dense fallback.
        let mut entries = entries;
        entries.sort_unstable_by_key(|(row, _)| *row);
        for (code, &count) in compact.counts_by_code.iter().enumerate() {
            if count == 0 {
                let zero_row = used_winner_rows[code];
                if entries
                    .binary_search_by_key(&zero_row, |(row, _)| *row)
                    .is_ok()
                {
                    return Ok(None);
                }
            }
        }

        // l_i_range accepts arbitrary cloned i32 rotations. Validate every
        // compact row before constructing its lazy iterator; rows outside
        // that interface fall back to the unchanged dense evaluator.
        for (row, _) in &entries {
            if i32::try_from(*row).is_err() {
                return Ok(None);
            }
        }

        let omega = domain.get_omega();
        for (row, multiplicity) in &entries {
            let rotation = i32::try_from(*row).expect("sparse row conversion checked above");
            if x == omega.pow_vartime([rotation as u64]) {
                return Ok(Some(*multiplicity));
            }
        }

        let xn = x.pow_vartime([logical_len]);
        if xn == C::Scalar::ONE {
            // Every supported base-domain root was handled above. All other
            // Lagrange basis values vanish at this domain point.
            return Ok(Some(C::Scalar::ZERO));
        }
        let rotations = entries.iter().map(|(row, _)| {
            i32::try_from(*row).expect("sparse row conversion checked above")
        });
        let lagrange_values = domain.l_i_range(x, xn, rotations);
        let mut evaluation = C::Scalar::ZERO;
        for ((_, multiplicity), lagrange_value) in entries.iter().zip(lagrange_values) {
            evaluation += *multiplicity * lagrange_value;
        }
        Ok(Some(evaluation))
    }
}

impl<C: CurveAffine> CompactLookupPolynomials<C> {
    fn materialize_lagrange(
        &self,
        kind: LookupPolynomialKind,
    ) -> io::Result<Polynomial<C::Scalar, LagrangeCoeff>>
    where
        C::Scalar: ff::WithSmallOrderMulGroup<3>,
    {
        match kind {
            LookupPolynomialKind::Multiplicity => {
                let mut values = Polynomial::<C::Scalar, LagrangeCoeff>::mapped_filled(
                    self.n,
                    C::Scalar::ZERO,
                )?;
                for (code, &count) in self.counts_by_code.iter().enumerate() {
                    let row = *self.table.winner_rows.get(code).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "lookup winner row is missing")
                    })?;
                    if row >= self.table.active_rows || row >= self.n {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "lookup winner row is outside active rows"));
                    }
                    values[row] = C::Scalar::from(count);
                }
                Ok(values)
            }
            LookupPolynomialKind::GrandSum => {
                if self.table.active_rows + self.blinding_factors + 1 != self.n
                    || self.phi_blinds.len() != self.blinding_factors
                    || self.table.codebook.len() != self.inverse_by_code.len()
                {
                    return Err(io::Error::new(io::ErrorKind::InvalidData, "compact lookup source dimensions are inconsistent"));
                }
                let mut delta = vec![C::Scalar::ZERO; self.table.active_rows];
                for input in 0..self.input_count {
                    let first_id = input.checked_mul(self.table.active_rows).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "lookup query tape offset overflow")
                    })?;
                    for (row, delta_row) in delta.iter_mut().enumerate() {
                        let tape_index = first_id.checked_add(row).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "lookup query row overflow")
                        })?;
                        let code = self.input_codes.read(tape_index)?;
                        let inverse = *self.inverse_by_code.get(code).ok_or_else(|| {
                            io::Error::new(io::ErrorKind::InvalidData, "lookup query ID exceeds codebook")
                        })?;
                        *delta_row += inverse;
                    }
                }
                for (code, &count) in self.counts_by_code.iter().enumerate() {
                    let row = *self.table.winner_rows.get(code).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "lookup winner row is missing")
                    })?;
                    if row >= delta.len() {
                        return Err(io::Error::new(io::ErrorKind::InvalidData, "lookup winner row is outside active data"));
                    }
                    let table_code = self.table.row_codes.read(row)?;
                    let inverse = *self.inverse_by_code.get(table_code).ok_or_else(|| {
                        io::Error::new(io::ErrorKind::InvalidData, "lookup table ID exceeds codebook")
                    })?;
                    delta[row] -= C::Scalar::from(count) * inverse;
                }
                let mut phi = Polynomial::<C::Scalar, LagrangeCoeff>::mapped_filled(self.n, C::Scalar::ZERO)?;
                let mut prefix = C::Scalar::ZERO;
                phi[0] = C::Scalar::ZERO;
                for (row, difference) in delta.into_iter().enumerate() {
                    prefix += difference;
                    phi[row + 1] = prefix;
                }
                let blind_start = self.table.active_rows + 1;
                for (offset, blind) in self.phi_blinds.iter().enumerate() {
                    phi[blind_start + offset] = *blind;
                }
                Ok(phi)
            }
        }
    }
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) source: Arc<LookupPolynomialSource<C>>,
    pub(in crate::plonk) commitment: C,
}

pub(in crate::plonk) fn native_compact_resource_layout<C: CurveAffine>() -> (usize, usize, usize) {
    (
        std::mem::size_of::<CompactTable<C::Scalar>>(),
        std::mem::size_of::<PackedIdTape>(),
        std::mem::size_of::<LookupPolynomialSource<C>>(),
    )
}

impl<C: SerdeCurveAffine> Committed<C> {
    #[allow(dead_code)]
    pub fn write<W: std::io::Write>(
        &self,
        writer: &mut W,
        format: SerdeFormat,
    ) -> std::io::Result<()>
    where
        <C as CurveAffine>::ScalarExt: crate::helpers::SerdePrimeField,
        C::Scalar: ff::WithSmallOrderMulGroup<3>,
    {
        self.source
            .materialize_coeff(LookupPolynomialKind::Multiplicity)?
            .write(writer, format)?;
        self.source
            .materialize_coeff(LookupPolynomialKind::GrandSum)?
            .write(writer, format)?;
        self.commitment.write(writer, format)
    }

    #[allow(dead_code)]
    pub fn read<R: std::io::Read>(reader: &mut R, format: SerdeFormat) -> std::io::Result<Self>
    where
        <C as CurveAffine>::ScalarExt: crate::helpers::SerdePrimeField,
    {
        let m_poly = Polynomial::read(reader, format)?;
        let phi_poly = Polynomial::read(reader, format)?;
        let commitment = C::read(reader, format)?;

        Ok(Committed {
            source: Arc::new(LookupPolynomialSource::dense(m_poly, phi_poly)),
            commitment,
        })
    }
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
    phi_eval: C::Scalar,
    phi_next_eval: C::Scalar,
    m_eval: C::Scalar,
}

impl<F: WithSmallOrderMulGroup<3>> Argument<F> {
    pub(in crate::plonk) fn prepare_table<'a, 'params: 'a, C, P: Params<'params, C>>(
        &self,
        vk: &VerifyingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [FixedColumnSource<C::Scalar>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
        cardinality_upper: usize,
    ) -> Result<PreparedTable<C>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
    {
        let n = params.n() as usize;
        let compressed_expression = compress_expressions::<C>(
            &self.table_expressions,
            vk,
            n,
            domain,
            theta,
            advice_values,
            fixed_values,
            instance_values,
            challenges,
        )?;

        let blinding_factors = vk.cs.blinding_factors();
        let chunk_size = n - blinding_factors - 1;
        if cardinality_upper == 0 {
            return Err(Error::NativeStorage(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "native mv-lookup table cardinality bound is zero",
            )));
        }
        let mut membership = HashMap::default();
        let mut codebook = Vec::new();
        let mut winner_rows = Vec::new();
        for (row, &value) in compressed_expression.iter().take(chunk_size).enumerate() {
            let representation = value.to_repr();
            let code = codebook.len();
            let inserted = insert_table_membership(
                &mut membership,
                representation.as_ref(),
                code,
                row,
                cardinality_upper,
            )?;
            if inserted {
                codebook.try_reserve(1).map_err(|error| Error::NativeStorage(io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    format!("cannot grow native mv-lookup codebook: {error}"),
                )))?;
                winner_rows.try_reserve(1).map_err(|error| Error::NativeStorage(io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    format!("cannot grow native mv-lookup winner rows: {error}"),
                )))?;
                codebook.push(value);
                winner_rows.push(row);
            } else {
                let entry = membership.get(representation.as_ref()).ok_or_else(|| Error::NativeStorage(
                    io::Error::new(io::ErrorKind::InvalidData, "native mv-lookup duplicate code disappeared"),
                ))?;
                winner_rows[entry.code] = row;
            }
        }
        let width = PackedIdWidth::for_cardinality(codebook.len()).map_err(Error::NativeStorage)?;
        let mut row_codes = PackedIdTape::zeroed(chunk_size, width).map_err(Error::NativeStorage)?;
        for (row, &value) in compressed_expression.iter().take(chunk_size).enumerate() {
            let representation = value.to_repr();
            let entry = membership.get(representation.as_ref()).ok_or_else(|| Error::NativeStorage(
                io::Error::new(io::ErrorKind::InvalidData, "native mv-lookup table row has no code"),
            ))?;
            row_codes.write(row, entry.code).map_err(Error::NativeStorage)?;
        }
        let compact = Arc::new(CompactTable {
            codebook,
            winner_rows,
            row_codes,
            active_rows: chunk_size,
            beta_inverses: Mutex::new(None),
        });
        Ok(PreparedTable { compact, membership })
    }

    pub(in crate::plonk) fn prepare<'a, 'params: 'a, C, P: Params<'params, C>>(
        &self,
        vk: &VerifyingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [FixedColumnSource<C::Scalar>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
        table: &PreparedTable<C>,
        argument_index: usize,
    ) -> Result<Prepared<C>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        let n = params.n() as usize;
        let blinding_factors = vk.cs.blinding_factors();
        let chunk_size = n - blinding_factors - 1;
        let input_count = self.inputs_expressions.len();
        let id_width = PackedIdWidth::for_cardinality(table.compact.codebook.len())
            .map_err(Error::NativeStorage)?;
        let id_count = input_count.checked_mul(chunk_size).ok_or_else(|| {
            Error::NativeStorage(io::Error::new(io::ErrorKind::InvalidInput, "native mv-lookup query ID count overflow"))
        })?;
        let mut input_codes = PackedIdTape::zeroed(id_count, id_width).map_err(Error::NativeStorage)?;

        let start = instant::Instant::now();
        let counts_by_code: Vec<u64> = {
            use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
            let mut counts = Vec::new();
            counts.try_reserve_exact(table.compact.codebook.len()).map_err(|error| {
                Error::NativeStorage(io::Error::new(io::ErrorKind::OutOfMemory, format!("cannot allocate sparse native lookup counts: {error}")))
            })?;
            counts.extend((0..table.compact.codebook.len()).map(|_| AtomicU64::new(0)));
            let first_out_of_range_row = AtomicUsize::new(usize::MAX);
            let count_overflow = AtomicBool::new(false);
            const LOOKUP_ENCODE_ROWS: usize = 65_536;

            // Q is evaluated one at a time. Its active rows become packed
            // table IDs; the full field-valued Q polynomial is then released.
            for (input_index, input_expressions) in self.inputs_expressions.iter().enumerate() {
                let compressed_input_expression = compress_expressions::<C>(
                    input_expressions,
                    vk,
                    n,
                    domain,
                    theta,
                    advice_values,
                    fixed_values,
                    instance_values,
                    challenges,
                )?;
                let first_id = input_index.checked_mul(chunk_size).ok_or_else(|| {
                    Error::NativeStorage(io::Error::new(io::ErrorKind::InvalidInput, "native mv-lookup query tape offset overflow"))
                })?;
                let mut encoded_ids = vec![usize::MAX; LOOKUP_ENCODE_ROWS.min(chunk_size)];
                for row_start in (0..chunk_size).step_by(LOOKUP_ENCODE_ROWS) {
                    let take = (chunk_size - row_start).min(LOOKUP_ENCODE_ROWS);
                    let input = &compressed_input_expression[row_start..row_start + take];
                    let output = &mut encoded_ids[..take];
                    output.fill(usize::MAX);
                    input
                        .par_iter()
                        .zip(output.par_iter_mut())
                        .enumerate()
                        .for_each(|(offset, (value, encoded))| {
                            let row = row_start + offset;
                            let representation = value.to_repr();
                            let Some(entry) = table.membership.get(representation.as_ref()) else {
                                first_out_of_range_row.fetch_min(row, Ordering::Relaxed);
                                return;
                            };
                            if entry.code >= table.compact.codebook.len() {
                                first_out_of_range_row.fetch_min(row, Ordering::Relaxed);
                                return;
                            }
                            *encoded = entry.code;
                            let Some(count) = counts.get(entry.code) else {
                                first_out_of_range_row.fetch_min(row, Ordering::Relaxed);
                                return;
                            };
                            if count
                                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| current.checked_add(1))
                                .is_err()
                            {
                                count_overflow.store(true, Ordering::Relaxed);
                            }
                        });
                    for (offset, code) in encoded_ids[..take].iter().copied().enumerate() {
                        if code != usize::MAX {
                            let index = first_id
                                .checked_add(row_start)
                                .and_then(|value| value.checked_add(offset))
                                .ok_or_else(|| Error::NativeStorage(io::Error::new(io::ErrorKind::InvalidInput, "native mv-lookup query tape index overflow")))?;
                            input_codes.write(index, code).map_err(Error::NativeStorage)?;
                        }
                    }
                }
                drop(compressed_input_expression);
            }
            let first_out_of_range_row = first_out_of_range_row.load(Ordering::Relaxed);
            if first_out_of_range_row != usize::MAX {
                return Err(Error::NativeStorage(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "native mv-lookup argument {} has an out-of-range query at row {} of {} active rows",
                        argument_index, first_out_of_range_row, chunk_size
                    ),
                )));
            }
            if count_overflow.load(Ordering::Relaxed) {
                return Err(Error::NativeStorage(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native mv-lookup sparse multiplicity counter overflow",
                )));
            }
            counts.into_iter().map(|count| count.load(Ordering::Relaxed)).collect()
        };
        log::trace!("m sparse counts and packed Q tapes {:?}", start.elapsed());
        #[cfg(feature = "sanity-checks")]
        let mut m_values = Polynomial::<F, LagrangeCoeff>::mapped_filled(n, F::ZERO)
            .map_err(Error::NativeStorage)?;
        #[cfg(feature = "sanity-checks")]
        for (code, &count) in counts_by_code.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let row = *table.compact.winner_rows.get(code).ok_or_else(|| Error::NativeStorage(
                io::Error::new(io::ErrorKind::InvalidData, "native mv-lookup winner row is missing"),
            ))?;
            if row >= chunk_size {
                return Err(Error::NativeStorage(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "native mv-lookup winner row is outside active rows",
                )));
            }
            m_values[row] = F::from(count);
        }

        #[cfg(feature = "sanity-checks")]
        {
            let compressed_table_expression = compress_expressions::<C>(
                &self.table_expressions,
                vk,
                n,
                domain,
                theta,
                advice_values,
                fixed_values,
                instance_values,
                challenges,
            )?;
            // check that m is zero after blinders
            let invalid_ms = m_values
                .iter()
                .skip(params.n() as usize - blinding_factors)
                .collect::<Vec<_>>();
            assert_eq!(invalid_ms.len(), blinding_factors);
            for mi in invalid_ms {
                assert_eq!(*mi, C::Scalar::ZERO);
            }

            // check sums
            // This auxiliary algebraic check must not consume the prover's
            // transcript/blinding RNG. Keep its random challenge local to the
            // sanity-only feature.
            let mut sanity_rng = rand_core::OsRng;
            let alpha = C::Scalar::random(&mut sanity_rng);
            let mut lhs_sum = C::Scalar::ZERO;
            for input_expressions in &self.inputs_expressions {
                let compressed_input_expression = compress_expressions::<C>(
                    input_expressions,
                    vk,
                    n,
                    domain,
                    theta,
                    advice_values,
                    fixed_values,
                    instance_values,
                    challenges,
                )?;
                for &fi in compressed_input_expression
                    .iter()
                    .take(params.n() as usize - blinding_factors - 1)
                {
                    lhs_sum += (fi + alpha).invert().unwrap();
                }
            }

            let mut rhs_sum = C::Scalar::ZERO;
            for (&ti, &mi) in compressed_table_expression.iter().zip(m_values.iter()) {
                rhs_sum += mi * (ti + alpha).invert().unwrap();
            }

            assert_eq!(lhs_sum, rhs_sum);
        }

        #[cfg(feature = "sanity-checks")]
        drop(m_values);

        // commit to m(X)
        let blind = Blind(C::Scalar::ZERO);
        let start = instant::Instant::now();
        let m_commitment = commit_multiset_counts(
            params,
            &counts_by_code,
            &table.compact.winner_rows,
            chunk_size,
            params.n(),
            blind,
        )
        .map_err(Error::NativeStorage)?
        .to_affine();
        log::trace!("m_commitment {:?}", start.elapsed());

        // write commitment of m(X) to transcript
        // transcript.write_point(m_commitment)?;

        Ok(Prepared {
            argument_index,
            theta,
            table: Arc::clone(&table.compact),
            input_codes: Arc::new(input_codes),
            input_count,
            counts_by_code,
            commitment: m_commitment,
        })
    }
}

fn sparse_multiset_entries<F>(
    counts_by_code: &[u64],
    winner_rows: &[usize],
    active_rows: usize,
    logical_len: u64,
    params_len: u64,
) -> io::Result<Option<Vec<(usize, F)>>>
where
    F: Field + From<u64>,
{
    if counts_by_code.len() != winner_rows.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "native mv-lookup count/row cardinality mismatch: {} counts, {} rows",
                counts_by_code.len(),
                winner_rows.len()
            ),
        ));
    }
    if logical_len > params_len {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "native mv-lookup logical commitment length {logical_len} exceeds parameter length {params_len}"
            ),
        ));
    }
    let logical_len_usize = usize::try_from(logical_len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "native mv-lookup logical commitment length does not fit the host index type",
        )
    })?;
    let mut nonzero_count = 0usize;
    let mut over_limit = false;
    for (&count, &row) in counts_by_code.iter().zip(winner_rows) {
        if count == 0 {
            continue;
        }
        if row >= active_rows || row >= logical_len_usize {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "native mv-lookup nonzero winner row {row} is outside active/logical rows ({active_rows}/{logical_len_usize})"
                ),
            ));
        }
        nonzero_count = nonzero_count.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "native mv-lookup nonzero support count overflow",
            )
        })?;
        if nonzero_count > crate::poly::commitment::MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS {
            over_limit = true;
        }
    }
    if over_limit {
        return Ok(None);
    }

    let mut entries = Vec::new();
    entries.try_reserve_exact(nonzero_count).map_err(|error| {
        io::Error::new(
            io::ErrorKind::OutOfMemory,
            format!("cannot allocate bounded sparse mv-lookup entries: {error}"),
        )
    })?;
    for (&count, &row) in counts_by_code.iter().zip(winner_rows) {
        if count != 0 {
            entries.push((row, F::from(count)));
        }
    }
    Ok(Some(entries))
}

fn commit_multiset_counts<'params, C, P>(
    params: &P,
    counts_by_code: &[u64],
    winner_rows: &[usize],
    active_rows: usize,
    logical_len: u64,
    blind: Blind<C::ScalarExt>,
) -> io::Result<C::CurveExt>
where
    C: CurveAffine,
    P: Params<'params, C>,
{
    let entries = sparse_multiset_entries::<C::ScalarExt>(
        counts_by_code,
        winner_rows,
        active_rows,
        logical_len,
        params.n(),
    )?;
    if let Some(entries) = entries {
        return params.commit_lagrange_sparse(logical_len, &entries, blind);
    }

    // Large supports retain the original mapped dense commitment path.
    let logical_len_usize = usize::try_from(logical_len).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "native mv-lookup logical commitment length does not fit the host index type",
        )
    })?;
    let mut values = Polynomial::<C::ScalarExt, LagrangeCoeff>::mapped_filled(
        logical_len_usize,
        C::ScalarExt::ZERO,
    )?;
    for (&count, &row) in counts_by_code.iter().zip(winner_rows) {
        if count != 0 {
            if values[row] != C::ScalarExt::ZERO {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("native mv-lookup nonzero winner row {row} occurs more than once"),
                ));
            }
            values[row] = C::ScalarExt::from(count);
        }
    }
    Ok(params.commit_lagrange(&values, blind))
}

#[cfg(test)]
mod native_table_map_tests {
    use super::{
        commit_multiset_counts, insert_table_membership, CompactLookupPolynomials,
        CompactTable, HashMap, LookupPolynomialKind, LookupPolynomialSource, PackedIdTape,
        PackedIdWidth, TableMembership,
    };
    use crate::arithmetic::{eval_polynomial, CurveAffine};
    use crate::poly::commitment::{
        Blind, Params, ParamsProver, MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS,
    };
    use crate::poly::{Coeff, EvaluationDomain, LagrangeCoeff, Polynomial};
    use ff::{Field, WithSmallOrderMulGroup};
    use halo2curves::{
        bn256::{Bn256, G1Affine},
        pasta::EpAffine,
    };
    use std::sync::{Arc, Mutex};

    #[test]
    fn duplicate_table_rows_keep_the_last_native_index() {
        let mut index = HashMap::default();
        assert!(insert_table_membership(&mut index, &[1, 2], 0, 0, 2).unwrap());
        assert!(insert_table_membership(&mut index, &[3, 4], 1, 1, 2).unwrap());
        assert!(!insert_table_membership(&mut index, &[1, 2], 0, 7, 2).unwrap());
        assert_eq!(index.len(), 2);
        assert_eq!(index.get(&[1, 2][..]), Some(&TableMembership { code: 0, winner_row: 7 }));
    }

    #[test]
    fn table_membership_rejects_cardinality_overflow() {
        let mut index = HashMap::default();
        insert_table_membership(&mut index, &[1], 0, 0, 1).unwrap();
        assert!(insert_table_membership(&mut index, &[2], 1, 1, 1).is_err());
    }

    fn assert_sparse_commit_matches_legacy<'params, C, P>(
        params: &P,
        counts: &[u64],
        winner_rows: &[usize],
        active_rows: usize,
        blind: Blind<C::ScalarExt>,
    ) where
        C: CurveAffine,
        P: Params<'params, C>,
    {
        let logical_len = params.n();
        let actual = commit_multiset_counts::<C, P>(
            params,
            counts,
            winner_rows,
            active_rows,
            logical_len,
            blind,
        )
        .unwrap();

        // Independent legacy oracle: reproduce the prior dense scatter and
        // call the unchanged dense commitment implementation.
        let mut dense_values = vec![C::ScalarExt::ZERO; logical_len as usize];
        for (&count, &row) in counts.iter().zip(winner_rows) {
            if count != 0 {
                dense_values[row] = C::ScalarExt::from(count);
            }
        }
        let dense = Polynomial::<C::ScalarExt, LagrangeCoeff>::from_vec(dense_values);
        let expected = params.commit_lagrange(&dense, blind);
        assert_eq!(actual, expected);
    }

    fn assert_sparse_api_rejects_invalid_indices<'params, C, P>(params: &P)
    where
        C: CurveAffine,
        P: Params<'params, C>,
    {
        let one = C::ScalarExt::ONE;
        let blind = Blind(C::ScalarExt::from(5));
        let logical_len = params.n();
        assert!(params
            .commit_lagrange_sparse(
                logical_len,
                &[(logical_len as usize, one)],
                blind,
            )
            .is_err());
        assert!(params
            .commit_lagrange_sparse(logical_len, &[(1, one), (1, C::ScalarExt::from(2))], blind)
            .is_err());
        assert!(params
            .commit_lagrange_sparse(logical_len + 1, &[], blind)
            .is_err());
        let over_limit: Vec<_> = (0..=MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS)
            .map(|row| (row, one))
            .collect();
        assert!(params
            .commit_lagrange_sparse(logical_len, &over_limit, blind)
            .is_err());
    }

    fn assert_sparse_commit_paths<'params, C, P>(params: &P)
    where
        C: CurveAffine,
        P: Params<'params, C>,
    {
        assert_sparse_commit_matches_legacy::<C, P>(
            params,
            &[4, 0, 1, 3],
            &[17, usize::MAX, 2, 9],
            24,
            Blind(C::ScalarExt::from(19)),
        );
        assert_sparse_commit_matches_legacy::<C, P>(
            params,
            &[],
            &[],
            24,
            Blind(C::ScalarExt::from(7)),
        );

        // The exact support limit remains on the sparse production path.
        let boundary_counts: Vec<_> = (1..=MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS as u64).collect();
        let boundary_rows: Vec<_> = (0..boundary_counts.len()).rev().collect();
        assert_sparse_commit_matches_legacy::<C, P>(
            params,
            &boundary_counts,
            &boundary_rows,
            40,
            Blind(C::ScalarExt::from(13)),
        );

        // This support exceeds the shared threshold and therefore exercises
        // the production helper's unchanged dense fallback.
        let dense_counts: Vec<_> = (1..=MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS as u64 + 1).collect();
        let dense_rows: Vec<_> = (0..dense_counts.len()).rev().collect();
        assert_sparse_commit_matches_legacy::<C, P>(
            params,
            &dense_counts,
            &dense_rows,
            40,
            Blind(C::ScalarExt::from(11)),
        );
        assert_sparse_api_rejects_invalid_indices::<C, P>(params);
    }

    fn compact_m_source<C: CurveAffine>(
        domain: &EvaluationDomain<C::Scalar>,
        counts_by_code: Vec<u64>,
        winner_rows: Vec<usize>,
        active_rows: usize,
    ) -> LookupPolynomialSource<C>
    where
        C::Scalar: WithSmallOrderMulGroup<3>,
    {
        let width = PackedIdWidth::for_cardinality(counts_by_code.len().max(1))
            .expect("test codebook width");
        let blinding_factors = usize::try_from(domain.get_n()).unwrap() - active_rows - 1;
        let codebook = vec![C::Scalar::ONE; counts_by_code.len()];
        let table = Arc::new(CompactTable {
            codebook,
            winner_rows,
            row_codes: PackedIdTape::zeroed(active_rows, width).expect("test row-code tape"),
            active_rows,
            beta_inverses: Mutex::new(None),
        });
        let input_codes = Arc::new(
            PackedIdTape::zeroed(0, width).expect("empty test input-code tape"),
        );
        LookupPolynomialSource::compact(CompactLookupPolynomials {
            domain: domain.clone(),
            n: usize::try_from(domain.get_n()).unwrap(),
            blinding_factors,
            table,
            input_codes,
            input_count: 0,
            inverse_by_code: Arc::new(vec![C::Scalar::ONE; counts_by_code.len()]),
            counts_by_code,
            phi_blinds: vec![C::Scalar::ZERO; blinding_factors],
        })
    }

    fn assert_sparse_m_eval_matches_legacy(
        domain: &EvaluationDomain<halo2curves::bn256::Fr>,
        counts_by_code: Vec<u64>,
        winner_rows: Vec<usize>,
        active_rows: usize,
    ) {
        use halo2curves::bn256::{Fr, G1Affine};

        let support_rows: Vec<_> = counts_by_code
            .iter()
            .zip(&winner_rows)
            .filter_map(|(&count, &row)| (count != 0).then_some(row))
            .collect();
        let source = compact_m_source::<G1Affine>(
            domain,
            counts_by_code.clone(),
            winner_rows.clone(),
            active_rows,
        );
        let legacy_poly = source
            .materialize_coeff(LookupPolynomialKind::Multiplicity)
            .expect("legacy compact multiplicity materialization");
        let mut points = vec![Fr::ZERO];
        if let Some(&row) = support_rows.first() {
            points.push(domain.get_omega().pow_vartime([row as u64]));
        }
        let unsupported_active_row = (0..active_rows)
            .find(|row| !support_rows.contains(row))
            .expect("the fixture leaves an unsupported active row");
        points.push(domain.get_omega().pow_vartime([unsupported_active_row as u64]));
        for row in active_rows..domain.get_n() as usize {
            points.push(domain.get_omega().pow_vartime([row as u64]));
        }
        let exterior = (2..=100u64)
            .map(Fr::from)
            .find(|x| x.pow_vartime([domain.get_n()]) != Fr::ONE)
            .expect("a small field element outside the evaluation domain");
        points.push(exterior);

        for x in points {
            let expected = eval_polynomial(&legacy_poly, x);
            let actual = source.evaluate_sparse_m(domain, x).unwrap();
            if support_rows.len() <= MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS {
                assert_eq!(actual, Some(expected));
            } else {
                assert_eq!(actual, None);
                let m_poly = source
                    .materialize_coeff(LookupPolynomialKind::Multiplicity)
                    .unwrap();
                assert_eq!(eval_polynomial(&m_poly, x), expected);
            }
        }
    }

    fn assert_sparse_m_eval_preserves_zero_row_contract(
        domain: &EvaluationDomain<halo2curves::bn256::Fr>,
        active_rows: usize,
    ) {
        use halo2curves::bn256::{Fr, G1Affine};

        for winner_rows in [vec![usize::MAX], vec![]] {
            let source = compact_m_source::<G1Affine>(domain, vec![0], winner_rows, active_rows);
            assert!(source.evaluate_sparse_m(domain, Fr::ZERO).is_err());
            assert!(source
                .materialize_coeff(LookupPolynomialKind::Multiplicity)
                .is_err());
        }

        let overlapping = compact_m_source::<G1Affine>(
            domain,
            vec![1, 0],
            vec![7, 7],
            active_rows,
        );
        let x = Fr::from(5u64);
        assert_eq!(overlapping.evaluate_sparse_m(domain, x).unwrap(), None);
        let legacy_poly = overlapping
            .materialize_coeff(LookupPolynomialKind::Multiplicity)
            .unwrap();
        assert_eq!(eval_polynomial(&legacy_poly, x), Fr::ZERO);
    }

    fn assert_dense_m_eval_uses_legacy_fallback() {
        use halo2curves::bn256::{Fr, G1Affine};

        let domain = EvaluationDomain::<Fr>::new(3, 6);
        let coefficients = vec![Fr::from(7u64), Fr::from(11u64)];
        let m_poly = Polynomial::<Fr, Coeff>::from_vec(coefficients.clone());
        let source = LookupPolynomialSource::<G1Affine>::dense(
            Polynomial::<Fr, Coeff>::from_vec(coefficients),
            Polynomial::<Fr, Coeff>::from_vec(vec![Fr::ZERO]),
        );
        let x = Fr::from(5u64);
        assert_eq!(source.evaluate_sparse_m(&domain, x).unwrap(), None);
        let fallback = source
            .materialize_coeff(LookupPolynomialKind::Multiplicity)
            .unwrap();
        assert_eq!(eval_polynomial(&fallback, x), eval_polynomial(&m_poly, x));
    }

    #[test]
    fn sparse_m_commit_matches_legacy_for_kzg_and_ipa() {
        let kzg_params = crate::poly::kzg::commitment::ParamsKZG::<Bn256>::new(6);
        assert_sparse_commit_paths::<G1Affine, _>(&kzg_params);

        let ipa_params = crate::poly::ipa::commitment::ParamsIPA::<EpAffine>::new(6);
        assert_sparse_commit_paths::<EpAffine, _>(&ipa_params);

        use halo2curves::bn256::Fr;
        let domain = EvaluationDomain::<Fr>::new(3, 6);
        let active_rows = 61;
        assert_sparse_m_eval_matches_legacy(
            &domain,
            vec![0, 0],
            vec![0, 1],
            active_rows,
        );
        for support in [
            MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS,
            MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS + 1,
        ] {
            let mut counts: Vec<_> = (0..support)
                .map(|index| u64::MAX - index as u64)
                .collect();
            let mut rows: Vec<_> = (0..support).map(|index| (index * 17) % active_rows).collect();
            counts.push(0);
            let zero_count_row = if support <= MAX_SPARSE_LAGRANGE_COMMITMENT_TERMS {
                (0..active_rows)
                    .find(|row| !rows.contains(row))
                    .expect("the sparse fixture leaves one active row")
            } else {
                (0..active_rows)
                    .find(|row| !rows.contains(row))
                    .expect("the fallback fixture leaves one active row")
            };
            rows.push(zero_count_row);
            assert_sparse_m_eval_matches_legacy(&domain, counts, rows, active_rows);
        }
        assert_sparse_m_eval_preserves_zero_row_contract(&domain, active_rows);
        assert_dense_m_eval_uses_legacy_fallback();
    }
}

impl<C: CurveAffine> Prepared<C> {
    pub(in crate::plonk) fn compact_storage_shape(&self) -> (usize, usize) {
        (
            PackedIdTape::byte_len(self.input_codes.ids(), self.input_codes.width())
                .unwrap_or(usize::MAX),
            self.counts_by_code.len(),
        )
    }

    pub(in crate::plonk) fn commit_grand_sum<'params, P: Params<'params, C>>(
        self,
        vk: &VerifyingKey<C>,
        params: &P,
        beta: ChallengeBeta<C>,
        advice_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &[FixedColumnSource<C::Scalar>],
        instance_values: &[Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &[C::Scalar],
        phi_blinds: &[C::Scalar],
    ) -> Result<Committed<C>, Error> {
        /*
            φ_i(X) = f_i(X) + α
            τ(X) = t(X) + α
            LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
            RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
        */

        let Prepared {
            argument_index,
            theta,
            table,
            input_codes,
            input_count,
            counts_by_code,
            commitment: _,
        } = self;
        let lookup_argument = vk
            .cs()
            .lookups()
            .get(argument_index)
            .ok_or_else(|| {
                Error::NativeStorage(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!(
                        "native mv-lookup argument index {} is out of range",
                        argument_index
                    ),
                ))
            })?;

        let n = params.n() as usize;
        let blinding_factors = vk.cs.blinding_factors();
        let active_rows = n.checked_sub(blinding_factors + 1).ok_or_else(|| {
            Error::NativeStorage(io::Error::new(io::ErrorKind::InvalidData, "mv-lookup active row count underflow"))
        })?;
        let expected_ids = input_count.checked_mul(active_rows).ok_or_else(|| {
            Error::NativeStorage(io::Error::new(io::ErrorKind::InvalidData, "mv-lookup query tape length overflow"))
        })?;
        if input_count != lookup_argument.inputs_expressions.len()
            || input_codes.ids() != expected_ids
            || table.active_rows != active_rows
            || counts_by_code.len() != table.codebook.len()
            || phi_blinds.len() != blinding_factors
        {
            return Err(Error::NativeStorage(io::Error::new(
                io::ErrorKind::InvalidData,
                "compact mv-lookup source does not match the configured argument",
            )));
        }

        let start = instant::Instant::now();
        // BatchInvert the distinct compressed table values once. Duplicate Q
        // and table entries receive the identical inverse, including the
        // native zero-to-zero result for beta + value == 0.
        let inverse_by_code = table.beta_inverses(*beta).map_err(Error::NativeStorage)?;
        let source = LookupPolynomialSource::compact(CompactLookupPolynomials {
            domain: vk.domain.clone(),
            n,
            blinding_factors,
            table,
            input_codes,
            input_count,
            counts_by_code,
            inverse_by_code,
            phi_blinds: phi_blinds.to_vec(),
        });
        let phi = match &source.backing {
            LookupPolynomialBacking::Compact(compact) => compact
                .materialize_lagrange(LookupPolynomialKind::GrandSum)
                .map_err(Error::NativeStorage)?,
            LookupPolynomialBacking::Dense { .. } => unreachable!("new lookup source must be compact"),
        };
        log::trace!(" - compact phi reconstruction {:?}", start.elapsed());

        #[cfg(feature = "sanity-checks")]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            let compact = match &source.backing {
                LookupPolynomialBacking::Compact(compact) => compact,
                LookupPolynomialBacking::Dense { .. } => unreachable!("new lookup source must be compact"),
            };
            let m_values = compact
                .materialize_lagrange(LookupPolynomialKind::Multiplicity)
                .map_err(Error::NativeStorage)?;
            let compressed_table_expression = compress_expressions::<C>(
                &lookup_argument.table_expressions,
                vk,
                n,
                &vk.domain,
                theta,
                advice_values,
                fixed_values,
                instance_values,
                challenges,
            )?;
            // While in Lagrange basis, check that product is correctly constructed
            let u = (params.n() as usize) - (blinding_factors + 1);

            /*
                φ_i(X) = f_i(X) + α
                τ(X) = t(X) + α
                LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
            */

            // q(X) = LHS - RHS mod zH(X)
            let mut input_products = vec![C::Scalar::ONE; u];
            let mut input_log_derivatives = vec![C::Scalar::ZERO; u];
            for input_expressions in &lookup_argument.inputs_expressions {
                let compressed_input_expression = compress_expressions::<C>(
                    input_expressions,
                    vk,
                    params.n() as usize,
                    &vk.domain,
                    theta,
                    advice_values,
                    fixed_values,
                    instance_values,
                    challenges,
                )?;
                for i in 0..u {
                    let shifted = *beta + compressed_input_expression[i];
                    input_products[i] *= shifted;
                    input_log_derivatives[i] += shifted.invert().unwrap();
                }
            }

            for i in 0..u {
                // LHS = τ(X) * Π(φ_i(X)) * (ϕ(gX) - ϕ(X))
                let lhs = {
                    (*beta + compressed_table_expression[i])
                        * input_products[i]
                        * (phi[i + 1] - phi[i])
                };

                // RHS = τ(X) * Π(φ_i(X)) * (∑ 1/(φ_i(X)) - m(X) / τ(X))))
                let rhs = {
                    (*beta + compressed_table_expression[i])
                        * input_products[i]
                        * (input_log_derivatives[i]
                            - m_values[i]
                                * (*beta + compressed_table_expression[i])
                                    .invert()
                                    .unwrap())
                };

                assert_eq!(lhs - rhs, C::Scalar::ZERO);
            }

            assert_eq!(phi[u], C::Scalar::ZERO);
        }

        let grand_sum_blind = Blind(C::Scalar::ZERO);
        let start = instant::Instant::now();
        let phi_commitment = params
            .commit_lagrange(&phi, grand_sum_blind.clone())
            .to_affine();
        log::trace!(" - phi_commitment {:?}", start.elapsed());

        // Hash grand sum commitment
        // transcript.write_point(phi_commitment)?;

        Ok(Committed {
            source: Arc::new(source),
            commitment: phi_commitment,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        vk: &VerifyingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &vk.domain;
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let phi_poly = self
            .source
            .materialize_coeff(LookupPolynomialKind::GrandSum)
            .map_err(Error::NativeStorage)?;
        let phi_eval = eval_polynomial(&phi_poly, *x);
        let phi_next_eval = eval_polynomial(&phi_poly, x_next);
        drop(phi_poly);
        let m_eval = match self
            .source
            .evaluate_sparse_m(domain, *x)
            .map_err(Error::NativeStorage)?
        {
            Some(m_eval) => m_eval,
            None => {
                let m_poly = self
                    .source
                    .materialize_coeff(LookupPolynomialKind::Multiplicity)
                    .map_err(Error::NativeStorage)?;
                let m_eval = eval_polynomial(&m_poly, *x);
                drop(m_poly);
                m_eval
            }
        };

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(phi_eval))
            .chain(Some(phi_next_eval))
            .chain(Some(m_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated {
            constructed: self,
            phi_eval,
            phi_next_eval,
            m_eval,
        })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            .chain(Some(ProverQuery::new_mv_lookup(
                *x,
                &self.constructed.source,
                LookupPolynomialKind::GrandSum,
                self.phi_eval,
                Blind(C::Scalar::ZERO),
            )))
            .chain(Some(ProverQuery::new_mv_lookup(
                x_next,
                &self.constructed.source,
                LookupPolynomialKind::GrandSum,
                self.phi_next_eval,
                Blind(C::Scalar::ZERO),
            )))
            .chain(Some(ProverQuery::new_mv_lookup(
                *x,
                &self.constructed.source,
                LookupPolynomialKind::Multiplicity,
                self.m_eval,
                Blind(C::Scalar::ZERO),
            )))
    }
}
