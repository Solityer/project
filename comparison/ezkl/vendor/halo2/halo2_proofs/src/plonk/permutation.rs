//! Implementation of permutation argument.

use super::circuit::{Any, Column};
use crate::{
    arithmetic::CurveAffine,
    helpers::SerdeCurveAffine,
    poly::{Coeff, EvaluationDomain, LagrangeCoeff, Polynomial},
    SerdeFormat,
};
use ff::{Field, PrimeField};

pub(crate) mod keygen;
pub(crate) mod prover;
pub(crate) mod verifier;

pub use keygen::Assembly;

use std::io;
use std::marker::PhantomData;

const MAPPING_PERSONAL: &[u8; 16] = b"EZKL-PermMap-v1!";

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
    successors: Vec<(u64, u64)>,
    /// Reconstructed native sigma coefficient polynomials. These are mapped
    /// under the active proof-session spill scope and are not serialized.
    polys: Vec<Polynomial<C::Scalar, Coeff>>,
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
        let mut successors = Vec::with_capacity(edge_count as usize);
        for _ in 0..edge_count {
            let source = read_hashed_u64(reader, &mut state)?;
            let target = read_hashed_u64(reader, &mut state)?;
            if source >= max_cells || target >= max_cells || source == target {
                return Err(invalid_mapping(
                    "successor entry is outside the domain or is identity",
                ));
            }
            if successors
                .last()
                .map(|entry: &(u64, u64)| entry.0 >= source)
                .unwrap_or(false)
            {
                return Err(invalid_mapping(
                    "successor sources are not strictly ordered",
                ));
            }
            successors.push((source, target));
        }
        let mut expected_digest = [0u8; 32];
        reader.read_exact(&mut expected_digest)?;
        let actual_digest = state.finalize();
        if actual_digest.as_bytes() != &expected_digest {
            return Err(invalid_mapping("successor mapping payload digest mismatch"));
        }
        validate_successor_permutation(&successors)?;
        let mut key = Self {
            domain_size,
            columns: expected_columns.to_vec(),
            successors,
            polys: Vec::new(),
            _marker: PhantomData,
        };
        key.rebuild_polys(domain)?;
        Ok(key)
    }

    /// Writes proving key for a single permutation argument to buffer using `Polynomial::write`.  
    pub(super) fn write<W: io::Write>(
        &self,
        writer: &mut W,
        _format: SerdeFormat,
    ) -> io::Result<()> {
        validate_successor_permutation(&self.successors)?;
        let mut state = mapping_hasher();
        write_hashed_u64(writer, &mut state, self.domain_size)?;
        write_hashed_u64(writer, &mut state, self.columns.len() as u64)?;
        for column in &self.columns {
            let (kind, phase, index) = column.native_storage_identity();
            write_hashed(writer, &mut state, &[kind, phase])?;
            write_hashed(writer, &mut state, &index.to_le_bytes())?;
        }
        write_hashed_u64(writer, &mut state, self.successors.len() as u64)?;
        let max_cells = self
            .domain_size
            .checked_mul(self.columns.len() as u64)
            .ok_or_else(|| invalid_mapping("permutation domain size overflow"))?;
        let mut previous_source = None;
        for &(source, target) in &self.successors {
            if source >= max_cells
                || target >= max_cells
                || source == target
                || previous_source
                    .map(|previous| previous >= source)
                    .unwrap_or(false)
            {
                return Err(invalid_mapping(
                    "invalid successor mapping during serialization",
                ));
            }
            write_hashed_u64(writer, &mut state, source)?;
            write_hashed_u64(writer, &mut state, target)?;
            previous_source = Some(source);
        }
        writer.write_all(state.finalize().as_bytes())?;
        Ok(())
    }
}

impl<C: CurveAffine> ProvingKey<C> {
    /// Gets the total number of bytes in the serialization of `self`
    pub(super) fn bytes_length(&self) -> usize {
        8 + 8 + self.columns.len() * 10 + 8 + self.successors.len() * 16 + 32
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
            successors,
            polys: Vec::new(),
            _marker: PhantomData,
        })
    }

    pub(super) fn successors(&self) -> &[(u64, u64)] {
        &self.successors
    }

    pub(super) fn domain_size(&self) -> u64 {
        self.domain_size
    }

    pub(super) fn polys(&self) -> &[Polynomial<C::Scalar, Coeff>] {
        &self.polys
    }

    pub(super) fn sigma_lagrange(
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
        let mut edge = self.successors.partition_point(|entry| entry.0 < start);
        let delta_powers = (0..self.columns.len())
            .map(|index| C::Scalar::DELTA.pow_vartime([index as u64]))
            .collect::<Vec<_>>();
        let mut sigma = Polynomial::mapped_filled(n as usize, C::Scalar::ZERO)?;
        for (row, value) in sigma.iter_mut().enumerate() {
            let source = start + row as u64;
            let target = if edge < self.successors.len() && self.successors[edge].0 == source {
                let target = self.successors[edge].1;
                edge += 1;
                target
            } else {
                source
            };
            let target_column = usize::try_from(target / n)
                .map_err(|_| invalid_mapping("target column does not fit usize"))?;
            let target_row = usize::try_from(target % n)
                .map_err(|_| invalid_mapping("target row does not fit usize"))?;
            *value = delta_powers[target_column] * omega_powers[target_row];
        }
        Ok(sigma)
    }

    fn rebuild_polys(&mut self, domain: &EvaluationDomain<C::Scalar>) -> io::Result<()> {
        let n = usize::try_from(self.domain_size)
            .map_err(|_| invalid_mapping("native domain size does not fit usize"))?;
        let mut omega_powers: Polynomial<C::Scalar, LagrangeCoeff> =
            Polynomial::mapped_filled(n, C::Scalar::ONE)?;
        let omega = domain.get_omega();
        for row in 1..n {
            omega_powers[row] = omega_powers[row - 1] * omega;
        }
        let mut polys = Vec::with_capacity(self.columns.len());
        for column in 0..self.columns.len() {
            let sigma = self.sigma_lagrange(domain, column, &omega_powers[..])?;
            polys.push(domain.lagrange_to_coeff(sigma));
        }
        self.polys = polys;
        Ok(())
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

        let mut reader = Cursor::new(encoded.clone());
        let decoded =
            ProvingKey::<G1Affine>::read(&mut reader, SerdeFormat::RawBytes, &domain, &columns)
                .expect("roundtrip compact native mapping");
        assert_eq!(decoded.successors(), successors.as_slice());
        assert_eq!(decoded.polys().len(), columns.len());

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
