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
    polynomial_slice_byte_length, read_polynomial_vec, write_polynomial_slice, SerdeCurveAffine,
    SerdePrimeField,
};
use crate::poly::{
    Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, PinnedEvaluationDomain,
    Polynomial,
};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};
use crate::SerdeFormat;
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
mod mv_lookup;
pub mod permutation;
mod shuffle;
mod vanishing;

use circuit::SelectorActivationSet;

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
use std::io;

const NATIVE_PK_MAGIC: &[u8; 8] = b"EZKLPK\0\x02";

/// Non-overlapping timing breakdown for loading the compact native proving
/// key. The first duration includes artifact reading, decoding, digest checks,
/// and mapping validation; it is not a pure file-I/O measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativePkReadTiming {
    /// Wall time for artifact reading, decoding, digest checks, and mapping validation.
    pub artifact_read_validate: Duration,
    /// Wall time for rebuilding native sigma coefficients; mapped spill I/O may be included.
    pub sigma_rebuild: Duration,
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
    selectors: Vec<SelectorActivationSet>,
    /// Whether selector compression is turned on or not.
    compress_selectors: bool,
}

// Current version of the VK
const VERSION: u8 = 0x03;

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
            assert!(self.selectors.is_empty());
        }
        // write self.selectors
        for selector in &self.selectors {
            selector.write_activation(writer)?;
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
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unexpected version byte",
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
            // read selectors
            let selectors: Vec<SelectorActivationSet> = (0..cs.num_selectors)
                .map(|_| SelectorActivationSet::read_activation(reader, 1usize << k))
                .collect::<io::Result<_>>()?;
            let (cs, _, _) = cs.compress_selector_activation_sets(&selectors);
            (cs, selectors)
        } else {
            // we still need to replace selectors with fixed Expressions in `cs`
            let fake_selectors = vec![vec![]; cs.num_selectors];
            let (cs, _) = cs.directly_convert_selectors_to_fixed(fake_selectors, false);
            (cs, vec![])
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
        10 + (self.fixed_commitments.len() * C::byte_length(format))
            + self.permutation.bytes_length(format)
            + self
                .selectors
                .iter()
                .map(SelectorActivationSet::encoded_byte_len)
                .sum::<usize>()
    }

    fn from_parts(
        domain: EvaluationDomain<C::Scalar>,
        fixed_commitments: Vec<C>,
        permutation: permutation::VerifyingKey<C>,
        cs: ConstraintSystem<C::Scalar>,
        selectors: Vec<SelectorActivationSet>,
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
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
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
    fn bytes_length(&self, format: SerdeFormat) -> usize
    where
        C: SerdeCurveAffine,
    {
        let scalar_len = C::Scalar::default().to_repr().as_ref().len();
        self.vk.bytes_length(format)
            + NATIVE_PK_MAGIC.len()
            + 12
            + scalar_len * (self.l0.len() + self.l_last.len() + self.l_active_row.len())
            + polynomial_slice_byte_length(&self.fixed_values)
            + polynomial_slice_byte_length(&self.fixed_polys)
            + self.permutation.bytes_length()
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
    /// Does so by first writing the verifying key and then serializing the rest of the data (in the form of field polynomials)
    pub fn write<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()> {
        writer.write_all(NATIVE_PK_MAGIC)?;
        self.vk.write(writer, format)?;
        self.l0.write(writer, format)?;
        self.l_last.write(writer, format)?;
        self.l_active_row.write(writer, format)?;
        write_polynomial_slice(&self.fixed_values, writer, format)?;
        write_polynomial_slice(&self.fixed_polys, writer, format)?;
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

    /// Reads and authenticates a compact proving key while separately timing
    /// the required native sigma coefficient reconstruction.
    pub fn read_with_native_backing_budget_and_timing<R: io::Read, ConcreteCircuit: Circuit<C::Scalar>>(
        reader: &mut R,
        format: SerdeFormat,
        max_total_backing_bytes: u64,
        #[cfg(feature = "circuit-params")] params: ConcreteCircuit::Params,
    ) -> io::Result<(Self, NativePkReadTiming)> {
        let load_started = Instant::now();
        let mut magic = [0u8; NATIVE_PK_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != NATIVE_PK_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "unsupported or legacy native proving key format",
            ));
        }
        let vk = VerifyingKey::<C>::read::<R, ConcreteCircuit>(
            reader,
            format,
            #[cfg(feature = "circuit-params")]
            params,
        )?;
        let l0 = Polynomial::read(reader, format)?;
        let l_last = Polynomial::read(reader, format)?;
        let l_active_row = Polynomial::read(reader, format)?;
        let fixed_values = read_polynomial_vec(reader, format)?;
        let fixed_polys = read_polynomial_vec(reader, format)?;
        let (permutation, sigma_rebuild) = permutation::ProvingKey::read_with_native_backing_budget_and_timing(
            reader,
            format,
            &vk.domain,
            &vk.cs.permutation().get_columns(),
            max_total_backing_bytes,
        )?;
        let artifact_read_validate = load_started.elapsed().saturating_sub(sigma_rebuild);
        let ev = Evaluator::new(vk.cs());
        Ok((
            Self {
                vk,
                l0,
                l_last,
                l_active_row,
                fixed_values,
                fixed_polys,
                permutation,
                ev,
            },
            NativePkReadTiming {
                artifact_read_validate,
                sigma_rebuild,
            },
        ))
    }

    /// Writes a proving key to a vector of bytes using [`Self::write`].
    pub fn to_bytes(&self, format: SerdeFormat) -> Vec<u8> {
        let mut bytes = Vec::<u8>::with_capacity(self.bytes_length(format));
        Self::write(self, &mut bytes, format).expect("Writing to vector should not fail");
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
