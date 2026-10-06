use crate::arithmetic::{best_multiexp, g_to_lagrange, parallelize};

use crate::helpers::SerdeCurveAffine;
use crate::poly::commitment::{Blind, CommitmentScheme, Params, ParamsProver, ParamsVerifier};
use crate::poly::{Coeff, LagrangeCoeff, Polynomial};
use crate::SerdeFormat;

use ff::{Field, PrimeField};
use group::{prime::PrimeCurveAffine, Curve, Group};
use halo2curves::pairing::Engine;
use halo2curves::CurveExt;
use rand_core::{OsRng, RngCore};
use std::fmt::Debug;
use std::marker::PhantomData;

use std::io;

use super::msm::MSMKZG;

/// These are the public parameters for the polynomial commitment scheme.
#[derive(Debug, Clone)]
pub struct ParamsKZG<E: Engine> {
    pub(crate) k: u32,
    pub(crate) n: u64,
    pub(crate) g: Vec<E::G1Affine>,
    pub(crate) g_lagrange: Vec<E::G1Affine>,
    pub(crate) g2: E::G2Affine,
    pub(crate) s_g2: E::G2Affine,
}

/// Umbrella commitment scheme construction for all KZG variants
#[derive(Debug)]
pub struct KZGCommitmentScheme<E: Engine> {
    _marker: PhantomData<E>,
}

impl<E: Engine + Debug> CommitmentScheme for KZGCommitmentScheme<E>
where
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
{
    type Scalar = E::Fr;
    type Curve = E::G1Affine;

    type ParamsProver = ParamsKZG<E>;
    type ParamsVerifier = ParamsVerifierKZG<E>;

    fn new_params(k: u32) -> Self::ParamsProver {
        ParamsKZG::new(k)
    }

    fn read_params<R: io::Read>(reader: &mut R) -> io::Result<Self::ParamsProver> {
        ParamsKZG::read(reader)
    }
}

impl<E: Engine + Debug> ParamsKZG<E>
where
    E::G1Affine: SerdeCurveAffine,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
{
    /// Initializes parameters for the curve, draws toxic secret from given rng.
    /// MUST NOT be used in production.
    pub fn setup<R: RngCore>(k: u32, rng: R) -> Self {
        // Largest root of unity exponent of the Engine is `2^E::Fr::S`, so we can
        // only support FFTs of polynomials below degree `2^E::Fr::S`.
        assert!(k <= E::Fr::S);
        let n: u64 = 1 << k;

        // Calculate g = [G1, [s] G1, [s^2] G1, ..., [s^(n-1)] G1] in parallel.
        let g1 = E::G1Affine::generator();
        let s = <E::Fr>::random(rng);

        let mut g_projective = vec![E::G1::identity(); n as usize];
        parallelize(&mut g_projective, |g, start| {
            let mut current_g: E::G1 = g1.into();
            current_g *= s.pow_vartime([start as u64]);
            for g in g.iter_mut() {
                *g = current_g;
                current_g *= s;
            }
        });

        let g = {
            let mut g = vec![E::G1Affine::identity(); n as usize];
            parallelize(&mut g, |g, starts| {
                E::G1::batch_normalize(&g_projective[starts..(starts + g.len())], g);
            });
            g
        };

        let mut g_lagrange_projective = vec![E::G1::identity(); n as usize];
        let mut root = E::Fr::ROOT_OF_UNITY;
        for _ in k..E::Fr::S {
            root = root.square();
        }
        let n_inv = E::Fr::from(n)
            .invert()
            .expect("inversion should be ok for n = 1<<k");
        let multiplier = (s.pow_vartime([n]) - E::Fr::ONE) * n_inv;
        parallelize(&mut g_lagrange_projective, |g, start| {
            for (idx, g) in g.iter_mut().enumerate() {
                let offset = start + idx;
                let root_pow = root.pow_vartime([offset as u64]);
                let scalar = multiplier * root_pow * (s - root_pow).invert().unwrap();
                *g = g1 * scalar;
            }
        });

        let g_lagrange = {
            let mut g_lagrange = vec![E::G1Affine::identity(); n as usize];
            parallelize(&mut g_lagrange, |g_lagrange, start| {
                let end = start + g_lagrange.len();
                E::G1::batch_normalize(&g_lagrange_projective[start..end], g_lagrange);
            });
            drop(g_lagrange_projective);
            g_lagrange
        };

        let g2 = <E::G2Affine as PrimeCurveAffine>::generator();
        let s_g2 = (g2 * s).into();

        Self {
            k,
            n,
            g,
            g_lagrange,
            g2,
            s_g2,
        }
    }

    /// Initializes parameters for the curve through existing parameters
    /// k, g, g_lagrange (optional), g2, s_g2
    pub fn from_parts(
        &self,
        k: u32,
        g: Vec<E::G1Affine>,
        g_lagrange: Option<Vec<E::G1Affine>>,
        g2: E::G2Affine,
        s_g2: E::G2Affine,
    ) -> Self {
        // let g_lagrange = if let Some(g_l) = g_lagrange {
        //     g_l
        // } else {
        //     g_to_lagrange(g.iter().map(PrimeCurveAffine::to_curve).collect(), k)
        // };

        let g_lagrange = match g_lagrange {
            Some(g_l) => g_l,
            None => g_to_lagrange(g.iter().map(PrimeCurveAffine::to_curve).collect(), k),
        };

        Self {
            k,
            n: 1 << k,
            g_lagrange,
            g,
            g2,
            s_g2,
        }
    }

    /// Returns gernerator on G2
    pub fn g2(&self) -> E::G2Affine {
        self.g2
    }

    /// Returns first power of secret on G2
    pub fn s_g2(&self) -> E::G2Affine {
        self.s_g2
    }

    /// Writes parameters to buffer
    pub fn write_custom<W: io::Write>(&self, writer: &mut W, format: SerdeFormat) -> io::Result<()>
    where
        E::G2Affine: SerdeCurveAffine,
    {
        writer.write_all(&self.k.to_le_bytes())?;
        for el in self.g.iter() {
            el.write(writer, format)?;
        }
        for el in self.g_lagrange.iter() {
            el.write(writer, format)?;
        }
        self.g2.write(writer, format)?;
        self.s_g2.write(writer, format)?;
        Ok(())
    }

    /// Reads params from a buffer.
    pub fn read_custom<R: io::Read>(reader: &mut R, format: SerdeFormat) -> io::Result<Self>
    where
        E::G2Affine: SerdeCurveAffine,
    {
        let mut k = [0u8; 4];
        reader.read_exact(&mut k[..])?;
        let k = u32::from_le_bytes(k);
        let n = 1 << k;

        let (g, g_lagrange) = match format {
            SerdeFormat::Processed => {
                use group::GroupEncoding;
                let load_points_from_file_parallelly =
                    |reader: &mut R| -> io::Result<Vec<Option<E::G1Affine>>> {
                        let mut points_compressed =
                            vec![<<E as Engine>::G1Affine as GroupEncoding>::Repr::default(); n];
                        for points_compressed in points_compressed.iter_mut() {
                            reader.read_exact((*points_compressed).as_mut())?;
                        }

                        let mut points = vec![Option::<E::G1Affine>::None; n];
                        parallelize(&mut points, |points, chunks| {
                            for (i, point) in points.iter_mut().enumerate() {
                                *point = Option::from(E::G1Affine::from_bytes(
                                    &points_compressed[chunks + i],
                                ));
                            }
                        });
                        Ok(points)
                    };

                let g = load_points_from_file_parallelly(reader)?;
                let g: Vec<<E as Engine>::G1Affine> = g
                    .iter()
                    .map(|point| {
                        point.ok_or_else(|| {
                            io::Error::new(io::ErrorKind::Other, "invalid point encoding")
                        })
                    })
                    .collect::<Result<_, _>>()?;
                let g_lagrange = load_points_from_file_parallelly(reader)?;
                let g_lagrange: Vec<<E as Engine>::G1Affine> = g_lagrange
                    .iter()
                    .map(|point| {
                        point.ok_or_else(|| {
                            io::Error::new(io::ErrorKind::Other, "invalid point encoding")
                        })
                    })
                    .collect::<Result<_, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytes => {
                let g = (0..n)
                    .map(|_| <E::G1Affine as SerdeCurveAffine>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                let g_lagrange = (0..n)
                    .map(|_| <E::G1Affine as SerdeCurveAffine>::read(reader, format))
                    .collect::<Result<Vec<_>, _>>()?;
                (g, g_lagrange)
            }
            SerdeFormat::RawBytesUnchecked => {
                // avoid try branching for performance
                let g = (0..n)
                    .map(|_| <E::G1Affine as SerdeCurveAffine>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                let g_lagrange = (0..n)
                    .map(|_| <E::G1Affine as SerdeCurveAffine>::read(reader, format).unwrap())
                    .collect::<Vec<_>>();
                (g, g_lagrange)
            }
        };

        let g2 = E::G2Affine::read(reader, format)?;
        let s_g2 = E::G2Affine::read(reader, format)?;

        Ok(Self {
            k,
            n: n as u64,
            g,
            g_lagrange,
            g2,
            s_g2,
        })
    }

    /// Reads an authenticated raw SRS directly at a smaller proof domain.
    ///
    /// The caller must authenticate the complete file against the trusted SRS
    /// digest before using this method.  For a smaller target domain, only the
    /// required prefix of `g` is decoded; the remainder and the full
    /// `g_lagrange` section are skipped, while `g2` and `s_g2` are read from the
    /// checked end of the file.  The target-domain Lagrange basis is then
    /// derived from the same `g` prefix as `downsize`, rather than taking a
    /// prefix of the larger-domain Lagrange basis.
    ///
    /// At the source domain this delegates to `read_custom` so the max-k path
    /// preserves its original full read and validation behavior.
    pub fn read_custom_for_k<R: io::Read + io::Seek>(
        reader: &mut R,
        format: SerdeFormat,
        target_k: u32,
        expected_source_k: u32,
    ) -> io::Result<Self>
    where
        E::G1Affine: SerdeCurveAffine<
            ScalarExt = <E as Engine>::Fr,
            CurveExt = <E as Engine>::G1,
        >,
        E::G1: CurveExt<AffineExt = E::G1Affine>,
        E::G2Affine: SerdeCurveAffine,
    {
        if !matches!(format, SerdeFormat::RawBytes) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "selective SRS loading requires the authenticated RawBytes format",
            ));
        }

        let file_len = reader.seek(io::SeekFrom::End(0))?;
        reader.seek(io::SeekFrom::Start(0))?;
        let mut encoded_k = [0u8; 4];
        reader.read_exact(&mut encoded_k)?;
        let source_k = u32::from_le_bytes(encoded_k);
        if source_k != expected_source_k {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("SRS header k={} does not match authenticated k={}", source_k, expected_source_k),
            ));
        }
        if target_k > source_k {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("requested SRS k={} exceeds source k={}", target_k, source_k),
            ));
        }
        if target_k == source_k {
            reader.seek(io::SeekFrom::Start(0))?;
            return Self::read_custom(reader, format);
        }

        let source_n = 1u64.checked_shl(source_k).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid SRS source domain")
        })?;
        let target_n = 1u64.checked_shl(target_k).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "invalid SRS target domain")
        })?;
        let g1_bytes = <E::G1Affine as SerdeCurveAffine>::byte_length(format) as u64;
        let g2_bytes = <E::G2Affine as SerdeCurveAffine>::byte_length(format) as u64;
        let expected_len = 4u64
            .checked_add(
                source_n
                    .checked_mul(g1_bytes)
                    .and_then(|bytes| bytes.checked_mul(2))
                    .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "SRS length overflow"))?,
            )
            .and_then(|bytes| bytes.checked_add(g2_bytes.checked_mul(2)?))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "SRS length overflow"))?;
        if file_len != expected_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("authenticated SRS has an invalid encoded length: expected {}, got {}", expected_len, file_len),
            ));
        }

        let target_n_usize = usize::try_from(target_n).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "SRS target domain exceeds address space")
        })?;
        let mut g = Vec::with_capacity(target_n_usize);
        for _ in 0..target_n_usize {
            g.push(<E::G1Affine as SerdeCurveAffine>::read(reader, format)?);
        }

        let skipped_points = source_n
            .checked_sub(target_n)
            .and_then(|remaining_g| remaining_g.checked_add(source_n))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "SRS skip length overflow"))?;
        let skipped_bytes = skipped_points
            .checked_mul(g1_bytes)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "SRS skip length overflow"))?;
        let skipped_bytes = i64::try_from(skipped_bytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "SRS skip length exceeds seek range")
        })?;
        reader.seek(io::SeekFrom::Current(skipped_bytes))?;

        let g2 = <E::G2Affine as SerdeCurveAffine>::read(reader, format)?;
        let s_g2 = <E::G2Affine as SerdeCurveAffine>::read(reader, format)?;
        if reader.seek(io::SeekFrom::Current(0))? != file_len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "selective SRS reader did not end at the authenticated file boundary",
            ));
        }

        let mut params = Self {
            k: source_k,
            n: source_n,
            g,
            g_lagrange: Vec::new(),
            g2,
            s_g2,
        };
        params.downsize(target_k);
        Ok(params)
    }
}

// TODO: see the issue at https://github.com/appliedzkp/halo2/issues/45
// So we probably need much smaller verifier key. However for new bases in g1 should be in verifier keys.
/// KZG multi-open verification parameters
pub type ParamsVerifierKZG<C> = ParamsKZG<C>;

impl<'params, E: Engine + Debug> Params<'params, E::G1Affine> for ParamsKZG<E>
where
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
{
    type MSM = MSMKZG<E>;

    fn k(&self) -> u32 {
        self.k
    }

    fn n(&self) -> u64 {
        self.n
    }

    fn downsize(&mut self, k: u32) {
        assert!(k <= self.k);

        self.k = k;
        self.n = 1 << k;

        self.g.truncate(self.n as usize);
        self.g.shrink_to_fit();
        drop(std::mem::take(&mut self.g_lagrange));
        self.g_lagrange = g_to_lagrange(self.g.iter().map(|g| g.to_curve()).collect(), k);
    }

    fn empty_msm(&'params self) -> MSMKZG<E> {
        MSMKZG::new()
    }

    fn commit_lagrange(&self, poly: &Polynomial<E::Fr, LagrangeCoeff>, _: Blind<E::Fr>) -> E::G1 {
        let mut scalars = Vec::with_capacity(poly.len());
        scalars.extend(poly.iter());
        let bases = &self.g_lagrange;
        let size = scalars.len();
        assert!(bases.len() >= size);

        best_multiexp(&scalars, &bases[0..size])
    }

    /// Writes params to a buffer.
    fn write<W: io::Write>(&self, writer: &mut W) -> io::Result<()> {
        self.write_custom(writer, SerdeFormat::RawBytes)
    }

    /// Reads params from a buffer.
    fn read<R: io::Read>(reader: &mut R) -> io::Result<Self> {
        Self::read_custom(reader, SerdeFormat::RawBytes)
    }
}

impl<'params, E: Engine + Debug> ParamsVerifier<'params, E::G1Affine> for ParamsKZG<E>
where
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
{
}

impl<'params, E: Engine + Debug> ParamsProver<'params, E::G1Affine> for ParamsKZG<E>
where
    E::G1Affine: SerdeCurveAffine<ScalarExt = <E as Engine>::Fr, CurveExt = <E as Engine>::G1>,
    E::G1: CurveExt<AffineExt = E::G1Affine>,
    E::G2Affine: SerdeCurveAffine,
{
    type ParamsVerifier = ParamsVerifierKZG<E>;

    fn verifier_params(&'params self) -> &'params Self::ParamsVerifier {
        self
    }

    fn new(k: u32) -> Self {
        Self::setup(k, OsRng)
    }

    fn commit(&self, poly: &Polynomial<E::Fr, Coeff>, _: Blind<E::Fr>) -> E::G1 {
        let mut scalars = Vec::with_capacity(poly.len());
        scalars.extend(poly.iter());
        let bases = &self.g;
        let size = scalars.len();
        assert!(bases.len() >= size);

        best_multiexp(&scalars, &bases[0..size])
    }

    fn get_g(&self) -> &[E::G1Affine] {
        &self.g
    }
}

#[cfg(test)]
mod test {
    use crate::poly::commitment::ParamsProver;
    use crate::poly::commitment::{Blind, Params};
    use crate::poly::kzg::commitment::ParamsKZG;
    use ff::Field;

    #[test]
    fn test_commit_lagrange() {
        const K: u32 = 6;

        use rand_core::OsRng;

        use crate::poly::EvaluationDomain;
        use halo2curves::bn256::{Bn256, Fr};

        let params = ParamsKZG::<Bn256>::new(K);
        let domain = EvaluationDomain::new(1, K);

        let mut a = domain.empty_lagrange();

        for (i, a) in a.iter_mut().enumerate() {
            *a = Fr::from(i as u64);
        }

        let b = domain.lagrange_to_coeff(a.clone());

        let alpha = Blind(Fr::random(OsRng));

        assert_eq!(params.commit(&b, alpha), params.commit_lagrange(&a, alpha));
    }

    #[test]
    fn test_parameter_serialisation_roundtrip() {
        const K: u32 = 4;

        use super::super::commitment::Params;
        use crate::halo2curves::bn256::Bn256;

        let params0 = ParamsKZG::<Bn256>::new(K);
        let mut data = vec![];
        <ParamsKZG<_> as Params<_>>::write(&params0, &mut data).unwrap();
        let params1: ParamsKZG<Bn256> = Params::read::<_>(&mut &data[..]).unwrap();

        assert_eq!(params0.k, params1.k);
        assert_eq!(params0.n, params1.n);
        assert_eq!(params0.g.len(), params1.g.len());
        assert_eq!(params0.g_lagrange.len(), params1.g_lagrange.len());

        assert_eq!(params0.g, params1.g);
        assert_eq!(params0.g_lagrange, params1.g_lagrange);
        assert_eq!(params0.g2, params1.g2);
        assert_eq!(params0.s_g2, params1.s_g2);
    }

    #[test]
    fn test_selective_raw_srs_matches_full_read_then_downsize() {
        const SOURCE_K: u32 = 5;
        const TARGET_K: u32 = 3;

        use crate::SerdeFormat;
        use halo2curves::bn256::Bn256;
        use std::io::Cursor;

        let source = ParamsKZG::<Bn256>::new(SOURCE_K);
        let mut encoded = Vec::new();
        source.write_custom(&mut encoded, SerdeFormat::RawBytes).unwrap();

        let mut full_reader = Cursor::new(encoded.as_slice());
        let mut expected = ParamsKZG::<Bn256>::read_custom(
            &mut full_reader,
            SerdeFormat::RawBytes,
        )
        .unwrap();
        expected.downsize(TARGET_K);

        let mut selective_reader = Cursor::new(encoded.as_slice());
        let actual = ParamsKZG::<Bn256>::read_custom_for_k(
            &mut selective_reader,
            SerdeFormat::RawBytes,
            TARGET_K,
            SOURCE_K,
        )
        .unwrap();

        assert_eq!(expected.k, actual.k);
        assert_eq!(expected.n, actual.n);
        assert_eq!(expected.g, actual.g);
        assert_eq!(expected.g_lagrange, actual.g_lagrange);
        assert_eq!(expected.g2, actual.g2);
        assert_eq!(expected.s_g2, actual.s_g2);
    }

    #[test]
    fn test_selective_raw_srs_rejects_invalid_target_and_length() {
        const SOURCE_K: u32 = 4;

        use crate::SerdeFormat;
        use halo2curves::bn256::Bn256;
        use std::io::Cursor;

        let source = ParamsKZG::<Bn256>::new(SOURCE_K);
        let mut encoded = Vec::new();
        source.write_custom(&mut encoded, SerdeFormat::RawBytes).unwrap();

        let mut too_large_reader = Cursor::new(encoded.as_slice());
        assert!(ParamsKZG::<Bn256>::read_custom_for_k(
            &mut too_large_reader,
            SerdeFormat::RawBytes,
            SOURCE_K + 1,
            SOURCE_K,
        )
        .is_err());

        let mut truncated = encoded;
        truncated.pop();
        let mut truncated_reader = Cursor::new(truncated.as_slice());
        assert!(ParamsKZG::<Bn256>::read_custom_for_k(
            &mut truncated_reader,
            SerdeFormat::RawBytes,
            SOURCE_K - 1,
            SOURCE_K,
        )
        .is_err());
    }
}
