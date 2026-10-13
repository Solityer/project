/// SRS generation, processing, verification and downloading
pub mod srs;

/// errors related to pfsys
pub mod errors;

pub use errors::PfsysError;
use itertools::chain;
use std::borrow::Borrow;

use crate::circuit::CheckMode;
use crate::graph::GraphWitness;
use crate::{EZKL_BUF_CAPACITY, EZKL_KEY_FORMAT};
use halo2_proofs::plonk::{
    create_proof, keygen_pk, keygen_vk_custom,
    keygen_vk_custom_with_native_gate, verify_proof, Circuit, NativeKeygenBudget,
    NativeKeygenCensus, NativeKeygenPreflight, NativePkReadTiming, ProvingKey, VerifyingKey,
};
#[cfg(not(feature = "formal-gpu"))]
use halo2_proofs::plonk::keygen_pk_with_native_gate;
#[cfg(feature = "formal-gpu")]
use halo2_proofs::plonk::keygen_pk_with_native_gate_compact_artifact;
use halo2_proofs::poly::commitment::{CommitmentScheme, Params, ParamsProver, Prover, Verifier};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer, TranscriptWriterBuffer};
use halo2curves::ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use halo2curves::serde::SerdeObject;
use halo2curves::{bn256, CurveAffine};
use instant::Instant;
use log::{debug, info, trace};
#[cfg(not(feature = "det-prove"))]
use rand::rngs::OsRng;
#[cfg(feature = "det-prove")]
use rand::rngs::StdRng;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use snark_verifier::verifier::plonk::PlonkProtocol;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Cursor, Write};
use std::ops::Deref;
use std::path::PathBuf;
use thiserror::Error as thisError;

#[cfg(feature = "python-bindings")]
use pyo3::types::PyDictMethods;

/// Converts a string to a `SerdeFormat`.
/// # Panics
/// Panics if the provided `s` is not a valid `SerdeFormat` (i.e. not one of "processed", "raw-bytes-unchecked", or "raw-bytes").
fn serde_format_from_str(s: &str) -> halo2_proofs::SerdeFormat {
    match s {
        "processed" => halo2_proofs::SerdeFormat::Processed,
        "raw-bytes-unchecked" => halo2_proofs::SerdeFormat::RawBytesUnchecked,
        "raw-bytes" => halo2_proofs::SerdeFormat::RawBytes,
        _ => panic!("invalid serde format"),
    }
}

/// Function signature of `verifyProof(bytes,uint256[])`.
pub const FN_SIG_VERIFY_PROOF: [u8; 4] = [0x1e, 0x8e, 0x1e, 0x13];

/// Function signature of `verifyProof(bytes,uint256[],bytes32[])`.
pub const FN_SIG_VERIFY_PROOF_WITH_VKA: [u8; 4] = [0x34, 0x09, 0xfc, 0x9f];

/// Function signature of verifyWithDataAttestation(address,bytes)
pub const FN_SIG_VERIFY_WITH_DATA_ATTESTATION: [u8; 4] = [0x4c, 0x79, 0x85, 0xd0];

/// Function signatore of registeredVkas(bytes32[]) 0xdc8b4094
pub const FN_SIG_REGISTER_VKA: [u8; 4] = [0xdc, 0x8b, 0x40, 0x94];

/// Encode proof into calldata to invoke `Halo2Verifier.verifyProof`.
///
/// For `vk_address`:
/// - Pass `None` if verifying key is embedded in `Halo2Verifier`
/// - Pass `Some(vka)` if verifying key is separated and already registered
pub fn encode_calldata(vka: Option<&[[u8; 32]]>, proof: &[u8], instances: &[bn256::Fr]) -> Vec<u8> {
    let (fn_sig, offset) = if vka.is_some() {
        (FN_SIG_VERIFY_PROOF_WITH_VKA, 0x60)
    } else {
        (FN_SIG_VERIFY_PROOF, 0x40)
    };
    let num_instances = instances.len();
    let (vka_offset, vka_data) = if let Some(vka) = vka {
        (
            to_be_bytes_32(offset + 0x40 + proof.len() + (num_instances * 0x20)).to_vec(),
            vka.to_vec(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    let num_vka_words = vka_data.len();
    chain![
        fn_sig,                                              // function signature
        to_be_bytes_32(offset),                              // offset of proof
        to_be_bytes_32(offset + 0x20 + proof.len()),         // offset of instances
        vka_offset,                                          // offset of vka
        to_be_bytes_32(proof.len()),                         // length of proof
        proof.iter().cloned(),                               // proof
        to_be_bytes_32(num_instances),                       // length of instances
        instances.iter().map(fr_to_bytes32).flatten(),       // instances
        to_be_bytes_32(num_vka_words),                       // vka length
        vka_data.iter().flat_map(|arr| arr.iter().cloned())  // vka words
    ]
    .collect()
}

fn to_be_bytes_32(value: usize) -> [u8; 32] {
    let mut bytes = [0u8; 32];
    // Convert the usize to big-endian bytes in the last 8 bytes (or however many needed)
    let value_bytes = value.to_be_bytes();
    let start_idx = 32 - value_bytes.len();
    bytes[start_idx..].copy_from_slice(&value_bytes);
    bytes
}

fn fr_to_bytes32(fe: impl Borrow<bn256::Fr>) -> [u8; 32] {
    fe_to_bytes32(fe)
}

fn fe_to_bytes32<F>(fe: impl Borrow<F>) -> [u8; 32]
where
    F: PrimeField<Repr = halo2_proofs::halo2curves::serde::Repr<32>>,
{
    let repr = fe.borrow().to_repr();
    // Note: we're converting from little-endian representation to big-endian bytes
    let mut bytes = [0u8; 32];
    let inner = repr.inner();
    for i in 0..32 {
        bytes[31 - i] = inner[i];
    }
    bytes
}

#[derive(thisError, Debug)]
/// Errors related to pfsys
pub enum PfSysError {
    /// Packing exponent is too large
    #[error("largest packing exponent exceeds max. try reducing the scale")]
    PackingExponent,
}

#[cfg(feature = "python-bindings")]
use halo2curves::bn256::G1Affine;

#[cfg(feature = "python-bindings")]
///
pub fn g1affine_to_pydict(g1affine_dict: &pyo3::Bound<'_, PyDict>, g1affine: &G1Affine) {
    let g1affine_x = field_to_string(&g1affine.x);
    let g1affine_y = field_to_string(&g1affine.y);
    g1affine_dict.set_item("x", g1affine_x).unwrap();
    g1affine_dict.set_item("y", g1affine_y).unwrap();
}

#[cfg(feature = "python-bindings")]
use halo2curves::bn256::G1;
#[cfg(feature = "python-bindings")]
///
pub fn g1_to_pydict(g1_dict: &pyo3::Bound<'_, PyDict>, g1: &G1) {
    let g1_x = field_to_string(&g1.x);
    let g1_y = field_to_string(&g1.y);
    let g1_z = field_to_string(&g1.z);
    g1_dict.set_item("x", g1_x).unwrap();
    g1_dict.set_item("y", g1_y).unwrap();
    g1_dict.set_item("z", g1_z).unwrap();
}

/// converts fp into a little endian Hex string
pub fn field_to_string<F: PrimeField + SerdeObject + Serialize>(fp: &F) -> String {
    let repr = serde_json::to_string(&fp).unwrap();
    let b: String = serde_json::from_str(&repr).unwrap();
    b
}

/// Encode the native BN254 Fr values that cross the authenticated GAT
/// statement boundary.
///
/// This is deliberately separate from [`field_to_string`], whose historical
/// ezkl wire format is used by a number of unrelated graph artifacts.  The
/// GAT statement format is stricter: it carries the canonical 32-byte
/// `PrimeField::Repr` emitted by halo2curves, in little-endian byte order,
/// with an explicit `0x` prefix and lower-case hexadecimal digits.
pub fn canonical_fr_to_string(value: &bn256::Fr) -> String {
    format!("0x{}", hex::encode(value.to_repr().as_ref()))
}

/// Decode one authenticated GAT BN254 Fr value without reduction or
/// alternate encodings.
pub fn canonical_string_to_fr(value: &str) -> Result<bn256::Fr, String> {
    if value.len() != 66 || !value.starts_with("0x") {
        return Err("native BN254 Fr must be encoded as 0x plus 64 hex digits".into());
    }
    if value[2..].bytes().any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase()) {
        return Err("native BN254 Fr encoding must use lower-case hexadecimal".into());
    }
    let mut repr = [0_u8; 32];
    hex::decode_to_slice(&value[2..], &mut repr).map_err(|error| format!("invalid native BN254 Fr encoding: {}", error))?;
    Option::from(bn256::Fr::from_repr(repr.into())).ok_or_else(|| "native BN254 Fr encoding is not a canonical field element".into())
}

/// converts a little endian Hex string into a field element
pub fn string_to_field<F: PrimeField + SerdeObject + Serialize + DeserializeOwned>(
    b: &String,
) -> F {
    let repr = serde_json::to_string(&b).unwrap();
    let fp: F = serde_json::from_str(&repr).unwrap();
    fp
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
/// Contains the instances of the circuit in human readable form
pub struct PrettyElements {
    /// the inputs as rescaled floats -- represented as a String for maximum compatibility with Python and JS
    pub rescaled_inputs: Vec<Vec<String>>,
    /// the inputs as felts but 0x strings -- represented as a String for maximum compatibility with Python and JS
    pub inputs: Vec<Vec<String>>,
    /// the processed inputs (eg. hash of the inputs) -- stays as a felt represented as a 0x string for maximum compatibility with Python and JS
    pub processed_inputs: Vec<Vec<String>>,
    /// the processed params (eg. hash of the params) -- stays as a felt represented as a 0x string for maximum compatibility with Python and JS
    pub processed_params: Vec<Vec<String>>,
    /// the processed outputs (eg. hash of the outputs) -- stays as a felt represented as a 0x string for maximum compatibility with Python and JS
    pub processed_outputs: Vec<Vec<String>>,
    /// the outputs as rescaled floats (if any) -- represented as a String for maximum compatibility with Python and JS
    pub rescaled_outputs: Vec<Vec<String>>,
    /// the outputs as felts but 0x strings (if any) -- represented as a String for maximum compatibility with Python and JS
    pub outputs: Vec<Vec<String>>,
}

/// An application snark with proof and instance variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snark<F: PrimeField + SerdeObject, C: CurveAffine>
where
    C::Scalar: Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    /// the protocol
    pub protocol: Option<PlonkProtocol<C>>,
    /// public instances of the snark
    pub instances: Vec<Vec<F>>,
    /// the proof
    pub proof: Vec<u8>,
    /// hex encoded proof
    pub hex_proof: Option<String>,
    /// the split proof
    pub split: Option<ProofSplitCommit>,
    /// the proof instances as rescaled floats
    pub pretty_public_inputs: Option<PrettyElements>,
    /// timestamp
    pub timestamp: Option<u128>,
    /// In-process native proof measurements. This is intentionally excluded
    /// from the verifier-facing proof artifact serialization.
    #[serde(skip)]
    pub proof_timing: Option<ProofTiming>,
    /// (optional) version of ezkl used to generate the proof
    version: Option<String>,
}

/// Native proof work measured by Halo2's scoped commitment hook.
#[derive(Debug, Clone, PartialEq)]
pub struct ProofTiming {
    /// Initial public-instance and phase-zero advice-polynomial commitments.
    pub commitment_ms: f64,
    /// Number of per-proof public instance commitments included in the timer.
    pub instance_commitment_count: usize,
    /// Number of per-proof phase-zero advice commitments included in the timer.
    pub phase0_advice_commitment_count: usize,
    /// Number of affine-normalization batches included in the timer.
    pub affine_normalization_batch_count: usize,
    /// Remaining proof construction and verifier-facing encoding.
    pub prove_ms: f64,
}

#[cfg(feature = "python-bindings")]
use pyo3::{types::PyDict, IntoPyObject, Python};
#[cfg(feature = "python-bindings")]
impl<'py, F: PrimeField + SerdeObject + Serialize, C: CurveAffine + Serialize> IntoPyObject<'py>
    for Snark<F, C>
where
    C::Scalar: Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    type Target = pyo3::PyAny;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let dict = PyDict::new(py);
        let field_elems: Vec<Vec<String>> = self
            .instances
            .iter()
            .map(|x| x.iter().map(|fp| field_to_string(fp)).collect())
            .collect::<Vec<_>>();
        dict.set_item("instances", field_elems).unwrap();
        let hex_proof = hex::encode(&self.proof);
        dict.set_item("proof", format!("0x{}", hex_proof)).unwrap();
        Ok(dict.into_any())
    }
}

impl<
        F: PrimeField + SerdeObject + Serialize + FromUniformBytes<64> + DeserializeOwned,
        C: CurveAffine + Serialize + DeserializeOwned,
    > Snark<F, C>
where
    C::Scalar: Serialize + DeserializeOwned,
    C::ScalarExt: Serialize + DeserializeOwned,
{
    /// Create a new application snark from proof and instance variables
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        protocol: Option<PlonkProtocol<C>>,
        instances: Vec<Vec<F>>,
        proof: Vec<u8>,
        hex_proof: Option<String>,
        split: Option<ProofSplitCommit>,
        pretty_public_inputs: Option<PrettyElements>,
    ) -> Self {
        Self {
            protocol,
            instances,
            proof,
            hex_proof,
            split,
            pretty_public_inputs,
            // unix timestamp
            timestamp: Some(
                instant::SystemTime::now()
                    .duration_since(instant::SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_millis(),
            ),
            proof_timing: None,
            version: Some(crate::version().to_string()),
        }
    }

    /// create hex proof from proof
    pub fn create_hex_proof(&mut self) {
        let hex_proof = hex::encode(&self.proof);
        self.hex_proof = Some(format!("0x{}", hex_proof));
    }

    /// Saves the Proof to a specified `proof_path`.
    pub fn save(&self, proof_path: &PathBuf) -> Result<(), PfsysError> {
        let file = std::fs::File::create(proof_path)
            .map_err(|e| PfsysError::SaveProof(format!("{}", e)))?;
        let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, file);
        serde_json::to_writer(&mut writer, &self)
            .map_err(|e| PfsysError::SaveProof(format!("{}", e)))?;
        Ok(())
    }

    /// Load a json serialized proof from the provided path.
    pub fn load<Scheme: CommitmentScheme<Curve = C, Scalar = F>>(
        proof_path: &PathBuf,
    ) -> Result<Self, PfsysError>
    where
        <C as CurveAffine>::ScalarExt: FromUniformBytes<64>,
    {
        trace!("reading proof");
        let file =
            std::fs::File::open(proof_path).map_err(|e| PfsysError::LoadProof(format!("{}", e)))?;
        let reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, file);
        let proof: Self =
            serde_json::from_reader(reader).map_err(|e| PfsysError::LoadProof(format!("{}", e)))?;
        Ok(proof)
    }

    /// Decode an already-read JSON proof artifact.
    pub fn decode_json_bytes(proof_bytes: &[u8]) -> Result<Self, PfsysError> {
        serde_json::from_slice(proof_bytes).map_err(|e| PfsysError::LoadProof(format!("{}", e)))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
/// A proof split commit
pub struct ProofSplitCommit {
    /// The start index of the output in the witness
    start: usize,
    /// The end index of the output in the witness
    end: usize,
}

impl From<GraphWitness> for Option<ProofSplitCommit> {
    fn from(witness: GraphWitness) -> Self {
        let mut elem_offset = 0;

        if let Some(input) = witness.processed_inputs {
            if let Some(polycommit) = input.polycommit {
                // flatten and count number of elements
                let num_elements = polycommit
                    .iter()
                    .map(|polycommit| polycommit.len())
                    .sum::<usize>();

                elem_offset += num_elements;
            }
        }

        if let Some(params) = witness.processed_params {
            if let Some(polycommit) = params.polycommit {
                // flatten and count number of elements
                let num_elements = polycommit
                    .iter()
                    .map(|polycommit| polycommit.len())
                    .sum::<usize>();

                elem_offset += num_elements;
            }
        }

        if let Some(output) = witness.processed_outputs {
            if let Some(polycommit) = output.polycommit {
                // flatten and count number of elements
                let num_elements = polycommit
                    .iter()
                    .map(|polycommit| polycommit.len())
                    .sum::<usize>();

                Some(ProofSplitCommit {
                    start: elem_offset,
                    end: elem_offset + num_elements,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

/// Creates a [VerifyingKey] and [ProvingKey] for a [crate::graph::GraphCircuit] (`circuit`) with specific [CommitmentScheme] parameters (`params`).
pub fn create_keys<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    circuit: &C,
    params: &'_ Scheme::ParamsProver,
    disable_selector_compression: bool,
) -> Result<ProvingKey<Scheme::Curve>, halo2_proofs::plonk::Error>
where
    <Scheme as CommitmentScheme>::Scalar: FromUniformBytes<64>,
{
    //	Real proof
    let empty_circuit = <C as Circuit<Scheme::Scalar>>::without_witnesses(circuit);

    // Initialize verifying key
    let now = Instant::now();
    trace!("preparing VK");
    let vk = keygen_vk_custom(params, &empty_circuit, !disable_selector_compression)?;
    let elapsed = now.elapsed();
    info!("VK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());

    // Initialize the proving key
    let now = Instant::now();
    let pk = keygen_pk(params, vk, &empty_circuit)?;
    let elapsed = now.elapsed();
    info!("PK took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());
    Ok(pk)
}

/// Create EZKL's native keypair while allowing the formal runner to gate the
/// actual Halo2 allocations and record the mandatory keygen synthesis census.
/// VK and PK each synthesize once as required by Halo2; their source-derived
/// geometry and observed census must agree exactly before the PK is accepted.
pub fn create_keys_with_native_gate<Scheme, C, Before, After>(
    circuit: &C,
    params: &Scheme::ParamsProver,
    disable_selector_compression: bool,
    mut before_assembly: Before,
    mut after_synthesis: After,
) -> Result<(ProvingKey<Scheme::Curve>, NativeKeygenCensus), halo2_proofs::plonk::Error>
where
    Scheme: CommitmentScheme,
    C: Circuit<Scheme::Scalar>,
    Scheme::Scalar: FromUniformBytes<64>,
    Before: FnMut(&str, &NativeKeygenPreflight) -> io::Result<NativeKeygenBudget>,
    After: FnMut(&str, &NativeKeygenCensus) -> io::Result<()>,
{
    let empty_circuit = <C as Circuit<Scheme::Scalar>>::without_witnesses(circuit);
    let mut vk_preflight = None;
    let mut vk_census = None;

    let now = Instant::now();
    trace!("preparing VK with native resource gate");
    let vk = keygen_vk_custom_with_native_gate(
        params,
        &empty_circuit,
        !disable_selector_compression,
        |preflight| {
            let budget = before_assembly("vk", preflight)?;
            if budget.max_selector_activation_bytes == u64::MAX || budget.max_permutation_backing_bytes == u64::MAX {
                return Err(io::Error::other("native VK keygen allocation caps must be finite"));
            }
            vk_preflight = Some(preflight.clone());
            Ok(budget)
        },
        |census| {
            after_synthesis("vk", census)?;
            vk_census = Some(census.clone());
            Ok(())
        },
    )?;
    let elapsed = now.elapsed();
    info!("VK with native resource gate took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());

    let expected_preflight = vk_preflight.ok_or(halo2_proofs::plonk::Error::Synthesis)?;
    let expected_census = vk_census.ok_or(halo2_proofs::plonk::Error::Synthesis)?;
    let now = Instant::now();
    trace!("preparing PK with native resource gate");
    #[cfg(feature = "formal-gpu")]
    let pk = keygen_pk_with_native_gate_compact_artifact(
        params,
        vk,
        &empty_circuit,
        |preflight| {
            if preflight != &expected_preflight {
                return Err(io::Error::other("native VK/PK keygen preflight geometry changed"));
            }
            let budget = before_assembly("pk", preflight)?;
            if budget.max_selector_activation_bytes == u64::MAX || budget.max_permutation_backing_bytes == u64::MAX {
                return Err(io::Error::other("native PK keygen allocation caps must be finite"));
            }
            Ok(budget)
        },
        |census| {
            if census != &expected_census {
                return Err(io::Error::other("native VK/PK keygen synthesis census changed"));
            }
            after_synthesis("pk", census)
        },
    )?;
    #[cfg(not(feature = "formal-gpu"))]
    let pk = keygen_pk_with_native_gate(
        params,
        vk,
        &empty_circuit,
        |preflight| {
            if preflight != &expected_preflight {
                return Err(io::Error::other("native VK/PK keygen preflight geometry changed"));
            }
            let budget = before_assembly("pk", preflight)?;
            if budget.max_selector_activation_bytes == u64::MAX || budget.max_permutation_backing_bytes == u64::MAX {
                return Err(io::Error::other("native PK keygen allocation caps must be finite"));
            }
            Ok(budget)
        },
        |census| {
            if census != &expected_census {
                return Err(io::Error::other("native VK/PK keygen synthesis census changed"));
            }
            after_synthesis("pk", census)
        },
    )?;
    let elapsed = now.elapsed();
    info!("PK with native resource gate took {}.{}", elapsed.as_secs(), elapsed.subsec_millis());
    Ok((pk, expected_census))
}

/// a wrapper around halo2's create_proof
#[allow(clippy::too_many_arguments)]
pub fn create_proof_circuit<
    'params,
    Scheme: CommitmentScheme,
    C: Circuit<Scheme::Scalar>,
    P: Prover<'params, Scheme>,
    V: Verifier<'params, Scheme>,
    Strategy: VerificationStrategy<'params, Scheme, V>,
    E: EncodedChallenge<Scheme::Curve>,
    TW: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
>(
    circuit: C,
    instances: Vec<Vec<Scheme::Scalar>>,
    params: &'params Scheme::ParamsProver,
    pk: &ProvingKey<Scheme::Curve>,
    check_mode: CheckMode,
    split: Option<ProofSplitCommit>,
    protocol: Option<PlonkProtocol<Scheme::Curve>>,
) -> Result<Snark<Scheme::Scalar, Scheme::Curve>, PfsysError>
where
    Scheme::ParamsVerifier: 'params,
    Scheme::Scalar: Serialize
        + DeserializeOwned
        + SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>,
    Scheme::Curve: Serialize + DeserializeOwned + SerdeObject,
    Scheme::ParamsProver: Send + Sync,
{
    let strategy = Strategy::new(params.verifier_params());
    let mut transcript = TranscriptWriterBuffer::<_, Scheme::Curve, _>::init(vec![]);
    #[cfg(feature = "det-prove")]
    let mut rng = <StdRng as rand::SeedableRng>::from_seed([0u8; 32]);
    #[cfg(not(feature = "det-prove"))]
    let mut rng = OsRng;

    let pi_inner = instances
        .iter()
        .map(|e| e.deref())
        .collect::<Vec<&[Scheme::Scalar]>>();
    let pi_inner: &[&[&[Scheme::Scalar]]] = &[&pi_inner];
    trace!("instances {:?}", instances);
    trace!(
        "pk num instance column: {:?}",
        pk.get_vk().cs().num_instance_columns()
    );

    info!("proof started...");
    // not wasm32 unknown
    let commitment_measurement = halo2_proofs::plonk::commitment_timing::begin()
        .map_err(|error| PfsysError::TimingError(error.to_string()))?;
    let now = Instant::now();

    create_proof::<Scheme, P, _, _, TW, _>(
        params,
        pk,
        &[circuit],
        pi_inner,
        &mut rng,
        &mut transcript,
    )?;
    let commitment_timing = commitment_measurement
        .finish()
        .map_err(|error| PfsysError::TimingError(error.to_string()))?;
    let commitment_elapsed = commitment_timing.elapsed;
    let proof = transcript.finalize();
    let hex_proof = format!("0x{}", hex::encode(&proof));
    let prove_elapsed = now.elapsed();

    let mut checkable_pf = Snark::new(protocol, instances, proof, Some(hex_proof), split, None);
    checkable_pf.proof_timing = Some(ProofTiming {
        commitment_ms: commitment_elapsed.as_secs_f64() * 1000.0,
        instance_commitment_count: commitment_timing.instance_commitments,
        phase0_advice_commitment_count: commitment_timing.advice_commitments,
        affine_normalization_batch_count: commitment_timing.normalization_batches,
        prove_ms: (prove_elapsed.as_secs_f64() * 1000.0
            - commitment_elapsed.as_secs_f64() * 1000.0)
            .max(0.0),
    });

    // sanity check that the generated proof is valid
    if check_mode == CheckMode::SAFE {
        debug!("verifying generated proof");
        let verifier_params = params.verifier_params();
        verify_proof_circuit::<V, Scheme, Strategy, E, TR>(
            &checkable_pf,
            verifier_params,
            pk.get_vk(),
            strategy,
            verifier_params.n(),
        )?;
    }
    let elapsed = now.elapsed();
    info!(
        "proof took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );

    Ok(checkable_pf)
}

/// Swaps the proof commitments to a new set in the proof
pub fn swap_proof_commitments<
    Scheme: CommitmentScheme,
    E: EncodedChallenge<Scheme::Curve>,
    TW: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
>(
    snark: &Snark<Scheme::Scalar, Scheme::Curve>,
    commitments: &[Scheme::Curve],
) -> Result<Snark<Scheme::Scalar, Scheme::Curve>, PfsysError>
where
    Scheme::Scalar: SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>
        + Ord
        + Serialize
        + DeserializeOwned,
    Scheme::Curve: Serialize + DeserializeOwned,
{
    let proof_first_bytes = get_proof_commitments::<Scheme, E, TW>(commitments)?;

    let mut snark_new = snark.clone();
    // swap the proof bytes for the new ones
    snark_new.proof[..proof_first_bytes.len()].copy_from_slice(&proof_first_bytes);
    snark_new.create_hex_proof();

    Ok(snark_new)
}

/// Returns the bytes encoded proof commitments
pub fn get_proof_commitments<
    Scheme: CommitmentScheme,
    E: EncodedChallenge<Scheme::Curve>,
    TW: TranscriptWriterBuffer<Vec<u8>, Scheme::Curve, E>,
>(
    commitments: &[Scheme::Curve],
) -> Result<Vec<u8>, PfsysError>
where
    Scheme::Scalar: SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>
        + Ord
        + Serialize
        + DeserializeOwned,
    Scheme::Curve: Serialize + DeserializeOwned,
{
    let mut transcript_new: TW = TranscriptWriterBuffer::<_, Scheme::Curve, _>::init(vec![]);

    // polycommit commitments are the first set of points in the proof, this will always be the first set of advice
    for commit in commitments {
        transcript_new
            .write_point(*commit)
            .map_err(|e| PfsysError::WritePoint(format!("{}", e)))?;
    }

    let proof_first_bytes = transcript_new.finalize();

    if commitments.is_empty() {
        log::warn!("no commitments found in witness");
    }

    Ok(proof_first_bytes)
}

/// A wrapper around halo2's verify_proof
pub fn verify_proof_circuit<
    'params,
    V: Verifier<'params, Scheme>,
    Scheme: CommitmentScheme,
    Strategy: VerificationStrategy<'params, Scheme, V>,
    E: EncodedChallenge<Scheme::Curve>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
>(
    snark: &Snark<Scheme::Scalar, Scheme::Curve>,
    params: &'params Scheme::ParamsVerifier,
    vk: &VerifyingKey<Scheme::Curve>,
    strategy: Strategy,
    orig_n: u64,
) -> Result<Strategy::Output, halo2_proofs::plonk::Error>
where
    Scheme::Scalar: SerdeObject
        + PrimeField
        + FromUniformBytes<64>
        + WithSmallOrderMulGroup<3>
        + Serialize
        + DeserializeOwned,
    Scheme::Curve: Serialize + DeserializeOwned,
{
    let pi_inner = snark
        .instances
        .iter()
        .map(|e| e.deref())
        .collect::<Vec<&[Scheme::Scalar]>>();
    let instances: &[&[&[Scheme::Scalar]]] = &[&pi_inner];
    trace!("instances {:?}", instances);

    let mut transcript = TranscriptReadBuffer::init(Cursor::new(snark.proof.clone()));
    verify_proof::<Scheme, V, _, TR, _>(params, vk, strategy, instances, &mut transcript, orig_n)
}

/// Loads a [VerifyingKey] at `path`.
pub fn load_vk<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
) -> Result<VerifyingKey<Scheme::Curve>, PfsysError>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    debug!("loading verification key from {:?}", path);
    let f = File::open(path.clone()).map_err(|e| PfsysError::LoadVk(format!("{}", e)))?;
    let mut reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, f);
    let vk = VerifyingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        serde_format_from_str(&EZKL_KEY_FORMAT),
        params,
    )
    .map_err(|e| PfsysError::LoadVk(format!("{}", e)))?;
    info!("loaded verification key ✅");
    Ok(vk)
}

/// Loads a [ProvingKey] at `path`.
pub fn load_pk<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
) -> Result<ProvingKey<Scheme::Curve>, PfsysError>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    debug!("loading proving key from {:?}", path);
    let start = instant::Instant::now();
    let f = File::open(path.clone()).map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    let mut reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, f);
    let pk = ProvingKey::<Scheme::Curve>::read::<_, C>(
        &mut reader,
        serde_format_from_str(&EZKL_KEY_FORMAT),
        params,
    )
    .map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    let elapsed = start.elapsed();
    info!("loaded proving key in {:?}", elapsed);
    Ok(pk)
}

/// Loads a formal native proving key while bounding the temporary sparse
/// permutation backing and target-membership validation store. The legacy
/// loader above remains available to nonformal callers.
pub fn load_pk_with_native_backing_budget<Scheme: CommitmentScheme, C: Circuit<Scheme::Scalar>>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
    max_total_backing_bytes: u64,
) -> Result<ProvingKey<Scheme::Curve>, PfsysError>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    debug!("loading proving key with native backing budget from {:?}", path);
    let start = instant::Instant::now();
    let f = File::open(path.clone()).map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    let mut reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, f);
    let pk = ProvingKey::<Scheme::Curve>::read_with_native_backing_budget::<_, C>(
        &mut reader,
        serde_format_from_str(&EZKL_KEY_FORMAT),
        max_total_backing_bytes,
        params,
    )
    .map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    let elapsed = start.elapsed();
    info!("loaded proving key with native backing budget in {:?}", elapsed);
    Ok(pk)
}

/// Loads a formal compact native proving key under its authenticated backing
/// budget and returns the non-overlapping artifact-validation and sigma
/// materialization wall-time breakdown. The legacy and untimed loaders remain
/// available to nonformal callers.
pub fn load_pk_with_native_backing_budget_and_timing<
    Scheme: CommitmentScheme,
    C: Circuit<Scheme::Scalar>,
>(
    path: PathBuf,
    params: <C as Circuit<Scheme::Scalar>>::Params,
    max_total_backing_bytes: u64,
) -> Result<(ProvingKey<Scheme::Curve>, NativePkReadTiming), PfsysError>
where
    Scheme::Curve: SerdeObject + CurveAffine,
    Scheme::Scalar: PrimeField + SerdeObject + FromUniformBytes<64>,
{
    debug!("loading proving key with native backing budget and timing from {:?}", path);
    let f = File::open(path.clone()).map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    let mut reader = BufReader::with_capacity(*EZKL_BUF_CAPACITY, f);
    let loaded = ProvingKey::<Scheme::Curve>::read_with_native_backing_budget_and_timing::<_, C>(
        &mut reader,
        serde_format_from_str(&EZKL_KEY_FORMAT),
        max_total_backing_bytes,
        params,
    )
    .map_err(|e| PfsysError::LoadPk(format!("{}", e)))?;
    info!("loaded proving key with native backing budget and timing");
    Ok(loaded)
}

/// Saves a [ProvingKey] to `path`.
pub fn save_pk<C: SerdeObject + CurveAffine>(
    path: &PathBuf,
    pk: &ProvingKey<C>,
) -> Result<(), io::Error>
where
    C::ScalarExt: FromUniformBytes<64> + SerdeObject,
{
    debug!("saving proving key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, f);
    pk.write(&mut writer, serde_format_from_str(&EZKL_KEY_FORMAT))?;
    writer.flush()?;
    info!("done saving proving key ✅");
    Ok(())
}

/// Saves a [VerifyingKey] to `path`.
pub fn save_vk<C: CurveAffine + SerdeObject>(
    path: &PathBuf,
    vk: &VerifyingKey<C>,
) -> Result<(), io::Error>
where
    C::ScalarExt: FromUniformBytes<64> + SerdeObject,
{
    debug!("saving verification key 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, f);
    vk.write(&mut writer, serde_format_from_str(&EZKL_KEY_FORMAT))?;
    writer.flush()?;
    info!("done saving verification key ✅");
    Ok(())
}

/// Saves [CommitmentScheme] parameters to `path`.
pub fn save_params<Scheme: CommitmentScheme>(
    path: &PathBuf,
    params: &'_ Scheme::ParamsVerifier,
) -> Result<(), io::Error> {
    debug!("saving parameters 💾");
    let f = File::create(path)?;
    let mut writer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, f);
    params.write(&mut writer)?;
    writer.flush()?;
    Ok(())
}

////////////////////////

#[cfg(test)]
#[cfg(all(feature = "ezkl", not(target_arch = "wasm32")))]
mod tests {

    use super::*;
    use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
    use halo2_proofs::poly::kzg::commitment::ParamsKZG;
    use halo2_proofs::SerdeFormat;
    use halo2curves::bn256::{Bn256, Fr, G1Affine};
    use std::io::Cursor;
    use tempfile::Builder;

    #[derive(Clone, Copy)]
    struct CopyCycleCircuit {
        mismatch: bool,
    }

    impl Circuit<Fr> for CopyCycleCircuit {
        type Config = (
            halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
            halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
            halo2_proofs::plonk::Column<halo2_proofs::plonk::Advice>,
        );
        type FloorPlanner = halo2_proofs::circuit::SimpleFloorPlanner;
        type Params = ();

        fn without_witnesses(&self) -> Self {
            Self { mismatch: false }
        }

        fn configure(meta: &mut halo2_proofs::plonk::ConstraintSystem<Fr>) -> Self::Config {
            let a = meta.advice_column();
            let b = meta.advice_column();
            let c = meta.advice_column();
            meta.enable_equality(a);
            meta.enable_equality(b);
            meta.enable_equality(c);
            (a, b, c)
        }

        fn synthesize(
            &self,
            columns: Self::Config,
            mut layouter: impl halo2_proofs::circuit::Layouter<Fr>,
        ) -> Result<(), halo2_proofs::plonk::Error> {
            use halo2_proofs::circuit::Value;
            layouter.assign_region(
                || "multi-column copy cycle equivalence",
                |mut region| {
                    let a0 = region.assign_advice(|| "a0", columns.0, 0, || Value::known(Fr::from(5)))?;
                    let a1 = region.assign_advice(|| "a1", columns.0, 1, || Value::known(Fr::from(7)))?;
                    let b2 = region.assign_advice(|| "b2", columns.1, 2, || Value::known(Fr::from(5)))?;
                    let b3 = region.assign_advice(|| "b3", columns.1, 3, || Value::known(Fr::from(7)))?;
                    let _c0 = region.assign_advice(|| "singleton", columns.2, 0, || Value::known(Fr::from(11)))?;
                    let c4_value = if self.mismatch { 6 } else { 5 };
                    let c4 = region.assign_advice(|| "c4", columns.2, 4, || Value::known(Fr::from(c4_value)))?;

                    region.constrain_equal(a0.cell(), b2.cell())?;
                    region.constrain_equal(b2.cell(), c4.cell())?;
                    region.constrain_equal(c4.cell(), a0.cell())?;
                    region.constrain_equal(a0.cell(), b2.cell())?;
                    region.constrain_equal(a1.cell(), b3.cell())?;
                    Ok(())
                },
            )
        }
    }

    #[test]
    fn test_copy_cycle_relation_acceptance_and_tamper_rejection() {
        use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
        use halo2_proofs::poly::kzg::strategy::SingleStrategy;
        use halo2_proofs::transcript::{
            Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer,
        };
        use rand::rngs::OsRng;

        let params = ParamsKZG::<Bn256>::setup(5, OsRng);
        let circuit = CopyCycleCircuit { mismatch: false };
        let vk = halo2_proofs::plonk::keygen_vk(&params, &circuit).unwrap();
        let pk = halo2_proofs::plonk::keygen_pk(&params, vk, &circuit).unwrap();

        let prove = |circuit: CopyCycleCircuit| {
            let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
            halo2_proofs::plonk::create_proof::<
                KZGCommitmentScheme<Bn256>,
                ProverSHPLONK<_>,
                _,
                _,
                _,
                _,
            >(&params, &pk, &[circuit], &[&[]], OsRng, &mut transcript)?;
            Ok::<_, halo2_proofs::plonk::Error>(transcript.finalize())
        };
        let verify = |proof: &[u8]| {
            let verifier_params = params.verifier_params();
            let strategy = SingleStrategy::new(verifier_params);
            let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(proof);
            halo2_proofs::plonk::verify_proof::<
                KZGCommitmentScheme<Bn256>,
                VerifierSHPLONK<_>,
                _,
                Blake2bRead<_, _, Challenge255<_>>,
                SingleStrategy<_>,
            >(
                verifier_params,
                pk.get_vk(),
                strategy,
                &[&[]],
                &mut transcript,
                params.n(),
            )
            .is_ok()
        };

        let valid_proof = prove(circuit).expect("valid copy-cycle proof generation");
        assert!(verify(&valid_proof), "valid multi-cycle proof rejected");

        let mismatched = prove(CopyCycleCircuit { mismatch: true });
        assert!(
            mismatched
                .as_ref()
                .map(|proof| verify(proof))
                .unwrap_or(false)
                == false,
            "mismatched copy-equivalence class was accepted",
        );

        let mut tampered = valid_proof;
        let last = tampered.len() - 1;
        tampered[last] ^= 1;
        assert!(!verify(&tampered), "tampered proof was accepted");
    }

    #[tokio::test]
    async fn test_can_load_saved_srs() {
        let tmp_dir = Builder::new().prefix("example").tempdir().unwrap();
        let fname = tmp_dir.path().join("polycommit.params");
        let srs = srs::gen_srs::<KZGCommitmentScheme<Bn256>>(1);
        let res = save_params::<KZGCommitmentScheme<Bn256>>(&fname, &srs);
        assert!(res.is_ok());
        let res = srs::load_srs_prover::<KZGCommitmentScheme<Bn256>>(fname);
        assert!(res.is_ok())
    }

    #[test]
    fn test_authenticated_srs_domain_reduction_matches_full_read() {
        const SOURCE_K: u32 = 5;
        const TARGET_K: u32 = 3;

        let source = <KZGCommitmentScheme<Bn256> as CommitmentScheme>::new_params(SOURCE_K);
        let mut encoded = Vec::new();
        source
            .write_custom(&mut encoded, SerdeFormat::RawBytes)
            .unwrap();

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

        let mut expected_bytes = Vec::new();
        expected
            .write_custom(&mut expected_bytes, SerdeFormat::RawBytes)
            .unwrap();
        let mut actual_bytes = Vec::new();
        actual
            .write_custom(&mut actual_bytes, SerdeFormat::RawBytes)
            .unwrap();
        assert_eq!(expected_bytes, actual_bytes);

        let mut wrong_header = Cursor::new(encoded.as_slice());
        assert!(ParamsKZG::<Bn256>::read_custom_for_k(
            &mut wrong_header,
            SerdeFormat::RawBytes,
            TARGET_K,
            SOURCE_K + 1,
        )
        .is_err());

        let mut truncated = encoded;
        truncated.pop();
        let mut truncated_reader = Cursor::new(truncated.as_slice());
        assert!(ParamsKZG::<Bn256>::read_custom_for_k(
            &mut truncated_reader,
            SerdeFormat::RawBytes,
            TARGET_K,
            SOURCE_K,
        )
        .is_err());
    }

    #[cfg(feature = "formal-gpu")]
    #[test]
    fn full_ezkl_repair_batch() {
        use crate::graph::{GraphCircuit, GraphWitness, NodeType, SupportedOp};
        use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
        use halo2_proofs::poly::kzg::strategy::SingleStrategy;
        use halo2_proofs::transcript::{Blake2bRead, Blake2bWrite, Challenge255};
        use rand::rngs::OsRng;
        use std::process::{Command, Stdio};
        crate::gat_adapter::run_geometry_and_mock_regressions();
        assert!(crate::execute::native_keygen_domains_within_capacity(26, 28));
        assert!(crate::execute::native_keygen_domains_within_capacity(25, 27));
        assert!(!crate::execute::native_keygen_domains_within_capacity(27, 28));
        assert!(!crate::execute::native_keygen_domains_within_capacity(26, 29));
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let temp = Builder::new().prefix("ezkl-repair-batch").tempdir().unwrap();
        let _spill = halo2_proofs::poly::native_spill_scope(temp.path()).unwrap();
        let fixture = root.join("tests/fixtures/gat_lookup_reuse");
        let compiled = GraphCircuit::load(fixture.join("compiled.circuit")).unwrap();
        let witness = GraphWitness::from_path(fixture.join("witness.json")).unwrap();
        let mut run_args = compiled.settings().run_args.clone();
        run_args.logrows = 13;
        let mut gat_spec = compiled.model().graph.nodes.values().find_map(|node| match node {
            NodeType::Node(node) => match &node.opkind { SupportedOp::Gat(op) => Some(op.spec.clone()), _ => None },
            NodeType::SubGraph { .. } => None,
        }).expect("tiny fixture includes the canonical GAT operation");
        run_args.param_scale = gat_spec.scales.s_w_hidden[0][0];
        assert_eq!((gat_spec.graph.n_total, gat_spec.graph.n, gat_spec.graph.e, gat_spec.feature_dim, gat_spec.graph.g_batch), (1, 1, 1, 1, 1));
        assert_eq!(gat_spec.model.hidden_layers.iter().map(|layer| layer.heads.len()).collect::<Vec<_>>(), vec![8, 8]);
        assert_eq!(gat_spec.model.output_heads.len(), 2);
        assert_eq!(gat_spec.model.classes, 1);
        assert_eq!(gat_spec.model.flattened_params_len().unwrap(), 126);
        assert_eq!(gat_spec.ledger.max_score_abs, 2_048);
        assert_eq!(witness.inputs.iter().map(Vec::len).collect::<Vec<_>>(), vec![1, 1, 1, 1, 2, 2]);
        assert_eq!(witness.params.iter().map(Vec::len).collect::<Vec<_>>(), vec![126]);
        assert_eq!(witness.outputs.iter().map(Vec::len).collect::<Vec<_>>(), vec![1]);
        let original_ledger = gat_spec.ledger.clone();
        let original_scales = gat_spec.scales.clone();
        let original_tables = gat_spec.tables.clone();
        let original_model_binding = gat_spec.model_binding.clone();
        let original_graph = gat_spec.graph.clone();
        // The committed tiny artifact predates ledger-closed table domains.
        // Regenerate only its exact tables from the unchanged fixture ledger,
        // graph degree, and scales by calling the production resource
        // generator's canonical row and digest routines.
        let table_rebuild = r#"import json,sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'scripts'))
import generate_gat_resources as gen
data=json.load(sys.stdin)
tables=data['tables']; ledger=data['ledger']; scales=data['scales']; graph=data['graph']
score=int(ledger['max_score_abs'])
delta=max(int(ledger['max_delta']),max(2*int(family['max_score_abs']) for family in ledger['families']))
if score < 0 or delta < 0:
    raise ValueError('tiny fixture ledger has a negative score/delta bound')
lrelu_output=max(abs(gen.table_value({'function_id':'lrelu'},-score)),abs(gen.table_value({'function_id':'lrelu'},score)))
exp_high=max(delta,2*lrelu_output)
exp_zero=gen.table_value({'function_id':'exp_neg'},0)
elu_high=0
for family in ledger['families']:
    if family['is_output']:
        continue
    layer=int(family['layer']); head=int(family['head'])
    numerator=int(family['max_aggregation_num']) * (1 << int(scales['s_agg_pre'][layer]))
    minimum_divisor=exp_zero * (1 << int(scales['s_h_prime'][layer][head]))
    if minimum_divisor <= 0:
        raise ValueError('tiny fixture hidden denominator is not positive')
    elu_high=max(elu_high,gen.round_ratio_away(numerator,minimum_divisor))
degrees={}
for start,end in zip(graph['edge_ptr'],graph['edge_ptr'][1:]):
    counts={}
    for dst in graph['dst'][start:end]:
        counts[dst]=counts.get(dst,0)+1
    degrees.update(counts)
max_degree=max(degrees.values(),default=0)
max_relu=max(2*lrelu_output,max_degree)
def table(key,low,high,expected_function):
    old=tables[key]
    if old['function_id'] != expected_function:
        raise ValueError(f"unexpected {key} table function {old['function_id']!r}")
    return gen.make_table(old['table_id'],old['function_id'],old['domain_id'],old['generator_id'],old['input_scale'],old['output_scale'],low,high)
rebuilt={
  'lrelu':table('lrelu',-score,score,'lrelu'),
  'exp':table('exp',0,exp_high,'exp_neg'),
  'elu':table('elu',-elu_high,elu_high,'elu'),
  'max_relu':table('max_relu',-max_relu,max_relu,'max_relu'),
  'static_tables_id':tables['static_tables_id'],
  'backend_aux_tables_id':tables['backend_aux_tables_id'],
}
rebuilt['static_tables_digest']=gen.static_tables_digest(rebuilt)
# The canonical resource generator represents rows by interval and function;
# CanonicalGat's Rust schema stores the same exact rows explicitly.
for key in gen.TABLE_ORDER:
    table=rebuilt[key]
    table['rows']=[[input_value,output_value] for input_value,output_value in gen.iter_table_rows(table)]
    del table['row_count']
print(json.dumps(rebuilt,separators=(',',':')))
"#;
        let table_input = serde_json::json!({
            "tables": &gat_spec.tables,
            "ledger": &gat_spec.ledger,
            "scales": &gat_spec.scales,
            "graph": &gat_spec.graph,
        });
        let python_for_tables = std::env::var_os("EZKL_REPAIR_TEST_PYTHON").unwrap_or_else(|| "python3".into());
        let mut child = Command::new(python_for_tables)
            .arg("-c")
            .arg(table_rebuild)
            .arg(&root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("Python is required for canonical tiny-fixture table regeneration");
        serde_json::to_writer(child.stdin.take().unwrap(), &table_input).unwrap();
        let generated = child.wait_with_output().unwrap();
        assert!(
            generated.status.success(),
            "canonical tiny-fixture table regeneration failed: {}",
            String::from_utf8_lossy(&generated.stderr),
        );
        gat_spec.tables = serde_json::from_slice::<crate::gat_adapter::schema::StaticTables>(&generated.stdout)
            .expect("production table generator returned malformed exact tables");
        assert_eq!(gat_spec.scales, original_scales, "table refresh must preserve fixed-point scales");
        for (label, before, after) in [
            ("lrelu", &original_tables.lrelu, &gat_spec.tables.lrelu),
            ("exp", &original_tables.exp, &gat_spec.tables.exp),
            ("elu", &original_tables.elu, &gat_spec.tables.elu),
            ("max_relu", &original_tables.max_relu, &gat_spec.tables.max_relu),
        ] {
            assert_eq!(before.table_id, after.table_id, "{} table id must remain fixed", label);
            assert_eq!(before.input_scale, after.input_scale, "{} table input scale must remain fixed", label);
            assert_eq!(before.output_scale, after.output_scale, "{} table output scale must remain fixed", label);
            assert_eq!(before.function_id, after.function_id, "{} function id must remain fixed", label);
            assert_eq!(before.domain_id, after.domain_id, "{} domain id must remain fixed", label);
            assert_eq!(before.generator_id, after.generator_id, "{} generator id must remain fixed", label);
        }
        assert_eq!(original_tables.static_tables_id, gat_spec.tables.static_tables_id);
        assert_eq!(original_tables.backend_aux_tables_id, gat_spec.tables.backend_aux_tables_id);
        assert_eq!(gat_spec.model_binding, original_model_binding, "table refresh must not change the private-model binding");
        assert_eq!(gat_spec.graph, original_graph, "table refresh must not change graph/topology");

        // This checked-in compile artifact predates full ledger/table closure:
        // its output families claim score=2048 but delta=32, and every family
        // claims table_output=32. Close only those stale under-bounds against
        // the unchanged score families and actual shared-table rows, using the
        // same common-table ledger semantics as the production generator.
        let table_output_floor = [
            &gat_spec.tables.lrelu,
            &gat_spec.tables.exp,
            &gat_spec.tables.elu,
            &gat_spec.tables.max_relu,
        ]
        .into_iter()
        .flat_map(|table| table.rows.iter().map(|(_, output)| output.checked_abs().expect("canonical table output absolute value")))
        .max()
        .expect("regenerated canonical tables contain rows");
        for family in &mut gat_spec.ledger.families {
            let required_delta = family.max_score_abs.checked_mul(2).expect("canonical GAT score-difference bound");
            family.max_delta = family.max_delta.max(required_delta);
            family.max_table_output_abs = family.max_table_output_abs.max(table_output_floor);
        }
        gat_spec.ledger.max_delta = gat_spec
            .ledger
            .families
            .iter()
            .map(|family| family.max_delta)
            .fold(gat_spec.ledger.max_delta, i128::max);
        gat_spec.ledger.max_table_output_abs = gat_spec.ledger.max_table_output_abs.max(table_output_floor);
        assert!(gat_spec.ledger.max_delta >= original_ledger.max_delta);
        assert!(gat_spec.ledger.max_table_output_abs >= original_ledger.max_table_output_abs);
        assert_eq!(gat_spec.ledger.max_score_abs, original_ledger.max_score_abs);
        assert_eq!(gat_spec.ledger.max_exp, original_ledger.max_exp);
        assert_eq!(gat_spec.ledger.max_sum, original_ledger.max_sum);
        assert_eq!(gat_spec.ledger.max_aggregation_num, original_ledger.max_aggregation_num);
        assert_eq!(gat_spec.ledger.max_dividend_abs, original_ledger.max_dividend_abs);
        assert_eq!(gat_spec.ledger.max_aux_abs, original_ledger.max_aux_abs);
        assert_eq!(gat_spec.ledger.max_divisor, original_ledger.max_divisor);
        assert_eq!(gat_spec.ledger.field_modulus, original_ledger.field_modulus);
        assert_eq!(gat_spec.ledger.ledger_id, original_ledger.ledger_id);
        assert_eq!(gat_spec.ledger.families.len(), original_ledger.families.len());
        for (before, after) in original_ledger.families.iter().zip(&gat_spec.ledger.families) {
            assert_eq!((before.layer, before.head, before.is_output), (after.layer, after.head, after.is_output));
            assert!(after.max_delta >= before.max_delta);
            assert!(after.max_table_output_abs >= before.max_table_output_abs);
            assert!(after.max_delta >= after.max_score_abs.checked_mul(2).expect("canonical GAT score-difference bound"));
            assert!(after.max_table_output_abs >= table_output_floor);
            assert_eq!(after.max_score_abs, before.max_score_abs);
            assert_eq!(after.max_weight_abs, before.max_weight_abs);
            assert_eq!(after.max_attention_abs, before.max_attention_abs);
            assert_eq!(after.max_bias_abs, before.max_bias_abs);
            assert_eq!(after.max_exp, before.max_exp);
            assert_eq!(after.max_sum, before.max_sum);
            assert_eq!(after.max_aggregation_num, before.max_aggregation_num);
            assert_eq!(after.max_dividend_abs, before.max_dividend_abs);
            assert_eq!(after.max_aux_abs, before.max_aux_abs);
            assert_eq!(after.max_divisor, before.max_divisor);
        }
        gat_spec.ledger.refresh_digest().unwrap();
        assert_eq!(gat_spec.ledger_identity.id, gat_spec.ledger.ledger_id);
        gat_spec.ledger_identity.digest = crate::gat_adapter::schema::sha256_digest(&gat_spec.ledger.canonical_bytes()).unwrap();
        eprintln!(
            "EZKL_TEST_FIXTURE_LEDGER_CLOSURE old_max_delta={} new_max_delta={} old_max_table_output_abs={} new_max_table_output_abs={} shared_table_output_floor={}",
            original_ledger.max_delta,
            gat_spec.ledger.max_delta,
            original_ledger.max_table_output_abs,
            gat_spec.ledger.max_table_output_abs,
            table_output_floor,
        );
        gat_spec.validate_for_compile().expect("canonical tiny fixture tables and widened ledger must close the full unchanged GAT relation and fixed scales");
        assert_eq!((gat_spec.tables.lrelu.input_min, gat_spec.tables.lrelu.input_max), (-2_048, 2_048));
        assert_eq!(gat_spec.tables.exp.input_max, gat_spec.ledger.max_delta);
        assert_eq!(gat_spec.ledger.max_delta, 4_096);
        assert_eq!(gat_spec.ledger.max_table_output_abs, 4_096);
        assert_eq!((gat_spec.tables.max_relu.input_min, gat_spec.tables.max_relu.input_max), (-4_096, 4_096));
        assert_eq!(gat_spec.tables.max_relu.rows.len(), 8_193);
        let max_table_rows = [
            gat_spec.tables.lrelu.rows.len(),
            gat_spec.tables.exp.rows.len(),
            gat_spec.tables.elu.rows.len(),
            gat_spec.tables.max_relu.rows.len(),
        ].into_iter().max().expect("tiny GAT fixture has registered tables");
        let table_rows = [
            ("lrelu", gat_spec.tables.lrelu.rows.len()),
            ("exp", gat_spec.tables.exp.rows.len()),
            ("elu", gat_spec.tables.elu.rows.len()),
            ("max_relu", gat_spec.tables.max_relu.rows.len()),
        ];
        let rows_with_blinding = max_table_rows.checked_add(crate::graph::RESERVED_BLINDING_ROWS).unwrap();
        assert!(max_table_rows <= (1usize << 14) - crate::graph::RESERVED_BLINDING_ROWS);
        let table_logrows = rows_with_blinding.checked_next_power_of_two().unwrap().trailing_zeros();
        let model = crate::gat_adapter::build_model_for_run_args(
            gat_spec.clone(),
            crate::gat_adapter::RoundingMode::NearestTiesAway,
            &run_args,
        ).unwrap();
        let module_logrows = crate::graph::GraphSettings::required_logrows_for_model(&model, &run_args).unwrap();
        let test_logrows = 13_u32.max(table_logrows).max(module_logrows);
        assert!(test_logrows <= 14, "tiny fixture needs k={} from table_rows_max={}, table_floor={}, module_floor={}; review capacity before any larger test SRS", test_logrows, max_table_rows, table_logrows, module_logrows);
        eprintln!(
            "EZKL_TEST_FIXTURE table_rows={:?} table_rows_max={} reserved_rows={} table_logrows={} module_logrows={} test_logrows={}",
            table_rows,
            max_table_rows,
            crate::graph::RESERVED_BLINDING_ROWS,
            table_logrows,
            module_logrows,
            test_logrows,
        );
        run_args.logrows = test_logrows;
        let params = ParamsKZG::<Bn256>::setup(test_logrows, OsRng);
        let zero_cap_result = create_keys_with_native_gate::<KZGCommitmentScheme<Bn256>, _, _, _>(
            &CopyCycleCircuit { mismatch: false },
            &params,
            false,
            |_, _| Ok(NativeKeygenBudget { max_selector_activation_bytes: 0, max_permutation_backing_bytes: 0 }),
            |_, _| Ok(()),
        );
        assert!(zero_cap_result.is_err(), "native keygen must reject a zero permutation allocation cap");
        // Match formal setup: derive complete GAT settings without invoking
        // GraphCircuit::new's additional full dummy layout.
        let settings = crate::gat_adapter::gat_settings_for_run_args(&gat_spec, &run_args).unwrap();
        let setup_circuit = GraphCircuit::new_gat_witnessless_from_settings(
            model.clone(),
            settings.clone(),
            run_args.check_mode,
        );
        let mut circuit = GraphCircuit::new_from_settings(model, settings, run_args.check_mode).unwrap();
        circuit.load_graph_witness(&witness).unwrap();
        circuit.ensure_complete_gat_witness().unwrap();
        assert!(circuit.settings().run_args.output_visibility.is_public());
        let public_inputs = circuit.prepare_public_inputs(&witness).unwrap();
        let output_count = witness.outputs.iter().map(Vec::len).sum::<usize>();
        assert!(output_count > 0 && output_count <= public_inputs.len());
        let mut native_stages = Vec::new();
        let (compact_pk, census) = create_keys_with_native_gate::<KZGCommitmentScheme<Bn256>, _, _, _>(
            &setup_circuit,
            &params,
            false,
            |_, preflight| {
                assert_eq!(preflight.logrows, test_logrows);
                assert_eq!(preflight.domain_size, 1 << test_logrows);
                Ok(NativeKeygenBudget { max_selector_activation_bytes: 256 * 1024 * 1024, max_permutation_backing_bytes: 256 * 1024 * 1024 })
            },
            |stage, native| {
                assert_eq!(native.preflight.logrows, test_logrows);
                native_stages.push((stage.to_owned(), native.clone()));
                Ok(())
            },
        ).unwrap();
        assert_eq!(native_stages.len(), 2);
        assert!(census.permutation_copy_calls > 0);
        assert!(census.selector_activation_storage_peak_bytes > 0);
        let compact_bytes = compact_pk.to_bytes(SerdeFormat::RawBytes);
        let pk_path = temp.path().join("tiny-gat.pk");
        save_pk(&pk_path, &compact_pk).unwrap();
        assert!(std::fs::metadata(&pk_path).unwrap().len() > 0);
        drop(compact_pk);
        assert!(load_pk_with_native_backing_budget::<KZGCommitmentScheme<Bn256>, GraphCircuit>(pk_path.clone(), circuit.params(), 0).is_err());
        let (restored_pk, timing) = load_pk_with_native_backing_budget_and_timing::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
            pk_path,
            circuit.params(),
            64 * 1024 * 1024,
        ).unwrap();
        assert!(timing.artifact_read_validate.as_secs_f64() > 0.0);
        assert!(timing.sigma_rebuild.as_secs_f64() > 0.0);
        assert_eq!(compact_bytes, restored_pk.to_bytes(SerdeFormat::RawBytes));
        let instance_columns: &[&[Fr]] = &[&public_inputs];
        let instances: &[&[&[Fr]]] = &[instance_columns];
        let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(vec![]);
        halo2_proofs::plonk::create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(
            &params,
            &restored_pk,
            &[circuit],
            instances,
            OsRng,
            &mut transcript,
        ).unwrap();
        let proof = transcript.finalize();
        assert!(!proof.is_empty());
        let verify = |statement: &[Fr]| {
            let columns: &[&[Fr]] = &[statement];
            let verifier_instances: &[&[&[Fr]]] = &[columns];
            let verifier_params = params.verifier_params();
            let mut reader = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(proof.as_slice());
            halo2_proofs::plonk::verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, SingleStrategy<_>>(
                verifier_params,
                restored_pk.get_vk(),
                SingleStrategy::new(verifier_params),
                verifier_instances,
                &mut reader,
                params.n(),
            )
        };
        assert!(verify(&public_inputs).is_ok(), "independent native verifier rejected the GAT public statement");
        let mut tampered_public = public_inputs.clone();
        let tampered_logit_index = tampered_public.len() - output_count;
        tampered_public[tampered_logit_index] += Fr::from(1u64);
        assert!(verify(&tampered_public).is_err(), "native verifier accepted an altered public logit");
        let python = r#"import ast,json,sys
from pathlib import Path
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'scripts'))
from formal_common import ContractError,validate_native_pk_read_timing
for relative in ('scripts/run_hybrid_formal.py','scripts/run_single_dataset.py','scripts/formal_common.py','scripts/collect_results.py','scripts/finalization_capacity.py'):
    ast.parse((root/relative).read_text(),filename=relative)
digest='ab'*32
pk='cd'*32
session='00112233445566778899aabbccddeeff'
timing={'schema':'ezkl.native-pk-read-timing.v1','operation':'authenticated_compact_native_pk_restore','session_id':session,'dataset':'tiny_gat_test','native_prove_budget_sha256':digest,'admission_sha256':digest,'resource_manifest_sha256':digest,'setup_receipt_sha256':digest,'native_keygen_budget_sha256':digest,'native_keygen_receipt_sha256':digest,'build_receipt_sha256':digest,'build_environment_sha256':digest,'binary_sha256':digest,'settings_sha256':digest,'compiled_circuit_sha256':digest,'vk_sha256':digest,'pk_sha256':pk,'measurement_scope':'artifact_read_decode_digest_and_mapping_validation_excluding_sigma_rebuild; sigma_wall_elapsed_may_include_mapped_or_spill_io','artifact_read_validate_ms':1.25,'sigma_rebuild_wall_ms':2.5}
path=Path(sys.argv[2])/'native_pk_read_timing.json'
path.write_text(json.dumps(timing))
validate_native_pk_read_timing(path,session_id=session,dataset='tiny_gat_test',native_prove_budget_sha256=digest,pk_sha256=pk)
timing['pk_sha256']='ef'*32
path.write_text(json.dumps(timing))
try:
    validate_native_pk_read_timing(path,session_id=session,dataset='tiny_gat_test',native_prove_budget_sha256=digest,pk_sha256=pk)
except ContractError:
    pass
else:
    raise AssertionError('native PK timing validator accepted a mismatched artifact identity')
"#;
        let python_executable = std::env::var_os("EZKL_REPAIR_TEST_PYTHON").unwrap_or_else(|| "python3".into());
        let status = Command::new(python_executable).arg("-c").arg(python).arg(&root).arg(temp.path()).status().expect("Python is required for the formal source/receipt regression");
        assert!(status.success(), "formal Python source/receipt regression failed");
    }

    #[test]
    fn test_snark_serialization_roundtrip() {
        let snark = Snark::<Fr, G1Affine> {
            proof: vec![1, 2, 3, 4, 5, 6, 7, 8],
            instances: vec![vec![Fr::from(1)], vec![Fr::from(2)]],
            protocol: None,
            hex_proof: None,
            split: None,
            pretty_public_inputs: None,
            timestamp: None,
            proof_timing: None,
            version: None,
        };

        snark
            .save(&"test_snark_serialization_roundtrip.json".into())
            .unwrap();
        let snark2 = Snark::<Fr, G1Affine>::load::<KZGCommitmentScheme<Bn256>>(
            &"test_snark_serialization_roundtrip.json".into(),
        )
        .unwrap();
        assert_eq!(snark.instances, snark2.instances);
        assert_eq!(snark.proof, snark2.proof);
    }
}
