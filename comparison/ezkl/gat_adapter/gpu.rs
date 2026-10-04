//! Minimal access to the pinned Halo2 native ICICLE backend.
use super::{GpuEvidence, GpuProvenance};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use crate::EZKLError;

const FRONTEND_COMMIT: &str = "5dfe006a0f1bc62ea82ca297709bbf3d22a2ca25";
const HALO2_COMMIT: &str = "f441c920be45f8f05d2c06a173d82e8885a5ed4d";

fn native_source_digest(directory: &Path) -> Result<String, EZKLError> {
    fn collect(directory: &Path, files: &mut Vec<PathBuf>) -> Result<(), EZKLError> {
        for entry in std::fs::read_dir(directory)? {
            let path = entry?.path();
            if path.is_symlink() { return Err("native ICICLE source must not contain symlinks".into()); }
            if path.is_dir() { collect(&path, files)?; } else if path.is_file() { files.push(path); }
        }
        Ok(())
    }
    let mut files = Vec::new(); collect(directory, &mut files)?; files.sort();
    let mut bytes = Vec::new();
    for path in files { let relative = path.strip_prefix(directory).map_err(|error| EZKLError::from(error.to_string()))?.to_string_lossy().replace('\\', "/"); bytes.extend_from_slice(&(relative.len() as u64).to_be_bytes()); bytes.extend_from_slice(relative.as_bytes()); bytes.extend_from_slice(crate::execute::get_file_hash(&path)?.as_bytes()); }
    super::schema::sha256_digest(&bytes).map_err(EZKLError::from)
}

fn manifest() -> Result<(GpuProvenance, Value), EZKLError> {
    let root = std::env::current_dir()?;
    let directory = root.join("artifacts/build/icicle-cuda");
    let path = directory.join("native_backend_manifest.json");
    let digest = crate::execute::get_file_hash(&path)?;
    if option_env!("ICICLE_NATIVE_MANIFEST_SHA256") != Some(digest.as_str()) { return Err("ICICLE native backend manifest does not match this GPU binary".into()); }
    let value: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    if value["frontend_commit"] != FRONTEND_COMMIT || value["halo2_commit"] != HALO2_COMMIT || value["abi"] != "icicle-2.8.0-bn254-static-cuda" { return Err("ICICLE backend source/ABI identity mismatch".into()); }
    if value["native_source_sha256"].as_str() != Some(native_source_digest(&root.join("vendor/icicle/icicle"))?.as_str()) { return Err("ICICLE native source does not match its built libraries".into()); }
    let libraries = value["libraries"].as_object().ok_or("ICICLE native library identities are missing")?;
    if libraries.len() != 2 { return Err("ICICLE native library identities must contain both field and curve libraries".into()); }
    for name in ["libingo_field_bn254.a", "libingo_curve_bn254.a"] {
        let relative = format!("artifacts/build/icicle-cuda/lib/{}", name);
        let path = root.join(&relative);
        if path.is_symlink() || !path.is_file() || libraries.get(&relative).and_then(Value::as_str) != Some(crate::execute::get_file_hash(&path)?.as_str()) { return Err(format!("ICICLE native library digest mismatch: {}", name).into()); }
    }
    let text = |key: &str| value[key].as_str().map(str::to_string).filter(|x| !x.is_empty()).ok_or_else(|| format!("ICICLE backend manifest missing {}", key));
    let provenance = GpuProvenance { frontend_commit: FRONTEND_COMMIT.into(), abi: text("abi")?, device: text("device")?, architecture: text("architecture")?, runtime: text("runtime")?, manifest_sha256: digest };
    Ok((provenance, value))
}

/// Check the exact native static libraries compiled into this binary.
pub fn validate_backend() -> Result<(), EZKLError> { manifest().map(|_| ()) }

/// Select the native CUDA device, failing before expensive formal computation.
pub fn initialize() -> Result<(), EZKLError> {
    validate_backend()?;
    let device: usize = std::env::var("ICICLE_DEVICE_ID").unwrap_or_else(|_| "0".into()).parse().map_err(|_| "ICICLE_DEVICE_ID must be an unsigned local device index")?;
    halo2_proofs::icicle::initialize_device(device).map_err(EZKLError::from)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProofRecord { proof_sha256: String, msm_calls: u64, ntt_calls: u64, synchronized_calls: u64, device_id: usize, evidence: GpuEvidence }

fn invocation_digest(record: &ProofRecord) -> Result<String, EZKLError> {
    super::schema::sha256_digest(&serde_json::to_vec(&(record.proof_sha256.as_str(), record.msm_calls, record.ntt_calls, record.synchronized_calls, record.device_id, &record.evidence.provenance))?).map_err(EZKLError::from)
}

/// Persist successful native GPU operation counters after the proof is saved.
pub fn write_proof_record(directory: &Path, proof_path: &Path) -> Result<(), EZKLError> {
    let (msm_calls, ntt_calls, synchronized_calls, device_id) = halo2_proofs::icicle::execution_stats();
    if msm_calls == 0 || ntt_calls == 0 || synchronized_calls != msm_calls + ntt_calls { return Err("native GPU proof did not complete synchronized MSM and NTT operations".into()); }
    let (provenance, _) = manifest()?;
    let evidence = GpuEvidence { backend: "CUDA".into(), gpu_backend: "icicle-cuda".into(), backend_loaded: true, cpu_fallback: false, device_synchronized: true, gpu_kernel_invocation_count: msm_calls + ntt_calls, provenance: Some(provenance), invocation_trace_sha256: String::new() };
    let mut record = ProofRecord { proof_sha256: crate::execute::get_file_hash(proof_path)?, msm_calls, ntt_calls, synchronized_calls, device_id, evidence };
    record.evidence.invocation_trace_sha256 = invocation_digest(&record)?;
    std::fs::write(directory.join("gpu_prover_record.json"), serde_json::to_vec_pretty(&record)?)?;
    Ok(())
}

/// Independent CPU verification retains proof-bound native prover evidence.
pub fn read_proof_evidence(directory: &Path, proof_path: &Path) -> Result<GpuEvidence, EZKLError> {
    if std::env::var_os("ENABLE_ICICLE_GPU").is_some() { return Err("independent formal verifier must execute on CPU".into()); }
    let path: PathBuf = directory.join("gpu_prover_record.json");
    if path.is_symlink() { return Err("GPU prover record must not be a symlink".into()); }
    let record: ProofRecord = serde_json::from_slice(&std::fs::read(&path)?)?;
    let (provenance, _) = manifest()?;
    let count = record.msm_calls + record.ntt_calls;
    if record.proof_sha256 != crate::execute::get_file_hash(proof_path)? || record.msm_calls == 0 || record.ntt_calls == 0 || record.synchronized_calls != count || record.evidence.gpu_kernel_invocation_count != count || record.evidence.provenance.as_ref() != Some(&provenance) || !record.evidence.backend_loaded || !record.evidence.device_synchronized || record.evidence.cpu_fallback || record.evidence.backend != "CUDA" || record.evidence.gpu_backend != "icicle-cuda" || record.evidence.invocation_trace_sha256 != invocation_digest(&record)? { return Err("GPU native prover execution evidence is inconsistent with the verified proof".into()); }
    Ok(record.evidence)
}
