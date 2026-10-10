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
    let provenance = GpuProvenance { frontend_commit: FRONTEND_COMMIT.into(), native_execution_policy: halo2_proofs::icicle::FORMAL_GPU_EXECUTION_POLICY.into(), abi: text("abi")?, device: text("device")?, architecture: text("architecture")?, runtime: text("runtime")?, manifest_sha256: digest };
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
struct ProofRecord { proof_sha256: String, native_execution_policy: String, msm_operations: u64, gpu_msm_chunk_calls: u64, ntt_calls: u64, cpu_fft_calls: u64, synchronized_calls: u64, device_id: usize, evidence: GpuEvidence }

fn invocation_digest(record: &ProofRecord) -> Result<String, EZKLError> {
    super::schema::sha256_digest(&serde_json::to_vec(&(record.proof_sha256.as_str(), record.native_execution_policy.as_str(), record.msm_operations, record.gpu_msm_chunk_calls, record.ntt_calls, record.cpu_fft_calls, record.synchronized_calls, record.device_id, &record.evidence.provenance))?).map_err(EZKLError::from)
}

fn validate_record(record: &ProofRecord) -> Result<u64, EZKLError> {
    let expected_policy = halo2_proofs::icicle::FORMAL_GPU_EXECUTION_POLICY;
    let invocations = record.gpu_msm_chunk_calls.checked_add(record.ntt_calls).ok_or("native GPU invocation counter overflow")?;
    if record.native_execution_policy != expected_policy
        || record.msm_operations == 0
        || record.gpu_msm_chunk_calls < record.msm_operations
        || record.ntt_calls != 0
        || record.cpu_fft_calls == 0
        || record.synchronized_calls != invocations
        || record.evidence.gpu_kernel_invocation_count != invocations
        || record.evidence.provenance.as_ref().map(|p| p.native_execution_policy.as_str()) != Some(expected_policy)
    {
        return Err("native GPU proof counters do not match the fixed CPU-FFT/GPU-MSM execution policy".into());
    }
    Ok(invocations)
}

/// Persist successful native GPU operation counters after the proof is saved.
pub fn write_proof_record(directory: &Path, proof_path: &Path) -> Result<(), EZKLError> {
    let (msm_operations, gpu_msm_chunk_calls, ntt_calls, cpu_fft_calls, synchronized_calls, device_id) = halo2_proofs::icicle::execution_stats();
    write_proof_record_with_stats(
        directory,
        proof_path,
        (msm_operations, gpu_msm_chunk_calls, ntt_calls, cpu_fft_calls, synchronized_calls, device_id),
    )
}

fn write_proof_record_with_stats(
    directory: &Path,
    proof_path: &Path,
    (msm_operations, gpu_msm_chunk_calls, ntt_calls, cpu_fft_calls, synchronized_calls, device_id): (u64, u64, u64, u64, u64, usize),
) -> Result<(), EZKLError> {
    let (provenance, _) = manifest()?;
    let policy = halo2_proofs::icicle::FORMAL_GPU_EXECUTION_POLICY.to_owned();
    let invocations = gpu_msm_chunk_calls.checked_add(ntt_calls).ok_or("native GPU invocation counter overflow")?;
    let evidence = GpuEvidence { backend: "CUDA".into(), gpu_backend: "icicle-cuda".into(), backend_loaded: true, cpu_fallback: false, device_synchronized: true, gpu_kernel_invocation_count: invocations, provenance: Some(provenance), invocation_trace_sha256: String::new() };
    let mut record = ProofRecord { proof_sha256: crate::execute::get_file_hash(proof_path)?, native_execution_policy: policy, msm_operations, gpu_msm_chunk_calls, ntt_calls, cpu_fft_calls, synchronized_calls, device_id, evidence };
    validate_record(&record)?;
    record.evidence.invocation_trace_sha256 = invocation_digest(&record)?;
    std::fs::write(directory.join("gpu_prover_record.json"), serde_json::to_vec_pretty(&record)?)?;
    // This sidecar describes the operation that produced the proof. The
    // independent verifier validates it against the immutable proof record,
    // but never replaces it with verifier-runtime evidence.
    std::fs::write(directory.join("gpu_evidence.json"), serde_json::to_vec_pretty(&record.evidence)?)?;
    Ok(())
}

/// Independent CPU verification retains proof-bound native prover evidence.
pub fn read_proof_evidence(directory: &Path, proof_path: &Path) -> Result<GpuEvidence, EZKLError> {
    if std::env::var_os("ENABLE_ICICLE_GPU").is_some() { return Err("independent formal verifier must execute on CPU".into()); }
    let path: PathBuf = directory.join("gpu_prover_record.json");
    if path.is_symlink() { return Err("GPU prover record must not be a symlink".into()); }
    let record: ProofRecord = serde_json::from_slice(&std::fs::read(&path)?)?;
    let (provenance, _) = manifest()?;
    let count = validate_record(&record)?;
    if record.proof_sha256 != crate::execute::get_file_hash(proof_path)? || record.evidence.provenance.as_ref() != Some(&provenance) || !record.evidence.backend_loaded || !record.evidence.device_synchronized || record.evidence.cpu_fallback || record.evidence.backend != "CUDA" || record.evidence.gpu_backend != "icicle-cuda" || record.evidence.invocation_trace_sha256 != invocation_digest(&record)? || count != record.gpu_msm_chunk_calls { return Err("GPU native prover execution evidence is inconsistent with the verified proof".into()); }
    let evidence_path = directory.join("gpu_evidence.json");
    if evidence_path.is_symlink() { return Err("GPU prover evidence must not be a symlink".into()); }
    let evidence: GpuEvidence = serde_json::from_slice(&std::fs::read(&evidence_path)?)?;
    if evidence != record.evidence { return Err("GPU prover evidence sidecar differs from the proof-bound native record".into()); }
    Ok(record.evidence)
}

#[cfg(test)]
mod proof_evidence_tests {
    use super::{read_proof_evidence, write_proof_record_with_stats, ProofRecord};
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_directory() -> PathBuf {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let directory = std::env::temp_dir().join(format!("ezkl-gpu-proof-evidence-{}-{nonce}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        directory
    }

    #[test]
    fn prove_writes_record_bound_sidecar_and_verify_is_read_only_fail_closed() {
        assert!(std::env::var_os("ENABLE_ICICLE_GPU").is_none(), "independent verifier test requires the CPU-only guard environment");
        let directory = temporary_directory();
        let proof_path = directory.join("proof.json");
        std::fs::write(&proof_path, b"synthetic proof bytes for evidence binding").unwrap();

        // Exercise the same production writer as Prove, with a small valid
        // native counter sample so the regression never initializes a GPU.
        write_proof_record_with_stats(&directory, &proof_path, (1, 1, 0, 1, 1, 0)).unwrap();
        let record_path = directory.join("gpu_prover_record.json");
        let evidence_path = directory.join("gpu_evidence.json");
        let record_bytes = std::fs::read(&record_path).unwrap();
        let evidence_bytes = std::fs::read(&evidence_path).unwrap();
        let record: ProofRecord = serde_json::from_slice(&record_bytes).unwrap();
        let evidence = serde_json::from_slice::<super::GpuEvidence>(&evidence_bytes).unwrap();
        assert_eq!(record.proof_sha256, crate::execute::get_file_hash(&proof_path).unwrap());
        assert_eq!(record.evidence, evidence);

        let verified_evidence = read_proof_evidence(&directory, &proof_path).unwrap();
        assert_eq!(verified_evidence, evidence);
        assert_eq!(std::fs::read(&record_path).unwrap(), record_bytes);
        assert_eq!(std::fs::read(&evidence_path).unwrap(), evidence_bytes);

        let mut tampered = evidence;
        tampered.gpu_kernel_invocation_count += 1;
        std::fs::write(&evidence_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(read_proof_evidence(&directory, &proof_path).is_err());
        assert_eq!(std::fs::read(&record_path).unwrap(), record_bytes);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
