//! Minimal access to the pinned Halo2 native ICICLE backend.
use super::{GpuEvidence, GpuProvenance};
use serde::ser::SerializeSeq;
use serde::{Deserialize, Serialize, Serializer};
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
struct ProofRecord {
    schema_version: u32,
    proof_sha256: String,
    admission_sha256: String,
    native_execution_policy: String,
    icicle_msm_max_concurrency: usize,
    icicle_msm_precompute_factor: usize,
    gpu_msm_precompute_calls: u64,
    gpu_msm_precomputed_chunk_calls: u64,
    gpu_msm_precomputed_cache_hits: u64,
    icicle_msm_precomputed_basis_device_upper_bytes: usize,
    icicle_msm_precomputed_basis_device_bytes: usize,
    icicle_msm_precomputed_basis_device_peak_bytes: usize,
    msm_operations: u64,
    gpu_msm_chunk_calls: u64,
    ntt_calls: u64,
    cpu_fft_calls: u64,
    synchronized_calls: u64,
    device_id: usize,
    gpu_ntt_elements: u64,
    ntt_max_size: usize,
    ntt_workspace_capacity_bytes: usize,
    ntt_domain_capacity_bytes: usize,
    ntt_host_staging_capacity_bytes: usize,
    ntt_device_peak_upper_bound_bytes: usize,
    gpu_h_eval_groups: u64,
    gpu_h_eval_calls: u64,
    gpu_h_eval_rows: u64,
    gpu_h_eval_fallbacks: u64,
    gpu_h_eval_chunks: u64,
    gpu_h_eval_syncs: u64,
    gpu_h_workspace_high_water_bytes: usize,
    gpu_h_local_memory_peak_upper_bytes: usize,
    evidence: GpuEvidence,
}

#[derive(Clone)]
struct NttProfile {
    admission_sha256: String,
    capacity_elements: u64,
    device_workspace_upper_bytes: u64,
    domain_cache_upper_bytes: u64,
    device_peak_upper_bytes: u64,
    host_staging_upper_bytes: u64,
    precomputed_basis_device_upper_bytes: u64,
    h_kernel_local_memory_upper_bytes: u64,
}

fn read_ntt_profile(directory: &Path) -> Result<NttProfile, EZKLError> {
    let admission_path = directory.join("admission.json");
    let budget_path = directory.join("native_prove_budget.json");
    if admission_path.is_symlink() || budget_path.is_symlink() {
        return Err("native GPU NTT profile inputs must not be symlinks".into());
    }
    let admission_bytes = std::fs::read(&admission_path)?;
    let admission_sha256 = crate::execute::get_file_hash(&admission_path)?;
    let admission: Value = serde_json::from_slice(&admission_bytes)?;
    let memory = admission.get("native_memory_model").ok_or("native GPU NTT profile has no Admission memory model")?;
    let read_u64 = |document: &Value, key: &str| -> Result<u64, EZKLError> {
        document.get(key).and_then(Value::as_u64).ok_or_else(|| format!("native GPU NTT profile lacks valid {key}").into())
    };
    if memory.get("schema").and_then(Value::as_str) != Some("ezkl-halo2-static-envelope-v20") {
        return Err("native GPU NTT profile requires the current Admission memory model".into());
    }
    let base_domain_size = read_u64(&admission, "domain_size")?;
    let proof_domain_size = read_u64(memory, "proof_domain_size")?;
    let capacity_elements = proof_domain_size.max(base_domain_size).min(1u64 << 25);
    let device_workspace_upper_bytes = capacity_elements.checked_mul(64).ok_or("native GPU NTT workspace bound overflow")?;
    let domain_cache_upper_bytes = capacity_elements
        .checked_add(1 + 64 + 6)
        .and_then(|elements| elements.checked_mul(32))
        .ok_or("native GPU NTT domain-cache bound overflow")?;
    let init_scratch_upper_bytes = 4 * 64 * 32;
    let device_peak_upper_bytes = device_workspace_upper_bytes
        .checked_add(domain_cache_upper_bytes)
        .and_then(|bytes| bytes.checked_add(init_scratch_upper_bytes))
        .ok_or("native GPU NTT device peak bound overflow")?;
    let host_staging_upper_bytes = capacity_elements.checked_mul(32).ok_or("native GPU NTT host staging bound overflow")?;
    let precomputed_basis_device_upper_bytes = base_domain_size
        .checked_mul(64 * 4)
        .ok_or("native resident MSM basis bound overflow")?;
    let point_staging_host_upper_bytes = base_domain_size.min(1 << 20)
        .checked_mul(64)
        .ok_or("native MSM point staging bound overflow")?;
    if read_u64(memory, "icicle_ntt_min_log_size")? != 16
        || read_u64(memory, "icicle_ntt_max_log_size")? != 25
        || read_u64(memory, "icicle_ntt_capacity_elements")? != capacity_elements
        || read_u64(memory, "icicle_ntt_device_workspace_upper_bytes")? != device_workspace_upper_bytes
        || read_u64(memory, "icicle_ntt_domain_cache_upper_bytes")? != domain_cache_upper_bytes
        || read_u64(memory, "icicle_ntt_init_scratch_upper_bytes")? != init_scratch_upper_bytes
        || read_u64(memory, "icicle_ntt_device_peak_upper_bytes")? != device_peak_upper_bytes
        || read_u64(memory, "icicle_ntt_host_staging_upper_bytes")? != host_staging_upper_bytes
        || read_u64(memory, "icicle_msm_max_concurrency")? != 1
        || read_u64(memory, "icicle_msm_precompute_factor")? != 4
        || read_u64(memory, "icicle_msm_precomputed_basis_count")? != 1
        || read_u64(memory, "icicle_msm_precomputed_basis_device_upper_bytes")? != precomputed_basis_device_upper_bytes
        || read_u64(memory, "icicle_msm_point_staging_host_upper_bytes")? != point_staging_host_upper_bytes
    {
        return Err("native GPU NTT Admission capacity does not match the fixed profile formulas".into());
    }
    let budget_bytes = std::fs::read(&budget_path)?;
    let budget: Value = serde_json::from_slice(&budget_bytes)?;
    if budget.get("schema").and_then(Value::as_str) != Some("ezkl.native-prove-budget.v6")
        || budget.get("admission_sha256").and_then(Value::as_str) != Some(admission_sha256.as_str())
        || budget.get("native_execution_policy").and_then(Value::as_str) != Some(halo2_proofs::icicle::FORMAL_GPU_EXECUTION_POLICY)
        || budget.get("h_workspace_borrows_ntt").and_then(Value::as_bool) != Some(true)
        || read_u64(&budget, "icicle_msm_max_concurrency")? != 1
        || read_u64(&budget, "icicle_ntt_min_log_size")? != 16
        || read_u64(&budget, "icicle_ntt_max_log_size")? != 25
        || read_u64(&budget, "icicle_ntt_capacity_elements")? != capacity_elements
        || read_u64(&budget, "icicle_ntt_device_workspace_upper_bytes")? != device_workspace_upper_bytes
        || read_u64(&budget, "icicle_ntt_domain_cache_upper_bytes")? != domain_cache_upper_bytes
        || read_u64(&budget, "icicle_ntt_init_scratch_upper_bytes")? != init_scratch_upper_bytes
        || read_u64(&budget, "icicle_ntt_device_peak_upper_bytes")? != device_peak_upper_bytes
        || read_u64(&budget, "icicle_ntt_host_staging_upper_bytes")? != host_staging_upper_bytes
        || read_u64(&budget, "icicle_msm_precompute_factor")? != 4
        || read_u64(&budget, "icicle_msm_precomputed_basis_count")? != 1
        || read_u64(&budget, "icicle_msm_precomputed_basis_device_upper_bytes")? != precomputed_basis_device_upper_bytes
        || read_u64(&budget, "icicle_msm_point_staging_host_upper_bytes")? != point_staging_host_upper_bytes
        || budget.get("icicle_msm_workspace_query_schema").and_then(Value::as_str)
            != Some("ezkl.icicle.cub_msm_workspace_query.v3")
    {
        return Err("native prove budget does not bind the current formal GPU NTT Admission profile".into());
    }
    let ordinary_workspace_peak = read_u64(&budget, "icicle_msm_ordinary_factor1_device_workspace_peak_bytes")?;
    let prepared_workspace_peak = read_u64(&budget, "icicle_msm_prepared_factor4_device_workspace_peak_bytes")?;
    let h_kernel_local_memory_upper_bytes = read_u64(&budget, "h_kernel_local_memory_upper_bytes")?;
    let h_program_metadata_upper_bytes = read_u64(&budget, "h_program_metadata_upper_bytes")?;
    let cuda_required_workspace_bytes = read_u64(&budget, "cuda_required_workspace_bytes")?;
    let expected_cuda_required_workspace_bytes = ordinary_workspace_peak
        .max(prepared_workspace_peak)
        .checked_add(device_peak_upper_bytes)
        .and_then(|bytes| bytes.checked_add(precomputed_basis_device_upper_bytes))
        .and_then(|bytes| bytes.checked_add(halo2_proofs::icicle::FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES as u64))
        .ok_or("native fused H CUDA requirement overflow")?;
    let integrated_device = budget.get("integrated_device").and_then(Value::as_bool)
        .ok_or("native prove budget lacks integrated-device memory identity")?;
    let host_required = read_u64(&budget, "host_required_known_bytes")?;
    let host_available = read_u64(&budget, "host_available_bytes")?;
    if ordinary_workspace_peak == 0
        || prepared_workspace_peak == 0
        || read_u64(&budget, "device_workspace_peak_bytes")? != ordinary_workspace_peak.max(prepared_workspace_peak)
        || h_kernel_local_memory_upper_bytes != halo2_proofs::icicle::FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES as u64
        || h_program_metadata_upper_bytes != halo2_proofs::icicle::FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES as u64
        || cuda_required_workspace_bytes != expected_cuda_required_workspace_bytes
        || host_required < h_program_metadata_upper_bytes
        || host_required < h_program_metadata_upper_bytes.saturating_add(if integrated_device { expected_cuda_required_workspace_bytes } else { 0 })
        || host_required > host_available
    {
        return Err("native prove budget has inconsistent two-path CUB workspace peaks".into());
    }
    Ok(NttProfile {
        admission_sha256,
        capacity_elements,
        device_workspace_upper_bytes,
        domain_cache_upper_bytes,
        device_peak_upper_bytes,
        host_staging_upper_bytes,
        precomputed_basis_device_upper_bytes,
        h_kernel_local_memory_upper_bytes,
    })
}

fn invocation_digest(record: &ProofRecord) -> Result<String, EZKLError> {
    let mut encoded = Vec::new();
    {
        let mut serializer = serde_json::Serializer::new(&mut encoded);
        let mut sequence = serializer.serialize_seq(Some(33))?;
        sequence.serialize_element(&record.schema_version)?;
        sequence.serialize_element(&record.proof_sha256.as_str())?;
        sequence.serialize_element(&record.admission_sha256.as_str())?;
        sequence.serialize_element(&record.native_execution_policy.as_str())?;
        sequence.serialize_element(&record.icicle_msm_max_concurrency)?;
        sequence.serialize_element(&record.msm_operations)?;
        sequence.serialize_element(&record.gpu_msm_chunk_calls)?;
        sequence.serialize_element(&record.icicle_msm_precompute_factor)?;
        sequence.serialize_element(&record.gpu_msm_precompute_calls)?;
        sequence.serialize_element(&record.gpu_msm_precomputed_chunk_calls)?;
        sequence.serialize_element(&record.gpu_msm_precomputed_cache_hits)?;
        sequence.serialize_element(&record.icicle_msm_precomputed_basis_device_upper_bytes)?;
        sequence.serialize_element(&record.icicle_msm_precomputed_basis_device_bytes)?;
        sequence.serialize_element(&record.icicle_msm_precomputed_basis_device_peak_bytes)?;
        sequence.serialize_element(&record.ntt_calls)?;
        sequence.serialize_element(&record.cpu_fft_calls)?;
        sequence.serialize_element(&record.synchronized_calls)?;
        sequence.serialize_element(&record.device_id)?;
        sequence.serialize_element(&record.gpu_ntt_elements)?;
        sequence.serialize_element(&record.ntt_max_size)?;
        sequence.serialize_element(&record.ntt_workspace_capacity_bytes)?;
        sequence.serialize_element(&record.ntt_domain_capacity_bytes)?;
        sequence.serialize_element(&record.ntt_host_staging_capacity_bytes)?;
        sequence.serialize_element(&record.ntt_device_peak_upper_bound_bytes)?;
        sequence.serialize_element(&record.gpu_h_eval_groups)?;
        sequence.serialize_element(&record.gpu_h_eval_calls)?;
        sequence.serialize_element(&record.gpu_h_eval_rows)?;
        sequence.serialize_element(&record.gpu_h_eval_fallbacks)?;
        sequence.serialize_element(&record.gpu_h_eval_chunks)?;
        sequence.serialize_element(&record.gpu_h_eval_syncs)?;
        sequence.serialize_element(&record.gpu_h_workspace_high_water_bytes)?;
        sequence.serialize_element(&record.gpu_h_local_memory_peak_upper_bytes)?;
        sequence.serialize_element(&record.evidence.provenance)?;
        sequence.end()?;
    }
    super::schema::sha256_digest(&encoded).map_err(EZKLError::from)
}

fn validate_record(record: &ProofRecord, ntt_profile: &NttProfile) -> Result<u64, EZKLError> {
    let expected_policy = halo2_proofs::icicle::FORMAL_GPU_EXECUTION_POLICY;
    let invocations = record.gpu_msm_chunk_calls
        .checked_add(record.gpu_msm_precompute_calls)
        .and_then(|value| value.checked_add(record.ntt_calls))
        .and_then(|value| value.checked_add(record.gpu_h_eval_chunks))
        .ok_or("native GPU invocation counter overflow")?;
    let synchronized_invocations = record.gpu_msm_chunk_calls
        .checked_add(record.gpu_msm_precompute_calls)
        .and_then(|value| value.checked_add(record.ntt_calls))
        .and_then(|value| value.checked_add(record.gpu_h_eval_syncs))
        .ok_or("native GPU synchronization counter overflow")?;
    let h_group_outcomes = record.gpu_h_eval_calls
        .checked_add(record.gpu_h_eval_fallbacks)
        .ok_or("native fused-H group counter overflow")?;
    let max_size = u64::try_from(record.ntt_max_size).map_err(|_| "native NTT maximum size exceeds u64")?;
    let expected_workspace = max_size.checked_mul(64).ok_or("native NTT workspace capacity overflow")?;
    let expected_domain = if max_size == 0 {
        0
    } else {
        max_size
            .checked_add(1 + 64 + 6)
            .and_then(|elements| elements.checked_mul(32))
            .ok_or("native NTT domain capacity overflow")?
    };
    let expected_host = max_size.checked_mul(32).ok_or("native NTT host staging capacity overflow")?;
    let expected_device_peak = if max_size == 0 {
        0
    } else {
        max_size.checked_mul(96).and_then(|bytes| bytes.checked_add(10464)).ok_or("native NTT device peak overflow")?
    };
    let minimum_ntt_elements = 1u64
        .checked_shl(16)
        .and_then(|minimum| minimum.checked_mul(record.ntt_calls))
        .unwrap_or(u64::MAX);
    let maximum_ntt_elements = max_size
        .checked_mul(record.ntt_calls)
        .unwrap_or(u64::MAX);
    let planned_ntt_below_minimum = ntt_profile.capacity_elements < (1u64 << 16);
    if record.native_execution_policy != expected_policy
        || record.schema_version != 4
        || record.admission_sha256 != ntt_profile.admission_sha256
        || record.icicle_msm_max_concurrency != 1
        || record.icicle_msm_max_concurrency != halo2_proofs::icicle::FORMAL_GPU_MSM_MAX_CONCURRENCY
        || record.icicle_msm_precompute_factor != halo2_proofs::icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR
        || record.icicle_msm_precomputed_basis_device_upper_bytes as u64 != ntt_profile.precomputed_basis_device_upper_bytes
        || record.icicle_msm_precomputed_basis_device_bytes > record.icicle_msm_precomputed_basis_device_upper_bytes
        || record.icicle_msm_precomputed_basis_device_peak_bytes < record.icicle_msm_precomputed_basis_device_bytes
        || record.icicle_msm_precomputed_basis_device_peak_bytes > record.icicle_msm_precomputed_basis_device_upper_bytes
        || record.msm_operations == 0
        || record.gpu_msm_chunk_calls < record.msm_operations
        || record.gpu_msm_precomputed_chunk_calls > record.gpu_msm_chunk_calls
        || record.gpu_msm_precompute_calls > record.gpu_msm_precomputed_chunk_calls
        || record.gpu_msm_precomputed_cache_hits > record.gpu_msm_precomputed_chunk_calls
        || record.gpu_msm_precompute_calls.checked_add(record.gpu_msm_precomputed_cache_hits)
            != Some(record.gpu_msm_precomputed_chunk_calls)
        || (planned_ntt_below_minimum && record.ntt_calls != 0)
        || (record.ntt_calls == 0 && record.gpu_ntt_elements != 0)
        || (record.ntt_calls > 0
            && (max_size < (1u64 << 16)
                || record.gpu_ntt_elements < minimum_ntt_elements
                || record.gpu_ntt_elements > maximum_ntt_elements))
        || record.gpu_h_eval_calls == 0
        || record.gpu_h_eval_groups == 0
        || record.gpu_h_eval_groups != h_group_outcomes
        || record.gpu_h_eval_rows == 0
        || record.gpu_h_eval_chunks == 0
        || record.gpu_h_eval_syncs != record.gpu_h_eval_chunks
        || record.gpu_h_workspace_high_water_bytes as u64 > ntt_profile.device_workspace_upper_bytes
        || record.gpu_h_local_memory_peak_upper_bytes as u64 > ntt_profile.h_kernel_local_memory_upper_bytes
        || max_size > ntt_profile.capacity_elements
        || record.ntt_workspace_capacity_bytes as u64 != expected_workspace
        || record.ntt_domain_capacity_bytes as u64 != expected_domain
        || record.ntt_host_staging_capacity_bytes as u64 != expected_host
        || record.ntt_device_peak_upper_bound_bytes as u64 != expected_device_peak
        || expected_workspace > ntt_profile.device_workspace_upper_bytes
        || expected_domain > ntt_profile.domain_cache_upper_bytes
        || expected_device_peak > ntt_profile.device_peak_upper_bytes
        || expected_host > ntt_profile.host_staging_upper_bytes
        || record.synchronized_calls != synchronized_invocations
        || record.evidence.gpu_kernel_invocation_count != invocations
        || record.evidence.provenance.as_ref().map(|p| p.native_execution_policy.as_str()) != Some(expected_policy)
    {
        return Err("native GPU proof counters/capacities do not match the authenticated fused H, NTT, and serialized MSM profile".into());
    }
    Ok(invocations)
}

/// Persist successful native GPU operation counters after the proof is saved.
pub fn write_proof_record(directory: &Path, proof_path: &Path) -> Result<(), EZKLError> {
    let stats = halo2_proofs::icicle::execution_stats();
    let ntt_profile = read_ntt_profile(directory)?;
    write_proof_record_with_stats(
        directory,
        proof_path,
        stats,
        &ntt_profile,
    )
}

fn write_proof_record_with_stats(
    directory: &Path,
    proof_path: &Path,
    (msm_operations, gpu_msm_chunk_calls, ntt_calls, cpu_fft_calls, synchronized_calls, device_id, gpu_ntt_elements, ntt_max_size, ntt_workspace_capacity_bytes, ntt_domain_capacity_bytes, ntt_host_staging_capacity_bytes, ntt_device_peak_upper_bound_bytes, gpu_msm_precompute_calls, gpu_msm_precomputed_chunk_calls, gpu_msm_precomputed_cache_hits, icicle_msm_precomputed_basis_device_bytes, icicle_msm_precomputed_basis_device_peak_bytes, gpu_h_eval_calls, gpu_h_eval_rows, gpu_h_eval_fallbacks, gpu_h_eval_chunks, gpu_h_eval_syncs, gpu_h_workspace_high_water_bytes, gpu_h_local_memory_peak_upper_bytes, gpu_h_eval_groups): (u64, u64, u64, u64, u64, usize, u64, usize, usize, usize, usize, usize, u64, u64, u64, usize, usize, u64, u64, u64, u64, u64, usize, usize, u64),
    ntt_profile: &NttProfile,
) -> Result<(), EZKLError> {
    let (provenance, _) = manifest()?;
    let policy = halo2_proofs::icicle::FORMAL_GPU_EXECUTION_POLICY.to_owned();
    let invocations = gpu_msm_chunk_calls.checked_add(gpu_msm_precompute_calls).and_then(|value| value.checked_add(ntt_calls)).and_then(|value| value.checked_add(gpu_h_eval_chunks)).ok_or("native GPU invocation counter overflow")?;
    let evidence = GpuEvidence { backend: "CUDA".into(), gpu_backend: "icicle-cuda".into(), backend_loaded: true, cpu_fallback: false, device_synchronized: true, gpu_kernel_invocation_count: invocations, provenance: Some(provenance), invocation_trace_sha256: String::new() };
    let mut record = ProofRecord { schema_version: 4, proof_sha256: crate::execute::get_file_hash(proof_path)?, admission_sha256: ntt_profile.admission_sha256.clone(), native_execution_policy: policy, icicle_msm_max_concurrency: halo2_proofs::icicle::FORMAL_GPU_MSM_MAX_CONCURRENCY, icicle_msm_precompute_factor: halo2_proofs::icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR, gpu_msm_precompute_calls, gpu_msm_precomputed_chunk_calls, gpu_msm_precomputed_cache_hits, icicle_msm_precomputed_basis_device_upper_bytes: usize::try_from(ntt_profile.precomputed_basis_device_upper_bytes).map_err(|_| "native resident MSM upper bound exceeds usize")?, icicle_msm_precomputed_basis_device_bytes, icicle_msm_precomputed_basis_device_peak_bytes, msm_operations, gpu_msm_chunk_calls, ntt_calls, cpu_fft_calls, synchronized_calls, device_id, gpu_ntt_elements, ntt_max_size, ntt_workspace_capacity_bytes, ntt_domain_capacity_bytes, ntt_host_staging_capacity_bytes, ntt_device_peak_upper_bound_bytes, gpu_h_eval_groups, gpu_h_eval_calls, gpu_h_eval_rows, gpu_h_eval_fallbacks, gpu_h_eval_chunks, gpu_h_eval_syncs, gpu_h_workspace_high_water_bytes, gpu_h_local_memory_peak_upper_bytes, evidence };
    validate_record(&record, ntt_profile)?;
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
    let ntt_profile = read_ntt_profile(directory)?;
    let (provenance, _) = manifest()?;
    let count = validate_record(&record, &ntt_profile)?;
    let expected_invocations = record.gpu_msm_chunk_calls.checked_add(record.gpu_msm_precompute_calls).and_then(|value| value.checked_add(record.ntt_calls)).and_then(|value| value.checked_add(record.gpu_h_eval_chunks)).ok_or("native GPU invocation counter overflow")?;
    if record.proof_sha256 != crate::execute::get_file_hash(proof_path)? || record.evidence.provenance.as_ref() != Some(&provenance) || !record.evidence.backend_loaded || !record.evidence.device_synchronized || record.evidence.cpu_fallback || record.evidence.backend != "CUDA" || record.evidence.gpu_backend != "icicle-cuda" || record.evidence.invocation_trace_sha256 != invocation_digest(&record)? || count != expected_invocations { return Err("GPU native prover execution evidence is inconsistent with the verified proof".into()); }
    let evidence_path = directory.join("gpu_evidence.json");
    if evidence_path.is_symlink() { return Err("GPU prover evidence must not be a symlink".into()); }
    let evidence: GpuEvidence = serde_json::from_slice(&std::fs::read(&evidence_path)?)?;
    if evidence != record.evidence { return Err("GPU prover evidence sidecar differs from the proof-bound native record".into()); }
    Ok(record.evidence)
}

#[cfg(test)]
mod proof_evidence_tests {
    use super::{read_proof_evidence, write_proof_record_with_stats, NttProfile, ProofRecord};
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

        let capacity_elements = 1u64 << 16;
        let device_workspace_upper_bytes = 64 * capacity_elements;
        let domain_cache_upper_bytes = 32 * (capacity_elements + 1 + 64 + 6);
        let init_scratch_upper_bytes = 4 * 64 * 32;
        let device_peak_upper_bytes = device_workspace_upper_bytes + domain_cache_upper_bytes + init_scratch_upper_bytes;
        let host_staging_upper_bytes = 32 * capacity_elements;
        let precomputed_basis_device_upper_bytes = capacity_elements * 64 * 4;
        let point_staging_host_upper_bytes = capacity_elements.min(1 << 20) * 64;
        let resident_basis_bytes = capacity_elements * 64 * 4;
        let cuda_required_workspace_bytes = 2000
            + device_peak_upper_bytes
            + precomputed_basis_device_upper_bytes
            + halo2_proofs::icicle::FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES as u64;
        let admission = serde_json::json!({
            "domain_size": capacity_elements,
            "native_memory_model": {
                "schema": "ezkl-halo2-static-envelope-v20",
                "domain_size": capacity_elements,
                "proof_domain_size": capacity_elements,
                "icicle_ntt_min_log_size": 16,
                "icicle_ntt_max_log_size": 25,
                "icicle_ntt_capacity_elements": capacity_elements,
                "icicle_ntt_device_workspace_upper_bytes": device_workspace_upper_bytes,
                "icicle_ntt_domain_cache_upper_bytes": domain_cache_upper_bytes,
                "icicle_ntt_init_scratch_upper_bytes": init_scratch_upper_bytes,
                "icicle_ntt_device_peak_upper_bytes": device_peak_upper_bytes,
                "icicle_ntt_host_staging_upper_bytes": host_staging_upper_bytes,
                "icicle_msm_max_concurrency": 1,
                "icicle_msm_precompute_factor": 4,
                "icicle_msm_precomputed_basis_count": 1,
                "icicle_msm_precomputed_basis_device_upper_bytes": precomputed_basis_device_upper_bytes,
                "icicle_msm_point_staging_host_upper_bytes": point_staging_host_upper_bytes
            }
        });
        let admission_path = directory.join("admission.json");
        std::fs::write(&admission_path, serde_json::to_vec(&admission).unwrap()).unwrap();
        let admission_sha256 = crate::execute::get_file_hash(&admission_path).unwrap();
        let budget = serde_json::json!({
            "schema": "ezkl.native-prove-budget.v6",
            "admission_sha256": admission_sha256,
            "native_execution_policy": halo2_proofs::icicle::FORMAL_GPU_EXECUTION_POLICY,
            "h_workspace_borrows_ntt": true,
            "h_kernel_local_memory_upper_bytes": halo2_proofs::icicle::FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES,
            "h_program_metadata_upper_bytes": halo2_proofs::icicle::FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES,
            "cuda_required_workspace_bytes": cuda_required_workspace_bytes,
            "integrated_device": false,
            "host_required_known_bytes": halo2_proofs::icicle::FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES,
            "host_available_bytes": 2 * halo2_proofs::icicle::FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES,
            "icicle_msm_max_concurrency": 1,
            "icicle_msm_precompute_factor": 4,
            "icicle_msm_precomputed_basis_count": 1,
            "icicle_msm_precomputed_basis_device_upper_bytes": precomputed_basis_device_upper_bytes,
            "icicle_msm_point_staging_host_upper_bytes": point_staging_host_upper_bytes,
            "icicle_msm_workspace_query_schema": "ezkl.icicle.cub_msm_workspace_query.v3",
            "icicle_msm_ordinary_factor1_device_workspace_peak_bytes": 1000,
            "icicle_msm_prepared_factor4_device_workspace_peak_bytes": 2000,
            "device_workspace_peak_bytes": 2000,
            "icicle_ntt_min_log_size": 16,
            "icicle_ntt_max_log_size": 25,
            "icicle_ntt_capacity_elements": capacity_elements,
            "icicle_ntt_device_workspace_upper_bytes": device_workspace_upper_bytes,
            "icicle_ntt_domain_cache_upper_bytes": domain_cache_upper_bytes,
            "icicle_ntt_init_scratch_upper_bytes": init_scratch_upper_bytes,
            "icicle_ntt_device_peak_upper_bytes": device_peak_upper_bytes,
            "icicle_ntt_host_staging_upper_bytes": host_staging_upper_bytes
        });
        std::fs::write(directory.join("native_prove_budget.json"), serde_json::to_vec(&budget).unwrap()).unwrap();
        let ntt_profile = NttProfile {
            admission_sha256,
            capacity_elements,
            device_workspace_upper_bytes,
            domain_cache_upper_bytes,
            device_peak_upper_bytes,
            host_staging_upper_bytes,
            precomputed_basis_device_upper_bytes,
            h_kernel_local_memory_upper_bytes: halo2_proofs::icicle::FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES as u64,
        };

        // Exercise the same production writer as Prove, with a small valid
        // native counter sample so the regression never initializes a GPU.
        let ntt_max_size = 1usize << 16;
        write_proof_record_with_stats(
            &directory,
            &proof_path,
            (
                1,
                1,
                1,
                1,
                4,
                0,
                capacity_elements,
                ntt_max_size,
                64 * ntt_max_size,
                32 * (ntt_max_size + 1 + 64 + 6),
                32 * ntt_max_size,
                96 * ntt_max_size + 10464,
                1,
                1,
                0,
                resident_basis_bytes as usize,
                resident_basis_bytes as usize,
                1,
                1,
                1,
                1,
                1,
                device_workspace_upper_bytes as usize,
                1 << 20,
                2,
            ),
            &ntt_profile,
        )
        .unwrap();
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

        let mut inconsistent_record: ProofRecord = serde_json::from_slice(&record_bytes).unwrap();
        inconsistent_record.gpu_h_eval_groups += 1;
        std::fs::write(&record_path, serde_json::to_vec_pretty(&inconsistent_record).unwrap()).unwrap();
        assert!(read_proof_evidence(&directory, &proof_path).is_err());
        std::fs::write(&record_path, &record_bytes).unwrap();

        let mut tampered = evidence;
        tampered.gpu_kernel_invocation_count += 1;
        std::fs::write(&evidence_path, serde_json::to_vec_pretty(&tampered).unwrap()).unwrap();
        assert!(read_proof_evidence(&directory, &proof_path).is_err());
        assert_eq!(std::fs::read(&record_path).unwrap(), record_bytes);
        std::fs::remove_dir_all(directory).unwrap();
    }
}
