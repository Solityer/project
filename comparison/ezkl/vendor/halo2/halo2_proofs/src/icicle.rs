use group::{ff::{PrimeField, Field}, Group};
use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarField};
use halo2curves::bn256::{Fr as Bn256Fr, G1Affine as Bn256G1Affine};
use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use crate::h_eval::{GateProgram, GATE_OP_ADD, GATE_OP_DOUBLE, GATE_OP_FOLD, GATE_OP_LOAD_CONST, GATE_OP_LOAD_QUERY, GATE_OP_MUL, GATE_OP_NEGATE, GATE_OP_SQUARE, GATE_OP_SUB};
use crate::poly::{LagrangeCoeff, Polynomial};
use crate::arithmetic::FftGroup;
use std::any::{TypeId, Any};
pub use halo2curves::CurveAffine;
use icicle_core::{
    curve::Affine,
    msm,
    ntt::{initialize_domain, release_domain, ntt, NTTConfig, NTTDir},
};
use std::{env, io, mem};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::fmt;
use icicle_cuda_runtime::{device, stream::CudaStream};

static MSM_OPERATIONS: AtomicU64 = AtomicU64::new(0);
static GPU_MSM_CHUNK_CALLS: AtomicU64 = AtomicU64::new(0);
static GPU_MSM_PRECOMPUTE_CALLS: AtomicU64 = AtomicU64::new(0);
static GPU_MSM_PRECOMPUTED_CHUNK_CALLS: AtomicU64 = AtomicU64::new(0);
static GPU_MSM_PRECOMPUTED_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
static GPU_MSM_PRECOMPUTED_DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);
static GPU_MSM_PRECOMPUTED_DEVICE_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static GPU_H_EVAL_CALLS: AtomicU64 = AtomicU64::new(0);
static GPU_H_EVAL_GROUPS: AtomicU64 = AtomicU64::new(0);
static GPU_H_EVAL_ROWS: AtomicU64 = AtomicU64::new(0);
static GPU_H_EVAL_FALLBACKS: AtomicU64 = AtomicU64::new(0);
static GPU_H_EVAL_CHUNKS: AtomicU64 = AtomicU64::new(0);
static GPU_H_EVAL_SYNCS: AtomicU64 = AtomicU64::new(0);
static GPU_H_WORKSPACE_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static GPU_H_LOCAL_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static GPU_H_RESOURCE_REPORT: AtomicUsize = AtomicUsize::new(0);
static GPU_H_FALLBACK_DIAGNOSTIC_REPORTED: AtomicUsize = AtomicUsize::new(0);
static NTT_CALLS: AtomicU64 = AtomicU64::new(0);
static CPU_FFT_CALLS: AtomicU64 = AtomicU64::new(0);
static SYNCHRONIZED_CALLS: AtomicU64 = AtomicU64::new(0);
static DEVICE_ID: AtomicUsize = AtomicUsize::new(0);
#[cfg(feature = "formal-gpu-msm")]
static FORMAL_GPU_MSM_GATE: Mutex<()> = Mutex::new(());
static NTT_DOMAINS: Mutex<Option<HashMap<usize, (u32, Bn256Fr)>>> = Mutex::new(None);
static NTT_WORKSPACES: Mutex<Option<HashMap<usize, NttWorkspace>>> = Mutex::new(None);
static PROVE_NTT_SCOPE_DEPTH: AtomicUsize = AtomicUsize::new(0);
static GPU_NTT_ELEMENTS: AtomicU64 = AtomicU64::new(0);
static NTT_MAX_SIZE: AtomicUsize = AtomicUsize::new(0);
static NTT_WORKSPACE_CAPACITY_BYTES: AtomicUsize = AtomicUsize::new(0);
static NTT_DOMAIN_CAPACITY_BYTES: AtomicUsize = AtomicUsize::new(0);
static NTT_HOST_STAGING_CAPACITY_BYTES: AtomicUsize = AtomicUsize::new(0);
static NTT_DEVICE_PEAK_UPPER_BOUND_BYTES: AtomicUsize = AtomicUsize::new(0);

struct NttWorkspace {
    capacity: usize,
    input: Option<DeviceVec<ScalarField>>,
    output: Option<DeviceVec<ScalarField>>,
    host_values: Vec<ScalarField>,
}

impl NttWorkspace {
    fn new() -> Self { Self { capacity: 0, input: None, output: None, host_values: Vec::new() } }
}

/// Maximum scalar/base slice handed to one native ICICLE MSM invocation.
pub const MSM_MAX_CHUNK_SCALARS: usize = 1 << 20;
/// Maximum number of concurrent formal GPU MSM invocations.
pub const FORMAL_GPU_MSM_MAX_CONCURRENCY: usize = 1;
/// Fixed formal Prove profile for resident KZG MSM bases.
pub const FORMAL_GPU_MSM_PRECOMPUTE_FACTOR: usize = 4;
/// Smallest transform length dispatched to ICICLE in the formal GPU Prove scope.
pub const FORMAL_GPU_NTT_MIN_SIZE: usize = 1 << 16;
/// Largest transform length dispatched to ICICLE in the formal GPU Prove scope.
pub const FORMAL_GPU_NTT_MAX_SIZE: usize = 1 << 25;
/// Upper bound, in bytes, for transient device scratch during NTT domain initialization.
pub const FORMAL_GPU_NTT_INIT_SCRATCH_BYTES: usize = 4 * 64 * std::mem::size_of::<ScalarField>();
/// Fixed formal GPU profile: bounded BN254 ICICLE NTT and chunked ICICLE MSM.
pub const FORMAL_GPU_EXECUTION_POLICY: &str = "halo2_icicle-bn254-gpu-hgraph-fused-ntt-16-25-msm-lagrange-resident-f4-v5";

const _: [(); 32] = [(); mem::size_of::<Bn256Fr>()];
const _: [(); 8] = [(); mem::align_of::<Bn256Fr>()];
const _: [(); 32] = [(); mem::size_of::<ScalarField>()];
const _: [(); 4] = [(); mem::align_of::<ScalarField>()];
/// Conservative reserve for per-thread CUDA local and stack allocation by the H kernel.
pub const FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES: usize = 64 * 1024 * 1024;
/// Maximum host-side graph metadata and its fixed device-staging copy for one H program.
pub const FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES: usize = 64 * 1024 * 1024;

/// The KZG basis being prepared; the cache is owned by one ParamsKZG instance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(crate) enum MsmBasisRole {
    Coefficient,
    Lagrange,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MsmDeviceIdentity {
    ordinal: usize,
    uuid: String,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct MsmCacheKey {
    role: MsmBasisRole,
    chunk_index: usize,
    factor: usize,
}

/// ParamsKZG-owned ICICLE points. `DeviceVec` frees against the current CUDA
/// device, so this wrapper selects the allocation's owner while releasing it.
struct DeviceBoundVec<T> {
    device_id: usize,
    bytes: usize,
    buffer: Option<DeviceVec<T>>,
}

impl<T> DeviceBoundVec<T> {
    fn allocate(len: usize, device_id: usize) -> Self {
        let bytes = len.checked_mul(mem::size_of::<T>()).expect("ICICLE resident MSM allocation size overflow");
        assert!(len > 0 && bytes > 0, "ICICLE resident MSM allocation must be nonempty");
        let buffer = DeviceVec::<T>::cuda_malloc_for_device(len, device_id)
            .expect("ICICLE resident MSM point allocation failed");
        let live = GPU_MSM_PRECOMPUTED_DEVICE_BYTES.fetch_add(bytes, Ordering::AcqRel)
            .checked_add(bytes)
            .expect("ICICLE resident MSM byte counter overflow");
        GPU_MSM_PRECOMPUTED_DEVICE_PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        Self { device_id, bytes, buffer: Some(buffer) }
    }

    fn device_slice(&self) -> &icicle_cuda_runtime::memory::DeviceSlice<T> {
        self.buffer.as_ref().expect("ICICLE resident MSM points were released")
    }

    fn device_slice_mut(&mut self) -> &mut icicle_cuda_runtime::memory::DeviceSlice<T> {
        self.buffer.as_mut().expect("ICICLE resident MSM points were released")
    }
}

impl<T> fmt::Debug for DeviceBoundVec<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("DeviceBoundVec").field("device_id", &self.device_id).field("bytes", &self.bytes).finish()
    }
}

impl<T> Drop for DeviceBoundVec<T> {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else { return; };
        let previous = device::get_device().expect("cannot read CUDA device while releasing resident MSM points");
        if previous != self.device_id {
            device::set_device(self.device_id).expect("cannot select owner device while releasing resident MSM points");
        }
        drop(buffer);
        if previous != self.device_id {
            device::set_device(previous).expect("cannot restore CUDA device after releasing resident MSM points");
        }
        let previous_bytes = GPU_MSM_PRECOMPUTED_DEVICE_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
        assert!(previous_bytes >= self.bytes, "ICICLE resident MSM byte counter underflow");
    }
}

#[derive(Debug)]
struct PreparedMsmBasis {
    device: MsmDeviceIdentity,
    base_len: usize,
    factor: usize,
    points: DeviceBoundVec<Affine<CurveCfg>>,
}

/// Shared by ParamsKZG clones whose coefficient and Lagrange SRS vectors are
/// unchanged. Replacing it on parameter mutation leaves existing clones intact.
#[derive(Default)]
pub(crate) struct PreparedMsmCache {
    active_device: Option<MsmDeviceIdentity>,
    entries: HashMap<MsmCacheKey, PreparedMsmBasis>,
}

impl fmt::Debug for PreparedMsmCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("PreparedMsmCache")
            .field("active_device", &self.active_device)
            .field("prepared_chunks", &self.entries.len())
            .finish()
    }
}

/// RAII guard that enables formal ICICLE NTT dispatch during native Prove work.
#[derive(Debug)]
pub struct NativeProveNttScope;

/// Enter the formal native Prove scope in which eligible transforms use ICICLE.
pub fn native_prove_ntt_scope() -> NativeProveNttScope {
    PROVE_NTT_SCOPE_DEPTH.fetch_add(1, Ordering::AcqRel);
    NativeProveNttScope
}

impl Drop for NativeProveNttScope {
    fn drop(&mut self) {
        let previous = PROVE_NTT_SCOPE_DEPTH.fetch_sub(1, Ordering::AcqRel);
        assert!(previous > 0, "native Prove NTT scope depth underflow");
    }
}

/// Return whether a native Prove NTT scope is active in this process.
pub fn native_prove_ntt_scope_active() -> bool { PROVE_NTT_SCOPE_DEPTH.load(Ordering::Acquire) > 0 }

/// Initialize the CUDA runtime used by this pinned ICICLE 2.8 backend.
pub fn initialize_device(device_id: usize) -> Result<(), String> {
    let count = device::get_device_count().map_err(|e| format!("ICICLE CUDA device enumeration failed: {:?}", e))?;
    if device_id >= count { return Err(format!("ICICLE device {} is unavailable ({} devices)", device_id, count)); }
    device::set_device(device_id).map_err(|e| format!("ICICLE CUDA device selection failed: {:?}", e))?;
    CudaStream::default().synchronize().map_err(|e| format!("ICICLE CUDA device synchronization failed: {:?}", e))?;
    DEVICE_ID.store(device_id, Ordering::Relaxed);
    Ok(())
}

/// Counts successful native operations and explicit synchronization in this process.
/// Tuple tail (indices 17..24): H program calls, rows, preflight fallbacks,
/// successful launches, actual H sync calls, borrowed NTT workspace bytes,
/// conservative local-plus-stack memory upper-bound bytes, and attempted groups.
pub fn execution_stats() -> (u64, u64, u64, u64, u64, usize, u64, usize, usize, usize, usize, usize, u64, u64, u64, usize, usize, u64, u64, u64, u64, u64, usize, usize, u64) {
    (
        MSM_OPERATIONS.load(Ordering::Relaxed),
        GPU_MSM_CHUNK_CALLS.load(Ordering::Relaxed),
        NTT_CALLS.load(Ordering::Relaxed),
        CPU_FFT_CALLS.load(Ordering::Relaxed),
        SYNCHRONIZED_CALLS.load(Ordering::Relaxed),
        DEVICE_ID.load(Ordering::Relaxed),
        GPU_NTT_ELEMENTS.load(Ordering::Relaxed),
        NTT_MAX_SIZE.load(Ordering::Relaxed),
        NTT_WORKSPACE_CAPACITY_BYTES.load(Ordering::Relaxed),
        NTT_DOMAIN_CAPACITY_BYTES.load(Ordering::Relaxed),
        NTT_HOST_STAGING_CAPACITY_BYTES.load(Ordering::Relaxed),
        NTT_DEVICE_PEAK_UPPER_BOUND_BYTES.load(Ordering::Relaxed),
        GPU_MSM_PRECOMPUTE_CALLS.load(Ordering::Relaxed),
        GPU_MSM_PRECOMPUTED_CHUNK_CALLS.load(Ordering::Relaxed),
        GPU_MSM_PRECOMPUTED_CACHE_HITS.load(Ordering::Relaxed),
        GPU_MSM_PRECOMPUTED_DEVICE_BYTES.load(Ordering::Acquire),
        GPU_MSM_PRECOMPUTED_DEVICE_PEAK_BYTES.load(Ordering::Relaxed),
        GPU_H_EVAL_CALLS.load(Ordering::Relaxed),
        GPU_H_EVAL_ROWS.load(Ordering::Relaxed),
        GPU_H_EVAL_FALLBACKS.load(Ordering::Relaxed),
        GPU_H_EVAL_CHUNKS.load(Ordering::Relaxed),
        GPU_H_EVAL_SYNCS.load(Ordering::Relaxed),
        GPU_H_WORKSPACE_PEAK_BYTES.load(Ordering::Relaxed),
        GPU_H_LOCAL_PEAK_BYTES.load(Ordering::Relaxed),
        GPU_H_EVAL_GROUPS.load(Ordering::Relaxed),
    )
}

/// Reset per-proof operation counters while retaining the actual persistent-cache footprint.
pub fn reset_execution_operation_stats() {
    MSM_OPERATIONS.store(0, Ordering::Relaxed);
    GPU_MSM_CHUNK_CALLS.store(0, Ordering::Relaxed);
    GPU_MSM_PRECOMPUTE_CALLS.store(0, Ordering::Relaxed);
    GPU_MSM_PRECOMPUTED_CHUNK_CALLS.store(0, Ordering::Relaxed);
    GPU_MSM_PRECOMPUTED_CACHE_HITS.store(0, Ordering::Relaxed);
    GPU_H_EVAL_CALLS.store(0, Ordering::Relaxed);
    GPU_H_EVAL_GROUPS.store(0, Ordering::Relaxed);
    GPU_H_EVAL_ROWS.store(0, Ordering::Relaxed);
    GPU_H_EVAL_FALLBACKS.store(0, Ordering::Relaxed);
    GPU_H_EVAL_CHUNKS.store(0, Ordering::Relaxed);
    GPU_H_EVAL_SYNCS.store(0, Ordering::Relaxed);
    GPU_H_FALLBACK_DIAGNOSTIC_REPORTED.store(0, Ordering::Relaxed);
    NTT_CALLS.store(0, Ordering::Relaxed);
    CPU_FFT_CALLS.store(0, Ordering::Relaxed);
    SYNCHRONIZED_CALLS.store(0, Ordering::Relaxed);
    GPU_NTT_ELEMENTS.store(0, Ordering::Relaxed);
}

/// Record one completed Halo2 CPU FFT in the fixed formal GPU execution profile.
pub fn record_cpu_fft() { CPU_FFT_CALLS.fetch_add(1, Ordering::Relaxed); }

/// Returns the selected CUDA device's current free and total memory in bytes.
pub fn device_memory_info() -> Result<(usize, usize), String> {
    select_device();
    device::get_memory_info().map_err(|e| format!("ICICLE CUDA memory query failed: {:?}", e))
}

/// Returns the selected CUDA UUID and current free/total memory in bytes.
pub fn device_capacity_identity() -> Result<(String, usize, usize), String> {
    select_device();
    let selected = DEVICE_ID.load(Ordering::Relaxed);
    let uuid = device::get_device_uuid(selected)
        .map_err(|e| format!("ICICLE CUDA UUID query failed: {:?}", e))?;
    let (free, total) = device::get_memory_info()
        .map_err(|e| format!("ICICLE CUDA memory query failed: {:?}", e))?;
    if total == 0 || free > total {
        return Err("ICICLE CUDA runtime returned invalid memory capacity".to_owned());
    }
    Ok((uuid, free, total))
}

fn select_device() { device::set_device(DEVICE_ID.load(Ordering::Relaxed)).expect("ICICLE CUDA thread device selection failed"); }

fn synchronized_operation(counter: &AtomicU64) {
    CudaStream::default().synchronize().expect("ICICLE CUDA operation synchronization failed");
    counter.fetch_add(1, Ordering::Relaxed);
    SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// Use the native CPU threshold for small multiexponentiations.
pub fn should_use_cpu_msm(size: usize) -> bool {
    #[cfg(feature = "formal-gpu-msm")]
    {
        return size <= 4;
    }
    #[cfg(not(feature = "formal-gpu-msm"))]
    {
        size <= (1
            << u8::from_str_radix(&env::var("ICICLE_SMALL_K").unwrap_or("2".to_string()), 10).unwrap())
    }
}

/// Use the native CPU threshold for small transforms.
pub fn should_use_cpu_fft(size: usize) -> bool {
    #[cfg(feature = "formal-gpu-msm")]
    {
        return !(FORMAL_GPU_NTT_MIN_SIZE..=FORMAL_GPU_NTT_MAX_SIZE).contains(&size);
    }
    #[cfg(not(feature = "formal-gpu-msm"))]
    {
        size <= (1
            << u8::from_str_radix(&env::var("ICICLE_SMALL_K_FFT").unwrap_or("2".to_string()), 10)
                .unwrap())
    }
}

/// Check whether the field has a native BN254 CUDA implementation.
pub fn is_gpu_supported_field<G: Any>(_sample_element: &G) -> bool {
    match TypeId::of::<G>() {
        id if id == TypeId::of::<Bn256Fr>() => true,
        _ => false,
    }
}

/// Check whether a curve type matches the BN254 G1 point representation used by ICICLE.
pub fn is_gpu_supported_curve<C: CurveAffine + 'static>() -> bool { TypeId::of::<C>() == TypeId::of::<Bn256G1Affine>() }

const BN254_MODULUS_LIMBS: [u64; 4] = [0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];
const BN254_MONTGOMERY_R_LIMBS: [u64; 4] = [0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0x0e0a77c19a07df2f];
const BN254_NEG_MONTGOMERY_R_LIMBS: [u64; 4] = [0x974bc177a0000006, 0xf13771b2da58a367, 0x51e1a2470908122e, 0x2259d6b14729c0fa];

extern "C" {
    fn bn254_h_eval_program(input: *const u32, output: *mut u32, rows: usize, instruction_count: usize, constant_count: usize, query_count: usize, scratch_slots: usize) -> i32;
    fn bn254_h_eval_kernel_attributes(local_bytes_per_thread: *mut usize, shared_bytes_per_block: *mut usize, stack_bytes_per_thread: *mut usize, registers_per_thread: *mut i32, max_threads_per_block: *mut i32) -> i32;
}

// SAFETY: Cargo.lock pins halo2curves to b753a832..., where BN254 Fr is a
// single [u64; 4] wrapper at offset zero. Compile-time size/alignment checks
// and the little-endian/modulus checks guard this exact type. Runtime
// ZERO/ONE/-ONE probes validate its Montgomery encoding before any caller
// copies raw limbs to the device; no generic field pointer is cast.
fn bn254_fr_raw_montgomery(value: &Bn256Fr) -> [u64; 4] { unsafe { std::ptr::read(value as *const Bn256Fr as *const [u64; 4]) } }

fn raw_montgomery_layout_supported<F: PrimeField + 'static>() -> bool {
    // The pinned halo2curves PrimeField::MODULUS string includes its `0x`
    // prefix (see halo2curves derive/src/field/mod.rs); keep this exact
    // modulus check while matching that published representation.
    if !cfg!(target_endian = "little") || TypeId::of::<F>() != TypeId::of::<Bn256Fr>() || F::MODULUS != "0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001" { return false; }
    bn254_fr_raw_montgomery(&Bn256Fr::ZERO) == [0; 4] && bn254_fr_raw_montgomery(&Bn256Fr::ONE) == BN254_MONTGOMERY_R_LIMBS && bn254_fr_raw_montgomery(&-Bn256Fr::ONE) == BN254_NEG_MONTGOMERY_R_LIMBS
}

fn fr_raw_montgomery_words(value: &Bn256Fr) -> [u32; 8] {
    let limbs = bn254_fr_raw_montgomery(value);
    [limbs[0] as u32, (limbs[0] >> 32) as u32, limbs[1] as u32, (limbs[1] >> 32) as u32, limbs[2] as u32, (limbs[2] >> 32) as u32, limbs[3] as u32, (limbs[3] >> 32) as u32]
}

fn canonical_field_words<F: PrimeField>(value: &F) -> Option<[u32; 8]> {
    let repr = value.to_repr();
    let bytes = repr.as_ref();
    if bytes.len() != 32 { return None; }
    let mut words = [0u32; 8];
    for (index, chunk) in bytes.chunks_exact(4).enumerate() { words[index] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]); }
    Some(words)
}

fn raw_scalar_words(value: &ScalarField) -> [u32; 8] {
    // SAFETY: the compile-time assertions pin ScalarField to 32 bytes with
    // 4-byte alignment; ICICLE Field is repr(C) with exactly eight u32 limbs.
    unsafe { mem::transmute_copy(value) }
}

fn words_to_u64_limbs(words: [u32; 8]) -> [u64; 4] { [words[0] as u64 | ((words[1] as u64) << 32), words[2] as u64 | ((words[3] as u64) << 32), words[4] as u64 | ((words[5] as u64) << 32), words[6] as u64 | ((words[7] as u64) << 32)] }

fn is_reduced_bn254_limb(value: &[u64; 4]) -> bool {
    for index in (0..4).rev() { if value[index] < BN254_MODULUS_LIMBS[index] { return true; } if value[index] > BN254_MODULUS_LIMBS[index] { return false; } }
    false
}

fn validate_gate_program<F: PrimeField>(program: &GateProgram<F>) -> io::Result<bool> {
    let mut initialized = Vec::new();
    if initialized.try_reserve_exact(program.scratch_slots).is_err() { return Ok(false); }
    initialized.resize(program.scratch_slots, false);
    let mut folds = 0usize;
    let check_slot = |slot: u32| -> io::Result<()> { if (slot as usize) < program.scratch_slots { Ok(()) } else { Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program scratch index is out of bounds")) } };
    let check_source = |slot: u32, initialized: &[bool]| -> io::Result<()> { check_slot(slot)?; if initialized[slot as usize] { Ok(()) } else { Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program reads an uninitialized scratch slot")) } };
    for instruction in &program.instructions {
        let op = instruction[0];
        let dst = instruction[1];
        if op == GATE_OP_LOAD_QUERY {
            if instruction[2] as usize >= program.queries.len() || instruction[3..].iter().any(|word| *word != 0) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H load-query instruction is malformed")); }
            check_slot(dst)?;
            initialized[dst as usize] = true;
        } else if op == GATE_OP_LOAD_CONST {
            if instruction[2] as usize >= program.constants.len() || instruction[3..].iter().any(|word| *word != 0) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H load-constant instruction is malformed")); }
            check_slot(dst)?;
            initialized[dst as usize] = true;
        } else if op == GATE_OP_ADD || op == GATE_OP_SUB || op == GATE_OP_MUL {
            if instruction[4..].iter().any(|word| *word != 0) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H binary instruction is malformed")); }
            check_slot(dst)?;
            check_source(instruction[2], &initialized)?;
            check_source(instruction[3], &initialized)?;
            initialized[dst as usize] = true;
        } else if op == GATE_OP_SQUARE || op == GATE_OP_DOUBLE || op == GATE_OP_NEGATE {
            if instruction[3..].iter().any(|word| *word != 0) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H unary instruction is malformed")); }
            check_slot(dst)?;
            check_source(instruction[2], &initialized)?;
            initialized[dst as usize] = true;
        } else if op == GATE_OP_FOLD {
            if instruction[2..].iter().any(|word| *word != 0) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H fold instruction is malformed")); }
            check_source(dst, &initialized)?;
            folds += 1;
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program uses an unknown opcode"));
        }
    }
    if folds != program.fold_count { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H fold count does not match its instruction stream")); }
    Ok(true)
}

fn query_column_len<F: PrimeField>(query_kind: u32, column: usize, fixed: &[Polynomial<F, LagrangeCoeff>], advice: &[Polynomial<F, LagrangeCoeff>], instance: &[Polynomial<F, LagrangeCoeff>]) -> Option<usize> {
    match query_kind { 0 => fixed.get(column).map(|poly| poly.len()), 1 => advice.get(column).map(|poly| poly.len()), 2 => instance.get(column).map(|poly| poly.len()), _ => None }
}

fn query_value<'a, F: PrimeField>(query_kind: u32, column: usize, row: usize, fixed: &'a [Polynomial<F, LagrangeCoeff>], advice: &'a [Polynomial<F, LagrangeCoeff>], instance: &'a [Polynomial<F, LagrangeCoeff>]) -> Option<&'a F> {
    match query_kind { 0 => fixed.get(column)?.get(row), 1 => advice.get(column)?.get(row), 2 => instance.get(column)?.get(row), _ => None }
}

pub(crate) fn record_gpu_h_eval_group() { GPU_H_EVAL_GROUPS.fetch_add(1, Ordering::Relaxed); }

pub(crate) fn record_gpu_h_eval_fallback() { GPU_H_EVAL_FALLBACKS.fetch_add(1, Ordering::Relaxed); }

fn unsupported_h_program(reason: &str) -> io::Result<bool> {
    record_gpu_h_eval_fallback();
    if GPU_H_FALLBACK_DIAGNOSTIC_REPORTED.swap(1, Ordering::Relaxed) == 0 {
        eprintln!("[ezkl-gpu-h] adapter preflight fallback: {}", reason);
    }
    Ok(false)
}

fn synchronize_h_stream(stream: &CudaStream) -> io::Result<()> {
    GPU_H_EVAL_SYNCS.fetch_add(1, Ordering::Relaxed);
    SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
    stream.synchronize().map_err(|e| io::Error::new(io::ErrorKind::Other, format!("ICICLE H stream synchronization failed: {:?}", e)))
}

/// Execute one grouped gate program on the shared ICICLE NTT workspace.
pub(crate) fn try_evaluate_h_program<F: PrimeField + 'static>(program: &GateProgram<F>, fixed: &[Polynomial<F, LagrangeCoeff>], advice: &[Polynomial<F, LagrangeCoeff>], instance: &[Polynomial<F, LagrangeCoeff>], part_index: usize, part_stride: usize, rot_scale: i32, isize: i32, y: F, values: &mut [F]) -> io::Result<bool> {
    if !native_prove_ntt_scope_active() { return unsupported_h_program("prove_ntt_scope_inactive"); }
    if env::var_os("ENABLE_ICICLE_GPU").is_none() { return unsupported_h_program("ENABLE_ICICLE_GPU_missing"); }
    if !raw_montgomery_layout_supported::<F>() { return unsupported_h_program("bn254_fr_raw_montgomery_guard_failed"); }
    if program.scratch_slots == 0 { return unsupported_h_program("program_has_no_scratch_slots"); }
    if values.len() < 32 { return unsupported_h_program(&format!("values_too_short len={} minimum=32", values.len())); }
    if part_stride == 0 { return unsupported_h_program("part_stride_zero"); }
    if part_index >= part_stride { return unsupported_h_program(&format!("part_index_out_of_range part_index={} part_stride={}", part_index, part_stride)); }
    if rot_scale <= 0 { return unsupported_h_program(&format!("rot_scale_nonpositive rot_scale={}", rot_scale)); }
    if isize <= 0 { return unsupported_h_program(&format!("domain_size_nonpositive isize={}", isize)); }
    if (isize as usize) % part_stride != 0 || values.len() != (isize as usize) / part_stride {
        return unsupported_h_program(&format!("domain_shape_mismatch isize={} part_stride={} expected_rows={} values_len={}", isize, part_stride, (isize as usize) / part_stride, values.len()));
    }
    let base_n = (isize as usize) / part_stride;
    if program.instructions.is_empty() { return unsupported_h_program("program_has_no_instructions"); }
    if program.fold_count == 0 { return unsupported_h_program("program_has_no_folds"); }
    let fixed_input = match program.instructions.len().checked_add(program.constants.len()).and_then(|n| n.checked_add(1)) { Some(value) => value, None => return unsupported_h_program("fixed_input_size_overflow") };
    let instruction_bytes = program.instructions.capacity().checked_mul(mem::size_of::<[u32; 8]>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H instruction metadata size overflow"))?;
    let constant_bytes = program.constants.capacity().checked_mul(mem::size_of::<F>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H constant metadata size overflow"))?;
    let query_bytes = program.queries.capacity().checked_mul(mem::size_of::<crate::h_eval::GateQuery>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H query metadata size overflow"))?;
    let validation_bytes = program.scratch_slots.checked_mul(mem::size_of::<bool>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H scratch metadata size overflow"))?;
    let staging_bytes = fixed_input.checked_mul(mem::size_of::<ScalarField>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H metadata staging size overflow"))?;
    let canonical_constant_bytes = program.constants.len().checked_mul(mem::size_of::<[u32; 8]>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H canonical constant size overflow"))?;
    let metadata_bytes = instruction_bytes.checked_add(constant_bytes).and_then(|n| n.checked_add(query_bytes)).and_then(|n| n.checked_add(validation_bytes)).and_then(|n| n.checked_add(staging_bytes)).and_then(|n| n.checked_add(canonical_constant_bytes)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H program metadata size overflow"))?;
    if metadata_bytes > FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES { return unsupported_h_program(&format!("program_metadata_over_budget bytes={} limit={}", metadata_bytes, FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES)); }
    if !validate_gate_program(program)? { return unsupported_h_program("program_validation_workspace_unavailable"); }
    for (query_index, query) in program.queries.iter().enumerate() {
        let actual_len = query_column_len(query.kind, query.column, fixed, advice, instance);
        if actual_len != Some(base_n) {
            return unsupported_h_program(&format!("query_column_shape_mismatch query_index={} kind={} column={} expected_rows={} actual_len={:?} isize={} part_stride={}", query_index, query.kind, query.column, base_n, actual_len, isize, part_stride));
        }
    }
    if canonical_field_words(&y).is_none() { return unsupported_h_program("y_is_not_canonical_field_element"); }
    if program.constants.iter().any(|value| canonical_field_words(value).is_none()) { return unsupported_h_program("constant_is_not_canonical_field_element"); }
    for (query_index, query) in program.queries.iter().enumerate() {
        if query.rotation.checked_mul(rot_scale).is_none() {
            return unsupported_h_program(&format!("query_rotation_overflow query_index={} rotation={} rot_scale={}", query_index, query.rotation, rot_scale));
        }
    }
    let device_id = DEVICE_ID.load(Ordering::Relaxed);
    if device::set_device(device_id).is_err() { return Err(io::Error::new(io::ErrorKind::Other, "ICICLE could not select the H evaluator device")); }
    let mut local_bytes_per_thread = 0usize;
    let mut shared_bytes_per_block = 0usize;
    let mut stack_bytes_per_thread = 0usize;
    let mut registers_per_thread = 0i32;
    let mut max_threads_per_block = 0i32;
    let attributes_status = unsafe { bn254_h_eval_kernel_attributes(&mut local_bytes_per_thread, &mut shared_bytes_per_block, &mut stack_bytes_per_thread, &mut registers_per_thread, &mut max_threads_per_block) };
    if attributes_status != 0 { return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H kernel resource query failed with CUDA status {}", attributes_status))); }
    if GPU_H_RESOURCE_REPORT.swap(1, Ordering::Relaxed) == 0 {
        eprintln!("[ezkl-gpu-h] kernel attrs local_bytes_per_thread={} stack_limit_bytes_per_thread={} shared_bytes_per_block={} registers_per_thread={} max_threads_per_block={} local_upper_formula=rounded_threads*(local+stack_limit)", local_bytes_per_thread, stack_bytes_per_thread, shared_bytes_per_block, registers_per_thread, max_threads_per_block);
    }
    if shared_bytes_per_block != 0 || max_threads_per_block < 128 { return unsupported_h_program(&format!("kernel_attributes_unsupported shared_bytes_per_block={} max_threads_per_block={}", shared_bytes_per_block, max_threads_per_block)); }
    let local_bytes_per_thread = match local_bytes_per_thread.checked_add(stack_bytes_per_thread) { Some(bytes) => bytes, None => return unsupported_h_program("kernel_local_bytes_overflow") };
    let local_thread_capacity = if local_bytes_per_thread == 0 { usize::MAX } else { FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES / local_bytes_per_thread };
    let local_row_capacity = local_thread_capacity / 128 * 128;
    if local_row_capacity < 128 { return unsupported_h_program(&format!("kernel_local_memory_reserve_too_small per_thread_bytes={} row_capacity={} reserve_bytes={}", local_bytes_per_thread, local_row_capacity, FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES)); }
    let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
    let workspaces = workspaces.get_or_insert_with(HashMap::new);
    let workspace = workspaces.entry(device_id).or_insert_with(NttWorkspace::new);
    let requested_capacity = workspace.capacity.max(values.len()).max(FORMAL_GPU_NTT_MIN_SIZE).max(workspace.host_values.capacity());
    let target_capacity = match requested_capacity.checked_next_power_of_two() { Some(value) => value, None => return unsupported_h_program("workspace_capacity_rounding_overflow") };
    if target_capacity > FORMAL_GPU_NTT_MAX_SIZE { return unsupported_h_program(&format!("workspace_capacity_over_policy_cap requested={} rounded={} cap={}", requested_capacity, target_capacity, FORMAL_GPU_NTT_MAX_SIZE)); }
    // Keep the shared host staging arena at the full authenticated NTT
    // workspace size. Reserving only a tile's active rows can make Vec grow
    // geometrically to a non-policy capacity, which breaks the strict
    // host-staging == ntt_max_size * sizeof(ScalarField) record contract.
    // No async operation can still reference host_values here: every prior H
    // tile synchronizes before returning, and the NTT workspace mutex is held.
    let replaced_host_arena = workspace.host_values.capacity() != target_capacity;
    if replaced_host_arena {
        // Drop the old allocation before requesting a larger one so the host
        // peak does not temporarily include both full staging arenas.
        workspace.host_values = Vec::new();
        workspace.host_values.try_reserve_exact(target_capacity).map_err(|error| io::Error::new(io::ErrorKind::Other, format!("ICICLE H host staging allocation failed: {}", error)))?;
        if workspace.host_values.capacity() != target_capacity {
            // Do not publish a capacity we cannot authenticate against the
            // NTT record or fall back to a successful CPU proof with an
            // unrecorded allocation peak.
            let actual_capacity = workspace.host_values.capacity();
            workspace.host_values = Vec::new();
            return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H host staging capacity {} does not match authenticated workspace size {}", actual_capacity, target_capacity)));
        }
    }
    if workspace.host_values.len() < target_capacity { workspace.host_values.resize_with(target_capacity, || ScalarField::from([0u32; 8])); }
    if workspace.host_values.len() != target_capacity || workspace.host_values.capacity() != target_capacity {
        if replaced_host_arena { workspace.host_values = Vec::new(); }
        return Err(io::Error::new(io::ErrorKind::Other, "ICICLE H host staging length/capacity does not match authenticated workspace size"));
    }
    let existing_input_capacity = workspace.input.as_ref().map_or(0, |buffer| buffer.len());
    let existing_output_capacity = workspace.output.as_ref().map_or(0, |buffer| buffer.len());
    if workspace.capacity < target_capacity || existing_input_capacity < target_capacity || existing_output_capacity < target_capacity {
        let new_input = match DeviceVec::<ScalarField>::cuda_malloc_for_device(target_capacity, device_id) {
            Ok(buffer) => buffer,
            Err(error) => {
                if replaced_host_arena { workspace.host_values = Vec::new(); }
                return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H input workspace allocation failed: {:?}", error)));
            }
        };
        let new_output = match DeviceVec::<ScalarField>::cuda_malloc_for_device(target_capacity, device_id) {
            Ok(buffer) => buffer,
            Err(error) => {
                if replaced_host_arena { workspace.host_values = Vec::new(); }
                return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H output workspace allocation failed: {:?}", error)));
            }
        };
        if new_input.len() < target_capacity || new_output.len() < target_capacity {
            if replaced_host_arena { workspace.host_values = Vec::new(); }
            return Err(io::Error::new(io::ErrorKind::Other, "ICICLE H allocated device workspace is below authenticated workspace size"));
        }
        workspace.input = Some(new_input);
        workspace.output = Some(new_output);
        workspace.capacity = target_capacity;
    }
    let input_capacity = workspace.input.as_ref().map_or(0, |buffer| buffer.len());
    let output_capacity = workspace.output.as_ref().map_or(0, |buffer| buffer.len());
    if input_capacity < target_capacity || output_capacity < target_capacity {
        if replaced_host_arena { workspace.host_values = Vec::new(); }
        return Err(io::Error::new(io::ErrorKind::Other, "ICICLE H device workspace length is below authenticated workspace size"));
    }
    let workspace_bytes = target_capacity.saturating_mul(2).saturating_mul(mem::size_of::<ScalarField>());
    NTT_WORKSPACE_CAPACITY_BYTES.fetch_max(workspace_bytes, Ordering::Relaxed);
    NTT_MAX_SIZE.fetch_max(target_capacity, Ordering::Relaxed);
    GPU_H_WORKSPACE_PEAK_BYTES.fetch_max(workspace_bytes, Ordering::Relaxed);
    NTT_HOST_STAGING_CAPACITY_BYTES.fetch_max(workspace.host_values.capacity().saturating_mul(mem::size_of::<ScalarField>()), Ordering::Relaxed);
    let query_count = program.queries.len();
    let rows_by_input = if query_count == 0 { input_capacity.saturating_sub(fixed_input) } else { input_capacity.saturating_sub(fixed_input) / query_count };
    let rows_by_output = output_capacity / program.scratch_slots.saturating_add(1);
    let max_chunk = (rows_by_input.min(rows_by_output).min(local_row_capacity).min(1 << 16) / 32) * 32;
    if max_chunk < 32 { return unsupported_h_program(&format!("workspace_chunk_too_small input_capacity={} output_capacity={} fixed_input={} query_count={} scratch_slots={} local_row_capacity={} max_chunk={}", input_capacity, output_capacity, fixed_input, query_count, program.scratch_slots, local_row_capacity, max_chunk)); }
    let instruction_count = program.instructions.len();
    let constant_count = program.constants.len();
    let constant_words: Vec<[u32; 8]> = program.constants.iter().map(|value| canonical_field_words(value).unwrap()).collect();
    let y_words = canonical_field_words(&y).unwrap();
    let stream = CudaStream::default();
    let mut start = 0usize;
    while start < values.len() {
        let rows = max_chunk.min(values.len() - start);
        let launched_threads = rows.saturating_add(127) / 128 * 128;
        let local_upper_bytes = launched_threads.checked_mul(local_bytes_per_thread).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H local-memory bound overflow"))?;
        if local_upper_bytes > FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES { return Err(io::Error::new(io::ErrorKind::InvalidInput, "GPU H tile exceeds the reserved local-memory upper bound")); }
        let input_elements = fixed_input.checked_add(query_count.checked_mul(rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H chunk query size overflow"))?).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H chunk input size overflow"))?;
        let output_elements = rows.checked_mul(program.scratch_slots.checked_add(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H output slot count overflow"))?).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H chunk output size overflow"))?;
        for (index, instruction) in program.instructions.iter().enumerate() { workspace.host_values[index] = ScalarField::from(*instruction); }
        for (index, words) in constant_words.iter().enumerate() { workspace.host_values[instruction_count + index] = ScalarField::from(*words); }
        workspace.host_values[instruction_count + constant_count] = ScalarField::from(y_words);
        for (query_index, query) in program.queries.iter().enumerate() {
            for row_offset in 0..rows {
                let row = start + row_offset;
                let idx = row.checked_mul(part_stride).and_then(|n| n.checked_add(part_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H row index overflow"))?;
                let rotated = ((idx as i32).checked_add(query.rotation.checked_mul(rot_scale).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H rotation overflow"))?).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H rotated index overflow"))?.rem_euclid(isize) as usize) / part_stride;
                let value = query_value(query.kind, query.column, rotated, fixed, advice, instance).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H query column index is inconsistent"))?;
                // SAFETY: the preflight above established exact TypeId<F> ==
                // Bn256Fr and passed the pinned-layout Montgomery probes.
                let fr = unsafe { &*(value as *const F as *const Bn256Fr) };
                workspace.host_values[fixed_input + query_index * rows + row_offset] = ScalarField::from(fr_raw_montgomery_words(fr));
            }
        }
        let input = workspace.input.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "ICICLE H input workspace is missing"))?;
        let output = workspace.output.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "ICICLE H output workspace is missing"))?;
        if let Err(error) = input[..input_elements].copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..input_elements]), &stream) {
            let _ = synchronize_h_stream(&stream);
            return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H input copy failed: {:?}", error)));
        }
        // SAFETY: exact BN254 Fr preflight plus the size/alignment assertions
        // makes each source scalar a 32-byte raw Montgomery value readable as
        // an ICICLE 32-byte limb container; this slice is only copied H2D.
        let h_source = unsafe { std::slice::from_raw_parts(values[start..start + rows].as_ptr() as *const ScalarField, rows) };
        if let Err(error) = output[..rows].copy_from_host_async(HostSlice::from_slice(h_source), &stream) {
            let _ = synchronize_h_stream(&stream);
            return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H accumulator copy failed: {:?}", error)));
        }
        // SAFETY: the selected DeviceVec slices own contiguous, 32-byte field
        // slots; the C ABI consumes their raw u32 words until this tile sync.
        let input_ptr = unsafe { input[..input_elements].as_ptr() } as *const u32;
        let output_ptr = unsafe { output[..output_elements].as_mut_ptr() } as *mut u32;
        let launch_status = unsafe { bn254_h_eval_program(input_ptr, output_ptr, rows, instruction_count, constant_count, query_count, program.scratch_slots) };
        if launch_status != 0 {
            let _ = synchronize_h_stream(&stream);
            return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE BN254 H kernel launch failed with CUDA status {}", launch_status)));
        }
        GPU_H_EVAL_CHUNKS.fetch_add(1, Ordering::Relaxed);
        GPU_H_LOCAL_PEAK_BYTES.fetch_max(local_upper_bytes, Ordering::Relaxed);
        if let Err(error) = output[..rows].copy_to_host_async(HostSlice::from_mut_slice(&mut workspace.host_values[..rows]), &stream) {
            let _ = synchronize_h_stream(&stream);
            return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H output copy failed: {:?}", error)));
        }
        synchronize_h_stream(&stream)?;
        for output in &workspace.host_values[..rows] { if !is_reduced_bn254_limb(&words_to_u64_limbs(raw_scalar_words(output))) { return Err(io::Error::new(io::ErrorKind::InvalidData, "ICICLE H output is not a reduced BN254 Montgomery residue")); } }
        for (destination, output) in values[start..start + rows].iter_mut().zip(workspace.host_values[..rows].iter()) {
            let words = words_to_u64_limbs(raw_scalar_words(output));
            // SAFETY: the enclosing exact-Fr preflight, pinned field-layout
            // guard, and reduced-limb check above permit restoring these raw
            // Montgomery limbs into this exact Fr value.
            unsafe { std::ptr::write(destination as *mut F as *mut [u64; 4], words); }
        }
        start += rows;
    }
    GPU_H_EVAL_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_H_EVAL_ROWS.fetch_add(values.len() as u64, Ordering::Relaxed);
    let _ = registers_per_thread;
    Ok(true)
}

fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32; 8]) -> <C as CurveAffine>::Base {
    let t: &[<<C as CurveAffine>::Base as PrimeField>::Repr] =
        unsafe { mem::transmute(&u32_arr[..]) };
    return PrimeField::from_repr(t[0]).unwrap();
}

fn icicle_scalars_from_c_scalars<G: PrimeField>(coeffs: &[G]) -> Vec<ScalarField> {
    coeffs.iter().map(|coef| {
        let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
        ScalarField::from(repr)
    }).collect()
}

fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<Affine<CurveCfg>> {
    assert!(is_gpu_supported_curve::<C>(), "formal ICICLE MSM received an unsupported curve");
    bases.iter().map(|p| {
        if bool::from(p.is_identity()) { return Affine::<CurveCfg>::zero(); }
        let coordinates = p.coordinates().unwrap();
        let x_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.x().to_repr()) };
        let y_repr: [u32; 8] = unsafe { mem::transmute_copy(&coordinates.y().to_repr()) };

        Affine::<CurveCfg>::from_limbs(x_repr, y_repr)
    }).collect()
}

fn c_from_icicle_point<C: CurveAffine>(point: &G1Projective) -> C::Curve {
    if point.z == G1Projective::zero().z { return C::Curve::identity(); }
    let (x, y) = {
        let affine: Affine<CurveCfg> = Affine::<CurveCfg>::from(*point);
        if affine == Affine::<CurveCfg>::zero() { return C::Curve::identity(); }

        (
            repr_from_u32::<C>(&affine.x.into()),
            repr_from_u32::<C>(&affine.y.into()),
        )
    };

    let affine = C::from_xy(x, y);

    return affine.unwrap().to_curve();
}

/// Perform the native ICICLE multiexponentiation and synchronize its result.
pub fn multiexp_on_device<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    #[cfg(feature = "formal-gpu-msm")]
    let _formal_msm_guard = FORMAL_GPU_MSM_GATE.lock().expect("formal ICICLE MSM gate poisoned");
    assert_eq!(coeffs.len(), bases.len(), "native ICICLE MSM scalar/base length mismatch");
    assert!(!coeffs.is_empty(), "native ICICLE MSM input must not be empty");
    assert!(is_gpu_supported_curve::<C>(), "native ICICLE MSM received an unsupported curve");
    assert!(is_gpu_supported_field(&coeffs[0]), "native ICICLE MSM received an unsupported scalar field");
    select_device();
    multiexp_on_device_inner(coeffs, bases)
}

fn multiexp_on_device_inner<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
    let mut result = C::Curve::identity();
    for (scalar_chunk, base_chunk) in coeffs
        .chunks(MSM_MAX_CHUNK_SCALARS)
        .zip(bases.chunks(MSM_MAX_CHUNK_SCALARS))
    {
        let scalar_binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(scalar_chunk);
        let scalar_slice = HostSlice::from_slice(&scalar_binding[..]);
        let point_binding = icicle_points_from_c(base_chunk);
        let point_slice = HostSlice::from_slice(&point_binding[..]);

        let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
        let mut cfg = msm::MSMConfig::default_for_device(DEVICE_ID.load(Ordering::Relaxed));
        cfg.c = 16;
        msm::msm(scalar_slice, point_slice, &cfg, &mut msm_results[..]).unwrap();

        let mut msm_host_result = vec![G1Projective::zero(); 1];
        msm_results
            .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
            .unwrap();
        CudaStream::default().synchronize().expect("ICICLE CUDA MSM chunk synchronization failed");
        GPU_MSM_CHUNK_CALLS.fetch_add(1, Ordering::Relaxed);
        SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
        result += c_from_icicle_point::<C>(&msm_host_result[0]);
    }
    MSM_OPERATIONS.fetch_add(1, Ordering::Relaxed);
    result
}

/// Return whether the current formal Prove scope selected ICICLE MSM for this
/// field and size. Other scopes keep using the original best_multiexp path.
pub(crate) fn formal_gpu_msm_selected<C: CurveAffine + 'static>(sample: &C::Scalar, size: usize) -> bool {
    native_prove_ntt_scope_active()
        && env::var_os("ENABLE_ICICLE_GPU").is_some()
        && !should_use_cpu_msm(size)
        && is_gpu_supported_curve::<C>()
        && is_gpu_supported_field(sample)
}

fn selected_msm_device_identity() -> MsmDeviceIdentity {
    select_device();
    let ordinal = DEVICE_ID.load(Ordering::Relaxed);
    let uuid = device::get_device_uuid(ordinal)
        .expect("ICICLE resident MSM device UUID query failed");
    MsmDeviceIdentity { ordinal, uuid }
}

fn prepare_resident_msm_basis<C: CurveAffine>(
    cache: &mut PreparedMsmCache,
    identity: &MsmDeviceIdentity,
    role: MsmBasisRole,
    chunk_index: usize,
    bases: &[C],
    factor: usize,
) -> bool {
    let key = MsmCacheKey { role, chunk_index, factor };
    if let Some(entry) = cache.entries.get(&key) {
        assert_eq!(entry.device, *identity, "resident MSM cache has the wrong CUDA device identity");
        assert_eq!(entry.base_len, bases.len(), "resident MSM cache basis chunk length changed");
        assert_eq!(entry.factor, factor, "resident MSM cache precompute factor changed");
        GPU_MSM_PRECOMPUTED_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
        return true;
    }

    let point_binding = icicle_points_from_c(bases);
    let point_slice = HostSlice::from_slice(&point_binding[..]);
    let point_count = point_binding.len();
    let output_len = point_count.checked_mul(factor).expect("ICICLE resident MSM precompute size overflow");
    let mut resident = DeviceBoundVec::<Affine<CurveCfg>>::allocate(output_len, identity.ordinal);
    let mut cfg = msm::MSMConfig::default_for_device(identity.ordinal);
    cfg.c = 16;
    cfg.precompute_factor = i32::try_from(factor).expect("ICICLE MSM precompute factor exceeds i32");
    msm::precompute_points::<CurveCfg>(
        point_slice,
        i32::try_from(point_count).expect("ICICLE MSM chunk length exceeds i32"),
        &cfg,
        resident.device_slice_mut(),
    ).expect("ICICLE resident MSM point precomputation failed");
    // precompute_points is asynchronous in this ICICLE revision. Keep both the
    // source points and cache mutex alive until the prepared basis is complete.
    CudaStream::default().synchronize().expect("ICICLE resident MSM precomputation synchronization failed");
    GPU_MSM_PRECOMPUTE_CALLS.fetch_add(1, Ordering::Relaxed);
    SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
    cache.entries.insert(key, PreparedMsmBasis {
        device: identity.clone(),
        base_len: bases.len(),
        factor,
        points: resident,
    });
    false
}

fn multiexp_on_device_cached_with_factor<C: CurveAffine>(
    coeffs: &[C::Scalar],
    bases: &[C],
    cache: &Arc<Mutex<PreparedMsmCache>>,
    role: MsmBasisRole,
    factor: usize,
) -> C::Curve {
    #[cfg(feature = "formal-gpu-msm")]
    let _formal_msm_guard = FORMAL_GPU_MSM_GATE.lock().expect("formal ICICLE MSM gate poisoned");
    assert!(!coeffs.is_empty(), "native ICICLE cached MSM input must not be empty");
    assert!(bases.len() >= coeffs.len(), "native ICICLE cached MSM basis is shorter than its scalars");
    assert!(is_gpu_supported_curve::<C>(), "native ICICLE cached MSM received an unsupported curve");
    assert!(is_gpu_supported_field(&coeffs[0]), "native ICICLE cached MSM received an unsupported scalar field");
    assert!(factor > 0 && factor <= i32::MAX as usize, "native ICICLE cached MSM factor is invalid");
    let identity = selected_msm_device_identity();
    let mut cache = cache.lock().expect("ParamsKZG resident MSM cache poisoned");
    if cache.active_device.as_ref() != Some(&identity) {
        cache.entries.clear();
        cache.active_device = Some(identity.clone());
    }

    let mut result = C::Curve::identity();
    let mut offset = 0usize;
    for (chunk_index, scalar_chunk) in coeffs.chunks(MSM_MAX_CHUNK_SCALARS).enumerate() {
        let available_basis_len = bases.len().saturating_sub(offset).min(MSM_MAX_CHUNK_SCALARS);
        let basis_chunk = &bases[offset..offset + scalar_chunk.len()];
        if available_basis_len == scalar_chunk.len() {
            prepare_resident_msm_basis(&mut cache, &identity, role, chunk_index, basis_chunk, factor);
            let prepared = cache.entries.get(&MsmCacheKey { role, chunk_index, factor })
                .expect("resident MSM cache entry was not installed");
            assert_eq!(prepared.base_len, scalar_chunk.len(), "resident MSM cache length differs from scalar chunk");
            let scalar_binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(scalar_chunk);
            let scalar_slice = HostSlice::from_slice(&scalar_binding[..]);
            let mut result_device = DeviceVec::<G1Projective>::cuda_malloc_for_device(1, identity.ordinal)
                .expect("ICICLE resident MSM result allocation failed");
            let mut cfg = msm::MSMConfig::default_for_device(identity.ordinal);
            cfg.c = 16;
            cfg.precompute_factor = i32::try_from(factor).expect("ICICLE MSM precompute factor exceeds i32");
            msm::msm(scalar_slice, prepared.points.device_slice(), &cfg, &mut result_device[..])
                .expect("ICICLE resident MSM execution failed");
            let mut host_result = [G1Projective::zero()];
            result_device.copy_to_host(HostSlice::from_mut_slice(&mut host_result[..]))
                .expect("ICICLE resident MSM result copy failed");
            CudaStream::default().synchronize().expect("ICICLE resident MSM chunk synchronization failed");
            GPU_MSM_CHUNK_CALLS.fetch_add(1, Ordering::Relaxed);
            GPU_MSM_PRECOMPUTED_CHUNK_CALLS.fetch_add(1, Ordering::Relaxed);
            SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
            result += c_from_icicle_point::<C>(&host_result[0]);
        } else {
            // A strict polynomial prefix has a shorter scalar tail than the
            // corresponding SRS chunk. Keep that tail on the ordinary GPU MSM.
            let scalar_binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(scalar_chunk);
            let scalar_slice = HostSlice::from_slice(&scalar_binding[..]);
            let point_binding = icicle_points_from_c(basis_chunk);
            let point_slice = HostSlice::from_slice(&point_binding[..]);
            let mut result_device = DeviceVec::<G1Projective>::cuda_malloc_for_device(1, identity.ordinal)
                .expect("ICICLE partial resident MSM result allocation failed");
            let mut cfg = msm::MSMConfig::default_for_device(identity.ordinal);
            cfg.c = 16;
            msm::msm(scalar_slice, point_slice, &cfg, &mut result_device[..])
                .expect("ICICLE partial resident MSM execution failed");
            let mut host_result = [G1Projective::zero()];
            result_device.copy_to_host(HostSlice::from_mut_slice(&mut host_result[..]))
                .expect("ICICLE partial resident MSM result copy failed");
            CudaStream::default().synchronize().expect("ICICLE partial resident MSM synchronization failed");
            GPU_MSM_CHUNK_CALLS.fetch_add(1, Ordering::Relaxed);
            SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
            result += c_from_icicle_point::<C>(&host_result[0]);
        }
        offset += scalar_chunk.len();
    }
    MSM_OPERATIONS.fetch_add(1, Ordering::Relaxed);
    result
}

/// Use a ParamsKZG-owned resident basis during formal Prove. The fixed
/// production factor is bound by the execution policy string.
pub(crate) fn multiexp_on_device_with_basis_cache<C: CurveAffine>(
    coeffs: &[C::Scalar],
    bases: &[C],
    cache: &Arc<Mutex<PreparedMsmCache>>,
    role: MsmBasisRole,
) -> C::Curve {
    multiexp_on_device_with_basis_cache_factor(coeffs, bases, cache, role, FORMAL_GPU_MSM_PRECOMPUTE_FACTOR)
}

pub(crate) fn multiexp_on_device_with_basis_cache_factor<C: CurveAffine>(
    coeffs: &[C::Scalar],
    bases: &[C],
    cache: &Arc<Mutex<PreparedMsmCache>>,
    role: MsmBasisRole,
    factor: usize,
) -> C::Curve {
    multiexp_on_device_cached_with_factor(coeffs, bases, cache, role, factor)
}

/// Perform the native ICICLE transform with safe domain lifecycle handling.
pub fn fft_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    omega: Scalar, 
    log_n: u32, 
    inverse: bool
) {
    select_device();
    let size = scalars.len();
    assert_eq!(size, 1usize << log_n, "native ICICLE FFT shape mismatch");
    assert!((FORMAL_GPU_NTT_MIN_SIZE..=FORMAL_GPU_NTT_MAX_SIZE).contains(&size), "native ICICLE NTT size is outside the fixed formal GPU policy");
    let sample = scalars[0];
    assert!(is_gpu_supported_field(&sample), "formal ICICLE NTT received an unsupported scalar field");
    // A raw byte bridge is safe only for this pinned Fr representation. Keep
    // the canonical conversion path for any generic field that fails a probe.
    let raw_montgomery_bridge = raw_montgomery_layout_supported::<G>();
    let omega = if inverse { omega.invert().unwrap() } else { omega };
    assert!(omega.pow_vartime([1u64 << log_n]) == Scalar::ONE && omega.pow_vartime([1u64 << (log_n - 1)]) != Scalar::ONE, "invalid native ICICLE FFT root");
    let root_repr = unsafe { mem::transmute_copy(&omega.to_repr()) };
    let root = Bn256Fr::from_repr(root_repr).unwrap();
    let device_id = DEVICE_ID.load(Ordering::Relaxed);
    let cfg = NTTConfig::<'_, ScalarField>::default_for_device(device_id);
    let dir = if inverse { NTTDir::kInverse } else { NTTDir::kForward };
    let mut domains = NTT_DOMAINS.lock().expect("ICICLE NTT domain mutex poisoned");
    let domains = domains.get_or_insert_with(HashMap::new);
    let replace = domains.get(&device_id).map_or(true, |(size, previous)| *size < log_n || previous.pow_vartime([1u64 << (*size - log_n)]) != root);
    if replace {
        CudaStream::default().synchronize().expect("ICICLE NTT domain synchronization failed");
        if domains.contains_key(&device_id) { release_domain::<ScalarField>(&cfg.ctx).expect("ICICLE NTT domain release failed"); }
        let primitive = icicle_scalars_from_c_scalars(&[root]);
        initialize_domain(primitive[0], &cfg.ctx, false).expect("ICICLE NTT domain initialization failed");
        domains.insert(device_id, (log_n, root));
        let domain_size = 1usize << log_n;
        let domain_bytes = (domain_size + 1 + 64 + 6) * mem::size_of::<ScalarField>();
        NTT_DOMAIN_CAPACITY_BYTES.fetch_max(domain_bytes, Ordering::Relaxed);
    }
    let mut workspaces = NTT_WORKSPACES.lock().expect("ICICLE NTT workspace mutex poisoned");
    let workspaces = workspaces.get_or_insert_with(HashMap::new);
    let workspace = workspaces.entry(device_id).or_insert_with(NttWorkspace::new);
    if workspace.capacity < size {
        workspace.input = None;
        workspace.output = None;
        workspace.capacity = 0;
        let input = DeviceVec::<ScalarField>::cuda_malloc_for_device(size, device_id).expect("ICICLE NTT input buffer allocation failed");
        let output = DeviceVec::<ScalarField>::cuda_malloc_for_device(size, device_id).expect("ICICLE NTT output buffer allocation failed");
        workspace.input = Some(input);
        workspace.output = Some(output);
        workspace.capacity = size;
        NTT_WORKSPACE_CAPACITY_BYTES.fetch_max(2 * size * mem::size_of::<ScalarField>(), Ordering::Relaxed);
    }
    if workspace.host_values.capacity() < size {
        workspace.host_values = Vec::new();
        workspace.host_values = Vec::with_capacity(size);
    }
    if workspace.host_values.len() < size { workspace.host_values.resize_with(size, || ScalarField::from([0u32; 8])); }
    if raw_montgomery_bridge {
        let byte_len = size.checked_mul(mem::size_of::<Bn256Fr>()).expect("ICICLE NTT raw input byte length overflow");
        assert_eq!(mem::size_of::<G>(), mem::size_of::<Bn256Fr>(), "guarded BN254 Fr input size changed");
        assert_eq!(mem::size_of::<ScalarField>(), mem::size_of::<Bn256Fr>(), "ICICLE BN254 scalar size changed");
        // SAFETY: the exact TypeId, target-endian, modulus, and ZERO/ONE/-ONE
        // raw-Montgomery probes above establish this G as the pinned Fr with
        // a 32-byte Montgomery representation. The shared host arena is an
        // initialized sequence of 32-byte ICICLE limbs; byte copying avoids
        // imposing Fr's 8-byte alignment on ScalarField's 4-byte alignment.
        // The source slice and shared staging allocation do not overlap.
        unsafe {
            std::ptr::copy_nonoverlapping(
                scalars.as_ptr().cast::<u8>(),
                workspace.host_values[..size].as_mut_ptr().cast::<u8>(),
                byte_len,
            );
        }
    } else {
        workspace.host_values[..size].iter_mut().zip(scalars.iter()).for_each(|(output, scalar)| {
            let repr: [u32; 8] = unsafe { mem::transmute_copy(&scalar.to_repr()) };
            *output = ScalarField::from(repr);
        });
    }
    NTT_HOST_STAGING_CAPACITY_BYTES.fetch_max(workspace.host_values.capacity() * mem::size_of::<ScalarField>(), Ordering::Relaxed);
    let stream = CudaStream::default();
    workspace.input.as_mut().expect("ICICLE NTT input buffer is missing")[..size].copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..size]), &stream).expect("ICICLE NTT host-to-device copy failed");
    let mut cfg = cfg;
    cfg.is_async = true;
    ntt::<ScalarField, ScalarField>(&workspace.input.as_ref().unwrap()[..size], dir, &cfg, &mut workspace.output.as_mut().unwrap()[..size]).expect("ICICLE NTT execution failed");
    workspace.output.as_ref().unwrap()[..size].copy_to_host_async(HostSlice::from_mut_slice(&mut workspace.host_values[..size]), &stream).expect("ICICLE NTT device-to-host copy failed");
    stream.synchronize().expect("ICICLE NTT operation synchronization failed");
    if raw_montgomery_bridge {
        // ICICLE's NTT field operations use canonical Barrett residues and
        // are linear, so raw Fr inputs R*x produce raw Fr outputs R*FFT(x).
        // Check every active output before restoring any Fr value; this keeps
        // the old from_repr(...).unwrap() fail-closed behavior without an
        // elementwise serialize/deserialize pass.
        let outputs_are_reduced = workspace.host_values[..size].iter().all(|scalar| {
            is_reduced_bn254_limb(&words_to_u64_limbs(raw_scalar_words(scalar)))
        });
        assert!(outputs_are_reduced, "ICICLE NTT returned a non-reduced BN254 scalar");
        let byte_len = size.checked_mul(mem::size_of::<Bn256Fr>()).expect("ICICLE NTT raw output byte length overflow");
        // SAFETY: the preceding per-element reduced check establishes every
        // output as a valid canonical representative of R*FFT(x), which is
        // precisely the pinned Fr Montgomery layout guarded above. Bytewise
        // copying avoids alignment assumptions and touches only active rows.
        unsafe {
            std::ptr::copy_nonoverlapping(
                workspace.host_values[..size].as_ptr().cast::<u8>(),
                scalars.as_mut_ptr().cast::<u8>(),
                byte_len,
            );
        }
        // ICICLE normalizes inverse NTTs; Halo2's FFT interface leaves them
        // unnormalized and the domain iFFT applies its own 1/N divisor.
        if inverse {
            let scale = G::from(size as u64);
            scalars.iter_mut().for_each(|value| *value *= scale);
        }
    } else {
        scalars.iter_mut().zip(workspace.host_values[..size].iter()).for_each(|(output, scalar)| {
            let repr: G::Repr = unsafe { mem::transmute_copy(scalar) };
            let mut value = G::from_repr(repr).unwrap();
            // ICICLE normalizes inverse NTTs; Halo2's FFT interface leaves them
            // unnormalized and the domain iFFT applies its own 1/N divisor.
            if inverse {
                value *= G::from(size as u64);
            }
            *output = value;
        });
    }
    NTT_CALLS.fetch_add(1, Ordering::Relaxed);
    GPU_NTT_ELEMENTS.fetch_add(size as u64, Ordering::Relaxed);
    SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
    let cached_domain_size = domains
        .get(&device_id)
        .map(|(domain_log, _)| 1usize << *domain_log)
        .unwrap_or(0);
    let retained_size = workspace.capacity.max(cached_domain_size).max(size);
    NTT_MAX_SIZE.fetch_max(retained_size, Ordering::Relaxed);
    let retained_high_water = NTT_MAX_SIZE.load(Ordering::Relaxed);
    let size_of_scalar = mem::size_of::<ScalarField>();
    let device_peak_upper_bound = if retained_high_water == 0 {
        0
    } else {
        retained_high_water
            .saturating_mul(3 * size_of_scalar)
            .saturating_add((1 + 64 + 6) * size_of_scalar)
            .saturating_add(FORMAL_GPU_NTT_INIT_SCRATCH_BYTES)
    };
    NTT_DEVICE_PEAK_UPPER_BOUND_BYTES.fetch_max(device_peak_upper_bound, Ordering::Relaxed);
}

#[cfg(all(test, feature = "formal-gpu-msm"))]
mod h_eval_gpu_fixture {
    use super::*;
    use std::time::Instant;

    fn random_full_field(state: &mut u64) -> Bn256Fr {
        loop {
            let mut repr = <Bn256Fr as PrimeField>::Repr::default();
            for chunk in repr.as_mut().chunks_exact_mut(8) {
                *state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut word = *state;
                word = (word ^ (word >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                word = (word ^ (word >> 27)).wrapping_mul(0x94d049bb133111eb);
                word ^= word >> 31;
                chunk.copy_from_slice(&word.to_le_bytes());
            }
            if let Some(value) = Option::<Bn256Fr>::from(Bn256Fr::from_repr(repr)) { return value; }
        }
    }

    fn rotated_row(row: usize, part: usize, stride: usize, rotation: i32, domain: usize) -> usize {
        ((row * stride + part) as i32 + rotation * stride as i32).rem_euclid(domain as i32) as usize / stride
    }

    fn cpu_row_oracle(initial: Bn256Fr, fixed: Bn256Fr, advice: Bn256Fr, instance: Bn256Fr, constant: Bn256Fr, y: Bn256Fr) -> Bn256Fr {
        let slot0 = fixed + constant;
        let slot1 = slot0 - advice;
        let slot2 = instance * instance;
        let slot2 = slot2 * slot2;
        let slot0 = slot0 + slot0;
        let slot3 = -slot2;
        (initial * y + slot1) * y + slot3
    }

    #[test]
    #[ignore = "manual CUDA numeric fixture; run only after the shared code batch is complete"]
    fn fused_h_rows_match_full_field_cpu_oracle_across_sources_parts_and_tiles() {
        assert!(env::var_os("ENABLE_ICICLE_GPU").is_some(), "set ENABLE_ICICLE_GPU=1 for this CUDA fixture");
        initialize_device(0).expect("initialize ICICLE GPU for H fixture");
        let device_id = DEVICE_ID.load(Ordering::Relaxed);
        CudaStream::default().synchronize().expect("synchronize before H fixture workspace reset");
        {
            let mut workspaces = NTT_WORKSPACES.lock().expect("NTT workspace mutex");
            if let Some(workspaces) = workspaces.as_mut() {
                workspaces.remove(&device_id);
            }
        }
        reset_execution_operation_stats();
        GPU_H_WORKSPACE_PEAK_BYTES.store(0, Ordering::Relaxed);
        GPU_H_LOCAL_PEAK_BYTES.store(0, Ordering::Relaxed);
        GPU_H_RESOURCE_REPORT.store(0, Ordering::Relaxed);

        let mut state = 0x4d595df4d0f33173u64;
        let rows = (1 << 16) + 37;
        let stride = 8;
        let domain = rows * stride;
        assert!(rows % 128 != 0, "exercise a tail block");
        let mut fixed_values: Vec<_> = (0..rows).map(|_| random_full_field(&mut state)).collect();
        let mut advice_values: Vec<_> = (0..rows).map(|_| random_full_field(&mut state)).collect();
        let mut instance_values: Vec<_> = (0..rows).map(|_| random_full_field(&mut state)).collect();
        fixed_values[0] = Bn256Fr::ZERO;
        fixed_values[1] = Bn256Fr::ONE;
        fixed_values[2] = -Bn256Fr::ONE;
        fixed_values[3] = Bn256Fr::ZERO;
        fixed_values[4] = Bn256Fr::ONE;
        fixed_values[5] = -Bn256Fr::ONE;
        advice_values[0] = Bn256Fr::ONE;
        advice_values[1] = -Bn256Fr::ONE;
        advice_values[rows - 5] = Bn256Fr::ZERO;
        advice_values[rows - 4] = Bn256Fr::ONE;
        advice_values[rows - 3] = -Bn256Fr::ONE;
        instance_values[0] = Bn256Fr::ZERO;
        instance_values[1] = Bn256Fr::ONE;
        instance_values[2] = -Bn256Fr::ONE;
        instance_values[rows - 1] = Bn256Fr::ZERO;
        instance_values[rows - 2] = Bn256Fr::ONE;
        instance_values[rows - 3] = -Bn256Fr::ONE;
        assert_eq!(bn254_fr_raw_montgomery(&Bn256Fr::ZERO), [0; 4]);
        assert_eq!(bn254_fr_raw_montgomery(&Bn256Fr::ONE), BN254_MONTGOMERY_R_LIMBS);
        assert_eq!(bn254_fr_raw_montgomery(&-Bn256Fr::ONE), BN254_NEG_MONTGOMERY_R_LIMBS);
        let fixed = vec![Polynomial::<Bn256Fr, LagrangeCoeff>::from_vec(fixed_values)];
        let advice = vec![Polynomial::<Bn256Fr, LagrangeCoeff>::from_vec(advice_values)];
        let instance = vec![Polynomial::<Bn256Fr, LagrangeCoeff>::from_vec(instance_values)];
        let constant = random_full_field(&mut state);
        let y = random_full_field(&mut state);
        let program = GateProgram {
            instructions: vec![
                [GATE_OP_LOAD_QUERY, 0, 0, 0, 0, 0, 0, 0],
                [GATE_OP_LOAD_QUERY, 1, 1, 0, 0, 0, 0, 0],
                [GATE_OP_LOAD_QUERY, 2, 2, 0, 0, 0, 0, 0],
                [GATE_OP_LOAD_CONST, 4, 0, 0, 0, 0, 0, 0],
                [GATE_OP_ADD, 0, 0, 4, 0, 0, 0, 0],
                [GATE_OP_SUB, 1, 0, 1, 0, 0, 0, 0],
                [GATE_OP_MUL, 2, 2, 2, 0, 0, 0, 0],
                [GATE_OP_SQUARE, 2, 2, 0, 0, 0, 0, 0],
                [GATE_OP_DOUBLE, 0, 0, 0, 0, 0, 0, 0],
                [GATE_OP_NEGATE, 3, 2, 0, 0, 0, 0, 0],
                [GATE_OP_FOLD, 1, 0, 0, 0, 0, 0, 0],
                [GATE_OP_FOLD, 3, 0, 0, 0, 0, 0, 0],
            ],
            constants: vec![constant],
            queries: vec![
                crate::h_eval::GateQuery { kind: 0, column: 0, rotation: 3 },
                crate::h_eval::GateQuery { kind: 1, column: 0, rotation: -5 },
                crate::h_eval::GateQuery { kind: 2, column: 0, rotation: 1 },
            ],
            scratch_slots: 5,
            fold_count: 2,
        };

        let _scope = native_prove_ntt_scope();
        let chunks_before = execution_stats().20;
        for part in 0..stride {
            let mut initial: Vec<_> = (0..rows).map(|_| random_full_field(&mut state)).collect();
            initial[0] = Bn256Fr::ZERO;
            initial[1] = Bn256Fr::ONE;
            initial[2] = -Bn256Fr::ONE;
            initial[rows - 1] = Bn256Fr::ONE;
            let cpu_started = if part == 0 { Some(Instant::now()) } else { None };
            let expected: Vec<_> = (0..rows).map(|row| {
                let fixed_row = rotated_row(row, part, stride, 3, domain);
                let advice_row = rotated_row(row, part, stride, -5, domain);
                let instance_row = rotated_row(row, part, stride, 1, domain);
                cpu_row_oracle(initial[row], fixed[0][fixed_row], advice[0][advice_row], instance[0][instance_row], constant, y)
            }).collect();
            let cpu_elapsed = cpu_started.map(|started| started.elapsed());
            record_gpu_h_eval_group();
            let gpu_started = if part == 0 { Some(Instant::now()) } else { None };
            let gpu_succeeded = try_evaluate_h_program(&program, &fixed, &advice, &instance, part, stride, stride as i32, domain as i32, y, &mut initial).expect("execute fused H rows");
            let gpu_elapsed = gpu_started.map(|started| started.elapsed());
            assert!(gpu_succeeded);
            assert_eq!(initial, expected, "full-field GPU result differs for part {}", part);
            if part == 0 {
                eprintln!(
                    "[ezkl-gpu-h-fixture] rows={} part={} cpu_oracle_elapsed_us={} gpu_adapter_elapsed_us={} measurement=single-representative-operation cpu_includes_expected-vector-allocation gpu_includes_first-call-host-and-device-workspace-allocation-query-bridge-all-tile-H2D-kernel-D2H-and-sync excludes_initialize_device-and-fixture-input-generation",
                    rows,
                    part,
                    cpu_elapsed.expect("timed representative CPU oracle").as_micros(),
                    gpu_elapsed.expect("timed representative GPU adapter").as_micros(),
                );
            }
        }
        let chunks_after_large = execution_stats().20;
        assert!(chunks_after_large - chunks_before > stride as u64, "large fixture must exercise multiple tiles and per-part tail tiles");
        let workspace_high_water = execution_stats().22;
        assert!(workspace_high_water > 0, "H fixture must record borrowed NTT workspace bytes");
        let mut small = vec![Bn256Fr::ZERO; 32];
        let small_fixed = vec![Polynomial::<Bn256Fr, LagrangeCoeff>::from_vec(fixed[0][0..32].to_vec())];
        let small_advice = vec![Polynomial::<Bn256Fr, LagrangeCoeff>::from_vec(advice[0][0..32].to_vec())];
        let small_instance = vec![Polynomial::<Bn256Fr, LagrangeCoeff>::from_vec(instance[0][0..32].to_vec())];
        let small_domain = small.len() * stride;
        let small_expected: Vec<_> = (0..small.len()).map(|row| {
            let fixed_row = rotated_row(row, stride - 1, stride, 3, small_domain);
            let advice_row = rotated_row(row, stride - 1, stride, -5, small_domain);
            let instance_row = rotated_row(row, stride - 1, stride, 1, small_domain);
            cpu_row_oracle(small[row], small_fixed[0][fixed_row], small_advice[0][advice_row], small_instance[0][instance_row], constant, y)
        }).collect();
        record_gpu_h_eval_group();
        assert!(try_evaluate_h_program(&program, &small_fixed, &small_advice, &small_instance, stride - 1, stride, stride as i32, small_domain as i32, y, &mut small).expect("reuse large H workspace for small input"));
        assert_eq!(small, small_expected, "small reused-workspace result differs from CPU oracle");
        assert_eq!(execution_stats().22, workspace_high_water, "small H call must reuse, not grow, the borrowed NTT workspace");
        let stats = execution_stats();
        assert_eq!(stats.24, stats.17 + stats.19, "H attempted groups must equal successful calls plus CPU fallbacks");
    }
}
