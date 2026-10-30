use group::{ff::{PrimeField, Field}, Group};
use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarField};
use halo2curves::bn256::{Fr as Bn256Fr, G1Affine as Bn256G1Affine};
use icicle_cuda_runtime::memory::{DeviceVec, HostOrDeviceSlice, HostSlice};
use crate::h_eval::{GateProgram, GATE_OP_ADD, GATE_OP_COMMIT_BLOCK, GATE_OP_DOUBLE, GATE_OP_FOLD, GATE_OP_FOLD_BLOCK, GATE_OP_LOAD_CONST, GATE_OP_LOAD_QUERY, GATE_OP_MUL, GATE_OP_MV_LOOKUP_FOLD, GATE_OP_NEGATE, GATE_OP_SCALE_ACCUMULATOR, GATE_OP_SQUARE, GATE_OP_STORE_OUTPUT, GATE_OP_SUB};
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
use std::sync::{Arc, Mutex, MutexGuard};
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
static GPU_H_TRANSFER_SYNCS: AtomicU64 = AtomicU64::new(0);
static GPU_H_WORKSPACE_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
static GPU_H_RESIDENT_CURRENT_BYTES: AtomicUsize = AtomicUsize::new(0);
static GPU_H_RESIDENT_PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);
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
    coset_powers: Option<DeviceVec<ScalarField>>,
    coset_key: Option<(usize, [u32; 8])>,
    host_values: Vec<ScalarField>,
}

/// A device allocation owned by the selected CUDA device and released while
/// that device is current. These buffers are scoped to one proof-part H
/// accumulation and never enter the persistent MSM cache.
struct OwnerBoundHBuffer {
    device_id: usize,
    bytes: usize,
    buffer: Option<DeviceVec<ScalarField>>,
}

impl OwnerBoundHBuffer {
    fn try_allocate(elements: usize, device_id: usize) -> io::Result<Self> {
        let bytes = elements.checked_mul(mem::size_of::<ScalarField>()).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "resident H allocation size overflow")
        })?;
        if elements == 0 || bytes == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "resident H allocation must be nonempty"));
        }
        let buffer = DeviceVec::<ScalarField>::cuda_malloc_for_device(elements, device_id)
            .map_err(|error| io::Error::new(io::ErrorKind::Other, format!("resident H device allocation failed: {:?}", error)))?;
        let previous = GPU_H_RESIDENT_CURRENT_BYTES.fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| current.checked_add(bytes))
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "resident H byte counter overflow"))?;
        let live = previous + bytes;
        GPU_H_RESIDENT_PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
        Ok(Self { device_id, bytes, buffer: Some(buffer) })
    }

    fn slice(&self) -> &icicle_cuda_runtime::memory::DeviceSlice<ScalarField> {
        self.buffer.as_ref().expect("resident H device allocation was released")
    }

    fn slice_mut(&mut self) -> &mut icicle_cuda_runtime::memory::DeviceSlice<ScalarField> {
        self.buffer.as_mut().expect("resident H device allocation was released")
    }
}

impl Drop for OwnerBoundHBuffer {
    fn drop(&mut self) {
        let Some(buffer) = self.buffer.take() else { return; };
        let previous = device::get_device().expect("cannot read CUDA device while releasing resident H workspace");
        if previous != self.device_id {
            device::set_device(self.device_id).expect("cannot select owner device while releasing resident H workspace");
        }
        drop(buffer);
        if previous != self.device_id {
            device::set_device(previous).expect("cannot restore CUDA device after releasing resident H workspace");
        }
        let previous_live = GPU_H_RESIDENT_CURRENT_BYTES.fetch_sub(self.bytes, Ordering::AcqRel);
        assert!(previous_live >= self.bytes, "resident H byte counter underflow");
    }
}

/// Per-part H accumulator kept on the selected device while a sequence of
/// multi-lookup programs is evaluated. Each program synchronizes before
/// releasing the shared NTT staging lock; CPU coset preparation can therefore
/// run between programs without racing a CUDA transfer.
pub(crate) struct ResidentHLookupSession {
    rows: usize,
    h: OwnerBoundHBuffer,
    device_id: usize,
    permutation: Option<ResidentPermutationBuffers>,
    kernel_local_bytes_per_thread: usize,
    kernel_local_row_capacity: usize,
}

struct ResidentPermutationBuffers {
    row_twiddles: OwnerBoundHBuffer,
    left: OwnerBoundHBuffer,
    right: OwnerBoundHBuffer,
}

#[derive(Clone, Copy)]
pub(crate) struct ResidentHKernelCapacity {
    local_bytes_per_thread: usize,
    row_capacity: usize,
}

impl NttWorkspace {
    fn new() -> Self { Self { capacity: 0, input: None, output: None, coset_powers: None, coset_key: None, host_values: Vec::new() } }
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
pub const FORMAL_GPU_EXECUTION_POLICY: &str = "halo2_icicle-bn254-gpu-hgraph-fused-coset-ntt-16-25-msm-lagrange-resident-f4-sparse-m-cpu32-v8";

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
/// Tuple tail (indices 17..26): H program calls, rows, preflight fallbacks,
/// successful kernel chunks, kernel syncs, borrowed NTT workspace bytes,
/// conservative local-plus-stack memory upper-bound bytes, attempted groups,
/// transfer-only syncs, and owner-bound resident H/permutation high-water bytes.
pub fn execution_stats() -> (u64, u64, u64, u64, u64, usize, u64, usize, usize, usize, usize, usize, u64, u64, u64, usize, usize, u64, u64, u64, u64, u64, usize, usize, u64, u64, usize) {
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
        GPU_H_TRANSFER_SYNCS.load(Ordering::Relaxed),
        GPU_H_RESIDENT_PEAK_BYTES.load(Ordering::Relaxed),
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
    GPU_H_TRANSFER_SYNCS.store(0, Ordering::Relaxed);
    GPU_H_RESIDENT_PEAK_BYTES.store(0, Ordering::Relaxed);
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

/// The fused coefficient-coset path requires the guarded raw BN254 Fr bridge.
pub fn is_gpu_supported_coset_field<G: PrimeField + 'static>() -> bool {
    raw_montgomery_layout_supported::<G>()
}

/// Check whether a curve type matches the BN254 G1 point representation used by ICICLE.
pub fn is_gpu_supported_curve<C: CurveAffine + 'static>() -> bool { TypeId::of::<C>() == TypeId::of::<Bn256G1Affine>() }

const BN254_MODULUS_LIMBS: [u64; 4] = [0x43e1f593f0000001, 0x2833e84879b97091, 0xb85045b68181585d, 0x30644e72e131a029];
const BN254_MONTGOMERY_R_LIMBS: [u64; 4] = [0xac96341c4ffffffb, 0x36fc76959f60cd29, 0x666ea36f7879462e, 0x0e0a77c19a07df2f];
const BN254_NEG_MONTGOMERY_R_LIMBS: [u64; 4] = [0x974bc177a0000006, 0xf13771b2da58a367, 0x51e1a2470908122e, 0x2259d6b14729c0fa];

extern "C" {
    fn bn254_h_eval_program(input: *const u32, output: *mut u32, rows: usize, instruction_count: usize, constant_count: usize, query_count: usize, scratch_slots: usize) -> i32;
    fn bn254_h_eval_mv_lookup_program(input: *const u32, h: *mut u32, scratch: *mut u32, rows: usize, instruction_count: usize, constant_count: usize, query_count: usize, scratch_slots: usize, graph_query_count: usize, lookup_count: usize) -> i32;
    fn bn254_h_permutation_scale(scale_words: *const u32, h: *mut u32, rows: usize) -> i32;
    fn bn254_h_permutation_twiddles(twiddles: *mut u32, rows: usize, row_offset: usize, omega_words: *const u32, part_stride: usize, part_index: usize) -> i32;
    fn bn254_h_permutation_begin_set(z: *const u32, l0: *const u32, h: *mut u32, left: *mut u32, right: *mut u32, rows: usize, row_offset: usize, full_size: usize, part_index: usize, part_stride: usize, rot_scale: i32, isize: i32, last_rotation: i32, set_index: usize, set_count: usize, scales: *const u32) -> i32;
    fn bn254_h_permutation_last_boundary(l_last: *const u32, h: *mut u32, right: *const u32, rows: usize, scale_words: *const u32) -> i32;
    fn bn254_h_permutation_column(source: *const u32, sigma: *const u32, row_twiddles: *const u32, left: *mut u32, right: *mut u32, rows: usize, scalars: *const u32) -> i32;
    fn bn254_h_permutation_copy_term(l_active: *const u32, left: *const u32, right: *const u32, h: *mut u32, rows: usize, scale_words: *const u32) -> i32;
    fn bn254_h_coset_scale(input_words: *mut u32, powers_words: *mut u32, row_offset: usize, rows: usize, factor_words: *const u32, generate_powers: bool) -> i32;
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
    let mut lookup_folds = 0usize;
    let mut lookup_output_stores = 0usize;
    let lookup = program.mv_lookup.as_ref();
    let weighted = program
        .instructions
        .first()
        .is_some_and(|instruction| instruction[0] == GATE_OP_SCALE_ACCUMULATOR);
    if weighted
        && !program
            .instructions
            .last()
            .is_some_and(|instruction| instruction[0] == GATE_OP_COMMIT_BLOCK)
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "GPU H weighted block program is missing its final commit",
        ));
    }
    let check_slot = |slot: u32| -> io::Result<()> { if (slot as usize) < program.scratch_slots { Ok(()) } else { Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program scratch index is out of bounds")) } };
    let check_source = |slot: u32, initialized: &[bool]| -> io::Result<()> { check_slot(slot)?; if initialized[slot as usize] { Ok(()) } else { Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program reads an uninitialized scratch slot")) } };
    for (instruction_index, instruction) in program.instructions.iter().enumerate() {
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
            if weighted || instruction[2..].iter().any(|word| *word != 0) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H fold instruction is malformed")); }
            check_source(dst, &initialized)?;
            folds += 1;
        } else if op == GATE_OP_SCALE_ACCUMULATOR {
            if !weighted || instruction_index != 0 || instruction[1] as usize >= program.constants.len() || instruction[2..].iter().any(|word| *word != 0) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H accumulator-scale instruction is malformed"));
            }
        } else if op == GATE_OP_FOLD_BLOCK {
            if !weighted || instruction[2..].iter().any(|word| *word != 0) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H block-fold instruction is malformed")); }
            check_source(dst, &initialized)?;
            folds += 1;
        } else if op == GATE_OP_COMMIT_BLOCK {
            if !weighted || instruction_index + 1 != program.instructions.len() || instruction[1] as usize >= program.constants.len() || instruction[2..].iter().any(|word| *word != 0) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H block-commit instruction is malformed"));
            }
        } else if op == GATE_OP_STORE_OUTPUT {
            let Some(lookup) = lookup else { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H output-store instruction requires multi-lookup metadata")); };
            let dst = instruction[1] as usize;
            let output_count = lookup.input_counts.iter().try_fold(0usize, |total, count| total.checked_add(count.checked_add(1)?)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H lookup output range overflows"))?;
            let output_end = lookup.output_base.checked_add(output_count).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H lookup output range overflows"))?;
            if instruction[3..].iter().any(|word| *word != 0) || dst < lookup.output_base || dst >= output_end || dst >= program.scratch_slots {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H output-store instruction is malformed"));
            }
            check_source(instruction[2], &initialized)?;
            if initialized[dst] { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H lookup output slot is written more than once")); }
            initialized[dst] = true;
            lookup_output_stores += 1;
        } else if op == GATE_OP_MV_LOOKUP_FOLD {
            let Some(lookup) = lookup else { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H multi-lookup fold requires lookup metadata")); };
            let lookup_index = instruction[3] as usize;
            let Some(input_count) = lookup.input_counts.get(lookup_index) else { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H multi-lookup index is invalid")); };
            let graph_offset = lookup.input_counts.iter().take(lookup_index).try_fold(0usize, |total, count| total.checked_add(count.checked_add(1)?)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H multi-lookup output offset overflows"))?;
            let output_start = lookup.output_base.checked_add(graph_offset).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H multi-lookup output offset overflows"))?;
            if weighted || lookup_index != lookup_folds || instruction[1] as usize != output_start || instruction[2] as usize != *input_count || instruction[4..].iter().any(|word| *word != 0) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H multi-lookup fold instruction is malformed"));
            }
            for output in output_start..=output_start + input_count {
                if !initialized.get(output).copied().unwrap_or(false) { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H multi-lookup reads an uninitialized graph output")); }
            }
            lookup_folds += 1;
            folds += 3;
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program uses an unknown opcode"));
        }
    }
    if let Some(lookup) = lookup {
        let output_count = lookup.input_counts.iter().try_fold(0usize, |total, count| total.checked_add(count.checked_add(1)?)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H lookup output count overflows"))?;
        if lookup_folds != lookup.input_counts.len() || lookup_output_stores != output_count || lookup.graph_query_count != program.queries.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H multi-lookup program metadata does not match its instruction stream"));
        }
    } else if lookup_folds != 0 || lookup_output_stores != 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H program has unbound lookup instructions"));
    }
    if folds != program.fold_count { return Err(io::Error::new(io::ErrorKind::InvalidData, "GPU H fold count does not match its instruction stream")); }
    Ok(true)
}

fn query_column_values<'a, F: PrimeField>(query_kind: u32, column: usize, fixed: &'a [Polynomial<F, LagrangeCoeff>], advice: &'a [Polynomial<F, LagrangeCoeff>], instance: &'a [Polynomial<F, LagrangeCoeff>]) -> Option<&'a [F]> {
    match query_kind { 0 => fixed.get(column).map(|poly| &poly[..]), 1 => advice.get(column).map(|poly| &poly[..]), 2 => instance.get(column).map(|poly| &poly[..]), _ => None }
}

struct ResolvedHQuery<'a> {
    values: &'a [Bn256Fr],
    rotation_offset: i32,
    rotation_rows: usize,
}

fn h_query_uses_bulk_copy(rot_scale: i32, part_stride: usize) -> bool {
    usize::try_from(rot_scale).ok() == Some(part_stride)
}

/// Pack one contiguous H-query row range directly into ICICLE's raw
/// Montgomery staging representation. The caller has already established
/// that the source field is the pinned BN254 Fr layout and has normalized
/// the query rotation for the base domain.
fn copy_bn254_h_query_chunk(source: &[Bn256Fr], start: usize, rows: usize, rotation_rows: usize, base_n: usize, destination: &mut [ScalarField]) -> io::Result<()> {
    if base_n == 0 || source.len() != base_n || start >= base_n || rows == 0 || rows > base_n || rotation_rows >= base_n || destination.len() != rows {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "ICICLE H query chunk shape is invalid"));
    }
    let origin = start.checked_add(rotation_rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ICICLE H query origin overflow"))? % base_n;
    let first_rows = rows.min(base_n - origin);
    let second_rows = rows - first_rows;
    let first_bytes = first_rows.checked_mul(mem::size_of::<ScalarField>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ICICLE H query copy size overflow"))?;
    let second_bytes = second_rows.checked_mul(mem::size_of::<ScalarField>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ICICLE H query copy size overflow"))?;

    // SAFETY: the caller established exact BN254 Fr identity and the pinned
    // raw-Montgomery representation probes before constructing `source`.
    // The compile-time size checks make both element layouts 32 bytes. The
    // slices are separately borrowed input polynomials and workspace-owned
    // staging memory, so the ranges do not overlap. Bounds and byte lengths
    // were checked above. At most one wrap is possible because rows <= N.
    unsafe {
        if first_rows != 0 {
            std::ptr::copy_nonoverlapping(source.as_ptr().add(origin).cast::<u8>(), destination.as_mut_ptr().cast::<u8>(), first_bytes);
        }
        if second_rows != 0 {
            std::ptr::copy_nonoverlapping(source.as_ptr().cast::<u8>(), destination.as_mut_ptr().add(first_rows).cast::<u8>(), second_bytes);
        }
    }
    Ok(())
}

/// Validate a complete ICICLE output chunk before replacing the caller's
/// accumulator, then restore the validated raw Montgomery values in bulk.
fn copy_validated_bn254_h_output_chunk(source: &[ScalarField], destination: &mut [Bn256Fr]) -> io::Result<()> {
    if source.len() != destination.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "ICICLE H output chunk shape is invalid"));
    }
    for output in source {
        if !is_reduced_bn254_limb(&words_to_u64_limbs(raw_scalar_words(output))) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "ICICLE H output is not a reduced BN254 Montgomery residue"));
        }
    }
    let bytes = source.len().checked_mul(mem::size_of::<ScalarField>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ICICLE H output copy size overflow"))?;
    // SAFETY: all source residues were checked against the BN254 modulus
    // above; both scalar types have compile-time-checked 32-byte layouts.
    // The source is NTT workspace staging, and the destination is the
    // caller's disjoint H accumulator slice. The checked length covers the
    // full destination and the entire chunk is validated before mutation.
    unsafe {
        std::ptr::copy_nonoverlapping(source.as_ptr().cast::<u8>(), destination.as_mut_ptr().cast::<u8>(), bytes);
    }
    Ok(())
}

/// Copy one already-validated BN254 Fr value into the ICICLE raw-Montgomery
/// staging buffer without canonical serialization.
fn copy_raw_bn254_fr(source: &Bn256Fr, destination: &mut ScalarField) {
    // SAFETY: the caller passed the pinned BN254 Fr raw-layout guard and both
    // field types have compile-time checked 32-byte layouts.
    unsafe {
        std::ptr::copy_nonoverlapping(source as *const Bn256Fr as *const u8, destination as *mut ScalarField as *mut u8, 32);
    }
}

pub(crate) fn record_gpu_h_eval_group() { GPU_H_EVAL_GROUPS.fetch_add(1, Ordering::Relaxed); }

pub(crate) fn record_gpu_h_eval_fallback_group() {
    record_gpu_h_eval_group();
    record_gpu_h_eval_fallback();
}

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

/// Synchronize H initialization/final readback transfers without counting
/// them as H-kernel launches. The aggregate native synchronization counter
/// still records every actual wait; `gpu_h_eval_syncs` remains paired with
/// successful H evaluation chunks for the strict proof receipt.
fn synchronize_h_transfer_stream(stream: &CudaStream) -> io::Result<()> {
    GPU_H_TRANSFER_SYNCS.fetch_add(1, Ordering::Relaxed);
    SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
    stream.synchronize().map_err(|e| io::Error::new(io::ErrorKind::Other, format!("ICICLE H transfer synchronization failed: {:?}", e)))
}

fn panic_after_ntt_stream_drain(stream: &CudaStream, operation: &'static str, error: impl fmt::Debug) -> ! {
    match stream.synchronize() {
        Ok(()) => panic!("ICICLE fused NTT {} failed: {:?}", operation, error),
        Err(sync_error) => panic!("ICICLE fused NTT {} failed: {:?}; draining the owner stream also failed: {:?}", operation, error, sync_error),
    }
}

/// Prepare shared H staging before the first device-side accumulator update.
/// The returned capacity is authenticated by the existing NTT workspace
/// envelope; the separate resident H allocation is accounted independently.
fn ensure_resident_h_staging(device_id: usize, rows: usize) -> io::Result<Option<usize>> {
    let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
    let workspaces = workspaces.get_or_insert_with(HashMap::new);
    let workspace = workspaces.entry(device_id).or_insert_with(NttWorkspace::new);
    let requested = workspace.capacity.max(rows).max(FORMAL_GPU_NTT_MIN_SIZE).max(workspace.host_values.capacity());
    let Some(target) = requested.checked_next_power_of_two() else { return Ok(None); };
    if target > FORMAL_GPU_NTT_MAX_SIZE { return Ok(None); }

    if workspace.host_values.capacity() != target {
        workspace.host_values = Vec::new();
        if workspace.host_values.try_reserve_exact(target).is_err() || workspace.host_values.capacity() != target {
            workspace.host_values = Vec::new();
            return Ok(None);
        }
    }
    if workspace.host_values.len() < target {
        workspace.host_values.resize_with(target, || ScalarField::from([0u32; 8]));
    }
    if workspace.host_values.len() != target || workspace.host_values.capacity() != target {
        return Ok(None);
    }

    let has_buffers = workspace.capacity >= target
        && workspace.input.as_ref().is_some_and(|buffer| buffer.len() >= target)
        && workspace.output.as_ref().is_some_and(|buffer| buffer.len() >= target)
        && workspace.coset_powers.as_ref().is_some_and(|buffer| buffer.len() >= target);
    if !has_buffers {
        workspace.input = None;
        workspace.output = None;
        workspace.coset_powers = None;
        workspace.coset_key = None;
        workspace.capacity = 0;
        let input = match DeviceVec::<ScalarField>::cuda_malloc_for_device(target, device_id) {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };
        let output = match DeviceVec::<ScalarField>::cuda_malloc_for_device(target, device_id) {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };
        let coset_powers = match DeviceVec::<ScalarField>::cuda_malloc_for_device(target, device_id) {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };
        if input.len() < target || output.len() < target || coset_powers.len() < target { return Ok(None); }
        workspace.input = Some(input);
        workspace.output = Some(output);
        workspace.coset_powers = Some(coset_powers);
        workspace.capacity = target;
    }

    let ntt_workspace_bytes = target.checked_mul(3).and_then(|n| n.checked_mul(mem::size_of::<ScalarField>()))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ICICLE NTT workspace size overflows"))?;
    let h_workspace_bytes = target.checked_mul(2).and_then(|n| n.checked_mul(mem::size_of::<ScalarField>()))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "ICICLE H staging size overflows"))?;
    NTT_WORKSPACE_CAPACITY_BYTES.fetch_max(ntt_workspace_bytes, Ordering::Relaxed);
    NTT_MAX_SIZE.fetch_max(target, Ordering::Relaxed);
    GPU_H_WORKSPACE_PEAK_BYTES.fetch_max(h_workspace_bytes, Ordering::Relaxed);
    NTT_HOST_STAGING_CAPACITY_BYTES.fetch_max(target.saturating_mul(mem::size_of::<ScalarField>()), Ordering::Relaxed);
    Ok(Some(target))
}

/// Start a BN254 resident H lookup session only when its complete persistent
/// H allocation and shared bounded staging can be admitted before any H fold.
pub(crate) fn resident_h_lookup_kernel_capacity() -> io::Result<Option<ResidentHKernelCapacity>> {
    if !native_prove_ntt_scope_active() || env::var_os("ENABLE_ICICLE_GPU").is_none() {
        return Ok(None);
    }
    let device_id = DEVICE_ID.load(Ordering::Relaxed);
    if device::set_device(device_id).is_err() { return Ok(None); }
    let mut local_bytes_per_thread = 0usize;
    let mut shared_bytes_per_block = 0usize;
    let mut stack_bytes_per_thread = 0usize;
    let mut registers_per_thread = 0i32;
    let mut max_threads_per_block = 0i32;
    let attr_status = unsafe {
        bn254_h_eval_kernel_attributes(
            &mut local_bytes_per_thread,
            &mut shared_bytes_per_block,
            &mut stack_bytes_per_thread,
            &mut registers_per_thread,
            &mut max_threads_per_block,
        )
    };
    if attr_status != 0 || shared_bytes_per_block != 0 || max_threads_per_block < 128 {
        return Ok(None);
    }
    let Some(local_bytes_per_thread) = local_bytes_per_thread
        .checked_add(stack_bytes_per_thread)
        .and_then(|bytes| bytes.checked_add(2 * mem::size_of::<ScalarField>()))
    else {
        return Ok(None);
    };
    let local_thread_capacity = if local_bytes_per_thread == 0 {
        usize::MAX
    } else {
        FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES / local_bytes_per_thread
    };
    let row_capacity = local_thread_capacity / 128 * 128;
    if row_capacity < 128 { return Ok(None); }
    let _ = registers_per_thread;
    Ok(Some(ResidentHKernelCapacity { local_bytes_per_thread, row_capacity }))
}

pub(crate) fn try_begin_resident_h_session<F: PrimeField + 'static>(values: &[F], kernel_capacity: ResidentHKernelCapacity, with_permutation: bool) -> io::Result<Option<ResidentHLookupSession>> {
    if !native_prove_ntt_scope_active() || env::var_os("ENABLE_ICICLE_GPU").is_none() { return Ok(None); }
    if !raw_montgomery_layout_supported::<F>() || values.len() < 32 { return Ok(None); }
    let device_id = DEVICE_ID.load(Ordering::Relaxed);
    if device::set_device(device_id).is_err() { return Ok(None); }
    if ensure_resident_h_staging(device_id, values.len())?.is_none() { return Ok(None); }
    let resident_elements = if with_permutation {
        values.len().checked_mul(4)
    } else {
        Some(values.len())
    }.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H and permutation allocation count overflows"))?;
    let resident_bytes = resident_elements.checked_mul(mem::size_of::<ScalarField>())
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H and permutation allocation size overflows"))?;
    let (free_bytes, total_bytes) = match device::get_memory_info() {
        Ok(memory) => memory,
        Err(error) => return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE resident H memory query failed: {:?}", error))),
    };
    let required = resident_bytes.checked_add(FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H admission size overflows"))?;
    if total_bytes == 0 || free_bytes > total_bytes || free_bytes < required {
        return Ok(None);
    }
    let mut h = match OwnerBoundHBuffer::try_allocate(values.len(), device_id) {
        Ok(buffer) => buffer,
        Err(_) => { return Ok(None); }
    };
    let permutation = if with_permutation {
        let row_twiddles = match OwnerBoundHBuffer::try_allocate(values.len(), device_id) {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };
        let left = match OwnerBoundHBuffer::try_allocate(values.len(), device_id) {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };
        let right = match OwnerBoundHBuffer::try_allocate(values.len(), device_id) {
            Ok(buffer) => buffer,
            Err(_) => return Ok(None),
        };
        Some(ResidentPermutationBuffers { row_twiddles, left, right })
    } else {
        None
    };
    let stream = CudaStream::default();
    // SAFETY: raw_montgomery_layout_supported established the exact field type
    // and Montgomery encoding before this pointer conversion.
    let values_bn = unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<Bn256Fr>(), values.len()) };
    let h_source = unsafe { std::slice::from_raw_parts(values_bn.as_ptr().cast::<ScalarField>(), values_bn.len()) };
    if let Err(error) = h.slice_mut().copy_from_host_async(HostSlice::from_slice(h_source), &stream) {
        synchronize_h_transfer_stream(&stream).map_err(|sync_error| io::Error::new(io::ErrorKind::Other, format!("resident H initialization transfer failed: {:?}; draining the owner stream also failed: {}", error, sync_error)))?;
        return Err(io::Error::new(io::ErrorKind::Other, format!("resident H initialization transfer failed after device mutation began: {:?}", error)));
    }
    synchronize_h_transfer_stream(&stream)?;
    Ok(Some(ResidentHLookupSession {
        rows: values_bn.len(),
        h,
        device_id,
        permutation,
        kernel_local_bytes_per_thread: kernel_capacity.local_bytes_per_thread,
        kernel_local_row_capacity: kernel_capacity.row_capacity,
    }))
}

impl ResidentHLookupSession {
    fn resident_transfer_error(stream: &CudaStream, operation: &'static str, error: impl std::fmt::Debug) -> io::Error {
        let transfer_error = io::Error::new(io::ErrorKind::Other, format!("resident H {} transfer failed: {:?}", operation, error));
        match synchronize_h_transfer_stream(stream) {
            Ok(()) => transfer_error,
            Err(sync_error) => io::Error::new(io::ErrorKind::Other, format!("{}; draining the owner stream also failed: {}", transfer_error, sync_error)),
        }
    }

    fn resident_kernel_launch_error(stream: &CudaStream, operation: &'static str, status: i32) -> io::Error {
        let launch_error = io::Error::new(io::ErrorKind::Other, format!("resident H {} kernel launch failed with CUDA status {}", operation, status));
        match synchronize_h_transfer_stream(stream) {
            Ok(()) => launch_error,
            Err(sync_error) => io::Error::new(io::ErrorKind::Other, format!("{}; draining the owner stream also failed: {}", launch_error, sync_error)),
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot_for_test<F: PrimeField + 'static>(&mut self) -> io::Result<Vec<F>> {
        if !raw_montgomery_layout_supported::<F>() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H test snapshot field layout is unsupported"));
        }
        let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
        let workspace = workspaces.as_mut().and_then(|all| all.get_mut(&self.device_id))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident H test snapshot staging workspace is missing"))?;
        if workspace.host_values.len() < self.rows {
            return Err(io::Error::new(io::ErrorKind::Other, "resident H test snapshot staging capacity is too small"));
        }
        let stream = CudaStream::default();
        let destination = HostSlice::from_mut_slice(&mut workspace.host_values[..self.rows]);
        if let Err(error) = self.h.slice().copy_to_host_async(destination, &stream) {
            return Err(Self::resident_transfer_error(&stream, "test snapshot readback", error));
        }
        synchronize_h_transfer_stream(&stream)?;
        let mut values = vec![F::ZERO; self.rows];
        // SAFETY: the field raw-layout guard and exact vector length passed
        // above, matching the production finish() conversion.
        let values_bn = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Bn256Fr>(), values.len()) };
        copy_validated_bn254_h_output_chunk(&workspace.host_values[..self.rows], values_bn)?;
        Ok(values)
    }

    fn permutation_scale<F: Field>(y: F, total_terms: usize, term_index: usize) -> io::Result<F> {
        let exponent = total_terms.checked_sub(term_index.checked_add(1).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "resident permutation term index overflows")
        })?).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation term exceeds fold length"))?;
        let exponent = u64::try_from(exponent).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation exponent exceeds field width"))?;
        Ok(y.pow_vartime([exponent]))
    }

    fn raw_field_scalar<F: PrimeField>(value: &F) -> io::Result<ScalarField> {
        canonical_field_words(value)
            .map(ScalarField::from)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident H scalar is not a canonical BN254 field element"))
    }

    // Record one successful chunk without waiting. The stage caller batches
    // chunks and synchronizes before reusing staging memory or stack scalars.
    fn record_permutation_kernel(rows: usize, local_bytes_per_thread: usize, status: i32, operation: &'static str, stream: &CudaStream) -> io::Result<()> {
        if status != 0 {
            let launch_error = io::Error::new(io::ErrorKind::Other, format!("resident permutation {} kernel launch failed with CUDA status {}", operation, status));
            return match synchronize_h_transfer_stream(stream) {
                Ok(()) => Err(launch_error),
                Err(sync_error) => Err(io::Error::new(io::ErrorKind::Other, format!("{}; draining the owner stream also failed: {}", launch_error, sync_error))),
            };
        }
        GPU_H_EVAL_CHUNKS.fetch_add(1, Ordering::Relaxed);
        let launched_threads = rows.saturating_add(127) / 128 * 128;
        let local_upper_bytes = match launched_threads.checked_mul(local_bytes_per_thread) {
            Some(bytes) => bytes,
            None => {
                let error = io::Error::new(io::ErrorKind::InvalidInput, "resident permutation local-memory upper bound overflows");
                return match synchronize_h_stream(stream) {
                    Ok(()) => Err(error),
                    Err(sync_error) => Err(io::Error::new(io::ErrorKind::Other, format!("{}; draining the owner stream also failed: {}", error, sync_error))),
                };
            }
        };
        GPU_H_LOCAL_PEAK_BYTES.fetch_max(local_upper_bytes, Ordering::Relaxed);
        if local_upper_bytes > FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES {
            let error = io::Error::new(io::ErrorKind::InvalidInput, "resident permutation kernel exceeds its local-memory reserve");
            return match synchronize_h_stream(stream) {
                Ok(()) => Err(error),
                Err(sync_error) => Err(io::Error::new(io::ErrorKind::Other, format!("{}; draining the owner stream also failed: {}", error, sync_error))),
            };
        }
        Ok(())
    }

    pub(crate) fn initialize_permutation_twiddles<F: PrimeField + 'static>(
        &mut self,
        extended_omega: F,
        part_stride: usize,
        part_index: usize,
    ) -> io::Result<()> {
        if !raw_montgomery_layout_supported::<F>() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation field layout changed after preflight"));
        }
        if self.rows < 32 || part_stride == 0 || part_index >= part_stride || self.rows.checked_mul(part_stride).is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation part geometry is invalid"));
        }
        let permutation = self.permutation.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident permutation buffers were not admitted"))?;
        let device_id = self.device_id;
        if device::set_device(device_id).is_err() { return Err(io::Error::new(io::ErrorKind::Other, "resident permutation could not select its owner device")); }
        let omega = Self::raw_field_scalar(&extended_omega)?;
        let omega_ptr = &omega as *const ScalarField as *const u32;
        let max_chunk = (self.kernel_local_row_capacity.min(1 << 20) / 128) * 128;
        if max_chunk < 128 { return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation tile is below one CUDA block")); }
        // Validate every `start * 8` chunk offset before any asynchronous work is queued.
        self.rows.checked_mul(8).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation twiddle pointer range overflows"))?;
        let stream = CudaStream::default();
        let mut start = 0usize;
        while start < self.rows {
            let rows = max_chunk.min(self.rows - start);
            let word_offset = start * 8;
            let twiddles_ptr = unsafe { (permutation.row_twiddles.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let status = unsafe { bn254_h_permutation_twiddles(twiddles_ptr, rows, start, omega_ptr, part_stride, part_index) };
            Self::record_permutation_kernel(rows, self.kernel_local_bytes_per_thread, status, "twiddle", &stream)?;
            start += rows;
        }
        synchronize_h_stream(&stream)
    }

    pub(crate) fn scale_permutation_h<F: PrimeField + 'static>(&mut self, y: F, total_terms: usize) -> io::Result<()> {
        if !raw_montgomery_layout_supported::<F>() || self.permutation.is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation scale was not preflighted"));
        }
        let scale = Self::raw_field_scalar(&y.pow_vartime([u64::try_from(total_terms).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation fold length exceeds field width"))?]))?;
        let scale_ptr = &scale as *const ScalarField as *const u32;
        let max_chunk = (self.kernel_local_row_capacity.min(1 << 20) / 128) * 128;
        if max_chunk < 128 { return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation tile is below one CUDA block")); }
        // Validate every `start * 8` chunk offset before any asynchronous work is queued.
        self.rows.checked_mul(8).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation scale pointer range overflows"))?;
        let stream = CudaStream::default();
        let mut start = 0usize;
        while start < self.rows {
            let rows = max_chunk.min(self.rows - start);
            let word_offset = start * 8;
            let h_ptr = unsafe { (self.h.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let status = unsafe { bn254_h_permutation_scale(scale_ptr, h_ptr, rows) };
            Self::record_permutation_kernel(rows, self.kernel_local_bytes_per_thread, status, "scale", &stream)?;
            start += rows;
        }
        synchronize_h_stream(&stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_permutation_set<F: PrimeField + 'static>(
        &mut self,
        z: &[F],
        l0: &[F],
        l_last: &[F],
        y: F,
        total_terms: usize,
        set_index: usize,
        set_count: usize,
        last_rotation: i32,
        part_index: usize,
        part_stride: usize,
        rot_scale: i32,
        isize: i32,
    ) -> io::Result<()> {
        if !raw_montgomery_layout_supported::<F>() || self.permutation.is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation set was not preflighted"));
        }
        let full_size = usize::try_from(isize).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "resident permutation extended size is invalid"))?;
        if part_stride == 0 || part_index >= part_stride || full_size % part_stride != 0 || full_size / part_stride != self.rows || z.len() != self.rows || l0.len() != full_size || l_last.len() != full_size || set_count == 0 || set_index >= set_count {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation set geometry is invalid"));
        }
        let z_bn = unsafe { std::slice::from_raw_parts(z.as_ptr().cast::<Bn256Fr>(), z.len()) };
        let z_raw = unsafe { std::slice::from_raw_parts(z_bn.as_ptr().cast::<ScalarField>(), z_bn.len()) };
        let l0_bn = unsafe { std::slice::from_raw_parts(l0.as_ptr().cast::<Bn256Fr>(), l0.len()) };
        let l_last_bn = unsafe { std::slice::from_raw_parts(l_last.as_ptr().cast::<Bn256Fr>(), l_last.len()) };
        let first_scale = if set_index == 0 { Self::permutation_scale(y, total_terms, 0)? } else { F::ZERO };
        let incoming_term = set_index.checked_add(1).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident incoming permutation term index overflows"))?;
        let incoming_scale = if set_index != 0 { Self::permutation_scale(y, total_terms, incoming_term)? } else { F::ZERO };
        let outgoing_term = set_index.checked_add(2).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident outgoing permutation term index overflows"))?;
        let outgoing_scale = if set_index + 1 < set_count { Self::permutation_scale(y, total_terms, outgoing_term)? } else { F::ZERO };
        let scales = [Self::raw_field_scalar(&first_scale)?, Self::raw_field_scalar(&incoming_scale)?, Self::raw_field_scalar(&outgoing_scale)?];
        let scales_ptr = scales.as_ptr() as *const u32;
        let device_id = self.device_id;
        if device::set_device(device_id).is_err() { return Err(io::Error::new(io::ErrorKind::Other, "resident permutation could not select its owner device")); }
        let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
        let workspace = workspaces.as_mut().and_then(|all| all.get_mut(&device_id)).ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation staging workspace is missing"))?;
        let input = workspace.input.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation input workspace is missing"))?;
        let output = workspace.output.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation output workspace is missing"))?;
        if input.len() < self.rows || output.len() < self.rows || workspace.host_values.len() < self.rows {
            return Err(io::Error::new(io::ErrorKind::Other, "resident permutation staging capacity is below its admitted bound"));
        }
        for row in 0..self.rows {
            let index = row.checked_mul(part_stride).and_then(|value| value.checked_add(part_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation L0 row index overflows"))?;
            let source = l0_bn.get(index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident permutation L0 source index is out of bounds"))?;
            copy_raw_bn254_fr(source, &mut workspace.host_values[row]);
        }
        let z_ptr = unsafe { input[..self.rows].as_ptr() as *const u32 };
        let max_chunk = (self.kernel_local_row_capacity.min(1 << 20) / 128) * 128;
        if max_chunk < 128 { return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation tile is below one CUDA block")); }
        // Validate every `start * 8` chunk offset before any asynchronous work is queued.
        self.rows.checked_mul(8).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation boundary pointer range overflows"))?;
        let stream = CudaStream::default();
        let permutation = self.permutation.as_mut().expect("permutation buffer was checked above");
        if let Err(error) = input[..self.rows].copy_from_host_async(HostSlice::from_slice(z_raw), &stream) {
            return Err(Self::resident_transfer_error(&stream, "Z", error));
        }
        if let Err(error) = output[..self.rows].copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..self.rows]), &stream) {
            return Err(Self::resident_transfer_error(&stream, "L0", error));
        }
        let mut start = 0usize;
        while start < self.rows {
            let rows = max_chunk.min(self.rows - start);
            let word_offset = start * 8;
            let l0_ptr = unsafe { (output[..self.rows].as_ptr() as *const u32).add(word_offset) };
            let h_ptr = unsafe { (self.h.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let left_ptr = unsafe { (permutation.left.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let right_ptr = unsafe { (permutation.right.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let status = unsafe { bn254_h_permutation_begin_set(z_ptr, l0_ptr, h_ptr, left_ptr, right_ptr, rows, start, full_size, part_index, part_stride, rot_scale, isize, last_rotation, set_index, set_count, scales_ptr) };
            Self::record_permutation_kernel(rows, self.kernel_local_bytes_per_thread, status, "set-boundary", &stream)?;
            start += rows;
        }
        // L0 is copied asynchronously from the reusable host staging arena.
        // Drain all begin-set kernels before replacing that arena with Llast.
        synchronize_h_stream(&stream)?;
        if set_index + 1 == set_count {
            let last_scale = Self::permutation_scale(y, total_terms, 1)?;
            let last_scale = Self::raw_field_scalar(&last_scale)?;
            let last_scale_ptr = &last_scale as *const ScalarField as *const u32;
            for row in 0..self.rows {
                let index = row.checked_mul(part_stride).and_then(|value| value.checked_add(part_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation Llast row index overflows"))?;
                let source = l_last_bn.get(index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident permutation Llast source index is out of bounds"))?;
                copy_raw_bn254_fr(source, &mut workspace.host_values[row]);
            }
            output[..self.rows].copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..self.rows]), &stream)
                .map_err(|error| Self::resident_transfer_error(&stream, "Llast", error))?;
            let mut start = 0usize;
            while start < self.rows {
                let rows = max_chunk.min(self.rows - start);
                let word_offset = start * 8;
                let right_ptr = unsafe { (permutation.right.slice().as_ptr() as *const u32).add(word_offset) };
                let h_ptr = unsafe { (self.h.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
                let llast_ptr = unsafe { (output[..self.rows].as_ptr() as *const u32).add(word_offset) };
                let status = unsafe { bn254_h_permutation_last_boundary(llast_ptr, h_ptr, right_ptr, rows, last_scale_ptr) };
                Self::record_permutation_kernel(rows, self.kernel_local_bytes_per_thread, status, "last-boundary", &stream)?;
                start += rows;
            }
            synchronize_h_stream(&stream)?;
        }
        Ok(())
    }

    pub(crate) fn multiply_permutation_column<F: PrimeField + 'static>(
        &mut self,
        source: &[F],
        sigma: &[F],
        beta: F,
        gamma: F,
        delta_start: F,
    ) -> io::Result<()> {
        if !raw_montgomery_layout_supported::<F>() || self.permutation.is_none() || source.len() != self.rows || sigma.len() != self.rows {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation column shape or admission is invalid"));
        }
        let source_bn = unsafe { std::slice::from_raw_parts(source.as_ptr().cast::<Bn256Fr>(), source.len()) };
        let sigma_bn = unsafe { std::slice::from_raw_parts(sigma.as_ptr().cast::<Bn256Fr>(), sigma.len()) };
        let source_raw = unsafe { std::slice::from_raw_parts(source_bn.as_ptr().cast::<ScalarField>(), source_bn.len()) };
        let sigma_raw = unsafe { std::slice::from_raw_parts(sigma_bn.as_ptr().cast::<ScalarField>(), sigma_bn.len()) };
        let scalars = [Self::raw_field_scalar(&beta)?, Self::raw_field_scalar(&gamma)?, Self::raw_field_scalar(&delta_start)?];
        let scalars_ptr = scalars.as_ptr() as *const u32;
        let device_id = self.device_id;
        if device::set_device(device_id).is_err() { return Err(io::Error::new(io::ErrorKind::Other, "resident permutation could not select its owner device")); }
        let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
        let workspace = workspaces.as_mut().and_then(|all| all.get_mut(&device_id)).ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation staging workspace is missing"))?;
        let input = workspace.input.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation input workspace is missing"))?;
        let output = workspace.output.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation output workspace is missing"))?;
        if input.len() < self.rows || output.len() < self.rows { return Err(io::Error::new(io::ErrorKind::Other, "resident permutation column staging capacity is below its admitted bound")); }
        let stream = CudaStream::default();
        let max_chunk = (self.kernel_local_row_capacity.min(1 << 20) / 128) * 128;
        if max_chunk < 128 { return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation tile is below one CUDA block")); }
        // Validate every `start * 8` chunk offset before any asynchronous work is queued.
        self.rows.checked_mul(8).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation column pointer range overflows"))?;
        let permutation = self.permutation.as_mut().expect("permutation buffer was checked above");
        if let Err(error) = input[..self.rows].copy_from_host_async(HostSlice::from_slice(source_raw), &stream) {
            return Err(Self::resident_transfer_error(&stream, "source", error));
        }
        if let Err(error) = output[..self.rows].copy_from_host_async(HostSlice::from_slice(sigma_raw), &stream) {
            return Err(Self::resident_transfer_error(&stream, "sigma", error));
        }
        let mut start = 0usize;
        while start < self.rows {
            let rows = max_chunk.min(self.rows - start);
            let word_offset = start * 8;
            let source_ptr = unsafe { (input[..self.rows].as_ptr() as *const u32).add(word_offset) };
            let sigma_ptr = unsafe { (output[..self.rows].as_ptr() as *const u32).add(word_offset) };
            let twiddle_ptr = unsafe { (permutation.row_twiddles.slice().as_ptr() as *const u32).add(word_offset) };
            let left_ptr = unsafe { (permutation.left.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let right_ptr = unsafe { (permutation.right.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let status = unsafe { bn254_h_permutation_column(source_ptr, sigma_ptr, twiddle_ptr, left_ptr, right_ptr, rows, scalars_ptr) };
            Self::record_permutation_kernel(rows, self.kernel_local_bytes_per_thread, status, "column", &stream)?;
            start += rows;
        }
        synchronize_h_stream(&stream)
    }

    pub(crate) fn finish_permutation_set<F: PrimeField + 'static>(
        &mut self,
        l_active: &[F],
        y: F,
        total_terms: usize,
        set_index: usize,
        set_count: usize,
        part_index: usize,
        part_stride: usize,
        isize: i32,
    ) -> io::Result<()> {
        if !raw_montgomery_layout_supported::<F>() || self.permutation.is_none() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation copy fold was not preflighted"));
        }
        let full_size = usize::try_from(isize).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "resident permutation copy size is invalid"))?;
        if set_count == 0 || set_index >= set_count || part_stride == 0 || part_index >= part_stride || full_size % part_stride != 0 || full_size / part_stride != self.rows || l_active.len() != full_size {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation copy-fold geometry is invalid"));
        }
        let term_index = set_count.checked_add(1).and_then(|index| index.checked_add(set_index))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation copy term index overflows"))?;
        let scale = Self::raw_field_scalar(&Self::permutation_scale(y, total_terms, term_index)?)?;
        let scale_ptr = &scale as *const ScalarField as *const u32;
        let l_active_bn = unsafe { std::slice::from_raw_parts(l_active.as_ptr().cast::<Bn256Fr>(), l_active.len()) };
        let device_id = self.device_id;
        if device::set_device(device_id).is_err() { return Err(io::Error::new(io::ErrorKind::Other, "resident permutation could not select its owner device")); }
        let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
        let workspace = workspaces.as_mut().and_then(|all| all.get_mut(&device_id)).ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation staging workspace is missing"))?;
        let output = workspace.output.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident permutation output workspace is missing"))?;
        if output.len() < self.rows || workspace.host_values.len() < self.rows { return Err(io::Error::new(io::ErrorKind::Other, "resident permutation copy staging capacity is below its admitted bound")); }
        let max_chunk = (self.kernel_local_row_capacity.min(1 << 20) / 128) * 128;
        if max_chunk < 128 { return Err(io::Error::new(io::ErrorKind::InvalidData, "resident permutation tile is below one CUDA block")); }
        // Validate every `start * 8` chunk offset before any asynchronous work is queued.
        self.rows.checked_mul(8).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation copy pointer range overflows"))?;
        for row in 0..self.rows {
            let index = row.checked_mul(part_stride).and_then(|value| value.checked_add(part_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident permutation active-row index overflows"))?;
            let source = l_active_bn.get(index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident permutation active-row source index is out of bounds"))?;
            copy_raw_bn254_fr(source, &mut workspace.host_values[row]);
        }
        let stream = CudaStream::default();
        let permutation = self.permutation.as_mut().expect("permutation buffer was checked above");
        if let Err(error) = output[..self.rows].copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..self.rows]), &stream) {
            return Err(Self::resident_transfer_error(&stream, "active-row", error));
        }
        let mut start = 0usize;
        while start < self.rows {
            let rows = max_chunk.min(self.rows - start);
            let word_offset = start * 8;
            let l_active_ptr = unsafe { (output[..self.rows].as_ptr() as *const u32).add(word_offset) };
            let left_ptr = unsafe { (permutation.left.slice().as_ptr() as *const u32).add(word_offset) };
            let right_ptr = unsafe { (permutation.right.slice().as_ptr() as *const u32).add(word_offset) };
            let h_ptr = unsafe { (self.h.slice_mut().as_mut_ptr() as *mut u32).add(word_offset) };
            let status = unsafe { bn254_h_permutation_copy_term(l_active_ptr, left_ptr, right_ptr, h_ptr, rows, scale_ptr) };
            Self::record_permutation_kernel(rows, self.kernel_local_bytes_per_thread, status, "copy-term", &stream)?;
            start += rows;
        }
        synchronize_h_stream(&stream)?;
        GPU_H_EVAL_CALLS.fetch_add(1, Ordering::Relaxed);
        GPU_H_EVAL_ROWS.fetch_add(self.rows as u64, Ordering::Relaxed);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn evaluate_mv_lookup_program<F: PrimeField + 'static>(
        &mut self,
        program: &GateProgram<F>,
        fixed: &[Polynomial<F, LagrangeCoeff>],
        advice: &[Polynomial<F, LagrangeCoeff>],
        instance: &[Polynomial<F, LagrangeCoeff>],
        phi: &[F],
        multiplicity: &[F],
        l0: &[F],
        l_last: &[F],
        l_active: &[F],
        part_index: usize,
        part_stride: usize,
        rot_scale: i32,
        isize: i32,
        y: F,
    ) -> io::Result<()> {
        if !raw_montgomery_layout_supported::<F>() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup field layout changed after preflight"));
        }
        let Some(lookup) = program.mv_lookup.as_ref() else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup received a non-lookup program"));
        };
        if lookup.input_counts.len() != 1 || lookup.input_counts[0] == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup program must contain one nonempty lookup"));
        }
        if program.scratch_slots == 0 || program.instructions.is_empty() || part_stride == 0 || part_index >= part_stride || rot_scale <= 0 || isize <= 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup geometry is invalid"));
        }
        let full_size = usize::try_from(isize).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "resident H lookup domain size is invalid"))?;
        if full_size % part_stride != 0 { return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup part geometry is invalid")); }
        let base_n = full_size / part_stride;
        if self.rows != base_n || phi.len() != base_n || multiplicity.len() != base_n || l0.len() != full_size || l_last.len() != full_size || l_active.len() != full_size {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup query source shape is invalid"));
        }
        if program.queries.len() != lookup.graph_query_count || !validate_gate_program(program)? {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup program failed validation"));
        }
        if canonical_field_words(&y).is_none() || program.constants.iter().any(|value| canonical_field_words(value).is_none()) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup field constant is not canonical"));
        }

        let instruction_count = program.instructions.len();
        let constant_count = program.constants.len();
        let graph_query_count = lookup.graph_query_count;
        let query_count = graph_query_count.checked_add(6).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup query count overflows"))?;
        let fixed_input = instruction_count.checked_add(constant_count).and_then(|count| count.checked_add(1)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup input size overflows"))?;
        let _metadata_bytes = instruction_count.checked_mul(mem::size_of::<[u32; 8]>())
            .and_then(|bytes| program.constants.capacity().checked_mul(mem::size_of::<F>()).and_then(|constant_bytes| bytes.checked_add(constant_bytes)))
            .and_then(|bytes| program.queries.capacity().checked_mul(mem::size_of::<crate::h_eval::GateQuery>()).and_then(|query_bytes| bytes.checked_add(query_bytes)))
            .and_then(|bytes| program.scratch_slots.checked_mul(mem::size_of::<bool>()).and_then(|validation_bytes| bytes.checked_add(validation_bytes)))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup metadata size overflows"))?;
        let raw_query_copy = h_query_uses_bulk_copy(rot_scale, part_stride);
        let mut resolved_queries = Vec::new();
        resolved_queries.try_reserve_exact(program.queries.len()).map_err(|error| io::Error::new(io::ErrorKind::Other, format!("resident H lookup query allocation failed: {}", error)))?;
        for (query_index, query) in program.queries.iter().enumerate() {
            let source = query_column_values(query.kind, query.column, fixed, advice, instance).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("resident H lookup query {} references a missing column", query_index)))?;
            if source.len() != base_n { return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup graph query has the wrong part length")); }
            let rotation_offset = query.rotation.checked_mul(rot_scale).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident H lookup query rotation overflows"))?;
            let rotation_rows = if raw_query_copy { (query.rotation as i64).rem_euclid(base_n as i64) as usize } else { 0 };
            // SAFETY: the raw BN254 representation guard passed above.
            let values = unsafe { std::slice::from_raw_parts(source.as_ptr().cast::<Bn256Fr>(), source.len()) };
            resolved_queries.push(ResolvedHQuery { values, rotation_offset, rotation_rows });
        }

        let device_id = self.device_id;
        if device::set_device(device_id).is_err() { return Err(io::Error::new(io::ErrorKind::Other, "resident H lookup could not select its owner device")); }
        let local_bytes_per_thread = self.kernel_local_bytes_per_thread;
        let local_row_capacity = self.kernel_local_row_capacity;
        if local_row_capacity < 128 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H lookup session has an invalid kernel capacity"));
        }

        // The shared NTT staging allocations are serialized by this same mutex;
        // the whole CUDA transfer/kernel sequence below is synchronized before
        // the guard leaves scope.
        let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
        let workspace = workspaces.as_mut().and_then(|all| all.get_mut(&device_id))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident H lookup staging workspace is missing"))?;
        let input_capacity = workspace.input.as_ref().map_or(0, |buffer| buffer.len());
        let output_capacity = workspace.output.as_ref().map_or(0, |buffer| buffer.len());
        if input_capacity < fixed_input || output_capacity < program.scratch_slots {
            return Err(io::Error::new(io::ErrorKind::Other, "resident H lookup staging capacity is below its preflight bound"));
        }
        if workspace.host_values.len() != workspace.capacity || workspace.host_values.capacity() != workspace.capacity {
            return Err(io::Error::new(io::ErrorKind::Other, "resident H lookup host staging arena is not authenticated"));
        }
        let rows_by_input = input_capacity.saturating_sub(fixed_input) / query_count;
        let rows_by_output = output_capacity / program.scratch_slots;
        let max_chunk = (rows_by_input.min(rows_by_output).min(local_row_capacity).min(1 << 16) / 32) * 32;
        if max_chunk < 32 { return Err(io::Error::new(io::ErrorKind::Other, "resident H lookup tile is below the supported minimum")); }
        let constant_words: Vec<[u32; 8]> = program.constants.iter().map(|value| canonical_field_words(value).expect("constants were prevalidated")).collect();
        let y_words = canonical_field_words(&y).expect("y was prevalidated");
        let instruction_words = program.instructions.iter().map(|instruction| ScalarField::from(*instruction)).collect::<Vec<_>>();
        let constant_values = constant_words.iter().copied().map(ScalarField::from).collect::<Vec<_>>();
        let stream = CudaStream::default();
        let mut start = 0usize;
        while start < self.rows {
            let rows = max_chunk.min(self.rows - start);
            let launched_threads = rows.saturating_add(127) / 128 * 128;
            let local_upper_bytes = launched_threads.checked_mul(local_bytes_per_thread).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup local memory bound overflows"))?;
            if local_upper_bytes > FORMAL_GPU_H_KERNEL_LOCAL_MEMORY_UPPER_BYTES { return Err(io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup tile exceeds its local-memory reserve")); }
            let query_elements = query_count.checked_mul(rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup query staging size overflows"))?;
            let input_elements = fixed_input.checked_add(query_elements).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup input staging size overflows"))?;
            let scratch_elements = program.scratch_slots.checked_mul(rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup scratch size overflows"))?;
            for (index, instruction) in instruction_words.iter().enumerate() { workspace.host_values[index] = *instruction; }
            for (index, constant) in constant_values.iter().enumerate() { workspace.host_values[instruction_count + index] = *constant; }
            workspace.host_values[instruction_count + constant_count] = ScalarField::from(y_words);
            for (query_index, query) in resolved_queries.iter().enumerate() {
                let query_start = fixed_input.checked_add(query_index.checked_mul(rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup query offset overflows"))?).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup query offset overflows"))?;
                let destination = workspace.host_values.get_mut(query_start..query_start + rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup query destination is out of bounds"))?;
                if raw_query_copy {
                    copy_bn254_h_query_chunk(query.values, start, rows, query.rotation_rows, base_n, destination)?;
                } else {
                    for offset in 0..rows {
                        let row = start + offset;
                        let full_index = row.checked_mul(part_stride).and_then(|n| n.checked_add(part_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup full row index overflows"))?;
                        let rotated = (i64::try_from(full_index).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup row exceeds signed index width"))? + i64::from(query.rotation_offset)).rem_euclid(i64::from(isize)) as usize / part_stride;
                        let source = query.values.get(rotated).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident H lookup rotated query index is inconsistent"))?;
                        copy_raw_bn254_fr(source, &mut destination[offset]);
                    }
                }
            }
            let auxiliary_base = fixed_input.checked_add(graph_query_count.checked_mul(rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup auxiliary offset overflows"))?).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup auxiliary offset overflows"))?;
            let auxiliary_sources = [
                (phi, true, 0usize),
                (phi, true, 1usize),
                (multiplicity, true, 0usize),
                (l0, false, 0usize),
                (l_last, false, 0usize),
                (l_active, false, 0usize),
            ];
            for (aux_index, (source, is_part, rotation)) in auxiliary_sources.iter().enumerate() {
                let query_start = auxiliary_base.checked_add(aux_index.checked_mul(rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup auxiliary query offset overflows"))?).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup auxiliary query offset overflows"))?;
                let destination = workspace.host_values.get_mut(query_start..query_start + rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup auxiliary query range is out of bounds"))?;
                // SAFETY: the field raw-layout guard passed before the session
                // began, and every auxiliary field has the same scalar type.
                let source = unsafe { std::slice::from_raw_parts(source.as_ptr().cast::<Bn256Fr>(), source.len()) };
                if *is_part {
                    copy_bn254_h_query_chunk(source, start, rows, *rotation, base_n, destination)?;
                } else {
                    for (offset, slot) in destination.iter_mut().enumerate() {
                        let row = start + offset;
                        let full_index = row.checked_mul(part_stride).and_then(|n| n.checked_add(part_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup auxiliary row overflows"))?;
                        let source_value = source.get(full_index).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "resident H lookup auxiliary source index is out of bounds"))?;
                        copy_raw_bn254_fr(source_value, slot);
                    }
                }
            }
            let input = workspace.input.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident H lookup input workspace is missing"))?;
            let scratch = workspace.output.as_mut().ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident H lookup scratch workspace is missing"))?;
            let h_end = start.checked_add(rows)
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup tile end overflows"))?;
            if h_end > self.rows {
                return Err(io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup tile exceeds the accumulator allocation"));
            }
            if let Err(error) = input[..input_elements].copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..input_elements]), &stream) {
                return Err(Self::resident_transfer_error(&stream, "lookup input", error));
            }
            // SAFETY: owner-bound H and NTT scratch buffers are contiguous and
            // remain live until the stream synchronization below completes.
            let input_ptr = unsafe { input[..input_elements].as_ptr() } as *const u32;
            let h_word_offset = start.checked_mul(mem::size_of::<ScalarField>() / mem::size_of::<u32>())
                .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "resident H lookup accumulator offset overflows"))?;
            let h_ptr = unsafe { (self.h.slice_mut().as_mut_ptr() as *mut u32).add(h_word_offset) };
            let scratch_ptr = unsafe { scratch[..scratch_elements].as_mut_ptr() } as *mut u32;
            let status = unsafe { bn254_h_eval_mv_lookup_program(input_ptr, h_ptr, scratch_ptr, rows, instruction_count, constant_count, query_count, program.scratch_slots, graph_query_count, 1) };
            if status != 0 {
                return Err(Self::resident_kernel_launch_error(&stream, "lookup", status));
            }
            GPU_H_EVAL_CHUNKS.fetch_add(1, Ordering::Relaxed);
            GPU_H_LOCAL_PEAK_BYTES.fetch_max(local_upper_bytes, Ordering::Relaxed);
            synchronize_h_stream(&stream)?;
            start += rows;
        }
        GPU_H_EVAL_CALLS.fetch_add(1, Ordering::Relaxed);
        GPU_H_EVAL_ROWS.fetch_add(self.rows as u64, Ordering::Relaxed);
        let _ = _metadata_bytes;
        Ok(())
    }

    pub(crate) fn finish<F: PrimeField + 'static>(self, values: &mut [F]) -> io::Result<()> {
        if !raw_montgomery_layout_supported::<F>() || values.len() != self.rows {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "resident H result destination does not match its preflight shape"));
        }
        let mut workspaces = NTT_WORKSPACES.lock().map_err(|_| io::Error::new(io::ErrorKind::Other, "ICICLE NTT workspace mutex poisoned"))?;
        let workspace = workspaces.as_mut().and_then(|all| all.get_mut(&self.device_id))
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "resident H output staging workspace is missing"))?;
        let rows = self.rows;
        if workspace.host_values.len() < rows { return Err(io::Error::new(io::ErrorKind::Other, "resident H output staging capacity is too small")); }
        let stream = CudaStream::default();
        let destination = HostSlice::from_mut_slice(&mut workspace.host_values[..rows]);
        if let Err(error) = self.h.slice().copy_to_host_async(destination, &stream) {
            return Err(Self::resident_transfer_error(&stream, "output readback", error));
        }
        synchronize_h_transfer_stream(&stream)?;
        // SAFETY: the raw BN254 layout guard and exact length check passed
        // immediately above.
        let values_bn = unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Bn256Fr>(), values.len()) };
        copy_validated_bn254_h_output_chunk(&workspace.host_values[..rows], values_bn)
    }
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
    let resolved_query_bytes = program.queries.len().checked_mul(mem::size_of::<ResolvedHQuery<'_>>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H resolved-query metadata size overflow"))?;
    let validation_bytes = program.scratch_slots.checked_mul(mem::size_of::<bool>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H scratch metadata size overflow"))?;
    let staging_bytes = fixed_input.checked_mul(mem::size_of::<ScalarField>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H metadata staging size overflow"))?;
    let canonical_constant_bytes = program.constants.len().checked_mul(mem::size_of::<[u32; 8]>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H canonical constant size overflow"))?;
    let other_metadata_bytes = instruction_bytes.checked_add(constant_bytes).and_then(|n| n.checked_add(query_bytes)).and_then(|n| n.checked_add(validation_bytes)).and_then(|n| n.checked_add(staging_bytes)).and_then(|n| n.checked_add(canonical_constant_bytes)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H program metadata size overflow"))?;
    let metadata_bytes = other_metadata_bytes.checked_add(resolved_query_bytes).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H program metadata size overflow"))?;
    if metadata_bytes > FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES { return unsupported_h_program(&format!("program_metadata_over_budget bytes={} limit={}", metadata_bytes, FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES)); }
    if !validate_gate_program(program)? { return unsupported_h_program("program_validation_workspace_unavailable"); }
    if canonical_field_words(&y).is_none() { return unsupported_h_program("y_is_not_canonical_field_element"); }
    if program.constants.iter().any(|value| canonical_field_words(value).is_none()) { return unsupported_h_program("constant_is_not_canonical_field_element"); }
    let raw_query_copy = h_query_uses_bulk_copy(rot_scale, part_stride);
    let mut resolved_queries = Vec::new();
    if resolved_queries.try_reserve_exact(program.queries.len()).is_err() {
        return unsupported_h_program("resolved_query_workspace_unavailable");
    }
    let actual_resolved_query_bytes = resolved_queries.capacity().checked_mul(mem::size_of::<ResolvedHQuery<'_>>()).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H resolved-query workspace size overflow"))?;
    let actual_metadata_bytes = other_metadata_bytes.checked_add(actual_resolved_query_bytes).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H program metadata size overflow"))?;
    if actual_metadata_bytes > FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES {
        return unsupported_h_program(&format!("program_metadata_over_budget bytes={} limit={}", actual_metadata_bytes, FORMAL_GPU_H_PROGRAM_METADATA_UPPER_BYTES));
    }
    for (query_index, query) in program.queries.iter().enumerate() {
        let Some(source) = query_column_values(query.kind, query.column, fixed, advice, instance) else {
            return unsupported_h_program(&format!("query_column_shape_mismatch query_index={} kind={} column={} expected_rows={} actual_len=None isize={} part_stride={}", query_index, query.kind, query.column, base_n, isize, part_stride));
        };
        if source.len() != base_n {
            return unsupported_h_program(&format!("query_column_shape_mismatch query_index={} kind={} column={} expected_rows={} actual_len=Some({}) isize={} part_stride={}", query_index, query.kind, query.column, base_n, source.len(), isize, part_stride));
        }
        let Some(rotation_offset) = query.rotation.checked_mul(rot_scale) else {
            return unsupported_h_program(&format!("query_rotation_overflow query_index={} rotation={} rot_scale={}", query_index, query.rotation, rot_scale));
        };
        let rotation_rows = if raw_query_copy { (query.rotation as i64).rem_euclid(base_n as i64) as usize } else { 0 };
        // SAFETY: raw_montgomery_layout_supported::<F>() passed before this
        // function body continued, including exact TypeId and ZERO/ONE/-ONE
        // representation probes for the pinned BN254 Fr layout.
        let raw_source = unsafe { std::slice::from_raw_parts(source.as_ptr() as *const Bn256Fr, source.len()) };
        resolved_queries.push(ResolvedHQuery { values: raw_source, rotation_offset, rotation_rows });
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
        eprintln!("[ezkl-gpu-h] kernel attrs local_bytes_per_thread={} stack_limit_bytes_per_thread={} extra_scalar_accumulator_reserve_bytes_per_thread={} shared_bytes_per_block={} registers_per_thread={} max_threads_per_block={} local_upper_formula=rounded_threads*(local+stack_limit+extra_scalar_accumulator_reserve)", local_bytes_per_thread, stack_bytes_per_thread, 2 * mem::size_of::<ScalarField>(), shared_bytes_per_block, registers_per_thread, max_threads_per_block);
    }
    if shared_bytes_per_block != 0 || max_threads_per_block < 128 { return unsupported_h_program(&format!("kernel_attributes_unsupported shared_bytes_per_block={} max_threads_per_block={}", shared_bytes_per_block, max_threads_per_block)); }
    let scalar_accumulator_reserve = match mem::size_of::<ScalarField>().checked_mul(2) { Some(bytes) => bytes, None => return unsupported_h_program("kernel_scalar_accumulator_reserve_overflow") };
    let local_bytes_per_thread = match local_bytes_per_thread.checked_add(stack_bytes_per_thread).and_then(|bytes| bytes.checked_add(scalar_accumulator_reserve)) { Some(bytes) => bytes, None => return unsupported_h_program("kernel_local_bytes_overflow") };
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
    let existing_coset_capacity = workspace.coset_powers.as_ref().map_or(0, |buffer| buffer.len());
    if workspace.capacity < target_capacity || existing_input_capacity < target_capacity || existing_output_capacity < target_capacity || existing_coset_capacity < target_capacity {
        // Drop all old device workspace before reserving the larger envelope;
        // replacement must not temporarily double the authenticated peak.
        workspace.input = None;
        workspace.output = None;
        workspace.coset_powers = None;
        workspace.coset_key = None;
        workspace.capacity = 0;
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
        let new_coset_powers = match DeviceVec::<ScalarField>::cuda_malloc_for_device(target_capacity, device_id) {
            Ok(buffer) => buffer,
            Err(error) => {
                if replaced_host_arena { workspace.host_values = Vec::new(); }
                return Err(io::Error::new(io::ErrorKind::Other, format!("ICICLE H coset-power workspace allocation failed: {:?}", error)));
            }
        };
        if new_input.len() < target_capacity || new_output.len() < target_capacity || new_coset_powers.len() < target_capacity {
            if replaced_host_arena { workspace.host_values = Vec::new(); }
            return Err(io::Error::new(io::ErrorKind::Other, "ICICLE H allocated device workspace is below authenticated workspace size"));
        }
        workspace.input = Some(new_input);
        workspace.output = Some(new_output);
        workspace.coset_powers = Some(new_coset_powers);
        workspace.capacity = target_capacity;
    }
    let input_capacity = workspace.input.as_ref().map_or(0, |buffer| buffer.len());
    let output_capacity = workspace.output.as_ref().map_or(0, |buffer| buffer.len());
    let coset_capacity = workspace.coset_powers.as_ref().map_or(0, |buffer| buffer.len());
    if input_capacity < target_capacity || output_capacity < target_capacity || coset_capacity < target_capacity {
        if replaced_host_arena { workspace.host_values = Vec::new(); }
        return Err(io::Error::new(io::ErrorKind::Other, "ICICLE H device workspace length is below authenticated workspace size"));
    }
    let ntt_workspace_bytes = target_capacity.saturating_mul(3).saturating_mul(mem::size_of::<ScalarField>());
    let h_workspace_bytes = target_capacity.saturating_mul(2).saturating_mul(mem::size_of::<ScalarField>());
    NTT_WORKSPACE_CAPACITY_BYTES.fetch_max(ntt_workspace_bytes, Ordering::Relaxed);
    NTT_MAX_SIZE.fetch_max(target_capacity, Ordering::Relaxed);
    GPU_H_WORKSPACE_PEAK_BYTES.fetch_max(h_workspace_bytes, Ordering::Relaxed);
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
    // Every unsupported-program return above this point occurs before the
    // first tile mutates `values`. After this point runtime errors are
    // returned as `Err`, never as a CPU-fallback `Ok(false)`: an earlier tile
    // may already have committed the weighted H update, and replaying it on
    // CPU would apply the scale/fold twice.
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
        for (query_index, query) in resolved_queries.iter().enumerate() {
            let query_start = query_index.checked_mul(rows).and_then(|offset| fixed_input.checked_add(offset)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H query staging offset overflow"))?;
            let query_end = query_start.checked_add(rows).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H query staging range overflow"))?;
            let query_destination = workspace.host_values.get_mut(query_start..query_end).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H query staging range is out of bounds"))?;
            if raw_query_copy {
                copy_bn254_h_query_chunk(query.values, start, rows, query.rotation_rows, base_n, query_destination)?;
            } else {
                for row_offset in 0..rows {
                    let row = start + row_offset;
                    let idx = row.checked_mul(part_stride).and_then(|n| n.checked_add(part_index)).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H row index overflow"))?;
                    let rotated = ((idx as i32).checked_add(query.rotation_offset).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "GPU H rotated index overflow"))?.rem_euclid(isize) as usize) / part_stride;
                    let fr = query.values.get(rotated).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GPU H query column index is inconsistent"))?;
                    query_destination[row_offset] = ScalarField::from(fr_raw_montgomery_words(fr));
                }
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
        // SAFETY: raw_montgomery_layout_supported::<F>() passed before any
        // tile ran, so this accumulator slice has the pinned BN254 Fr layout.
        let destination = unsafe { std::slice::from_raw_parts_mut(values[start..start + rows].as_mut_ptr() as *mut Bn256Fr, rows) };
        copy_validated_bn254_h_output_chunk(&workspace.host_values[..rows], destination)?;
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
    fft_on_device_inner(scalars, omega, log_n, inverse, None);
}

/// Perform coefficient coset scaling on device, then execute the ordinary
/// forward NTT with `coset_gen = 1`. The per-device power vector is cached by
/// exact `(length, coset)` and shares the normal NTT workspace lifetime.
pub fn fft_on_device_with_coset<G: Field + ff::PrimeField + FftGroup<G> + 'static>(
    scalars: &mut [G],
    omega: G,
    log_n: u32,
    coset_gen: G,
    host_factors: Option<&[G]>,
) {
    fft_on_device_inner(scalars, omega, log_n, false, Some((coset_gen, host_factors)));
}

fn fft_on_device_inner<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G],
    omega: Scalar,
    log_n: u32,
    inverse: bool,
    coset: Option<(G, Option<&[G]>)>,
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
    let coset_key = coset.as_ref().map(|(factor, _)| {
        assert!(!inverse && raw_montgomery_bridge, "fused ICICLE coset NTT requires the BN254 Fr forward path");
        assert!(is_gpu_supported_field(factor), "fused ICICLE coset NTT received an unsupported field");
        let words = canonical_field_words(factor).expect("fused ICICLE coset factor is not canonical");
        words
    });
    if let Some((_, Some(factors))) = coset.as_ref() {
        assert_eq!(factors.len(), size, "fused ICICLE coset factor vector has the wrong size");
    }
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
    if workspace.capacity < size
        || workspace.input.as_ref().map_or(true, |buffer| buffer.len() < size)
        || workspace.output.as_ref().map_or(true, |buffer| buffer.len() < size)
        || workspace.coset_powers.as_ref().map_or(true, |buffer| buffer.len() < size)
    {
        workspace.input = None;
        workspace.output = None;
        workspace.coset_powers = None;
        workspace.coset_key = None;
        workspace.capacity = 0;
        let input = DeviceVec::<ScalarField>::cuda_malloc_for_device(size, device_id).expect("ICICLE NTT input buffer allocation failed");
        let output = DeviceVec::<ScalarField>::cuda_malloc_for_device(size, device_id).expect("ICICLE NTT output buffer allocation failed");
        let coset_powers = DeviceVec::<ScalarField>::cuda_malloc_for_device(size, device_id).expect("ICICLE NTT coset-power buffer allocation failed");
        workspace.input = Some(input);
        workspace.output = Some(output);
        workspace.coset_powers = Some(coset_powers);
        workspace.capacity = size;
        let workspace_bytes = 3usize.checked_mul(size).and_then(|elements| elements.checked_mul(mem::size_of::<ScalarField>())).expect("ICICLE NTT workspace size overflow");
        NTT_WORKSPACE_CAPACITY_BYTES.fetch_max(workspace_bytes, Ordering::Relaxed);
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
    let cache_key = coset_key.map(|words| (size, words));
    let refresh_coset_powers = cache_key.is_some_and(|key| workspace.coset_key != Some(key));
    if refresh_coset_powers {
        // Never leave an old key published after its device vector is about
        // to be overwritten. Publish the new key only after the whole stream
        // has completed successfully below.
        workspace.coset_key = None;
    }
    let coset_factor_scalar = coset.as_ref().map(|(factor, _)| {
        // SAFETY: the exact Fr type and raw Montgomery representation were
        // checked above, before this bridge to ICICLE's pinned scalar type.
        let factor_bn = unsafe { &*(factor as *const G).cast::<Bn256Fr>() };
        ScalarField::from(fr_raw_montgomery_words(factor_bn))
    });
    if refresh_coset_powers {
        if let Some((_, Some(host_factors))) = coset.as_ref() {
            // SAFETY: the factor helper produces this exact field type and
            // the Fr bridge was checked above. The caller's factors remain
            // borrowed until the owner stream is synchronized.
            let host_factors = unsafe {
                std::slice::from_raw_parts(host_factors.as_ptr().cast::<ScalarField>(), host_factors.len())
            };
            let powers = workspace.coset_powers.as_mut().expect("ICICLE NTT coset-power buffer is missing");
            if let Err(error) = powers[..size].copy_from_host_async(HostSlice::from_slice(host_factors), &stream) {
                panic_after_ntt_stream_drain(&stream, "coset-power upload", error);
            }
        }
    }
    let input = workspace.input.as_mut().expect("ICICLE NTT input buffer is missing");
    if let Err(error) = input[..size].copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..size]), &stream) {
        panic_after_ntt_stream_drain(&stream, "host-to-device input copy", error);
    }
    if let Some(factor) = coset_factor_scalar.as_ref() {
        let input_ptr = unsafe { input[..size].as_mut_ptr() } as *mut u32;
        let powers = workspace.coset_powers.as_mut().expect("ICICLE NTT coset-power buffer is missing");
        let powers_ptr = unsafe { powers[..size].as_mut_ptr() } as *mut u32;
        let factor_ptr = factor as *const ScalarField as *const u32;
        let generate_powers = refresh_coset_powers && coset.as_ref().is_some_and(|(_, factors)| factors.is_none());
        let mut row_offset = 0usize;
        const COSET_SCALE_TILE_ROWS: usize = 1 << 20;
        while row_offset < size {
            let rows = COSET_SCALE_TILE_ROWS.min(size - row_offset);
            let status = unsafe { bn254_h_coset_scale(input_ptr, powers_ptr, row_offset, rows, factor_ptr, generate_powers) };
            if status != 0 {
                panic_after_ntt_stream_drain(&stream, "coset-scale kernel launch", status);
            }
            row_offset += rows;
        }
    }
    let mut cfg = cfg;
    cfg.is_async = true;
    if let Err(error) = ntt::<ScalarField, ScalarField>(&input[..size], dir, &cfg, &mut workspace.output.as_mut().expect("ICICLE NTT output buffer is missing")[..size]) {
        panic_after_ntt_stream_drain(&stream, "transform", error);
    }
    if let Err(error) = workspace.output.as_ref().expect("ICICLE NTT output buffer is missing")[..size]
        .copy_to_host_async(HostSlice::from_mut_slice(&mut workspace.host_values[..size]), &stream)
    {
        panic_after_ntt_stream_drain(&stream, "device-to-host output copy", error);
    }
    if let Err(error) = stream.synchronize() {
        panic!("ICICLE NTT operation synchronization failed: {:?}", error);
    }
    if refresh_coset_powers {
        workspace.coset_key = cache_key;
    }
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
            .saturating_mul(4 * size_of_scalar)
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
    fn raw_h_query_pack_and_writeback_match_legacy_word_path() {
        assert!(raw_montgomery_layout_supported::<Bn256Fr>());
        assert!(!raw_montgomery_layout_supported::<halo2curves::pasta::Fp>());
        assert!(h_query_uses_bulk_copy(3, 3));
        assert!(!h_query_uses_bulk_copy(1, 3), "non-row-aligned contracts must retain the checked legacy index path");
        assert!(!h_query_uses_bulk_copy(-3, 3));

        let base_n = 37usize;
        let source: Vec<_> = (0..base_n).map(|row| Bn256Fr::from((row as u64 + 1) * 97)).collect();
        let rotations = [3i32, -5, 1_000_003, -1_000_007];
        for stride in [1usize, 3, 8] {
            for part in 0..stride {
                for rotation in rotations {
                    assert!(rotation.checked_mul(stride as i32).is_some());
                    for (start, rows) in [(0usize, 13usize), (base_n - 5, 5usize)] {
                        let rotation_rows = (rotation as i64).rem_euclid(base_n as i64) as usize;
                        let mut packed: Vec<_> = (0..rows).map(|_| ScalarField::from([0u32; 8])).collect();
                        copy_bn254_h_query_chunk(&source, start, rows, rotation_rows, base_n, &mut packed).expect("bulk raw-Montgomery query packing");

                        let mut legacy: Vec<_> = (0..rows).map(|_| ScalarField::from([0u32; 8])).collect();
                        let domain = base_n * stride;
                        let rotation_offset = rotation.checked_mul(stride as i32).expect("bounded test rotation");
                        for row_offset in 0..rows {
                            let row = start + row_offset;
                            let index = row * stride + part;
                            let rotated = ((index as i32 + rotation_offset).rem_euclid(domain as i32) as usize) / stride;
                            legacy[row_offset] = ScalarField::from(fr_raw_montgomery_words(&source[rotated]));
                        }
                        assert_eq!(packed.iter().map(raw_scalar_words).collect::<Vec<_>>(), legacy.iter().map(raw_scalar_words).collect::<Vec<_>>(), "packing differs for stride={} part={} rotation={} start={}", stride, part, rotation, start);
                    }
                }
            }
        }

        let mut weighted_input: Vec<_> = source.clone();
        weighted_input[0] = Bn256Fr::ZERO;
        weighted_input[1] = Bn256Fr::ONE;
        weighted_input[2] = -Bn256Fr::ONE;
        let weighted_input_before = weighted_input.clone();
        let fold_values: Vec<_> = (0..base_n).map(|row| Bn256Fr::from((row as u64 + 11) * 19)).collect();
        let y = Bn256Fr::from(29);
        let expected_output: Vec<_> = weighted_input.iter().zip(&fold_values).map(|(accumulator, fold)| *accumulator * y + fold).collect();
        let raw_output: Vec<_> = expected_output.iter().map(|value| ScalarField::from(fr_raw_montgomery_words(value))).collect();

        let mut legacy_output = weighted_input.clone();
        for (destination, output) in legacy_output.iter_mut().zip(&raw_output) {
            let limbs = words_to_u64_limbs(raw_scalar_words(output));
            // SAFETY: these test values came from canonical BN254 Fr values
            // and are restored using the same pinned raw-Montgomery layout
            // as the former element-by-element output path.
            unsafe { std::ptr::write(destination as *mut Bn256Fr as *mut [u64; 4], limbs); }
        }
        let mut bulk_output = weighted_input.clone();
        copy_validated_bn254_h_output_chunk(&raw_output, &mut bulk_output).expect("validated bulk H output writeback");
        assert_eq!(bulk_output, expected_output, "bulk output writeback differs from the weighted H field oracle");
        assert_eq!(bulk_output, legacy_output, "bulk output writeback differs from the legacy word path");
        assert_eq!(weighted_input, weighted_input_before, "query packing must not mutate the weighted H input accumulator");

        let invalid_residue = ScalarField::from([
            BN254_MODULUS_LIMBS[0] as u32, (BN254_MODULUS_LIMBS[0] >> 32) as u32,
            BN254_MODULUS_LIMBS[1] as u32, (BN254_MODULUS_LIMBS[1] >> 32) as u32,
            BN254_MODULUS_LIMBS[2] as u32, (BN254_MODULUS_LIMBS[2] >> 32) as u32,
            BN254_MODULUS_LIMBS[3] as u32, (BN254_MODULUS_LIMBS[3] >> 32) as u32,
        ]);
        let invalid_chunk = [raw_output[0].clone(), invalid_residue];
        let mut unchanged = [Bn256Fr::from(101), Bn256Fr::from(202)];
        let before_invalid_write = unchanged;
        let error = copy_validated_bn254_h_output_chunk(&invalid_chunk, &mut unchanged).expect_err("noncanonical H output must fail");
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert_eq!(unchanged, before_invalid_write, "invalid output must not partially mutate the accumulator chunk");
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
            mv_lookup: None,
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
