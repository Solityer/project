use group::{ff::{PrimeField, Field}, Group};
use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarField};
use halo2curves::bn256::{Fr as Bn256Fr, G1Affine as Bn256G1Affine};
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use crate::arithmetic::FftGroup;
use std::any::{TypeId, Any};
pub use halo2curves::CurveAffine;
use icicle_core::{
    curve::Affine,
    msm,
    ntt::{initialize_domain, release_domain, ntt, NTTConfig, NTTDir},
};
use std::{env, mem};
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
pub const FORMAL_GPU_EXECUTION_POLICY: &str = "halo2_icicle-bn254-gpu-ntt-16-25-msm-lagrange-resident-f4-v4";

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
pub fn execution_stats() -> (u64, u64, u64, u64, u64, usize, u64, usize, usize, usize, usize, usize, u64, u64, u64, usize, usize) {
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
    )
}

/// Reset per-proof operation counters while retaining the actual persistent-cache footprint.
pub fn reset_execution_operation_stats() {
    MSM_OPERATIONS.store(0, Ordering::Relaxed);
    GPU_MSM_CHUNK_CALLS.store(0, Ordering::Relaxed);
    GPU_MSM_PRECOMPUTE_CALLS.store(0, Ordering::Relaxed);
    GPU_MSM_PRECOMPUTED_CHUNK_CALLS.store(0, Ordering::Relaxed);
    GPU_MSM_PRECOMPUTED_CACHE_HITS.store(0, Ordering::Relaxed);
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
    workspace.host_values[..size].iter_mut().zip(scalars.iter()).for_each(|(output, scalar)| {
        let repr: [u32; 8] = unsafe { mem::transmute_copy(&scalar.to_repr()) };
        *output = ScalarField::from(repr);
    });
    NTT_HOST_STAGING_CAPACITY_BYTES.fetch_max(workspace.host_values.capacity() * mem::size_of::<ScalarField>(), Ordering::Relaxed);
    let stream = CudaStream::default();
    workspace.input.as_mut().expect("ICICLE NTT input buffer is missing").copy_from_host_async(HostSlice::from_slice(&workspace.host_values[..size]), &stream).expect("ICICLE NTT host-to-device copy failed");
    let mut cfg = cfg;
    cfg.is_async = true;
    ntt::<ScalarField, ScalarField>(&workspace.input.as_ref().unwrap()[..size], dir, &cfg, &mut workspace.output.as_mut().unwrap()[..size]).expect("ICICLE NTT execution failed");
    workspace.output.as_ref().unwrap().copy_to_host_async(HostSlice::from_mut_slice(&mut workspace.host_values[..size]), &stream).expect("ICICLE NTT device-to-host copy failed");
    stream.synchronize().expect("ICICLE NTT operation synchronization failed");
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
