use group::{ff::{PrimeField, Field}, Group};
use icicle_bn254::curve::{CurveCfg, G1Projective, ScalarField};
use halo2curves::bn256::Fr as Bn256Fr;
use icicle_cuda_runtime::memory::{DeviceVec, HostSlice};
use crate::arithmetic::FftGroup;
use std::any::{TypeId, Any};
pub use halo2curves::CurveAffine;
use icicle_core::{
    curve::Affine,
    msm,
    ntt::{initialize_domain, release_domain, ntt_inplace, NTTConfig, NTTDir},
};
use maybe_rayon::iter::IntoParallelRefIterator;
use maybe_rayon::iter::ParallelIterator;
use std::{env, mem};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::collections::HashMap;
use icicle_cuda_runtime::{device, stream::CudaStream};

static MSM_OPERATIONS: AtomicU64 = AtomicU64::new(0);
static GPU_MSM_CHUNK_CALLS: AtomicU64 = AtomicU64::new(0);
static NTT_CALLS: AtomicU64 = AtomicU64::new(0);
static CPU_FFT_CALLS: AtomicU64 = AtomicU64::new(0);
static SYNCHRONIZED_CALLS: AtomicU64 = AtomicU64::new(0);
static DEVICE_ID: AtomicUsize = AtomicUsize::new(0);
static NTT_DOMAINS: Mutex<Option<HashMap<usize, (u32, Bn256Fr)>>> = Mutex::new(None);
// Serialize all large CUDA allocations, including host-to-device conversion
// buffers, so an MSM cannot overlap the NTT workspace (or vice versa).
static GPU_OPERATION_LOCK: Mutex<()> = Mutex::new(());

/// Maximum scalar/base slice handed to one native ICICLE MSM invocation.
pub const MSM_MAX_CHUNK_SCALARS: usize = 1 << 20;
/// Fixed formal GPU profile: Halo2 CPU FFTs and chunked BN254 ICICLE MSMs.
pub const FORMAL_GPU_EXECUTION_POLICY: &str = "halo2-cpu-fft_icicle-bn254-gpu-msm-c16-chunk-1048576-v1";

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
pub fn execution_stats() -> (u64, u64, u64, u64, u64, usize) {
    (
        MSM_OPERATIONS.load(Ordering::Relaxed),
        GPU_MSM_CHUNK_CALLS.load(Ordering::Relaxed),
        NTT_CALLS.load(Ordering::Relaxed),
        CPU_FFT_CALLS.load(Ordering::Relaxed),
        SYNCHRONIZED_CALLS.load(Ordering::Relaxed),
        DEVICE_ID.load(Ordering::Relaxed),
    )
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
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K_FFT").unwrap_or("2".to_string()), 10).unwrap())
}

/// Check whether the field has a native BN254 CUDA implementation.
pub fn is_gpu_supported_field<G: Any>(_sample_element: &G) -> bool {
    match TypeId::of::<G>() {
        id if id == TypeId::of::<Bn256Fr>() => true,
        _ => false,
    }
}

fn repr_from_u32<C: CurveAffine>(u32_arr: &[u32; 8]) -> <C as CurveAffine>::Base {
    let t: &[<<C as CurveAffine>::Base as PrimeField>::Repr] =
        unsafe { mem::transmute(&u32_arr[..]) };
    return PrimeField::from_repr(t[0]).unwrap();
}

fn icicle_scalars_from_c_scalars<G: PrimeField>(coeffs: &[G]) -> Vec<ScalarField> {
    coeffs.par_iter().map(|coef| {
        let repr: [u32; 8] = unsafe { mem::transmute_copy(&coef.to_repr()) };
        ScalarField::from(repr)
    }).collect()
}

fn c_scalars_from_icicle_scalars<G: PrimeField>(scalars: &[ScalarField]) -> Vec<G> {
    scalars.par_iter().map(|scalar| {
        let repr: G::Repr = unsafe { mem::transmute_copy(scalar) };
        G::from_repr(repr).unwrap()
    }).collect()
}

fn icicle_points_from_c<C: CurveAffine>(bases: &[C]) -> Vec<Affine<CurveCfg>> {
    bases.par_iter().map(|p| {
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
    let _operation = GPU_OPERATION_LOCK.lock().expect("ICICLE GPU operation lock poisoned");
    select_device();
    assert_eq!(coeffs.len(), bases.len(), "native ICICLE MSM scalar/base length mismatch");
    assert!(!coeffs.is_empty(), "native ICICLE MSM input must not be empty");

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

/// Perform the native ICICLE transform with safe domain lifecycle handling.
pub fn fft_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    omega: Scalar, 
    log_n: u32, 
    inverse: bool
) {
    let _operation = GPU_OPERATION_LOCK.lock().expect("ICICLE GPU operation lock poisoned");
    select_device();
    let cfg = NTTConfig::<'_, ScalarField>::default_for_device(DEVICE_ID.load(Ordering::Relaxed));
    let dir = if inverse { NTTDir::kInverse } else { NTTDir::kForward };

    assert_eq!(scalars.len(), 1usize << log_n, "native ICICLE FFT shape mismatch");
    let omega = if inverse { omega.invert().unwrap() } else { omega };
    assert!(omega.pow_vartime([1u64 << log_n]) == Scalar::ONE && (log_n == 0 || omega.pow_vartime([1u64 << (log_n - 1)]) != Scalar::ONE), "invalid native ICICLE FFT root");
    let root_repr = unsafe { mem::transmute_copy(&omega.to_repr()) };
    let root = Bn256Fr::from_repr(root_repr).unwrap();
    let mut domains = NTT_DOMAINS.lock().expect("ICICLE NTT domain mutex poisoned");
    let domains = domains.get_or_insert_with(HashMap::new);
    let device_id = DEVICE_ID.load(Ordering::Relaxed);
    let replace = domains.get(&device_id).map_or(true, |(size, previous)| *size < log_n || previous.pow_vartime([1u64 << (*size - log_n)]) != root);
    if replace {
        CudaStream::default().synchronize().expect("ICICLE NTT domain synchronization failed");
        if domains.contains_key(&device_id) { release_domain::<ScalarField>(&cfg.ctx).expect("ICICLE NTT domain release failed"); }
        let primitive = icicle_scalars_from_c_scalars(&[root]);
        initialize_domain(primitive[0], &cfg.ctx, log_n >= 3).expect("ICICLE NTT domain initialization failed");
        domains.insert(device_id, (log_n, root));
    }

    let mut icicle_scalars: Vec<ScalarField> = icicle_scalars_from_c_scalars(scalars);
    let host_scalars = HostSlice::from_mut_slice(&mut icicle_scalars);

    ntt_inplace::<ScalarField, ScalarField>(
        host_scalars,
        dir,
        &cfg,
    ).unwrap();

    let c_scalars = &c_scalars_from_icicle_scalars::<G>(&mut host_scalars.as_slice())[..];
    scalars.copy_from_slice(&c_scalars);
    if inverse { let factor = G::from(scalars.len() as u64); for scalar in scalars.iter_mut() { *scalar *= factor; } }
    synchronized_operation(&NTT_CALLS);
}
