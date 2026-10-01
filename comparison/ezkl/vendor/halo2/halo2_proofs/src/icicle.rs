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

static MSM_CALLS: AtomicU64 = AtomicU64::new(0);
static NTT_CALLS: AtomicU64 = AtomicU64::new(0);
static SYNCHRONIZED_CALLS: AtomicU64 = AtomicU64::new(0);
static DEVICE_ID: AtomicUsize = AtomicUsize::new(0);
static NTT_DOMAINS: Mutex<Option<HashMap<usize, (u32, Bn256Fr)>>> = Mutex::new(None);

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
pub fn execution_stats() -> (u64, u64, u64, usize) { (MSM_CALLS.load(Ordering::Relaxed), NTT_CALLS.load(Ordering::Relaxed), SYNCHRONIZED_CALLS.load(Ordering::Relaxed), DEVICE_ID.load(Ordering::Relaxed)) }

fn select_device() { device::set_device(DEVICE_ID.load(Ordering::Relaxed)).expect("ICICLE CUDA thread device selection failed"); }

fn synchronized_operation(counter: &AtomicU64) {
    CudaStream::default().synchronize().expect("ICICLE CUDA operation synchronization failed");
    counter.fetch_add(1, Ordering::Relaxed);
    SYNCHRONIZED_CALLS.fetch_add(1, Ordering::Relaxed);
}


/// Use the native CPU threshold for small multiexponentiations.
pub fn should_use_cpu_msm(size: usize) -> bool {
    size <= (1
        << u8::from_str_radix(&env::var("ICICLE_SMALL_K").unwrap_or("2".to_string()), 10).unwrap())
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
    select_device();
    let binding = icicle_scalars_from_c_scalars::<C::ScalarExt>(coeffs);
    let coeffs = HostSlice::from_slice(&binding[..]);
    let binding = icicle_points_from_c(bases);
    let bases = HostSlice::from_slice(&binding[..]);

    let mut msm_results = DeviceVec::<G1Projective>::cuda_malloc(1).unwrap();
    let cfg = msm::MSMConfig::default_for_device(DEVICE_ID.load(Ordering::Relaxed));

    msm::msm(coeffs, bases, &cfg, &mut msm_results[..]).unwrap();

    let mut msm_host_result = vec![G1Projective::zero(); 1];
    msm_results
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result[..]))
        .unwrap();

    synchronized_operation(&MSM_CALLS);
    let msm_point = c_from_icicle_point::<C>(&msm_host_result[0]);

    msm_point
}

/// Perform the native ICICLE transform with safe domain lifecycle handling.
pub fn fft_on_device<Scalar: ff::PrimeField, G: FftGroup<Scalar> + ff::PrimeField>(
    scalars: &mut [G], 
    omega: Scalar, 
    log_n: u32, 
    inverse: bool
) {
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
        initialize_domain(primitive[0], &cfg.ctx, true).expect("ICICLE NTT domain initialization failed");
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