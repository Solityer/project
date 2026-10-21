#![cfg(feature = "gpu-accelerated")]

use halo2_proofs::{
    arithmetic::{best_fft, best_fft_cpu, try_gpu_fft_if_eligible, Field},
    fft::recursive::FFTData,
    icicle,
    poly::EvaluationDomain,
};
use halo2curves::bn256::Fr;
use std::time::Instant;

const LOG_N: u32 = 20;
static GPU_NTT_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn deterministic_values(size: usize) -> Vec<Fr> {
    let mut state = Fr::from(u64::MAX);
    (0..size)
        .map(|index| {
            let value = match index {
                0 => Fr::ZERO,
                1 => Fr::ONE,
                2 => -Fr::ONE,
                3 => Fr::from(u64::MAX),
                _ if index % 4 == 0 => -state,
                _ => state,
            };
            state = state.square() + Fr::from((index as u64).wrapping_add(0x85eb_ca6b));
            value
        })
        .collect()
}

#[test]
fn formal_gpu_ntt_matches_cpu_and_domain_ifft() {
    let _guard = GPU_NTT_TEST_LOCK.lock().expect("formal GPU NTT test lock poisoned");
    let device_id = std::env::var("ICICLE_DEVICE_ID")
        .unwrap_or_else(|_| "0".to_owned())
        .parse::<usize>()
        .expect("ICICLE_DEVICE_ID must be a local device index");
    icicle::reset_execution_operation_stats();

    let size = 1usize << LOG_N;
    let domain = EvaluationDomain::<Fr>::new(2, LOG_N);
    let omega = domain.get_omega();
    let omega_inv = domain.get_omega_inv();
    let fft_data = FFTData::new(size, omega, omega_inv);
    let input = deterministic_values(size);

    let cpu_started = Instant::now();
    let mut cpu_forward = input.clone();
    let cpu_forward_started = Instant::now();
    best_fft_cpu(&mut cpu_forward, omega, LOG_N, &fft_data, false);
    let cpu_forward_ms = cpu_forward_started.elapsed().as_secs_f64() * 1000.0;
    let mut cpu_inverse = cpu_forward.clone();
    let cpu_inverse_started = Instant::now();
    best_fft_cpu(&mut cpu_inverse, omega_inv, LOG_N, &fft_data, true);
    let cpu_inverse_ms = cpu_inverse_started.elapsed().as_secs_f64() * 1000.0;
    let cpu_forward_inverse_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

    let gpu_started = Instant::now();
    icicle::initialize_device(device_id).expect("initialize formal ICICLE device");
    let scope = icicle::native_prove_ntt_scope();
    assert!(icicle::native_prove_ntt_scope_active());
    let mut gpu_forward = input.clone();
    let gpu_forward_started = Instant::now();
    best_fft(&mut gpu_forward, omega, LOG_N, &fft_data, false);
    let gpu_forward_ms = gpu_forward_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(gpu_forward, cpu_forward);

    let mut gpu_inverse = cpu_forward.clone();
    let gpu_inverse_started = Instant::now();
    best_fft(&mut gpu_inverse, omega_inv, LOG_N, &fft_data, true);
    let gpu_inverse_ms = gpu_inverse_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(gpu_inverse, cpu_inverse);
    let gpu_forward_inverse_ms_including_device_init = gpu_started.elapsed().as_secs_f64() * 1000.0;

    let coefficients = domain.lagrange_to_coeff(domain.lagrange_from_vec(cpu_forward.clone()));
    let inverse_size = Fr::from(size as u64).invert().unwrap();
    let expected_coefficients = cpu_inverse
        .iter()
        .map(|coefficient| *coefficient * inverse_size)
        .collect::<Vec<_>>();
    assert_eq!(&coefficients[..], expected_coefficients.as_slice());

    let mut recovered = coefficients.to_vec();
    best_fft(&mut recovered, omega, LOG_N, &fft_data, false);
    assert_eq!(recovered, cpu_forward);
    drop(scope);
    assert!(!icicle::native_prove_ntt_scope_active());

    let stats = icicle::execution_stats();
    let scalar_bytes = std::mem::size_of::<Fr>();
    assert!(stats.2 >= 4, "eligible NTTs did not use the GPU dispatcher");
    assert!(stats.3 >= 2, "CPU oracle did not run");
    assert_eq!(stats.6, stats.2 * size as u64);
    assert_eq!(stats.7, size, "NTT high-water size must include resident capacity");
    assert_eq!(stats.8, 2 * size * scalar_bytes);
    assert_eq!(stats.9, (size + 1 + 64 + 6) * scalar_bytes);
    assert!(stats.10 <= size * scalar_bytes);
    assert_eq!(stats.11, 96 * size + 10_464);
    assert_eq!(
        icicle::FORMAL_GPU_EXECUTION_POLICY,
        "halo2_icicle-bn254-gpu-ntt-16-25-msm-lagrange-resident-f4-v4"
    );
    assert_eq!(icicle::FORMAL_GPU_MSM_MAX_CONCURRENCY, 1);
    eprintln!(
        "FORMAL_GPU_NTT_TIMING log_n={} elements={} cpu_forward_inverse_ms={:.3} gpu_forward_inverse_ms_including_device_and_first_ntt_init={:.3}",
        LOG_N,
        size,
        cpu_forward_inverse_ms,
        gpu_forward_inverse_ms_including_device_init,
    );
    eprintln!(
        "FORMAL_GPU_NTT_TRANSFORM_TIMING log_n={} elements={} cpu_forward_ms={:.3} cpu_inverse_ms={:.3} gpu_forward_ms={:.3} gpu_inverse_ms={:.3} forward_cpu_over_gpu={:.3} inverse_cpu_over_gpu={:.3}",
        LOG_N,
        size,
        cpu_forward_ms,
        cpu_inverse_ms,
        gpu_forward_ms,
        gpu_inverse_ms,
        cpu_forward_ms / gpu_forward_ms,
        cpu_inverse_ms / gpu_inverse_ms,
    );

    icicle::reset_execution_operation_stats();
    let reset_stats = icicle::execution_stats();
    assert_eq!(reset_stats.2, 0);
    assert_eq!(reset_stats.6, 0);
    assert_eq!(reset_stats.7, size);
    assert_eq!(reset_stats.8, stats.8);
    assert_eq!(reset_stats.9, stats.9);
    assert_eq!(reset_stats.10, stats.10);
    assert_eq!(reset_stats.11, stats.11);
}

#[test]
fn formal_gpu_ntt_policy_schedules_out_of_range_sizes_on_cpu() {
    assert!(icicle::should_use_cpu_fft(icicle::FORMAL_GPU_NTT_MIN_SIZE - 1));
    assert!(!icicle::should_use_cpu_fft(icicle::FORMAL_GPU_NTT_MIN_SIZE));
    assert!(!icicle::should_use_cpu_fft(icicle::FORMAL_GPU_NTT_MAX_SIZE));
    assert!(icicle::should_use_cpu_fft(icicle::FORMAL_GPU_NTT_MAX_SIZE + 1));
    assert!(icicle::should_use_cpu_fft(1usize << 28));
    assert!(icicle::should_use_cpu_fft(1usize << 10));
    assert_eq!(icicle::FORMAL_GPU_MSM_MAX_CONCURRENCY, 1);
}
