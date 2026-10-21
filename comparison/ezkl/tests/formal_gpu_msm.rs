#![cfg(feature = "formal-gpu")]

use halo2_proofs::{
    arithmetic::{best_multiexp_cpu, best_multiexp_gpu, Field},
    icicle,
    poly::{commitment::{Blind, Params, ParamsProver}, EvaluationDomain},
    poly::kzg::commitment::ParamsKZG,
};
use halo2curves::group::{prime::PrimeCurveAffine, Curve};
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::PrimeField;
use rand::rngs::OsRng;
use std::time::Instant;

const LOG_N: u32 = 21;

fn scalars(size: usize, salt: u64) -> Vec<Fr> {
    const FR_MODULUS_TOP_LIMB: u64 = 0x3064_4e72_e131_a029;
    assert!(size >= 6, "full-field MSM fixture requires its six explicit boundary scalars");
    let mut values = vec![
        Fr::ZERO,
        Fr::from(1),
        -Fr::from(1),
        Fr::from(u64::MAX),
        Fr::from(u64::MAX - 1),
        Fr::from_raw([0, 1, 0, 0]), // 2^64
    ];
    values.reserve(size - values.len());
    for index in values.len()..size {
        let mut state = salt ^ (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut next_limb = || {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        };
        // Keeping the highest limb below the modulus limb makes the generated
        // four-limb integer canonical while exercising its upper scalar windows.
        let low_limbs = [next_limb(), next_limb(), next_limb()];
        let top_limb = if index == 6 {
            FR_MODULUS_TOP_LIMB - 1
        } else {
            next_limb() % FR_MODULUS_TOP_LIMB
        };
        let raw = [low_limbs[0], low_limbs[1], low_limbs[2], top_limb];
        values.push(Fr::from_raw(raw));
    }
    assert_eq!(values[0], Fr::ZERO, "full-field fixture lost its zero scalar");
    assert_eq!(values[1], Fr::from(1), "full-field fixture lost its one scalar");
    assert_eq!(values[2], -Fr::from(1), "full-field fixture lost its minus-one scalar");
    assert_eq!(values[3], Fr::from(u64::MAX), "full-field fixture lost its u64 maximum scalar");
    assert_eq!(values[4], Fr::from(u64::MAX - 1), "full-field fixture lost its adjacent u64 boundary scalar");
    assert_eq!(values[5], Fr::from_raw([0, 1, 0, 0]), "full-field fixture lost its 2^64 scalar");
    assert!(
        values[6..].iter().any(|scalar| scalar.to_repr()[8..].iter().any(|byte| *byte != 0)),
        "full-field fixture samples must include scalars above 64 bits"
    );
    assert!(
        values[6..].iter().any(|scalar| scalar.to_repr()[31] & 0x30 != 0),
        "full-field fixture samples must exercise a nonzero highest effective scalar window"
    );
    values
}

#[test]
fn formal_gpu_msm_factor4_matches_cpu_and_reuses_owned_bases() {
    assert_eq!(std::env::var("ENABLE_ICICLE_GPU").unwrap_or_default(), "1", "formal GPU MSM comparison must not silently use CPU");
    let device_id = std::env::var("ICICLE_DEVICE_ID").unwrap_or_else(|_| "0".into()).parse::<usize>().expect("ICICLE_DEVICE_ID must be a local device index");
    let init_started = Instant::now();
    icicle::initialize_device(device_id).expect("initialize formal ICICLE MSM device");
    let device_init_ms = init_started.elapsed().as_secs_f64() * 1000.0;
    let module_warmup_started = Instant::now();
    let warmup_base = [G1Affine::generator()];
    let warmup_scalar = [Fr::from(1)];
    assert_eq!(best_multiexp_gpu::<G1Affine>(&warmup_scalar, &warmup_base), warmup_base[0].to_curve(), "ICICLE MSM module warm-up differs from its one-point CPU result");
    let msm_module_warmup_ms = module_warmup_started.elapsed().as_secs_f64() * 1000.0;
    icicle::reset_execution_operation_stats();
    assert_eq!(icicle::execution_stats().15, 0, "formal MSM comparison requires an empty resident-basis cache");
    let _prove_scope = icicle::native_prove_ntt_scope();
    let size = 1usize << LOG_N;
    let expected_peak_upper = 2 * 64 * size * icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR;
    let point_cycle = (1u64..=16).map(|value| (G1Affine::generator() * Fr::from(value)).to_affine()).collect::<Vec<_>>();
    let known_bases = (0..size).map(|index| point_cycle[index % point_cycle.len()]).collect::<Vec<_>>();
    let mut rng = OsRng;
    let seed_params = ParamsKZG::<Bn256>::setup(4, &mut rng);
    let params_factor1 = seed_params.from_parts(LOG_N, known_bases.clone(), Some(known_bases.clone()), seed_params.g2(), seed_params.s_g2());
    let domain = EvaluationDomain::<Fr>::new(2, LOG_N);
    let coefficients_one = scalars(size, 1);
    let coefficients_two = scalars(size, 29);
    assert_ne!(coefficients_one, coefficients_two, "MSM comparison must use distinct full-field scalar vectors");
    let cpu_started = Instant::now();
    let cpu_one = best_multiexp_cpu::<G1Affine>(&coefficients_one, &known_bases);
    let cpu_two = best_multiexp_cpu::<G1Affine>(&coefficients_two, &known_bases);
    let cpu_oracle_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
    let ordinary_started = Instant::now();
    let ordinary_one = best_multiexp_gpu::<G1Affine>(&coefficients_one, &known_bases);
    let ordinary_factor1_ms = ordinary_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(ordinary_one, cpu_one, "ordinary factor-1 ICICLE MSM differs from the CPU oracle");
    let prepared_factor1_cold_started = Instant::now();
    let prepared_factor1_cold_one = params_factor1.commit_coefficients_with_icicle_precompute_factor_for_test(&coefficients_one, 1);
    let prepared_factor1_first_total_ms = prepared_factor1_cold_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(prepared_factor1_cold_one, cpu_one, "cold resident factor-1 KZG MSM differs from the CPU oracle");
    let cloned_params_factor1 = params_factor1.clone();
    let prepared_factor1_warm_started = Instant::now();
    let prepared_factor1_warm_two = cloned_params_factor1.commit_coefficients_with_icicle_precompute_factor_for_test(&coefficients_two, 1);
    let prepared_factor1_warm_online_ms = prepared_factor1_warm_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(prepared_factor1_warm_two, cpu_two, "warm resident factor-1 KZG MSM differs from the CPU oracle for new scalars");
    drop(cloned_params_factor1);
    let factor1_stats = icicle::execution_stats();
    assert_eq!(factor1_stats.12, 2, "resident factor-1 coefficient basis must be prepared as two chunks");
    assert_eq!(factor1_stats.13, 4, "resident factor-1 cold and warm commitments must each use both prepared chunks");
    assert_eq!(factor1_stats.14, 2, "resident factor-1 warm commitment must hit both shared ParamsKZG cache chunks");
    assert_eq!(factor1_stats.15, size * 64, "resident factor-1 basis must occupy one canonical point per SRS element");
    assert!(factor1_stats.16 >= factor1_stats.15, "resident factor-1 peak must cover its live basis");
    assert!(factor1_stats.16 <= expected_peak_upper, "resident factor-1 peak exceeds the factor-4 two-basis upper bound");
    drop(params_factor1);
    assert_eq!(icicle::execution_stats().15, 0, "dropping the resident factor-1 ParamsKZG owner must release its cache before factor-4 comparison");
    let params = seed_params.from_parts(LOG_N, known_bases.clone(), Some(known_bases.clone()), seed_params.g2(), seed_params.s_g2());
    let prepared_factor4_cold_started = Instant::now();
    let prepared_factor4_cold_one = params.commit_coefficients_with_icicle_precompute_factor_for_test(&coefficients_one, icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR);
    let prepared_factor4_first_total_ms = prepared_factor4_cold_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(prepared_factor4_cold_one, cpu_one, "cold resident factor-4 KZG MSM differs from the CPU oracle");
    let cloned_params = params.clone();
    let prepared_factor4_warm_started = Instant::now();
    let prepared_warm_two = cloned_params.commit_coefficients_with_icicle_precompute_factor_for_test(&coefficients_two, icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR);
    let prepared_factor4_warm_online_ms = prepared_factor4_warm_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(prepared_warm_two, cpu_two, "warm factor-4 KZG MSM differs from the CPU oracle for new scalars");
    drop(cloned_params);
    let lagrange_values = scalars(size, 43);
    let cpu_lagrange = best_multiexp_cpu::<G1Affine>(&lagrange_values, &known_bases);
    let lagrange_poly = domain.lagrange_from_vec(lagrange_values);
    let gpu_lagrange = params.commit_lagrange(&lagrange_poly, Blind(Fr::ZERO));
    assert_eq!(gpu_lagrange, cpu_lagrange, "resident Lagrange-basis MSM differs from the CPU oracle");
    let live_after_prepared = icicle::execution_stats();
    let expected_live = size * 64 * icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR * 2;
    assert_eq!(live_after_prepared.12, 6, "resident factor-1 coefficient and factor-4 coefficient/Lagrange bases must each prepare two chunks");
    assert_eq!(live_after_prepared.13, 10, "resident factor-1 and factor-4 cold/warm comparisons must use ten prepared chunks");
    assert_eq!(live_after_prepared.14, 4, "resident factor-1 and factor-4 warm commitments must each hit both cached chunks");
    assert_eq!(live_after_prepared.15, expected_live, "resident coefficient and Lagrange basis bytes differ from their typed allocation size");
    assert!(live_after_prepared.16 >= live_after_prepared.15, "resident MSM peak must cover the current live factor-4 coefficient and Lagrange bases");
    assert!(live_after_prepared.16 <= expected_peak_upper, "resident MSM peak exceeds two factor-4 SRS bases");
    let partial_size = icicle::MSM_MAX_CHUNK_SCALARS + icicle::MSM_MAX_CHUNK_SCALARS / 2;
    let partial_values = coefficients_two[..partial_size].to_vec();
    let cpu_partial = best_multiexp_cpu::<G1Affine>(&partial_values, &known_bases[..partial_size]);
    let stats_before_partial = icicle::execution_stats();
    let partial_started = Instant::now();
    let gpu_partial = params.commit_coefficients_with_icicle_precompute_factor_for_test(&partial_values, icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR);
    let partial_mixed_gpu_ms = partial_started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(gpu_partial, cpu_partial, "truncated polynomial GPU MSM differs from the CPU oracle");
    let stats_after_partial = icicle::execution_stats();
    assert_eq!(stats_after_partial.12, stats_before_partial.12, "a truncated polynomial tail must not trigger another resident basis preparation");
    assert_eq!(stats_after_partial.13, stats_before_partial.13 + 1, "the full first partial-polynomial chunk should hit one resident basis and its shorter tail should remain ordinary");
    assert_eq!(stats_after_partial.14, stats_before_partial.14 + 1, "only the full first partial-polynomial chunk should hit the resident cache");
    assert_eq!(stats_after_partial.1, stats_before_partial.1 + 2, "partial polynomial should use one cached full chunk and one ordinary GPU tail chunk");
    let prepared_factor1_estimated_prepare_ms = (prepared_factor1_first_total_ms - prepared_factor1_warm_online_ms).max(0.0);
    let prepared_factor1_saving_ms = ordinary_factor1_ms - prepared_factor1_warm_online_ms;
    let prepared_factor1_break_even_total_calls = if prepared_factor1_saving_ms > 0.0 { Some((prepared_factor1_estimated_prepare_ms / prepared_factor1_saving_ms).ceil().max(1.0) as u64) } else { None };
    let prepared_factor4_estimated_prepare_ms = (prepared_factor4_first_total_ms - prepared_factor4_warm_online_ms).max(0.0);
    let prepared_factor4_saving_ms = ordinary_factor1_ms - prepared_factor4_warm_online_ms;
    let prepared_factor4_break_even_total_calls = if prepared_factor4_saving_ms > 0.0 { Some((prepared_factor4_estimated_prepare_ms / prepared_factor4_saving_ms).ceil().max(1.0) as u64) } else { None };
    eprintln!("FORMAL_GPU_MSM_TIMING log_n={} elements={} device_init_ms={:.3} one_point_msm_module_warmup_ms={:.3} cpu_oracle_two_calls_ms={:.3} ordinary_factor1_total_ms_after_warmup={:.3} resident_factor1_first_total_ms={:.3} resident_factor1_estimated_prepare_ms={:.3} resident_factor1_warm_online_ms={:.3} resident_factor1_break_even_total_calls={:?} resident_factor4_first_total_ms={:.3} resident_factor4_estimated_prepare_ms={:.3} resident_factor4_warm_online_ms={:.3} resident_factor4_break_even_total_calls={:?} partial_factor4_mixed_cached_gpu_ms={:.3} external_gpu_concurrency_must_be_recorded_by_runner", LOG_N, size, device_init_ms, msm_module_warmup_ms, cpu_oracle_ms, ordinary_factor1_ms, prepared_factor1_first_total_ms, prepared_factor1_estimated_prepare_ms, prepared_factor1_warm_online_ms, prepared_factor1_break_even_total_calls, prepared_factor4_first_total_ms, prepared_factor4_estimated_prepare_ms, prepared_factor4_warm_online_ms, prepared_factor4_break_even_total_calls, partial_mixed_gpu_ms);
    drop(params);
    assert_eq!(icicle::execution_stats().15, 0, "dropping the ParamsKZG owner must release its cached device bases");
    let mut lifecycle_params = ParamsKZG::<Bn256>::setup(8, &mut rng);
    let lifecycle_domain = EvaluationDomain::<Fr>::new(2, 8);
    let lifecycle_values = scalars(1 << 8, 71);
    let lifecycle_cpu = best_multiexp_cpu::<G1Affine>(&lifecycle_values, lifecycle_params.get_g());
    assert_eq!(lifecycle_params.commit_coefficients_with_icicle_precompute_factor_for_test(&lifecycle_values, icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR), lifecycle_cpu, "initial lifecycle resident MSM differs from CPU");
    assert!(icicle::execution_stats().15 > 0, "initial lifecycle commit did not allocate a resident basis");
    lifecycle_params.downsize(7);
    assert_eq!(icicle::execution_stats().15, 0, "downsize must replace and release the old resident-basis cache before mutating SRS bases");
    let downsized_values = scalars(1 << 7, 97);
    let downsized_domain = EvaluationDomain::<Fr>::new(2, 7);
    let downsized_cpu = best_multiexp_cpu::<G1Affine>(&downsized_values, &lifecycle_params.get_g()[..]);
    assert_eq!(lifecycle_params.commit_coefficients_with_icicle_precompute_factor_for_test(&downsized_values, icicle::FORMAL_GPU_MSM_PRECOMPUTE_FACTOR), downsized_cpu, "downsized fresh-cache MSM differs from CPU");
    drop(lifecycle_params);
    assert_eq!(icicle::execution_stats().15, 0, "dropping the downsized ParamsKZG owner must release its cached basis");
}
