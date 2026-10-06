use halo2_proofs::{
    arithmetic::{best_fft, best_multiexp, best_multiexp_cpu, Field},
    circuit::{Layouter, SimpleFloorPlanner, Value},
    fft::recursive::FFTData,
    plonk::{create_proof, keygen_pk, keygen_vk, verify_proof, Advice, Circuit, Column, ConstraintSystem, Error, Instance, Selector, VerifyingKey},
    poly::{commitment::Params, kzg::{commitment::{KZGCommitmentScheme, ParamsKZG}, multiopen::{ProverSHPLONK, VerifierSHPLONK}, strategy::SingleStrategy}, Rotation},
    transcript::{Blake2bRead, Blake2bWrite, Challenge255, TranscriptReadBuffer, TranscriptWriterBuffer},
    SerdeFormat,
};
use halo2curves::{bn256::{Bn256, Fr, G1, G1Affine}, ff::PrimeField, group::{Curve, Group, prime::PrimeCurveAffine}};
use rand_core::OsRng;
use std::{fs::File, path::Path};

#[derive(Clone)]
struct Config { advice: Column<Advice>, instance: Column<Instance>, selector: Selector }
#[derive(Clone, Default)]
struct TinyCircuit;
impl Circuit<Fr> for TinyCircuit {
    type Config = Config;
    type FloorPlanner = SimpleFloorPlanner;
    type Params = ();
    fn without_witnesses(&self) -> Self { Self }
    fn configure(meta: &mut ConstraintSystem<Fr>) -> Config {
        let advice = meta.advice_column(); let instance = meta.instance_column(); let selector = meta.selector();
        meta.enable_equality(advice); meta.enable_equality(instance);
        meta.create_gate("square is four", |meta| { let q = meta.query_selector(selector); let value = meta.query_advice(advice, Rotation::cur()); vec![q * (value.clone() * value - halo2_proofs::plonk::Expression::Constant(Fr::from(4))) ] });
        Config { advice, instance, selector }
    }
    fn synthesize(&self, config: Config, mut layouter: impl Layouter<Fr>) -> Result<(), Error> {
        let cell = layouter.assign_region(|| "private value", |mut region| { config.selector.enable(&mut region, 0)?; region.assign_advice(|| "value", config.advice, 0, || Value::known(Fr::from(2))) })?;
        layouter.constrain_instance(cell.cell(), config.instance, 0)
    }
}

fn cpu_verify(directory: &Path) {
    assert!(std::env::var_os("ENABLE_ICICLE_GPU").is_none());
    let params = ParamsKZG::<Bn256>::read(&mut File::open(directory.join("params.bin")).unwrap()).unwrap();
    let vk = VerifyingKey::<G1Affine>::read::<_, TinyCircuit>(&mut File::open(directory.join("vk.bin")).unwrap(), SerdeFormat::Processed, ()).unwrap();
    let proof = std::fs::read(directory.join("proof.bin")).unwrap();
    let public = [Fr::from(2)];
    let mut transcript = Blake2bRead::<_, G1Affine, Challenge255<_>>::init(&proof[..]);
    verify_proof::<KZGCommitmentScheme<Bn256>, VerifierSHPLONK<_>, _, _, _>(&params, &vk, SingleStrategy::new(&params), &[&[&public]], &mut transcript, params.n()).unwrap();
    let stats = halo2_proofs::icicle::execution_stats();
    assert_eq!(stats.0, 0);
    assert_eq!(stats.1, 0);
    println!("NATIVE_CPU_INDEPENDENT_VERIFY_OK");
}

fn gpu_fft_equivalence() {
    assert!(std::env::var_os("ENABLE_ICICLE_GPU").is_some());
    for log_n in [5u32, 7u32, 5u32] {
        let n = 1usize << log_n;
        let input = (0..n).map(|x| Fr::from(x as u64 + 1)).collect::<Vec<_>>();
        let omega = Fr::ROOT_OF_UNITY.pow_vartime([1u64 << (Fr::S - log_n)]);
        let mut gpu = input.clone();
        halo2_proofs::arithmetic::best_fft_gpu(&mut gpu, omega, log_n, false);
        let expected = (0..n).map(|i| {
            let step = omega.pow_vartime([i as u64]);
            let mut weight = Fr::ONE;
            input.iter().fold(Fr::ZERO, |sum, value| { let term = *value * weight; weight *= step; sum + term })
        }).collect::<Vec<_>>();
        assert_eq!(gpu, expected);
        halo2_proofs::arithmetic::best_fft_gpu(&mut gpu, omega.invert().unwrap(), log_n, true);
        assert_eq!(gpu, input.iter().map(|x| *x * Fr::from(n as u64)).collect::<Vec<_>>());
    }
    println!("NATIVE_GPU_FFT_EQUIVALENCE_OK");
}

fn main() {
    let root = std::env::current_dir().unwrap();
    let directory = root.join("tests/artifacts/native_gpu_check");
    match std::env::args().nth(1).as_deref() {
        Some("verify") => { cpu_verify(&directory); return; }
        Some("gpu-fft-check") => { gpu_fft_equivalence(); return; }
        _ => {}
    }

    halo2_proofs::icicle::initialize_device(0).unwrap();
    std::env::set_var("ENABLE_ICICLE_GPU", "1");
    let fft_child = std::process::Command::new(std::env::current_exe().unwrap()).arg("gpu-fft-check").status().unwrap();
    assert!(fft_child.success());

    let scalars = (1..=32).map(Fr::from).collect::<Vec<_>>();
    let bases = (1..=32).map(|x| (G1::generator() * Fr::from(x)).to_affine()).collect::<Vec<_>>();
    assert_eq!(best_multiexp::<G1Affine>(&scalars, &bases), best_multiexp_cpu::<G1Affine>(&scalars, &bases));
    let mut mixed = bases.clone(); mixed[3] = G1Affine::identity();
    assert_eq!(best_multiexp::<G1Affine>(&scalars, &mixed), best_multiexp_cpu::<G1Affine>(&scalars, &mixed));
    assert_eq!(best_multiexp::<G1Affine>(&vec![Fr::ZERO; 32], &bases), G1::identity());

    let before_chunk_test = halo2_proofs::icicle::execution_stats();
    let chunk_size = halo2_proofs::icicle::MSM_MAX_CHUNK_SCALARS + 1;
    let large_scalars = vec![Fr::ONE; chunk_size];
    let repeated_base = G1Affine::generator();
    let large_bases = vec![repeated_base; chunk_size];
    let chunked = best_multiexp::<G1Affine>(&large_scalars, &large_bases);
    assert_eq!(chunked, G1::generator() * Fr::from(chunk_size as u64));
    let after_chunk_test = halo2_proofs::icicle::execution_stats();
    assert_eq!(after_chunk_test.0 - before_chunk_test.0, 1);
    assert_eq!(after_chunk_test.1 - before_chunk_test.1, 2);
    assert_eq!(after_chunk_test.2 - before_chunk_test.2, 0);
    assert_eq!(after_chunk_test.4 - before_chunk_test.4, 2);

    let log_n = 5u32;
    let n = 1usize << log_n;
    let input = (0..n).map(|x| Fr::from(x as u64 + 1)).collect::<Vec<_>>();
    let omega = Fr::ROOT_OF_UNITY.pow_vartime([1u64 << (Fr::S - log_n)]);
    let fft_data = FFTData::new(n, omega, omega.invert().unwrap());
    let mut cpu_fft = input.clone();
    best_fft(&mut cpu_fft, omega, log_n, &fft_data, false);
    let best = (0..n).map(|i| {
        let step = omega.pow_vartime([i as u64]); let mut weight = Fr::ONE;
        input.iter().fold(Fr::ZERO, |sum, value| { let term = *value * weight; weight *= step; sum + term })
    }).collect::<Vec<_>>();
    assert_eq!(cpu_fft, best);
    let stats_after_dispatch = halo2_proofs::icicle::execution_stats();
    assert_eq!(stats_after_dispatch.2, 0);
    assert!(stats_after_dispatch.3 > 0);

    std::fs::create_dir_all(&directory).unwrap();
    let params = ParamsKZG::<Bn256>::setup(5, OsRng);
    let vk = keygen_vk(&params, &TinyCircuit).unwrap(); let pk = keygen_pk(&params, vk, &TinyCircuit).unwrap();
    let public = [Fr::from(2)]; let mut transcript = Blake2bWrite::<_, G1Affine, Challenge255<_>>::init(Vec::new());
    let before_prove = halo2_proofs::icicle::execution_stats();
    create_proof::<KZGCommitmentScheme<Bn256>, ProverSHPLONK<_>, _, _, _, _>(&params, &pk, &[TinyCircuit], &[&[&public]], OsRng, &mut transcript).unwrap();
    let after_prove = halo2_proofs::icicle::execution_stats();
    let msm_operations = after_prove.0 - before_prove.0;
    let gpu_msm_chunk_calls = after_prove.1 - before_prove.1;
    let ntt_calls = after_prove.2 - before_prove.2;
    let cpu_fft_calls = after_prove.3 - before_prove.3;
    let synchronized_calls = after_prove.4 - before_prove.4;
    assert!(msm_operations > 0 && gpu_msm_chunk_calls >= msm_operations);
    assert_eq!(ntt_calls, 0);
    assert!(cpu_fft_calls > 0);
    assert_eq!(synchronized_calls, gpu_msm_chunk_calls);
    std::fs::write(directory.join("proof.bin"), transcript.finalize()).unwrap();
    params.write(&mut File::create(directory.join("params.bin")).unwrap()).unwrap(); pk.get_vk().write(&mut File::create(directory.join("vk.bin")).unwrap(), SerdeFormat::Processed).unwrap();
    let child = std::process::Command::new(std::env::current_exe().unwrap()).arg("verify").env_remove("ENABLE_ICICLE_GPU").status().unwrap(); assert!(child.success());
    println!("NATIVE_GPU_CPU_FFT_MSM_POLICY_OK logical_msm={} physical_msm_chunks={} cpu_fft={} ntt={} synchronized={}", msm_operations, gpu_msm_chunk_calls, cpu_fft_calls, ntt_calls, synchronized_calls);
}
