use crate::circuit::region::RegionSettings;
use crate::circuit::CheckMode;
use crate::commands::CalibrationTarget;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::deploy_contract_via_solidity;
#[cfg(all(feature = "eth", feature = "reusable-verifier", not(target_arch = "wasm32")))]
use crate::eth::register_vka_via_rv;
#[allow(unused_imports)]
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::{get_contract_artifacts, verify_proof_via_solidity};
use crate::gat_adapter::{
    build_model_for_run_args, load_dataset_with_manifest, load_relation_resources_with_manifest,
    gat_settings_for_run_args, write_generated_metadata, ArtifactFile, ArtifactManifest, BoundCircuitRef, CoreRunManifest,
    FormalCliContract, GpuEvidence, PublicStatement, RoundingMode, SrsProvenance,
};
use crate::graph::input::GraphData;
use crate::graph::modules::ModuleForwardResult;
use crate::graph::{GraphCircuit, GraphSettings, GraphWitness, Model, SupportedOp, Visibility};
use crate::pfsys::{
    create_keys, create_keys_with_native_gate, load_pk,
    load_pk_with_native_backing_budget_and_timing, load_vk, save_params, save_pk,
    swap_proof_commitments, Snark,
};
use crate::pfsys::{create_proof_circuit, verify_proof_circuit, ProofSplitCommit};
use crate::pfsys::{
    canonical_fr_to_string, canonical_string_to_fr, save_vk, srs::*,
};
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::pfsys::encode_calldata;
use crate::tensor::TensorError;
use crate::RunArgs;
use crate::EZKL_BUF_CAPACITY;
use crate::{commands::*, EZKLError};
use colored::Colorize;
#[cfg(unix)]
use gag::Gag;
use halo2_proofs::dev::VerifyFailure;
use halo2_proofs::plonk::{self, Circuit};
use halo2_proofs::poly::commitment::Verifier;
use halo2_proofs::poly::commitment::{CommitmentScheme, Params};

use halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme;
use halo2_proofs::poly::kzg::multiopen::{ProverSHPLONK, VerifierSHPLONK};
use halo2_proofs::poly::kzg::{
    commitment::ParamsKZG, strategy::SingleStrategy as KZGSingleStrategy,
};
use halo2_proofs::poly::VerificationStrategy;
use halo2_proofs::transcript::{EncodedChallenge, TranscriptReadBuffer};
use halo2_proofs::SerdeFormat;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use halo2_solidity_verifier;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
#[cfg(any(feature = "gpu-accelerated", test))]
use halo2curves::ff::Field;
use halo2curves::ff::{FromUniformBytes, PrimeField, WithSmallOrderMulGroup};
use halo2curves::serde::SerdeObject;
use indicatif::{ProgressBar, ProgressStyle};
use instant::Instant;
use itertools::Itertools;
use lazy_static::lazy_static;
use log::debug;
use log::{info, trace, warn};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use snark_verifier::system::halo2::transcript::evm::EvmTranscript;
use std::fs::File;
#[cfg(any(feature = "eth", feature = "reusable-verifier"))]
use std::io::Write;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Cursor;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tabled::Tabled;
use thiserror::Error;
use tract_onnx::prelude::IntoTensor;
use tract_onnx::prelude::Tensor as TractTensor;

lazy_static! {
    #[derive(Debug)]
    /// The path to the ezkl related data.
    pub static ref EZKL_REPO_PATH: String =
        std::env::var("EZKL_REPO_PATH").unwrap_or_else(|_|
            // $HOME/.ezkl/
            format!("{}/.ezkl", std::env::var("HOME").unwrap())
        );

    /// The path to the ezkl related data (SRS)
    pub static ref EZKL_SRS_REPO_PATH: String = format!("{}/srs", *EZKL_REPO_PATH);

}

/// Set the device used for computation.
#[cfg(feature = "gpu-accelerated")]
pub fn set_device() -> Result<(), EZKLError> {
    if std::env::var_os("ENABLE_ICICLE_GPU").is_some() { crate::gat_adapter::gpu::initialize()?; }
    Ok(())
}

/// A wrapper for execution errors
#[derive(Debug, Error)]
pub enum ExecutionError {
    /// verification failed
    #[error("verification failed:\n{}", .0.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n"))]
    VerifyError(Vec<VerifyFailure>),
    /// Prover error
    #[error("[mock] {0}")]
    MockProverError(String),
}

lazy_static::lazy_static! {
    // read from env EZKL_WORKING_DIR var or default to current dir
    static ref WORKING_DIR: PathBuf = {
        let wd = std::env::var("EZKL_WORKING_DIR").unwrap_or_else(|_| ".".to_string());
        PathBuf::from(wd)
    };
}

/// Run an ezkl command with given args
pub async fn run(command: Commands) -> Result<String, EZKLError> {
    #[cfg(feature = "gpu-accelerated")]
    set_device()?;
    // set working dir
    std::env::set_current_dir(WORKING_DIR.as_path())?;

    match command {
        #[cfg(feature = "empty-cmd")]
        Commands::Empty => Ok(String::new()),
        Commands::GenSrs { srs_path, logrows } => gen_srs_cmd(srs_path, logrows as u32),
        Commands::GetSrs {
            srs_path,
            settings_path,
            logrows,
        } => get_srs_cmd(srs_path, settings_path, logrows).await,
        Commands::Table { model, args } => table(model.unwrap_or(DEFAULT_MODEL.into()), args),
        Commands::GenSettings {
            model,
            settings_path,
            args,
        } => gen_circuit_settings(
            model.unwrap_or(DEFAULT_MODEL.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            args,
        ),
        Commands::GenRandomData {
            model,
            data,
            variables,
            seed,
            min,
            max,
        } => gen_random_data(
            model.unwrap_or(DEFAULT_MODEL.into()),
            data.unwrap_or(DEFAULT_DATA.into()),
            variables,
            seed,
            min,
            max,
        ),
        Commands::CalibrateSettings {
            model,
            settings_path,
            data,
            target,
            lookup_safety_margin,
            scales,
            scale_rebase_multiplier,
            max_logrows,
        } => calibrate(
            model.unwrap_or(DEFAULT_MODEL.into()),
            data.unwrap_or(DEFAULT_DATA.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            target,
            lookup_safety_margin,
            scales,
            scale_rebase_multiplier,
            max_logrows,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::GenWitness {
            data,
            compiled_circuit,
            output,
            vk_path,
            srs_path,
        } => gen_witness(
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            data.unwrap_or(DataField(DEFAULT_DATA.into())).to_string(),
            Some(output.unwrap_or(DEFAULT_WITNESS.into())),
            vk_path,
            srs_path,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::GatPrepare {
            dataset,
            phase,
            resource_manifest,
            output_dir,
            admission_dir,
            srs_path,
            public_statement,
            public_output,
            producer_manifest,
            gpu,
        } => gat_prepare(
            dataset,
            phase,
            resource_manifest,
            output_dir,
            admission_dir,
            srs_path,
            public_statement,
            public_output,
            producer_manifest,
            gpu,
        ),
        Commands::GatProduce {
            dataset,
            resource_manifest,
            output_dir,
            gpu,
        } => gat_produce(dataset, resource_manifest, output_dir, gpu),
        Commands::GatWitness {
            dataset,
            resource_manifest,
            compiled_circuit,
            statement,
            output,
            gpu,
        } => gat_witness(
            dataset,
            resource_manifest,
            compiled_circuit,
            statement,
            output,
            gpu,
        ),
        Commands::GatContract { output } => {
            FormalCliContract::write(&output)?;
            Ok(output.display().to_string())
        }
        Commands::Mock { model, witness } => mock(
            model.unwrap_or(DEFAULT_MODEL.into()),
            witness.unwrap_or(DEFAULT_WITNESS.into()),
        ),
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::CreateEvmVerifier {
            vk_path,
            srs_path,
            settings_path,
            sol_code_path,
            abi_path,
            reusable,
        } => {
            create_evm_verifier(
                vk_path.unwrap_or(DEFAULT_VK.into()),
                srs_path,
                settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
                sol_code_path.unwrap_or(DEFAULT_SOL_CODE.into()),
                abi_path.unwrap_or(DEFAULT_VERIFIER_ABI.into()),
                reusable.unwrap_or(DEFAULT_RENDER_REUSABLE.parse().unwrap()),
            )
            .await
        }
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::EncodeEvmCalldata {
            proof_path,
            calldata_path,
            vka_path,
        } => encode_evm_calldata(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            calldata_path.unwrap_or(DEFAULT_CALLDATA.into()),
            vka_path,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
        Commands::CreateEvmVka {
            vk_path,
            srs_path,
            settings_path,
            vka_path,
            decimals,
        } => {
            create_evm_vka(
                vk_path.unwrap_or(DEFAULT_VK.into()),
                srs_path,
                settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
                vka_path.unwrap_or(DEFAULT_VKA.into()),
                decimals.unwrap_or(DEFAULT_DECIMALS.parse().unwrap()),
            )
            .await
        }

        Commands::CompileCircuit {
            model,
            compiled_circuit,
            settings_path,
        } => compile_circuit(
            model.unwrap_or(DEFAULT_MODEL.into()),
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
        ),
        Commands::Setup {
            compiled_circuit,
            srs_path,
            vk_path,
            pk_path,
            witness,
            disable_selector_compression,
        } => setup(
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            srs_path,
            vk_path.unwrap_or(DEFAULT_VK.into()),
            pk_path.unwrap_or(DEFAULT_PK.into()),
            witness,
            disable_selector_compression
                .unwrap_or(DEFAULT_DISABLE_SELECTOR_COMPRESSION.parse().unwrap()),
        ),
        Commands::SwapProofCommitments {
            proof_path,
            witness_path,
        } => swap_proof_commitments_cmd(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            witness_path.unwrap_or(DEFAULT_WITNESS.into()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),

        Commands::Prove {
            witness,
            compiled_circuit,
            pk_path,
            proof_path,
            srs_path,
            check_mode,
        } => prove(
            witness.unwrap_or(DEFAULT_WITNESS.into()),
            compiled_circuit.unwrap_or(DEFAULT_COMPILED_CIRCUIT.into()),
            pk_path.unwrap_or(DEFAULT_PK.into()),
            Some(proof_path.unwrap_or(DEFAULT_PROOF.into())),
            srs_path,
            check_mode.unwrap_or(DEFAULT_CHECKMODE.parse().unwrap()),
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        Commands::Verify {
            proof_path,
            settings_path,
            vk_path,
            srs_path,
            reduced_srs,
            statement,
            public_output,
            artifact_manifest,
            resource_manifest,
        } => verify_with_formal_inputs(
            proof_path.unwrap_or(DEFAULT_PROOF.into()),
            settings_path.unwrap_or(DEFAULT_SETTINGS.into()),
            vk_path.unwrap_or(DEFAULT_VK.into()),
            srs_path,
            reduced_srs.unwrap_or(DEFAULT_USE_REDUCED_SRS_FOR_VERIFICATION.parse().unwrap()),
            statement,
            public_output,
            artifact_manifest,
            resource_manifest,
        )
        .map(|e| serde_json::to_string(&e).unwrap()),
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::DeployEvm {
            sol_code_path,
            rpc_url,
            addr_path,
            optimizer_runs,
            private_key,
            #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
            contract,
        } => {
            deploy_evm(
                sol_code_path.unwrap_or(DEFAULT_SOL_CODE.into()),
                rpc_url,
                addr_path.unwrap_or(DEFAULT_CONTRACT_ADDRESS.into()),
                optimizer_runs,
                private_key,
                #[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
                contract,
                #[cfg(not(all(feature = "reusable-verifier", not(target_arch = "wasm32"))))]
                ContractType::default(),
            )
            .await
        }
        #[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
        Commands::VerifyEvm {
            proof_path,
            addr_verifier,
            rpc_url,
            vka_path,
            encoded_calldata,
        } => {
            verify_evm(
                proof_path.unwrap_or(DEFAULT_PROOF.into()),
                addr_verifier,
                rpc_url,
                vka_path,
                encoded_calldata,
            )
            .await
        }
        #[cfg(all(feature = "eth", feature = "reusable-verifier"))]
        Commands::RegisterVka {
            addr_verifier,
            vka_path,
            rpc_url,
            vka_digest_path,
            private_key,
        } => {
            register_vka(
                rpc_url,
                addr_verifier,
                vka_path.unwrap_or(DEFAULT_VKA.into()),
                vka_digest_path.unwrap_or(DEFAULT_VKA_DIGEST.into()),
                private_key,
            )
            .await
        }
        #[cfg(not(feature = "no-update"))]
        Commands::Update { version } => update_ezkl_binary(&version).map(|e| e.to_string()),
    }
}

#[cfg(not(feature = "no-update"))]
/// Assert that the version is valid
fn assert_version_is_valid(version: &str) -> Result<(), EZKLError> {
    let err_string = "Invalid version string. Must be in the format v0.0.0";
    if version.is_empty() {
        return Err(err_string.into());
    }
    // safe to unwrap since we know the length is not 0
    if !version.starts_with('v') {
        return Err(err_string.into());
    }

    semver::Version::parse(&version[1..])
        .map_err(|_| "Invalid version string. Must be in the format v0.0.0")?;

    Ok(())
}

#[cfg(not(feature = "no-update"))]
const INSTALL_BYTES: &[u8] = include_bytes!("../install_ezkl_cli.sh");

#[cfg(not(feature = "no-update"))]
fn update_ezkl_binary(version: &Option<String>) -> Result<String, EZKLError> {
    // run the install script with the version
    let install_script = std::str::from_utf8(INSTALL_BYTES)?;
    //  now run as sh script with the version as an argument

    // check if bash is installed
    let command = if std::process::Command::new("bash")
        .arg("--version")
        .status()
        .is_err()
    {
        log::warn!(
            "bash is not installed on this system, trying to run the install script with sh (may fail)"
        );
        "sh"
    } else {
        "bash"
    };

    let mut command = std::process::Command::new(command);
    let mut command = command.arg("-c").arg(install_script);

    if let Some(version) = version {
        assert_version_is_valid(version)?;
        command = command.arg(version)
    };
    let output = command.output()?;

    if output.status.success() {
        info!("updated binary");
        Ok("".to_string())
    } else {
        Err(format!(
            "failed to update binary: {}, {}",
            std::str::from_utf8(&output.stdout)?,
            std::str::from_utf8(&output.stderr)?
        )
        .into())
    }
}

/// Get the srs path
pub fn get_srs_path(logrows: u32, srs_path: Option<PathBuf>) -> PathBuf {
    if let Some(srs_path) = srs_path {
        srs_path
    } else {
        if !Path::new(&*EZKL_SRS_REPO_PATH).exists() {
            std::fs::create_dir_all(&*EZKL_SRS_REPO_PATH).unwrap();
        }
        Path::new(&*EZKL_SRS_REPO_PATH).join(format!("kzg{}.srs", logrows))
    }
}

fn srs_exists_check(logrows: u32, srs_path: Option<PathBuf>) -> bool {
    Path::new(&get_srs_path(logrows, srs_path)).exists()
}

pub(crate) fn gen_srs_cmd(srs_path: PathBuf, logrows: u32) -> Result<String, EZKLError> {
    let params = gen_srs::<KZGCommitmentScheme<Bn256>>(logrows);
    save_params::<KZGCommitmentScheme<Bn256>>(&srs_path, &params)?;
    Ok(String::new())
}

async fn fetch_srs(uri: &str) -> Result<Vec<u8>, EZKLError> {
    let pb = {
        let pb = init_spinner();
        pb.set_message("Downloading SRS (this may take a while) ...");
        pb
    };
    let client = reqwest::Client::new();
    // wasm doesn't require it to be mutable
    #[allow(unused_mut)]
    let mut resp = client.get(uri).body(vec![]).send().await?;
    let mut buf = vec![];
    while let Some(chunk) = resp.chunk().await? {
        buf.extend(chunk.to_vec());
    }

    pb.finish_with_message("SRS downloaded.");
    Ok(std::mem::take(&mut buf))
}

pub(crate) fn get_file_hash(path: &Path) -> Result<String, EZKLError> {
    crate::gat_adapter::schema::sha256_file_with_progress(path, "native file identity")
        .map(|(_, digest)| digest)
        .map_err(EZKLError::from)
}

fn check_srs_hash(logrows: u32, srs_path: Option<PathBuf>) -> Result<String, EZKLError> {
    let path = get_srs_path(logrows, srs_path);
    let hash = get_file_hash(&path)?;

    let predefined_hash = match crate::srs_sha::PUBLIC_SRS_SHA256_HASHES.get(&logrows) {
        Some(h) => h,
        None => return Err(format!("SRS (k={}) hash not found in public set", logrows).into()),
    };

    if hash != *predefined_hash {
        // delete file
        warn!("removing SRS file at {}", path.display());
        std::fs::remove_file(path)?;
        return Err(
            "SRS hash does not match the expected hash. Remote SRS may have been tampered with."
                .into(),
        );
    }
    Ok(hash)
}

pub(crate) async fn get_srs_cmd(
    srs_path: Option<PathBuf>,
    settings_path: Option<PathBuf>,
    logrows: Option<u32>,
) -> Result<String, EZKLError> {
    // logrows overrides settings

    let err_string = "You will need to provide a valid settings file to use the settings option. You should run gen-settings to generate a settings file (and calibrate-settings to pick optimal logrows).";

    let k = if let Some(k) = logrows {
        k
    } else if let Some(settings_p) = &settings_path {
        if settings_p.exists() {
            let settings = GraphSettings::load(settings_p)?;
            settings.run_args.logrows
        } else {
            return Err(err_string.into());
        }
    } else {
        return Err(err_string.into());
    };

    if !srs_exists_check(k, srs_path.clone()) {
        info!("SRS does not exist, downloading...");
        let srs_uri = format!("{}{}.srs", PUBLIC_SRS_URL, k);
        let mut reader = Cursor::new(fetch_srs(&srs_uri).await?);
        // check the SRS
        let pb = init_spinner();
        pb.set_message("Validating SRS (this may take a while) ...");
        let params = ParamsKZG::<Bn256>::read(&mut reader)?;
        pb.finish_with_message("SRS validated.");

        info!("Saving SRS to disk...");
        let computed_srs_path = get_srs_path(k, srs_path.clone());
        let mut file = std::fs::File::create(&computed_srs_path)?;
        let mut buffer = BufWriter::with_capacity(*EZKL_BUF_CAPACITY, &mut file);
        params.write(&mut buffer)?;

        info!(
            "Saved SRS to {}.",
            computed_srs_path.as_os_str().to_str().unwrap_or("disk")
        );

        info!("SRS downloaded");
    } else {
        info!("SRS already exists at that path");
    };
    // check the hash
    check_srs_hash(k, srs_path.clone())?;

    Ok(String::new())
}

pub(crate) fn table(model: PathBuf, run_args: RunArgs) -> Result<String, EZKLError> {
    let model = Model::from_run_args(&run_args, &model)?;
    info!("\n {}", model.table_nodes());
    Ok(String::new())
}

pub(crate) fn gen_witness(
    compiled_circuit_path: PathBuf,
    data: String,
    output: Option<PathBuf>,
    vk_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
) -> Result<GraphWitness, EZKLError> {
    // these aren't real values so the sanity checks are mostly meaningless

    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;
    let data = GraphData::from_str(&data)?;
    let settings = circuit.settings().clone();

    let vk = if let Some(vk) = vk_path {
        Some(load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
            vk,
            settings.clone(),
        )?)
    } else {
        None
    };

    let mut input = circuit.load_graph_input(&data)?;
    #[cfg(any(not(feature = "ezkl"), target_arch = "wasm32"))]
    let mut input = circuit.load_graph_input(&data)?;

    // if any of the settings have kzg visibility then we need to load the srs

    let region_settings =
        RegionSettings::all_true(settings.run_args.decomp_base, settings.run_args.decomp_legs);

    let start_time = Instant::now();
    let witness = if settings.module_requires_polycommit() {
        if get_srs_path(settings.run_args.logrows, srs_path.clone()).exists() {
            let srs: ParamsKZG<Bn256> = load_params_prover::<KZGCommitmentScheme<Bn256>>(
                srs_path.clone(),
                settings.run_args.logrows,
            )?;
            circuit.forward::<KZGCommitmentScheme<_>>(
                &mut input,
                vk.as_ref(),
                Some(&srs),
                region_settings,
            )?
        } else {
            warn!("SRS for poly commit does not exist (will be ignored)");
            circuit.forward::<KZGCommitmentScheme<Bn256>>(
                &mut input,
                vk.as_ref(),
                None,
                region_settings,
            )?
        }
    } else {
        circuit.forward::<KZGCommitmentScheme<Bn256>>(
            &mut input,
            vk.as_ref(),
            None,
            region_settings,
        )?
    };

    // print each variable tuple (symbol, value) as symbol=value
    trace!(
        "witness generation {:?} took {:?}",
        circuit
            .settings()
            .run_args
            .variables
            .iter()
            .map(|v| { format!("{}={}", v.0, v.1) })
            .collect::<Vec<_>>(),
        start_time.elapsed()
    );

    if let Some(output_path) = output {
        witness.save(output_path)?;
    }

    // print the witness in debug
    debug!("witness: \n {}", witness.as_json()?.to_colored_json_auto()?);

    Ok(witness)
}

/// Generate a circuit settings file
pub(crate) fn gen_circuit_settings(
    model_path: PathBuf,
    params_output: PathBuf,
    run_args: RunArgs,
) -> Result<String, EZKLError> {
    let circuit = GraphCircuit::from_run_args(&run_args, &model_path)?;
    let params = circuit.settings();
    params.save(&params_output)?;
    Ok(String::new())
}

/// Generate a circuit settings file
pub(crate) fn gen_random_data(
    model_path: PathBuf,
    data_path: PathBuf,
    variables: Vec<(String, usize)>,
    seed: u64,
    min: Option<f32>,
    max: Option<f32>,
) -> Result<String, EZKLError> {
    let mut file = std::fs::File::open(&model_path).map_err(|e| {
        crate::graph::errors::GraphError::ReadWriteFileError(
            model_path.display().to_string(),
            e.to_string(),
        )
    })?;

    let (tract_model, _symbol_values) = Model::load_onnx_using_tract(&mut file, &variables)?;

    let input_facts = tract_model
        .input_outlets()
        .map_err(|e| EZKLError::from(e.to_string()))?
        .iter()
        .map(|&i| tract_model.outlet_fact(i))
        .collect::<tract_onnx::prelude::TractResult<Vec<_>>>()
        .map_err(|e| EZKLError::from(e.to_string()))?;

    let min = min.unwrap_or(0.0);
    let max = max.unwrap_or(1.0);

    /// Generates a random tensor of a given size and type.
    fn random(
        sizes: &[usize],
        datum_type: tract_onnx::prelude::DatumType,
        seed: u64,
        min: f32,
        max: f32,
    ) -> TractTensor {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut tensor = TractTensor::zero::<f32>(sizes).unwrap();
        let slice = tensor.as_slice_mut::<f32>().unwrap();
        slice.iter_mut().for_each(|x| *x = rng.gen_range(min..max));
        tensor.cast_to_dt(datum_type).unwrap().into_owned()
    }

    fn tensor_for_fact(
        fact: &tract_onnx::prelude::TypedFact,
        seed: u64,
        min: f32,
        max: f32,
    ) -> TractTensor {
        if let Some(value) = &fact.konst {
            return value.clone().into_tensor();
        }

        random(
            fact.shape
                .as_concrete()
                .expect("Expected concrete shape, found: {fact:?}"),
            fact.datum_type,
            seed,
            min,
            max,
        )
    }

    let generated = input_facts
        .iter()
        .map(|v| tensor_for_fact(v, seed, min, max))
        .collect_vec();

    let data = GraphData::from_tract_data(&generated)?;

    data.save(data_path)?;

    Ok(String::new())
}

// not for wasm targets
pub(crate) fn init_spinner() -> ProgressBar {
    let pb = indicatif::ProgressBar::new_spinner();
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.enable_steady_tick(Duration::from_millis(200));
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] {spinner:.blue} {msg}")
            .unwrap()
            .tick_strings(&[
                "------ - ✨ ",
                "------ - ⏳ ",
                "------ - 🌎 ",
                "------ - 🔎 ",
                "------ - 🥹 ",
                "------ - 🫠 ",
                "------ - 👾 ",
            ]),
    );
    pb
}

// not for wasm targets
pub(crate) fn init_bar(len: u64) -> ProgressBar {
    let pb = ProgressBar::new(len);
    pb.set_draw_target(indicatif::ProgressDrawTarget::stdout());
    pb.enable_steady_tick(Duration::from_millis(200));
    let sty = ProgressStyle::with_template(
        "[{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}",
    )
    .unwrap()
    .progress_chars("##-");
    pb.set_style(sty);
    pb
}

use colored_json::ToColoredJson;

#[derive(Debug, Clone, Tabled)]
/// Accuracy tearsheet
pub struct AccuracyResults {
    mean_error: f32,
    median_error: f32,
    max_error: f32,
    min_error: f32,
    mean_abs_error: f32,
    median_abs_error: f32,
    max_abs_error: f32,
    min_abs_error: f32,
    mean_squared_error: f32,
    mean_percent_error: f32,
    mean_abs_percent_error: f32,
}

impl AccuracyResults {
    /// Create a new accuracy results struct
    pub fn new(
        mut original_preds: Vec<crate::tensor::Tensor<f32>>,
        mut calibrated_preds: Vec<crate::tensor::Tensor<f32>>,
    ) -> Result<Self, EZKLError> {
        let mut errors = vec![];
        let mut abs_errors = vec![];
        let mut squared_errors = vec![];
        let mut percentage_errors = vec![];
        let mut abs_percentage_errors = vec![];

        for (original, calibrated) in original_preds.iter_mut().zip(calibrated_preds.iter_mut()) {
            original.flatten();
            calibrated.flatten();
            let error = (original.clone() - calibrated.clone())?;
            let abs_error = error.map(|x| x.abs());
            let squared_error = error.map(|x| x.powi(2));
            let percentage_error = error.enum_map(|i, x| {
                // if everything is 0 then we can't divide by 0 so we just return 0
                let res = if original[i] == 0.0 && x == 0.0 {
                    0.0
                } else {
                    x / original[i]
                };
                Ok::<f32, TensorError>(res)
            })?;
            let abs_percentage_error = percentage_error.map(|x| x.abs());

            errors.extend(error);
            abs_errors.extend(abs_error);
            squared_errors.extend(squared_error);
            percentage_errors.extend(percentage_error);
            abs_percentage_errors.extend(abs_percentage_error);
        }

        let mean_percent_error =
            percentage_errors.iter().sum::<f32>() / percentage_errors.len() as f32;
        let mean_abs_percent_error =
            abs_percentage_errors.iter().sum::<f32>() / abs_percentage_errors.len() as f32;
        let mean_error = errors.iter().sum::<f32>() / errors.len() as f32;
        let median_error = errors[errors.len() / 2];
        let max_error = *errors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let min_error = *errors
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let mean_abs_error = abs_errors.iter().sum::<f32>() / abs_errors.len() as f32;
        let median_abs_error = abs_errors[abs_errors.len() / 2];
        let max_abs_error = *abs_errors
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        let min_abs_error = *abs_errors
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();

        let mean_squared_error = squared_errors.iter().sum::<f32>() / squared_errors.len() as f32;

        Ok(Self {
            mean_error,
            median_error,
            max_error,
            min_error,
            mean_abs_error,
            median_abs_error,
            max_abs_error,
            min_abs_error,
            mean_squared_error,
            mean_percent_error,
            mean_abs_percent_error,
        })
    }
}

/// Calibrate the circuit parameters to a given a dataset
#[allow(trivial_casts)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn calibrate(
    model_path: PathBuf,
    data: String,
    settings_path: PathBuf,
    target: CalibrationTarget,
    lookup_safety_margin: f64,
    scales: Option<Vec<crate::Scale>>,
    scale_rebase_multiplier: Vec<u32>,
    max_logrows: Option<u32>,
) -> Result<GraphSettings, EZKLError> {
    use log::error;
    use std::collections::HashMap;
    use tabled::Table;

    use crate::fieldutils::IntegerRep;

    let data = GraphData::from_str(&data)?;
    // load the pre-generated settings
    let settings = GraphSettings::load(&settings_path)?;
    // now retrieve the run args
    // we load the model to get the input and output shapes

    let model = Model::from_run_args(&settings.run_args, &model_path)?;

    let input_shapes = model.graph.input_shapes()?;

    let chunks = data.split_into_batches(input_shapes)?;
    info!("num calibration batches: {}", chunks.len());

    debug!("running onnx predictions...");
    let original_predictions = Model::run_onnx_predictions(
        &settings.run_args,
        &model_path,
        &chunks,
        model.graph.input_shapes()?,
    )?;

    let range = if let Some(scales) = scales {
        scales
    } else {
        (11..14).collect::<Vec<crate::Scale>>()
    };

    let mut found_params: Vec<GraphSettings> = vec![];

    // 2 x 2 grid
    let range_grid = range
        .iter()
        .cartesian_product(range.iter())
        .map(|(a, b)| (*a, *b))
        .collect::<Vec<(crate::Scale, crate::Scale)>>();

    // remove all entries where input_scale > param_scale
    let mut range_grid = range_grid
        .into_iter()
        .filter(|(a, b)| a <= b)
        .collect::<Vec<(crate::Scale, crate::Scale)>>();

    // if all integers
    let all_scale_0 = model
        .graph
        .get_input_types()?
        .iter()
        .all(|t| t.is_integer());
    if all_scale_0 {
        // set all a values to 0 then dedup
        range_grid = range_grid
            .iter()
            .map(|(_, b)| (0, *b))
            .sorted()
            .dedup()
            .collect::<Vec<(crate::Scale, crate::Scale)>>();
    }

    let range_grid = range_grid
        .iter()
        .cartesian_product(scale_rebase_multiplier.iter())
        .map(|(a, b)| (*a, *b))
        .collect::<Vec<((crate::Scale, crate::Scale), u32)>>();

    let mut forward_pass_res = HashMap::new();

    let pb = init_bar(range_grid.len() as u64);
    pb.set_message("calibrating...");

    let mut num_failed = 0;
    let mut num_passed = 0;
    let mut failure_reasons = vec![];

    for ((input_scale, param_scale), scale_rebase_multiplier) in range_grid {
        pb.set_message(format!(
            "i-scale: {}, p-scale: {}, rebase-(x): {}, fail: {}, pass: {}",
            input_scale.to_string().blue(),
            param_scale.to_string().blue(),
            scale_rebase_multiplier.to_string().yellow(),
            num_failed.to_string().red(),
            num_passed.to_string().green()
        ));

        let key = (input_scale, param_scale, scale_rebase_multiplier);
        forward_pass_res.insert(key, vec![]);

        let local_run_args = RunArgs {
            input_scale,
            param_scale,
            scale_rebase_multiplier,
            lookup_range: (IntegerRep::MIN, IntegerRep::MAX),
            ..settings.run_args.clone()
        };

        // if unix get a gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _r = Gag::stdout().ok();
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        let _g = Gag::stderr().ok();

        let mut circuit = match GraphCircuit::from_run_args(&local_run_args, &model_path) {
            Ok(c) => c,
            Err(e) => {
                error!("circuit creation from run args failed: {:?}", e);
                failure_reasons.push(format!(
                    "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                    input_scale.to_string().blue(),
                    param_scale.to_string().blue(),
                    scale_rebase_multiplier.to_string().yellow(),
                    e
                ));
                pb.inc(1);
                num_failed += 1;
                continue;
            }
        };

        let forward_res = chunks
            .iter()
            .map(|chunk| {
                let chunk = chunk.clone();

                let data = circuit
                    .load_graph_input(&chunk)
                    .map_err(|e| format!("failed to load circuit inputs: {}", e))?;

                let forward_res = circuit
                    .forward::<KZGCommitmentScheme<Bn256>>(
                        &mut data.clone(),
                        None,
                        None,
                        RegionSettings::all_true(
                            settings.run_args.decomp_base,
                            settings.run_args.decomp_legs,
                        ),
                    )
                    .map_err(|e| format!("failed to forward: {}", e))?;

                // push result to the hashmap
                forward_pass_res
                    .get_mut(&key)
                    .ok_or("key not found")?
                    .push(forward_res);

                Ok(()) as Result<(), String>
            })
            .collect::<Result<Vec<()>, String>>();

        match forward_res {
            Ok(_) => (),
            // typically errors will be due to the circuit overflowing the i64 limit
            Err(e) => {
                error!("forward pass failed: {:?}", e);
                pb.inc(1);
                num_failed += 1;
                failure_reasons.push(format!(
                    "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                    input_scale.to_string().blue(),
                    param_scale.to_string().blue(),
                    scale_rebase_multiplier.to_string().yellow(),
                    e
                ));
                continue;
            }
        }

        // drop the gag
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_r);
        #[cfg(all(not(not(feature = "ezkl")), unix))]
        drop(_g);

        let result = forward_pass_res.get(&key).ok_or("key not found")?;

        let min_lookup_range = result
            .iter()
            .map(|x| x.min_lookup_inputs)
            .min()
            .unwrap_or(0);

        let max_lookup_range = result
            .iter()
            .map(|x| x.max_lookup_inputs)
            .max()
            .unwrap_or(0);

        let max_range_size = result.iter().map(|x| x.max_range_size).max().unwrap_or(0);

        let res = circuit.calc_min_logrows(
            (min_lookup_range, max_lookup_range),
            max_range_size,
            max_logrows,
            lookup_safety_margin,
        );

        if res.is_ok() {
            let new_settings = circuit.settings().clone();

            let found_run_args = RunArgs {
                input_scale: new_settings.run_args.input_scale,
                param_scale: new_settings.run_args.param_scale,
                lookup_range: new_settings.run_args.lookup_range,
                logrows: new_settings.run_args.logrows,
                scale_rebase_multiplier: new_settings.run_args.scale_rebase_multiplier,
                ..settings.run_args.clone()
            };

            let found_settings = GraphSettings {
                run_args: found_run_args,
                required_lookups: new_settings.required_lookups,
                required_range_checks: new_settings.required_range_checks,
                model_output_scales: new_settings.model_output_scales,
                model_input_scales: new_settings.model_input_scales,
                num_rows: new_settings.num_rows,
                total_assignments: new_settings.total_assignments,
                total_const_size: new_settings.total_const_size,
                dynamic_lookup_params: new_settings.dynamic_lookup_params,
                shuffle_params: new_settings.shuffle_params,
                einsum_params: new_settings.einsum_params,
                ..settings.clone()
            };

            found_params.push(found_settings.clone());

            debug!(
                "found settings: \n {}",
                found_settings.as_json()?.to_colored_json_auto()?
            );
            num_passed += 1;
        } else if let Err(res) = res {
            failure_reasons.push(format!(
                "i-scale: {}, p-scale: {}, rebase-(x): {}, reason: {}",
                input_scale.to_string().blue(),
                param_scale.to_string().blue(),
                scale_rebase_multiplier.to_string().yellow(),
                res.to_string().red()
            ));
            num_failed += 1;
        }

        pb.inc(1);
    }

    pb.finish_with_message("Calibration Done.");

    if found_params.is_empty() {
        if !failure_reasons.is_empty() {
            error!("Calibration failed for the following reasons:");
            for reason in failure_reasons {
                error!("{}", reason);
            }
        }

        return Err("calibration failed, could not find any suitable parameters given the calibration dataset".into());
    }

    debug!("Found {} sets of parameters", found_params.len());

    // now find the best params according to the target
    let mut best_params = match target {
        CalibrationTarget::Resources { .. } => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| p.run_args.logrows);

            let min_logrows = param_iterator
                .next()
                .ok_or("no params found")?
                .run_args
                .logrows;

            // pick the ones that have the minimum logrows but also the largest scale:
            // this is the best tradeoff between resource usage and accuracy
            found_params
                .iter()
                .filter(|p| p.run_args.logrows == min_logrows)
                .max_by_key(|p| {
                    (
                        p.run_args.input_scale,
                        p.run_args.param_scale,
                        // we want the largest rebase multiplier as it means we can use less constraints
                        p.run_args.scale_rebase_multiplier,
                    )
                })
                .ok_or("no params found")?
                .clone()
        }
        CalibrationTarget::Accuracy => {
            let mut param_iterator = found_params.iter().sorted_by_key(|p| {
                (
                    p.run_args.input_scale,
                    p.run_args.param_scale,
                    // we want the largest rebase multiplier as it means we can use less constraints
                    p.run_args.scale_rebase_multiplier,
                )
            });

            let last = param_iterator.next_back().ok_or("no params found")?;
            let max_scale = (
                last.run_args.input_scale,
                last.run_args.param_scale,
                last.run_args.scale_rebase_multiplier,
            );

            // pick the ones that have the max scale but also the smallest logrows:
            // this is the best tradeoff between resource usage and accuracy
            found_params
                .iter()
                .filter(|p| {
                    (
                        p.run_args.input_scale,
                        p.run_args.param_scale,
                        p.run_args.scale_rebase_multiplier,
                    ) == max_scale
                })
                .min_by_key(|p| p.run_args.logrows)
                .ok_or("no params found")?
                .clone()
        }
    };

    let outputs = forward_pass_res
        .get(&(
            best_params.run_args.input_scale,
            best_params.run_args.param_scale,
            best_params.run_args.scale_rebase_multiplier,
        ))
        .ok_or("no params found")?
        .iter()
        .map(|x| x.get_float_outputs(&best_params.model_output_scales))
        .collect::<Vec<_>>();

    let accuracy_res = AccuracyResults::new(
        original_predictions.into_iter().flatten().collect(),
        outputs.into_iter().flatten().collect(),
    )?;

    let tear_sheet_table = Table::new(vec![accuracy_res]);

    warn!(
        "\n\n <------------- Numerical Fidelity Report (input_scale: {}, param_scale: {}, scale_input_multiplier: {}) ------------->\n\n{}\n\n",
        best_params.run_args.input_scale,
        best_params.run_args.param_scale,
        best_params.run_args.scale_rebase_multiplier,
        tear_sheet_table.to_string().as_str()
    );

    if matches!(target, CalibrationTarget::Resources { col_overflow: true }) {
        let lookup_log_rows = best_params.lookup_log_rows_with_blinding();
        let module_log_row = best_params.module_constraint_logrows_with_blinding();
        let instance_logrows = best_params.log2_total_instances_with_blinding();
        let dynamic_lookup_logrows =
            best_params.min_dynamic_lookup_and_shuffle_logrows_with_blinding();

        let range_check_logrows = best_params.range_check_log_rows_with_blinding();

        let mut reduction = std::cmp::max(lookup_log_rows, module_log_row);
        reduction = std::cmp::max(reduction, range_check_logrows);
        reduction = std::cmp::max(reduction, instance_logrows);
        reduction = std::cmp::max(reduction, dynamic_lookup_logrows);
        reduction = std::cmp::max(reduction, crate::graph::MIN_LOGROWS);

        info!(
            "logrows > bits, shrinking logrows: {} -> {}",
            best_params.run_args.logrows, reduction
        );

        best_params.run_args.logrows = reduction;
    }

    best_params.save(&settings_path)?;

    debug!("Saved parameters.");

    Ok(best_params)
}

pub(crate) fn mock(
    compiled_circuit_path: PathBuf,
    data_path: PathBuf,
) -> Result<String, EZKLError> {
    // mock should catch any issues by default so we set it to safe
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    let data = GraphWitness::from_path(data_path)?;

    circuit.load_graph_witness(&data)?;

    let public_inputs = circuit.prepare_public_inputs(&data)?;

    info!("Mock proof");

    let prover = halo2_proofs::dev::MockProver::run(
        circuit.settings().run_args.logrows,
        &circuit,
        vec![public_inputs],
    )
    .map_err(|e| ExecutionError::MockProverError(e.to_string()))?;

    prover.verify().map_err(ExecutionError::VerifyError)?;
    Ok(String::new())
}

#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn create_evm_verifier(
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    settings_path: PathBuf,
    sol_code_path: PathBuf,
    abi_path: PathBuf,
    reusable: bool,
) -> Result<String, EZKLError> {
    let settings = GraphSettings::load(&settings_path)?;
    let params =
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, settings.run_args.logrows)?;

    let num_instance = settings.total_instances();
    // create a scales array that is the same length as the number of instances, all populated with 0
    let scales = vec![0; num_instance.len()];
    // let poseidon_instance = settings.module_sizes.num_instances().iter().sum::<usize>();

    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)?;
    trace!("params computed");

    let generator = halo2_solidity_verifier::SolidityGenerator::new(
        &params,
        &vk,
        halo2_solidity_verifier::BatchOpenScheme::Bdfg21,
        &num_instance,
        &scales,
        0,
        0,
    );
    let (verifier_solidity, name) = if reusable {
        (generator.render_separately()?.0, "Halo2VerifierReusable") // ignore the rendered vk artifact for now and generate it in create_evm_vka
    } else {
        (generator.render()?, "Halo2Verifier")
    };

    File::create(sol_code_path.clone())?.write_all(verifier_solidity.as_bytes())?;

    // fetch abi of the contract
    let (abi, _, _) = get_contract_artifacts(sol_code_path, name, 0).await?;
    // save abi to file
    serde_json::to_writer(std::fs::File::create(abi_path)?, &abi)?;

    Ok(String::new())
}

#[cfg(feature = "reusable-verifier")]
pub(crate) async fn create_evm_vka(
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    settings_path: PathBuf,
    vka_path: PathBuf,
    decimals: usize,
) -> Result<String, EZKLError> {
    log::warn!("Reusable verifier support is experimental and may change in the future. Use at your own risk.");
    let settings = GraphSettings::load(&settings_path)?;
    let params =
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, settings.run_args.logrows)?;

    let num_poseidon_instance = settings.module_sizes.num_instances().iter().sum::<usize>();
    let num_fixed_point_instance = settings
        .model_instance_shapes
        .iter()
        .map(|x| x.iter().product::<usize>())
        .collect_vec();

    let scales = settings.get_model_instance_scales();
    let vk = load_vk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(vk_path, settings)?;
    trace!("params computed");
    // assert that the decimals must be less than or equal to 38 to prevent overflow
    if decimals > 38 {
        return Err("decimals must be less than or equal to 38".into());
    }

    let generator = halo2_solidity_verifier::SolidityGenerator::new(
        &params,
        &vk,
        halo2_solidity_verifier::BatchOpenScheme::Bdfg21,
        &num_fixed_point_instance,
        &scales,
        decimals,
        num_poseidon_instance,
    );

    let vka_words: Vec<[u8; 32]> = generator.render_separately_vka_words()?.1;
    let serialized_vka_words = bincode::serialize(&vka_words).or_else(|e| {
        Err(EZKLError::from(format!(
            "Failed to serialize vka words: {}",
            e
        )))
    })?;

    File::create(vka_path.clone())?.write_all(&serialized_vka_words)?;

    // Load in the vka words and deserialize them and check that they match the original
    let bytes = std::fs::read(vka_path)?;
    let vka_buf: Vec<[u8; 32]> = bincode::deserialize(&bytes)
        .map_err(|e| EZKLError::from(format!("Failed to deserialize vka words: {e}")))?;
    if vka_buf != vka_words {
        return Err("vka words do not match".into());
    };

    Ok(String::new())
}
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn deploy_evm(
    sol_code_path: PathBuf,
    rpc_url: String,
    addr_path: PathBuf,
    runs: usize,
    private_key: Option<String>,
    contract: ContractType,
) -> Result<String, EZKLError> {
    let contract_name = match contract {
        ContractType::Verifier { reusable: false } => "Halo2Verifier",
        ContractType::Verifier { reusable: true } => "Halo2VerifierReusable",
    };
    let contract_address = deploy_contract_via_solidity(
        sol_code_path,
        &rpc_url,
        runs,
        private_key.as_deref(),
        contract_name,
    )
    .await?;

    info!("Contract deployed at: {:#?}", contract_address);

    let mut f = File::create(addr_path)?;
    write!(f, "{:#?}", contract_address)?;
    Ok(String::new())
}

#[cfg(all(feature = "eth", feature = "reusable-verifier", not(target_arch = "wasm32")))]
pub(crate) async fn register_vka(
    rpc_url: String,
    rv_addr: H160Flag,
    vka_path: PathBuf,
    vka_digest_path: PathBuf,
    private_key: Option<String>,
) -> Result<String, EZKLError> {
    log::warn!("Reusable verifier support is experimental and may change in the future. Use at your own risk.");
    // Load the vka, which is bincode serialized, from the vka_path
    let bytes = std::fs::read(vka_path)?;
    let vka_buf: Vec<[u8; 32]> = bincode::deserialize(&bytes)
        .map_err(|e| EZKLError::from(format!("Failed to deserialize vka words: {e}")))?;
    let vka_digest = register_vka_via_rv(
        rpc_url.as_ref(),
        private_key.as_deref(),
        rv_addr.into(),
        &vka_buf,
    )
    .await?;

    info!("VKA digest: {:#?}", vka_digest);

    let mut f = File::create(vka_digest_path)?;
    write!(f, "{:#?}", vka_digest)?;
    Ok(String::new())
}

/// Encodes the calldata for the EVM verifier
/// TODO: Add a "RV address param" which will query the "RegisteredVKA" events to fetch the
/// VKA from the vka_digest.
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) fn encode_evm_calldata(
    proof_path: PathBuf,
    calldata_path: PathBuf,
    vka_path: Option<PathBuf>,
) -> Result<Vec<u8>, EZKLError> {
    let snark = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;

    let flattened_instances = snark.instances.into_iter().flatten();

    // Load the vka, which is bincode serialized, from the vka_path
    let vka_buf: Option<Vec<[u8; 32]>> =
        match vka_path {
            Some(path) => {
                let bytes = std::fs::read(path)?;
                Some(bincode::deserialize(&bytes).map_err(|e| {
                    EZKLError::from(format!("Failed to deserialize vka words: {e}"))
                })?)
            }
            None => None,
        };

    let vka: Option<&[[u8; 32]]> = vka_buf.as_deref();
    let encoded = encode_calldata(vka, &snark.proof, &flattened_instances.collect::<Vec<_>>());

    log::debug!("Encoded calldata: {:?}", encoded);

    File::create(calldata_path)?.write_all(encoded.as_slice())?;

    Ok(encoded)
}

/// TODO: Add an optional vka_digest param that will allow us to fetch the associated VKA
/// from the RegisteredVKA events on the RV.
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
pub(crate) async fn verify_evm(
    proof_path: PathBuf,
    addr_verifier: H160Flag,
    rpc_url: String,
    vka_path: Option<PathBuf>,
    encoded_calldata: Option<PathBuf>,
) -> Result<String, EZKLError> {
    let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;

    let result = verify_proof_via_solidity(
        proof.clone(),
        addr_verifier.into(),
        vka_path.map(|s| s.into()),
        rpc_url.as_ref(),
        encoded_calldata.map(|s| s.into()),
    )
    .await?;

    info!("Solidity verification result: {}", result);

    if !result {
        return Err("Solidity verification failed".into());
    }

    Ok(String::new())
}

pub(crate) fn compile_circuit(
    model_path: PathBuf,
    compiled_circuit: PathBuf,
    settings_path: PathBuf,
) -> Result<String, EZKLError> {
    let settings = GraphSettings::load(&settings_path)?;
    let circuit = GraphCircuit::from_settings(&settings, &model_path, CheckMode::UNSAFE)?;
    circuit.save(compiled_circuit)?;
    Ok(String::new())
}

fn enforce_gat_gpu(requested: bool) -> Result<(), EZKLError> {
    if !requested {
        return Ok(());
    }
    #[cfg(feature = "gpu-accelerated")]
    { crate::gat_adapter::gpu::validate_backend() }
    #[cfg(not(feature = "gpu-accelerated"))]
    {
        Err("GPU formal run requested but ezkl was not built with gpu-accelerated".into())
    }
}

fn require_authentic_srs(path: &Path) -> Result<String, EZKLError> {
    require_formal_local_path(path, "SRS")?;
    if std::fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(format!("official SRS must not be a symlink: {}", path.display()).into());
    }
    let metadata = std::fs::metadata(path)
        .map_err(|e| format!("cannot stat official SRS {}: {}", path.display(), e))?;
    if !metadata.is_file() || metadata.len() == 0 {
        return Err(format!(
            "official SRS is not a non-empty regular file: {}",
            path.display()
        )
        .into());
    }
    for marker in [
        "dummy",
        "reduced",
        "test",
        "generated",
        "placeholder",
        "fake",
    ] {
        if path
            .components()
            .filter_map(|component| component.as_os_str().to_str())
            .any(|component| component.to_ascii_lowercase().contains(marker))
        {
            return Err(
                format!("refusing non-authentic SRS path marker: {}", path.display()).into(),
            );
        }
    }
    let (_, digest) = crate::gat_adapter::schema::sha256_file_with_progress(
        path,
        "official SRS authentication",
    )
    .map_err(EZKLError::from)?;
    let expected = crate::srs_sha::PUBLIC_SRS_SHA256_HASHES
        .get(&crate::graph::MAX_PUBLIC_SRS)
        .ok_or_else(|| {
            format!(
                "native trusted SRS hash is unavailable for maximum domain k={}",
                crate::graph::MAX_PUBLIC_SRS
            )
        })?;
    if digest != *expected {
        return Err(format!(
            "official SRS digest {} does not match the native trusted maximum-domain digest {}",
            digest, expected
        )
        .into());
    }
    Ok(digest)
}

fn load_formal_kzg_params(
    path: &Path,
    target_k: u32,
    authenticated_sha256: &str,
) -> Result<ParamsKZG<Bn256>, EZKLError> {
    let source_k = crate::graph::MAX_PUBLIC_SRS;
    let trusted_sha256 = crate::srs_sha::PUBLIC_SRS_SHA256_HASHES
        .get(&source_k)
        .ok_or_else(|| format!("native trusted SRS hash is unavailable for k={}", source_k))?;
    if authenticated_sha256 != *trusted_sha256 {
        return Err("formal SRS digest is not the native trusted maximum-domain SRS".into());
    }
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let params = ParamsKZG::<Bn256>::read_custom_for_k(
        &mut reader,
        SerdeFormat::RawBytes,
        target_k,
        source_k,
    )?;
    Ok(params)
}

fn fp_values(values: &[i128]) -> Vec<Fr> {
    values
        .iter()
        .map(|value| crate::fieldutils::integer_rep_to_felt::<Fr>(*value))
        .collect()
}

fn gat_input_tensors(
    spec: &crate::gat_adapter::CanonicalGat,
) -> Result<Vec<crate::tensor::Tensor<Fr>>, EZKLError> {
    let graph = &spec.graph;
    let node_ids = graph
        .node_ids
        .iter()
        .map(|x| *x as i128)
        .collect::<Vec<_>>();
    let src = graph.src.iter().map(|x| *x as i128).collect::<Vec<_>>();
    let dst = graph.dst.iter().map(|x| *x as i128).collect::<Vec<_>>();
    let node_ptr = graph
        .node_ptr
        .iter()
        .map(|x| *x as i128)
        .collect::<Vec<_>>();
    let edge_ptr = graph
        .edge_ptr
        .iter()
        .map(|x| *x as i128)
        .collect::<Vec<_>>();
    let inputs = [
        (
            spec.features.as_slice(),
            vec![graph.n_total, spec.feature_dim],
            spec.scales.s_input,
        ),
        (node_ids.as_slice(), vec![graph.n], 0),
        (src.as_slice(), vec![graph.e], 0),
        (dst.as_slice(), vec![graph.e], 0),
        (node_ptr.as_slice(), vec![graph.g_batch + 1], 0),
        (edge_ptr.as_slice(), vec![graph.g_batch + 1], 0),
    ];
    inputs
        .into_iter()
        .map(|(values, dims, scale)| {
            let mut tensor = crate::tensor::Tensor::new(Some(&fp_values(values)), &dims)
                .map_err(EZKLError::from)?;
            tensor.set_scale(scale);
            Ok(tensor)
        })
        .collect()
}

fn gat_parameter_tensor(
    spec: &crate::gat_adapter::CanonicalGat,
) -> Result<crate::tensor::Tensor<Fr>, EZKLError> {
    let values = spec.model.flattened_params();
    if values.is_empty() {
        return Err("canonical GAT model parameter closure is empty".into());
    }
    let mut tensor = crate::tensor::Tensor::new(Some(&fp_values(&values)), &[values.len()])
        .map_err(EZKLError::from)?;
    tensor.set_scale(spec.scales.s_w_hidden[0][0]);
    Ok(tensor)
}

fn native_hash_string(value: Fr) -> String {
    canonical_fr_to_string(&value)
}

fn assert_native_gat_identity(
    circuit: &GraphCircuit,
    expected: &crate::gat_adapter::CanonicalGat,
) -> Result<(), EZKLError> {
    match circuit.settings().gat_range_gadget_version {
        Some(version)
            if version == crate::gat_adapter::range_gadget::GAT_RANGE_GADGET_VERSION => {}
        Some(version) => {
            return Err(format!(
                "formal GAT settings use unsupported bounded-range gadget version {}",
                version
            )
            .into())
        }
        None => {
            return Err(
                "formal GAT settings lack the authenticated bounded-range gadget version; regenerate resources under the current source"
                    .into(),
            )
        }
    }
    let mut found = 0usize;
    for node in circuit.model().graph.nodes.values() {
        if let crate::graph::NodeType::Node(node) = node {
            if let SupportedOp::Gat(op) = &node.opkind {
                found += 1;
                if op.spec.graph.topology_digest != expected.graph.topology_digest
                    || op.spec.model_binding.hash != expected.model_binding.hash
                    || op.spec.model_binding.canonical_model_digest
                        != expected.model_binding.canonical_model_digest
                    || op.spec.quantization.digest != expected.quantization.digest
                    || op.spec.tables.static_tables_digest != expected.tables.static_tables_digest
                    || op.spec.ledger_identity.digest != expected.ledger_identity.digest
                {
                    return Err(
                        "compiled circuit GatOp identity disagrees with authenticated resources"
                            .into(),
                    );
                }
                if !op.spec.private_params_redacted
                    || op
                        .spec
                        .model
                        .flattened_params()
                        .iter()
                        .any(|value| *value != 0)
                {
                    return Err(
                        "compiled circuit retains private GAT parameter values; refusing formal use"
                            .into(),
                    );
                }
            }
        }
    }
    if found != 1 {
        return Err(format!(
            "compiled circuit must contain exactly one native GatOp, found {}",
            found
        )
        .into());
    }
    Ok(())
}

fn assert_native_gat_public_identity(
    circuit: &GraphCircuit,
    statement: &PublicStatement,
) -> Result<(), EZKLError> {
    let mut found = 0usize;
    for node in circuit.model().graph.nodes.values() {
        if let crate::graph::NodeType::Node(node) = node {
            if let SupportedOp::Gat(op) = &node.opkind {
                found += 1;
                if op.spec.topology != statement.public_ezkl.topology
                    || op.spec.quantization != statement.public_ezkl.quant
                    || op.spec.tables.static_tables_id != statement.public_ezkl.static_tables.id
                    || op.spec.tables.static_tables_digest
                        != statement.public_ezkl.static_tables.digest
                    || op.spec.ledger_identity != statement.public_ezkl.ledger
                    || op.spec.model_binding.hash != statement.public_ezkl.model.hash
                    || op.spec.model_binding.hash_domain != statement.public_ezkl.model.hash_domain
                    || op.spec.model_binding.canonical_model_digest
                        != statement.public_ezkl.model.canonical_model_digest
                    || op.spec.model.model_arch_id != statement.model_arch_id
                    || op.spec.ezkl_security_profile_id != statement.ezkl_security_profile_id
                    || statement.public_ezkl.circuit.native.model_graph_id
                        != format!("gat:{}", op.spec.model.model_arch_id)
                {
                    return Err(
                        "compiled GatOp public identity disagrees with the formal statement".into(),
                    );
                }
                if !op.spec.private_params_redacted
                    || op
                        .spec
                        .model
                        .flattened_params()
                        .iter()
                        .any(|value| *value != 0)
                {
                    return Err(
                        "compiled circuit retains private GAT parameter values; refusing formal verification"
                            .into(),
                    );
                }
            }
        }
    }
    if found != 1 {
        return Err(format!(
            "compiled circuit must contain exactly one native GatOp, found {}",
            found
        )
        .into());
    }
    Ok(())
}

fn source_tree_hash(root: &Path) -> Result<String, EZKLError> {
    fn collect(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), EZKLError> {
        let mut entries = std::fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path)?;
            if metadata.file_type().is_symlink() {
                continue;
            }
            if metadata.is_dir() {
                collect(&path, files)?;
            } else if metadata.is_file() {
                files.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    for relative in ["src", "gat_adapter", ".cargo"] {
        let path = root.join(relative);
        if path.is_dir() {
            collect(&path, &mut files)?;
        }
    }
    for relative in ["Cargo.toml", "Cargo.lock", "build.rs", "rust-toolchain"] {
        let path = root.join(relative);
        if path.is_file() && !path.is_symlink() {
            files.push(path);
        }
    }
    files.sort_by_key(|path| path.strip_prefix(root).unwrap().to_owned());
    if files.is_empty() {
        return Err("official ezkl source tree is empty".into());
    }
    let mut canonical = Vec::new();
    for path in files {
        let relative = path
            .strip_prefix(root)
            .map_err(|e| format!("source path is outside baseline: {}", e))?
            .to_string_lossy()
            .replace('\\', "/");
        let digest = get_file_hash(&path)?;
        canonical.extend_from_slice(&(relative.len() as u64).to_be_bytes());
        canonical.extend_from_slice(relative.as_bytes());
        canonical.extend_from_slice(digest.as_bytes());
    }
    crate::gat_adapter::schema::sha256_digest(&canonical).map_err(EZKLError::from)
}

fn resolved_circuit_ref(
    loaded: &crate::gat_adapter::LoadedGat,
    settings_sha256: String,
    compiled_sha256: String,
    vk_sha256: String,
) -> Result<crate::gat_adapter::CircuitRef, EZKLError> {
    let model_graph_bytes = serde_json::to_vec(&loaded.spec.model)?;
    let model_graph_digest =
        crate::gat_adapter::schema::sha256_digest(&model_graph_bytes).map_err(EZKLError::from)?;
    Ok(crate::gat_adapter::CircuitRef {
        compiled_circuit_id: format!("compiled:{}", &compiled_sha256),
        compiled_circuit_digest: compiled_sha256,
        settings_id: format!("settings:{}", &settings_sha256),
        settings_digest: settings_sha256,
        vk_id: format!("vk:{}", &vk_sha256),
        vk_digest: vk_sha256,
        model_graph_id: format!("gat:{}", loaded.spec.model.model_arch_id),
        model_graph_digest,
        ezkl_security_profile_id: loaded.spec.ezkl_security_profile_id.clone(),
    })
}

fn copy_external_file(source: &Path, destination: &Path) -> Result<(), EZKLError> {
    if source != destination {
        if let Ok(metadata) = std::fs::symlink_metadata(destination) {
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "bound artifact destination must not be a symlink: {}",
                    destination.display()
                )
                .into());
            }
        }

        let source_size = std::fs::metadata(source)?.len();
        let source_digest = get_file_hash(source)?;
        std::fs::copy(source, destination)?;
        let destination_size = std::fs::metadata(destination)?.len();
        let destination_digest = get_file_hash(destination)?;
        if destination_size != source_size || destination_digest != source_digest {
            return Err(format!(
                "external artifact copy changed: {}",
                destination.display()
            )
            .into());
        }
    } else {
        // Keep the same-path case fail-closed on unreadable or missing inputs.
        get_file_hash(source)?;
    }
    Ok(())
}

static NEXT_SRS_TEMP_ID: AtomicU64 = AtomicU64::new(0);

fn next_srs_temp_path(destination: &Path) -> Result<PathBuf, EZKLError> {
    let parent = destination
        .parent()
        .ok_or_else(|| "bound SRS destination has no parent directory".to_string())?;
    let file_name = destination
        .file_name()
        .ok_or_else(|| "bound SRS destination has no file name".to_string())?;
    let sequence = NEXT_SRS_TEMP_ID.fetch_add(1, Ordering::Relaxed);
    let mut temporary_name = file_name.to_os_string();
    temporary_name.push(format!(".tmp.{}.{}", std::process::id(), sequence));
    Ok(parent.join(temporary_name))
}

fn sync_parent_directory(path: &Path) -> Result<(), EZKLError> {
    let parent = path
        .parent()
        .ok_or_else(|| "bound SRS destination has no parent directory".to_string())?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn materialize_authenticated_srs(
    source: &Path,
    destination: &Path,
    expected_sha256: &str,
) -> Result<(), EZKLError> {
    let source_metadata = std::fs::symlink_metadata(source)?;
    if source_metadata.file_type().is_symlink() || !source_metadata.is_file() {
        return Err("authenticated SRS source must be a regular non-symlink file".into());
    }
    let source_size = source_metadata.len();
    let source_sha256 = get_file_hash(source)?;
    if source_sha256 != expected_sha256 {
        return Err("authenticated SRS source digest changed before artifact binding".into());
    }

    let parent = destination
        .parent()
        .ok_or_else(|| "bound SRS destination has no parent directory".to_string())?;
    std::fs::create_dir_all(parent)?;
    match std::fs::symlink_metadata(destination) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(format!(
                    "bound SRS destination must be a regular non-symlink file: {}",
                    destination.display()
                )
                .into());
            }
            if metadata.len() == source_size && get_file_hash(destination)? == expected_sha256 {
                return Ok(());
            }
            return Err(format!(
                "existing bound SRS does not match the authenticated source: {}",
                destination.display()
            )
            .into());
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let mut temporary = None;
    let mut needs_copy = false;
    for _ in 0..128 {
        let candidate = next_srs_temp_path(destination)?;
        match std::fs::hard_link(source, &candidate) {
            Ok(()) => {
                temporary = Some(candidate);
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::CrossesDevices | std::io::ErrorKind::Unsupported
                ) =>
            {
                needs_copy = true;
                break;
            }
            Err(error) => return Err(error.into()),
        }
    }

    if needs_copy {
        for _ in 0..128 {
            let candidate = next_srs_temp_path(destination)?;
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&candidate)
            {
                Ok(_) => {
                    temporary = Some(candidate);
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error.into()),
            }
        }
        let candidate = temporary
            .as_deref()
            .ok_or_else(|| "could not reserve a unique temporary SRS path".to_string())?;
        if let Err(error) = std::fs::copy(source, candidate) {
            let _ = std::fs::remove_file(candidate);
            return Err(error.into());
        }
    }

    let temporary = temporary
        .ok_or_else(|| "could not create a unique temporary SRS link".to_string())?;
    let install_result = (|| -> Result<(), EZKLError> {
        let temporary_metadata = std::fs::metadata(&temporary)?;
        let temporary_sha256 = get_file_hash(&temporary)?;
        if temporary_metadata.len() != source_size || temporary_sha256 != expected_sha256 {
            return Err("temporary bound SRS does not match the authenticated source".into());
        }
        File::open(&temporary)?.sync_all()?;

        // The official runner serializes this producer stage. Recheck before
        // rename so retries never replace a prior artifact or shared SRS link.
        match std::fs::symlink_metadata(destination) {
            Ok(_) => {
                return Err(format!(
                    "bound SRS destination appeared while installing: {}",
                    destination.display()
                )
                .into())
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
        std::fs::rename(&temporary, destination)?;
        sync_parent_directory(destination)?;
        Ok(())
    })();
    if install_result.is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
    install_result
}

fn validate_statement_against_loaded(
    statement: &PublicStatement,
    spec: &crate::gat_adapter::CanonicalGat,
) -> Result<(), EZKLError> {
    if statement.model_arch_id != spec.model.model_arch_id
        || statement.quant_cfg_id != spec.quant_cfg_id
        || statement.static_table_id != spec.tables.static_tables_id
        || statement.degree_bound_id != spec.ledger.ledger_id
        || statement.ezkl_security_profile_id != spec.ezkl_security_profile_id
    {
        return Err(
            "external public statement registry/model identity disagrees with canonical resources"
                .into(),
        );
    }
    let public = &statement.public_ezkl;
    if public.topology != spec.topology
        || public.quant != spec.quantization
        || public.ledger != spec.ledger_identity
        || public.static_tables.id != spec.tables.static_tables_id
        || public.static_tables.digest != spec.tables.static_tables_digest
        || public.model.hash != spec.model_binding.hash
        || public.model.hash_domain != spec.model_binding.hash_domain
        || public.model.canonical_model_digest != spec.model_binding.canonical_model_digest
        || public.feature_table != spec.features
        || public.node_ids != spec.graph.node_ids
        || public.src != spec.graph.src
        || public.dst != spec.graph.dst
        || public.node_ptr != spec.graph.node_ptr
        || public.edge_ptr != spec.graph.edge_ptr
    {
        return Err("external public statement graph/features/registry binding disagrees with canonical resources".into());
    }
    let expected_model_graph_id = spec
        .circuit_ref
        .as_ref()
        .map(|circuit_ref| circuit_ref.model_graph_id.clone())
        .unwrap_or_else(|| format!("gat:{}", spec.model.model_arch_id));
    if public.circuit.native.model_graph_id != expected_model_graph_id {
        return Err(
            "external public statement model graph identity disagrees with canonical resources"
                .into(),
        );
    }
    if let Some(circuit_ref) = spec.circuit_ref.as_ref() {
        if public.circuit.native != *circuit_ref {
            return Err(
                "external public statement CircuitRef disagrees with bound native setup".into(),
            );
        }
    }
    Ok(())
}

/// Rebuild the exact nested native instance columns through the official
/// `GraphCircuit::prepare_public_inputs` helper.  The adapter only supplies a
/// public-only `GraphWitness` shell; it never reads or reconstructs W/trace.
fn public_instances_from_statement(
    circuit: &mut GraphCircuit,
    statement: &PublicStatement,
) -> Result<Vec<Vec<Fr>>, EZKLError> {
    let input_shapes = circuit.model().graph.input_shapes()?;
    if input_shapes.len() != 6 {
        return Err(format!(
            "compiled GAT circuit exposes {} public inputs; expected 6",
            input_shapes.len()
        )
        .into());
    }
    let values = [
        statement.public_ezkl.feature_table.clone(),
        statement
            .public_ezkl
            .node_ids
            .iter()
            .map(|value| *value as i128)
            .collect::<Vec<_>>(),
        statement
            .public_ezkl
            .src
            .iter()
            .map(|value| *value as i128)
            .collect::<Vec<_>>(),
        statement
            .public_ezkl
            .dst
            .iter()
            .map(|value| *value as i128)
            .collect::<Vec<_>>(),
        statement
            .public_ezkl
            .node_ptr
            .iter()
            .map(|value| *value as i128)
            .collect::<Vec<_>>(),
        statement
            .public_ezkl
            .edge_ptr
            .iter()
            .map(|value| *value as i128)
            .collect::<Vec<_>>(),
    ];
    let mut inputs = Vec::with_capacity(values.len());
    for (index, (shape, values)) in input_shapes.iter().zip(values).enumerate() {
        let expected = shape.iter().product::<usize>();
        if values.len() != expected {
            return Err(format!(
                "external statement input {} has {} values; compiled circuit requires {}",
                index,
                values.len(),
                expected
            )
            .into());
        }
        inputs.push(fp_values(&values));
    }
    let output_shapes = circuit.model().graph.output_shapes()?;
    if output_shapes.len() != 1 {
        return Err("compiled GAT circuit must expose exactly one output".into());
    }
    let output_len = output_shapes[0].iter().product::<usize>();
    if statement.public_ezkl.y.len() != output_len {
        return Err(format!(
            "external public Y has {} values; compiled circuit requires {}",
            statement.public_ezkl.y.len(),
            output_len
        )
        .into());
    }
    let parameter_shapes = circuit.model().graph.param_shapes()?;
    if parameter_shapes.len() != 1 {
        return Err("compiled GAT circuit must expose exactly one runtime parameter tensor".into());
    }
    let mut witness = GraphWitness::new(inputs, vec![fp_values(&statement.public_ezkl.y)]);
    witness.params = parameter_shapes
        .iter()
        .map(|shape| vec![Fr::from(0u64); shape.iter().product()])
        .collect();
    witness.processed_params = Some(ModuleForwardResult {
        poseidon_hash: Some(vec![canonical_string_to_fr(&statement.public_ezkl.model_hash)
            .map_err(EZKLError::from)?]),
        poseidon_hash_duration: None,
        polycommit: None,
    });
    circuit.load_graph_witness(&witness)?;
    Ok(vec![circuit.prepare_public_inputs(&witness)?])
}

fn gat_prepare(
    dataset: String,
    phase: GatPreparePhase,
    resource_manifest: PathBuf,
    output_dir: PathBuf,
    admission_dir: Option<PathBuf>,
    srs_path: PathBuf,
    public_statement_path: Option<PathBuf>,
    public_output_path: Option<PathBuf>,
    producer_manifest_path: Option<PathBuf>,
    gpu: bool,
) -> Result<String, EZKLError> {
    enforce_gat_gpu(gpu)?;
    require_formal_local_path(&resource_manifest, "resource manifest")?;
    require_formal_local_path(&output_dir, "output directory")?;
    require_formal_local_path(&srs_path, "SRS")?;
    if let Some(path) = admission_dir.as_ref() {
        if !matches!(phase, GatPreparePhase::Setup) {
            return Err("--admission-dir is accepted only during native GAT setup".into());
        }
        require_formal_local_path(path, "native admission directory")?;
    }
    if let Some(path) = public_statement_path.as_ref() {
        require_formal_local_path(path, "public statement")?;
    }
    if let Some(path) = public_output_path.as_ref() {
        require_formal_local_path(path, "public output")?;
    }
    if let Some(path) = producer_manifest_path.as_ref() {
        require_formal_local_path(path, "producer manifest")?;
    }
    if matches!(phase, GatPreparePhase::Final) {
        require_formal_regular_file(
            public_statement_path
                .as_ref()
                .ok_or_else(|| "final GAT preparation requires --public-statement".to_string())?,
            "public statement",
        )?;
        require_formal_regular_file(
            public_output_path
                .as_ref()
                .ok_or_else(|| "final GAT preparation requires --public-output".to_string())?,
            "public output",
        )?;
        require_formal_regular_file(
            producer_manifest_path
                .as_ref()
                .ok_or_else(|| "final GAT preparation requires --producer-manifest".to_string())?,
            "producer manifest",
        )?;
    }
    if let Ok(metadata) = std::fs::symlink_metadata(&output_dir) {
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "output directory must not be a symlink: {}",
                output_dir.display()
            )
            .into());
        }
    }
    let baseline_root = std::env::current_dir()?;
    let loaded = load_dataset_with_manifest(&baseline_root, &resource_manifest, &dataset)
        .map_err(EZKLError::from)?;
    let srs_sha256 = require_authentic_srs(&srs_path)?;
    let backend = gat_backend(gpu)?;
    match phase {
        GatPreparePhase::Admission => gat_prepare_admission(loaded, output_dir),
        GatPreparePhase::Setup => {
            gat_prepare_setup(loaded, output_dir, srs_path, srs_sha256, gpu, admission_dir)
        }
        GatPreparePhase::Final => gat_prepare_final(
            loaded,
            output_dir,
            srs_path,
            srs_sha256,
            public_statement_path
                .ok_or_else(|| "final GAT preparation requires --public-statement".to_string())?,
            public_output_path
                .ok_or_else(|| "final GAT preparation requires --public-output".to_string())?,
            producer_manifest_path
                .ok_or_else(|| "final GAT preparation requires --producer-manifest".to_string())?,
            backend,
        ),
    }
}

fn gat_backend(gpu: bool) -> Result<String, EZKLError> {
    let backend = std::env::var("EZKL_GAT_BACKEND").unwrap_or_else(|_| "cpu".into());
    if backend != "cpu" && backend != "gpu" {
        return Err(format!("unsupported formal backend: {}", backend).into());
    }
    if (backend == "gpu") != gpu {
        return Err("GAT backend environment and --gpu flag disagree".into());
    }
    Ok(backend)
}

fn native_capacity_logrows_with_blinding(rows: usize) -> Result<u32, EZKLError> {
    let rows_with_blinding = rows
        .checked_add(crate::graph::RESERVED_BLINDING_ROWS)
        .ok_or_else(|| EZKLError::from("native GAT capacity row count overflow"))?;
    let ceil_log2 = if rows_with_blinding <= 1 {
        1
    } else {
        usize::BITS - (rows_with_blinding - 1).leading_zeros()
    };
    Ok(ceil_log2)
}

#[doc(hidden)]
pub fn build_canonical_gat_settings(
    loaded: &crate::gat_adapter::LoadedGat,
) -> Result<(Model, GraphSettings), EZKLError> {
    println!(
        "EZKL_NATIVE_PHASE START stage=model_build dataset={} n_total={} n={} e={} graphs={} feature_dim={} classes={} layers={} hidden_heads={} output_heads={}",
        loaded.dataset,
        loaded.spec.graph.n_total,
        loaded.spec.graph.n,
        loaded.spec.graph.e,
        loaded.spec.graph.g_batch,
        loaded.spec.feature_dim,
        loaded.spec.model.classes,
        loaded.spec.model.hidden_layers.len(),
        loaded.spec.model.hidden_layers.iter().map(|layer| layer.heads.len()).sum::<usize>(),
        loaded.spec.model.output_heads.len(),
    );
    let model_build_started = Instant::now();
    let mut run_args = RunArgs::default();
    run_args.input_visibility = Visibility::Public;
    run_args.output_visibility = Visibility::Public;
    run_args.param_visibility = Visibility::Hashed {
        hash_is_public: true,
        outlets: vec![],
    };
    run_args.input_scale = loaded.spec.scales.s_input;
    run_args.param_scale = loaded.spec.scales.s_w_hidden[0][0];
    run_args.lookup_range = (
        loaded.spec.tables.lrelu.input_min,
        loaded.spec.tables.lrelu.input_max,
    );
    run_args.check_mode = CheckMode::UNSAFE;
    run_args.validate().map_err(EZKLError::from)?;
    let model = build_model_for_run_args(
        loaded.spec.clone(),
        RoundingMode::NearestTiesAway,
        &run_args,
    )
    .map_err(EZKLError::from)?;
    println!(
        "EZKL_NATIVE_PHASE DONE stage=model_build dataset={} elapsed_ms={:.3}",
        loaded.dataset,
        model_build_started.elapsed().as_secs_f64() * 1000.0,
    );
    let shape_started = Instant::now();
    let model_floor_logrows = GraphSettings::required_logrows_for_model(&model, &run_args)?;
    let exact_table_rows = [
        loaded.spec.tables.lrelu.rows.len(),
        loaded.spec.tables.exp.rows.len(),
        loaded.spec.tables.elu.rows.len(),
        loaded.spec.tables.max_relu.rows.len(),
    ];
    let exact_table_total_rows = exact_table_rows.iter().try_fold(0usize, |total, rows| {
        total
            .checked_add(*rows)
            .ok_or_else(|| EZKLError::from("native GAT exact-table row count overflow"))
    })?;
    let max_exact_table_rows = exact_table_rows.iter().copied().max().unwrap_or(0);
    let exact_table_total_floor_logrows =
        native_capacity_logrows_with_blinding(exact_table_total_rows)?;
    let exact_table_max_floor_logrows =
        native_capacity_logrows_with_blinding(max_exact_table_rows)?;
    let mut selected_logrows = run_args
        .logrows
        .max(model_floor_logrows)
        .max(exact_table_total_floor_logrows)
        .max(exact_table_max_floor_logrows)
        .max(crate::graph::MIN_LOGROWS);
    let mut settings = loop {
        run_args.logrows = selected_logrows;
        if selected_logrows > crate::graph::MAX_PUBLIC_SRS {
            return Err(format!(
                "canonical GAT setup requires logrows {}, exceeding official SRS maximum {}",
                selected_logrows,
                crate::graph::MAX_PUBLIC_SRS
            )
            .into());
        }
        run_args.validate().map_err(EZKLError::from)?;
        let settings = gat_settings_for_run_args(&loaded.spec, &run_args).map_err(EZKLError::from)?;
        let settings_minimum_logrows = [
            settings.log2_total_instances_with_blinding(),
            settings.module_constraint_logrows_with_blinding(),
            settings.dynamic_lookup_and_shuffle_logrows_with_blinding(),
            settings.min_dynamic_lookup_and_shuffle_logrows_with_blinding(),
            crate::graph::MIN_LOGROWS,
        ]
        .into_iter()
        .max()
        .unwrap_or(crate::graph::MIN_LOGROWS);
        if selected_logrows < settings_minimum_logrows {
            selected_logrows = settings_minimum_logrows;
            continue;
        }
        break settings;
    };
    // The shape-only adapter computes full GraphSettings without materializing
    // a witness or running GraphCircuit's expensive dummy synthesis.  Setup
    // later reconstructs the native circuit from this exact settings object.
    settings.run_args = run_args.clone();
    let domain_size = 1u64
        .checked_shl(selected_logrows)
        .ok_or_else(|| EZKLError::from("native GAT selected domain size overflow"))?;
    println!(
        "EZKL_NATIVE_PHASE DONE stage=static_capacity_preflight dataset={} model_instance_module_floor_k={} exact_table_rows_total={} exact_table_total_floor_k={} exact_table_rows_max={} exact_table_max_floor_k={} selected_k={} domain={} official_srs_max_k={} geometry=shape_only_full_relation",
        loaded.dataset,
        model_floor_logrows,
        exact_table_total_rows,
        exact_table_total_floor_logrows,
        max_exact_table_rows,
        exact_table_max_floor_logrows,
        selected_logrows,
        domain_size,
        crate::graph::MAX_PUBLIC_SRS,
    );
    info!("canonical GAT relation logrows: {}", run_args.logrows);
    println!(
        "EZKL_NATIVE_PHASE DONE stage=shape_only_gat_geometry dataset={} elapsed_ms={:.3} logical_rows={} logical_assignments={} dynamic_lookup_coord={} max_dynamic_input_len={} dynamic_lookups={}",
        loaded.dataset,
        shape_started.elapsed().as_secs_f64() * 1000.0,
        settings.num_rows,
        settings.total_assignments,
        settings.dynamic_lookup_params.total_dynamic_col_size,
        settings.dynamic_lookup_params.max_dynamic_input_len,
        settings.dynamic_lookup_params.num_dynamic_lookups,
    );
    Ok((model, settings))
}

fn build_canonical_gat_circuit(
    loaded: &crate::gat_adapter::LoadedGat,
) -> Result<GraphCircuit, EZKLError> {
    let (model, settings) = build_canonical_gat_settings(loaded)?;
    Ok(GraphCircuit::new_gat_witnessless_from_settings(
        model,
        settings,
        CheckMode::UNSAFE,
    ))
}

fn configure_canonical_constraint_system(circuit: &GraphCircuit) -> plonk::ConstraintSystem<Fr> {
    let mut cs = plonk::ConstraintSystem::<Fr>::default();
    let _config = GraphCircuit::configure_with_params(&mut cs, circuit.params());
    // halo2_proofs enables mv-lookup directly in Cargo.toml. The parent
    // ezkl feature named mv-lookup is not part of the official formal-gpu
    // feature set, so cfg(feature = "mv-lookup") silently skipped converting
    // lookup trackers into the exact chunked arguments consumed by the native
    // prover. Always use the dependency's configured native path here.
    let cs = cs.chunk_lookups();
    cs
}

#[cfg(any(feature = "gpu-accelerated", test))]
struct NativeMvLookupShape {
    argument_count: u64,
    query_polynomial_count: u64,
    max_query_polynomials_per_argument: u64,
    table_cardinality_upper_bounds: Vec<u64>,
    argument_table_group_indices: Vec<u64>,
}

#[cfg(feature = "gpu-accelerated")]
fn shape_scalar_elements(shapes: &[Vec<usize>]) -> Result<u64, EZKLError> {
    shapes.iter().try_fold(0u64, |total, shape| {
        let elements = shape.iter().try_fold(1u64, |count, dimension| {
            count
                .checked_mul(u64::try_from(*dimension).map_err(|_| {
                    "native GAT witness shape dimension exceeds u64".to_string()
                })?)
                .ok_or_else(|| "native GAT witness shape element count overflow".to_string())
        })?;
        total
            .checked_add(elements)
            .ok_or_else(|| "native GAT witness scalar count overflow".to_string().into())
    })
}

#[cfg(any(feature = "gpu-accelerated", test))]
fn selector_advice_product(
    expression: &plonk::Expression<Fr>,
) -> Option<usize> {
    let plonk::Expression::Product(left, right) = expression else {
        return None;
    };
    match (&**left, &**right) {
        (plonk::Expression::Selector(selector), plonk::Expression::Advice(_))
        | (plonk::Expression::Advice(_), plonk::Expression::Selector(selector)) => {
            Some(selector.index())
        }
        _ => None,
    }
}

#[cfg(any(feature = "gpu-accelerated", test))]
fn table_expression_cardinality_upper(
    table: &[plonk::Expression<Fr>],
    selector_activation_counts: &[u64],
    fixed_assignment_callback_counts: &[u64],
    fixed_value_explicit_zero_seen: &[bool],
    domain_size: u64,
) -> Result<u64, EZKLError> {
    if table.len() >= 2
        && matches!(&table[0], plonk::Expression::Constant(value) if *value == Fr::ONE)
    {
        let mut table_selector = None;
        let dynamic_table = table[1..].iter().all(|expression| {
            let Some(selector) = selector_advice_product(expression) else {
                return false;
            };
            match table_selector {
                Some(previous) => previous == selector,
                None => {
                    table_selector = Some(selector);
                    true
                }
            }
        });
        if dynamic_table {
            let selector = table_selector
                .ok_or_else(|| "native mv-lookup dynamic table has no activation selector".to_string())?;
            let active_rows = *selector_activation_counts
                .get(selector)
                .ok_or_else(|| "native mv-lookup table selector is outside the selector bound vector".to_string())?;
            return Ok(active_rows.saturating_add(1).min(domain_size));
        }
    }

    let mut fixed_values_upper = 0u64;
    let mut fixed_columns = 0usize;
    let mut fixed_zero_seen = false;
    for expression in table {
        match expression {
            plonk::Expression::Constant(_) => {}
            plonk::Expression::Fixed(query) => {
                fixed_columns += 1;
                // Distinct rows in a tuple of multiple independently filled
                // fixed columns cannot be bounded by the sum of each column's
                // assignment callback count. Fall back to the full domain for
                // those unsupported tuple forms instead of under-admitting a
                // table map.
                if fixed_columns > 1 {
                    return Ok(domain_size);
                }
                // Each fixed assignment or fill callback can introduce at
                // most one distinct fixed value. `fill_from_row` repeats one
                // Assigned value over its suffix, so counting every filled
                // row would overstate the table cardinality and its map cost.
                let count = *fixed_assignment_callback_counts
                    .get(query.column_index())
                    .ok_or_else(|| "native mv-lookup fixed table column is outside the fixed callback bound vector".to_string())?;
                fixed_zero_seen = *fixed_value_explicit_zero_seen
                    .get(query.column_index())
                    .ok_or_else(|| "native mv-lookup fixed zero census is outside the fixed-column bound vector".to_string())?;
                fixed_values_upper = fixed_values_upper
                    .checked_add(count)
                    .ok_or_else(|| "native mv-lookup fixed table value bound overflow".to_string())?;
            }
            _ => return Ok(domain_size),
        }
    }
    // Unassigned fixed rows evaluate to zero. If synthesis explicitly
    // assigned a supported zero representation, that default value is
    // already included in the callback bound; otherwise reserve one value
    // for the implicit zero.
    let implicit_zero = if fixed_columns == 0 || !fixed_zero_seen { 1 } else { 0 };
    Ok(fixed_values_upper.saturating_add(implicit_zero).min(domain_size))
}

#[cfg(feature = "gpu-accelerated")]
fn native_mv_lookup_shape_with_bounds(
    cs: &plonk::ConstraintSystem<Fr>,
    selector_activation_counts_upper: &[u64],
    fixed_assignment_callback_counts_upper: &[u64],
    fixed_value_explicit_zero_seen: &[bool],
    domain_size: u64,
) -> Result<NativeMvLookupShape, EZKLError> {
    let argument_count = u64::try_from(cs.lookups().len())
        .map_err(|_| "native mv-lookup argument count exceeds u64".to_string())?;
    let mut query_polynomial_count = 0u64;
    let mut max_query_polynomials_per_argument = 0u64;
    let mut table_groups: Vec<(Vec<plonk::Expression<Fr>>, u64)> = Vec::new();
    let mut argument_table_group_indices = Vec::with_capacity(cs.lookups().len());
    for lookup in cs.lookups() {
        let input_count = u64::try_from(lookup.input_expressions().len())
            .map_err(|_| "native mv-lookup query polynomial count exceeds u64".to_string())?;
        if input_count == 0 {
            return Err("native mv-lookup argument has no query polynomials".into());
        }
        query_polynomial_count = query_polynomial_count
            .checked_add(input_count)
            .ok_or_else(|| "native mv-lookup query polynomial count overflow".to_string())?;
        max_query_polynomials_per_argument = max_query_polynomials_per_argument.max(input_count);
        let table = lookup.table_expressions();
        let table_group_index = if let Some((index, _)) = table_groups
            .iter()
            .enumerate()
            .find(|(_, (expressions, _))| expressions == table)
        {
            index
        } else {
            let bound = table_expression_cardinality_upper(
                table,
                selector_activation_counts_upper,
                fixed_assignment_callback_counts_upper,
                fixed_value_explicit_zero_seen,
                domain_size,
            )?;
            if bound == 0 || bound > domain_size {
                return Err("native mv-lookup table cardinality bound is out of domain".into());
            }
            table_groups.push((table.clone(), bound));
            table_groups.len() - 1
        };
        argument_table_group_indices.push(
            u64::try_from(table_group_index)
                .map_err(|_| "native mv-lookup table group index exceeds u64".to_string())?,
        );
    }
    Ok(NativeMvLookupShape {
        argument_count,
        query_polynomial_count,
        max_query_polynomials_per_argument,
        table_cardinality_upper_bounds: table_groups.into_iter().map(|(_, bound)| bound).collect(),
        argument_table_group_indices,
    })
}

#[cfg(any(feature = "gpu-accelerated", test))]
fn native_mv_lookup_argument_table_group_indices(
    cs: &plonk::ConstraintSystem<Fr>,
) -> Result<Vec<u64>, EZKLError> {
    let mut groups: Vec<Vec<plonk::Expression<Fr>>> = Vec::new();
    let mut indices = Vec::with_capacity(cs.lookups().len());
    for lookup in cs.lookups() {
        let table = lookup.table_expressions();
        let group = if let Some(index) = groups.iter().position(|expressions| expressions == table) {
            index
        } else {
            groups.push(table.to_vec());
            groups.len() - 1
        };
        indices.push(
            u64::try_from(group)
                .map_err(|_| "native mv-lookup table group index exceeds u64".to_string())?,
        );
    }
    Ok(indices)
}

fn cap_native_selector_resource_shape(
    mut shape: plonk::NativeEvaluationShape,
    cs: &plonk::ConstraintSystem<Fr>,
    selector_fixed_columns: usize,
) -> Result<plonk::NativeEvaluationShape, EZKLError> {
    if selector_fixed_columns > cs.num_selectors() {
        return Err("native selector compression exceeds configured selector count".into());
    }

    let max_columns = cs
        .num_fixed_columns()
        .checked_add(selector_fixed_columns)
        .and_then(|n| n.checked_add(cs.num_advice_columns()))
        .and_then(|n| n.checked_add(cs.num_instance_columns()))
        .ok_or_else(|| "native evaluation resource column count overflow".to_string())?;

    for count in shape
        .custom_gate_dependency_columns
        .iter_mut()
        .chain(shape.lookup_dependency_columns.iter_mut())
        .chain(shape.shuffle_dependency_columns.iter_mut())
    {
        *count = (*count).min(max_columns);
    }

    Ok(shape)
}

#[cfg(feature = "gpu-accelerated")]
fn gat_prepare_admission(
    loaded: crate::gat_adapter::LoadedGat,
    output_dir: PathBuf,
) -> Result<String, EZKLError> {
    let output_path = output_dir.join("admission.json");
    let settings_path = output_dir.join("settings.json");
    for path in [&output_path, &settings_path] {
        if std::fs::symlink_metadata(path).is_ok() {
            return Err(format!("native GAT admission refuses to reuse output: {}", path.display()).into());
        }
    }
    std::fs::create_dir_all(&output_dir)?;
    let (model, settings) = build_canonical_gat_settings(&loaded)?;
    let circuit = GraphCircuit::new_gat_witnessless_from_settings(
        model,
        settings.clone(),
        CheckMode::UNSAFE,
    );
    let configure_started = Instant::now();
    println!(
        "EZKL_NATIVE_PHASE START stage=configure_constraint_system dataset={} k={} domain={}",
        loaded.dataset,
        settings.run_args.logrows,
        1u64 << settings.run_args.logrows,
    );
    let cs = configure_canonical_constraint_system(&circuit);

    let (selector_evaluation_cs, _) = cs
        .clone()
        .directly_convert_selectors_to_fixed(vec![Vec::new(); cs.num_selectors()], false);
    let unshared_evaluation_shape =
        plonk::native_evaluation_resource_shape::<G1Affine>(&selector_evaluation_cs);
    drop(selector_evaluation_cs);

    let logrows = settings.run_args.logrows;
    let domain_size = 1u64
        .checked_shl(logrows)
        .ok_or_else(|| format!("invalid native GAT logrows for admission: {}", logrows))?;
    // This pre-Setup envelope uses only configured constraint-system geometry.
    // The mandatory VK/PK synthesis in Setup records the exact census and
    // rejects any observation above these source-derived bounds.
    let selector_activation_counts_upper = vec![domain_size; cs.num_selectors()];
    let fixed_assignment_callback_counts_upper = vec![domain_size; cs.num_fixed_columns()];
    let fixed_value_explicit_zero_seen_upper = vec![false; cs.num_fixed_columns()];
    let selector_fixed_column_count_upper = cs.num_selectors();
    let lookup_shape = native_mv_lookup_shape_with_bounds(
        &cs,
        &selector_activation_counts_upper,
        &fixed_assignment_callback_counts_upper,
        &fixed_value_explicit_zero_seen_upper,
        domain_size,
    )?;
    let evaluation_shape = cap_native_selector_resource_shape(
        unshared_evaluation_shape,
        &cs,
        selector_fixed_column_count_upper,
    )?;
    let lookup_table_group_count = lookup_shape.table_cardinality_upper_bounds.len();
    let graph_witness_scalar_element_count = shape_scalar_elements(
        &circuit.core.model.graph.input_shapes()?
            .into_iter()
            .chain(circuit.core.model.graph.param_shapes()?)
            .chain(circuit.core.model.graph.output_shapes()?)
            .collect::<Vec<_>>(),
    )?;
    let selector_fixed_column_count = selector_fixed_column_count_upper;
    let fixed_column_count_post_compression = cs
        .num_fixed_columns()
        .checked_add(selector_fixed_column_count)
        .ok_or_else(|| "post-compression fixed-column count overflow".to_string())?;
    let permutation_cells = (cs.num_permutation_columns() as u64)
        .checked_mul(1u64.checked_shl(logrows).ok_or_else(|| "native GAT domain size overflow".to_string())?)
        .ok_or_else(|| "native permutation cell count overflow".to_string())?;
    // A copy callback can touch at most one permutation cell. The Setup
    // synthesis census narrows this configured bound before accepting the key.
    let permutation_copy_call_count_upper = permutation_cells;
    if permutation_copy_call_count_upper > permutation_cells {
        return Err("native GAT shape census exceeds the permutation-cell copy bound".into());
    }
    let permutation_successor_entry_count_upper = permutation_cells.min(
        permutation_copy_call_count_upper
            .checked_mul(2)
            .ok_or_else(|| "native permutation successor bound overflow".to_string())?,
    );
    let selector_activation_bitmap_bytes = u128::from(cs.num_selectors() as u64)
        .checked_mul((u128::from(1u64 << logrows) + 7) / 8)
        .ok_or_else(|| "selector activation bitmap byte count overflow".to_string())?;
    let selector_activation_storage_bytes_upper = selector_activation_bitmap_bytes
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add((cs.num_selectors() as u128) * 64))
        .ok_or_else(|| "selector activation storage upper bound overflow".to_string())?;
    let fixed_polynomial_read_chunk_fields_upper = 16 * 1024usize;
    let serialized_fr_bytes = Fr::default().to_repr().as_ref().len();
    let fixed_polynomial_read_raw_chunk_bytes_upper =
        fixed_polynomial_read_chunk_fields_upper * serialized_fr_bytes;
    let fixed_polynomial_read_decoded_chunk_bytes_upper =
        fixed_polynomial_read_chunk_fields_upper * std::mem::size_of::<Fr>();
    let fixed_polynomial_read_heap_scratch_bytes_upper =
        fixed_polynomial_read_raw_chunk_bytes_upper
            + fixed_polynomial_read_decoded_chunk_bytes_upper;
    let gat_transient_witness_bytes_upper =
        crate::gat_adapter::op::gat_transient_witness_bytes_upper(&loaded.spec)?;
    let advice_event_record_bytes_upper = 9usize
        .checked_add(serialized_fr_bytes)
        .ok_or_else(|| "advice event record width overflow".to_string())?;
    let advice_event_decoder_record_bytes_upper = 9usize
        .checked_add(2usize.checked_mul(serialized_fr_bytes).ok_or_else(|| "advice event decoder record width overflow".to_string())?)
        .ok_or_else(|| "advice event decoder record width overflow".to_string())?;
    let advice_event_assignment_counts_upper = vec![domain_size; cs.num_advice_columns()];
    let advice_event_assignment_count_upper = u64::try_from(cs.num_advice_columns())
        .ok()
        .and_then(|columns| columns.checked_mul(domain_size))
        .ok_or_else(|| "advice event source-bound count overflow".to_string())?;
    let advice_event_batch_size_upper = 8192usize;
    let advice_event_decoded_entry_bytes_upper =
        plonk::native_assigned_event_bytes::<Fr>();
    let advice_event_pending_batch_bytes_upper = advice_event_batch_size_upper
        .checked_mul(advice_event_decoded_entry_bytes_upper)
        .ok_or_else(|| "advice event pending-batch byte count overflow".to_string())?;
    let advice_event_flush_scratch_bytes_upper = advice_event_batch_size_upper
        .checked_mul(
            std::mem::size_of::<Option<Fr>>()
                .checked_add(std::mem::size_of::<(Fr, &mut Fr)>())
                .ok_or_else(|| "advice event flush-scratch entry width overflow".to_string())?,
        )
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<Option<Fr>>>()))
        .ok_or_else(|| "advice event flush-scratch size overflow".to_string())?;
    let advice_event_reader_buffer_bytes_upper = 8192usize
        .checked_add(std::mem::size_of::<std::io::BufReader<std::fs::File>>())
        .ok_or_else(|| "advice event reader-buffer size overflow".to_string())?;
    let advice_event_read_buffer_bytes_upper = advice_event_batch_size_upper
        .checked_mul(
            advice_event_decoded_entry_bytes_upper
                .checked_add(std::mem::size_of::<Option<Fr>>())
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<(Fr, &mut Fr)>()))
                .ok_or_else(|| "advice event read-buffer width overflow".to_string())?,
        )
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<Vec<u8>>()))
        .and_then(|bytes| bytes.checked_add(advice_event_reader_buffer_bytes_upper))
        .and_then(|bytes| bytes.checked_add(advice_event_decoder_record_bytes_upper))
        .ok_or_else(|| "advice event read-buffer byte count overflow".to_string())?;
    // One `AssignedEventColumn` state and its record buffer are retained for
    // every configured advice column during prover synthesis. `SpillLog`
    // contributes a path, optional buffered file, byte counters and Arc; the
    // explicit 64-byte allowance covers enum/alignment padding in that source
    // field sum. The writer uses std's default 8 KiB BufWriter capacity.
    let advice_event_column_metadata_bytes_upper = std::mem::size_of::<std::path::PathBuf>()
        .checked_add(std::mem::size_of::<Option<std::io::BufWriter<std::fs::File>>>())
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<u64>()))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<std::sync::Arc<()>>()))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<u8>>()))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<plonk::Assigned<Fr>>>()))
        .and_then(|bytes| bytes.checked_add(3 * std::mem::size_of::<u64>()))
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or_else(|| "advice event column metadata bound overflow".to_string())?;
    let advice_event_writer_buffer_bytes_upper = 8192usize;
    let advice_event_column_bytes_upper = advice_event_column_metadata_bytes_upper
        .checked_add(advice_event_record_bytes_upper)
        .ok_or_else(|| "advice event per-column resident bound overflow".to_string())?;
    println!(
        "EZKL_NATIVE_PHASE DONE stage=configured_geometry_source_bounds dataset={} selectors={} selector_fixed_columns_upper={} fixed_post_compression_upper={} permutation_copy_calls_upper={} successor_entries_upper={} activation_bitmap_bytes_upper={} activation_storage_bytes_upper={} activation_storage_kind=rle_bitmap_hybrid_v1 evidence=configured_bound_before_native_synthesis",
        loaded.dataset,
        cs.num_selectors(),
        selector_fixed_column_count,
        fixed_column_count_post_compression,
        permutation_copy_call_count_upper,
        permutation_successor_entry_count_upper,
        selector_activation_bitmap_bytes,
        selector_activation_storage_bytes_upper,
    );
    let scalar_columns = cs.num_fixed_columns() + cs.num_advice_columns() + cs.num_instance_columns();
    let scalar_domain_bytes = u128::from(domain_size)
        * scalar_columns as u128
        * std::mem::size_of::<Fr>() as u128;
    let degree = cs.degree();
    let quotient_degree = degree
        .checked_sub(1)
        .ok_or_else(|| "native GAT constraint degree is below two".to_string())?;
    let quotient_extent = domain_size
        .checked_mul(
            u64::try_from(quotient_degree)
                .map_err(|_| "native GAT quotient degree exceeds the supported size")?,
        )
        .ok_or_else(|| "native GAT extended proof domain overflow".to_string())?;
    let proof_domain_size = quotient_extent
        .checked_next_power_of_two()
        .ok_or_else(|| "native GAT extended proof domain is not representable".to_string())?;
    let proof_logrows = proof_domain_size.trailing_zeros();
    let uses_recursive_fft = !cfg!(target_arch = "x86_64");
    let recursive_fft_twiddle_field_elements = if uses_recursive_fft {
        u64::try_from(
            2u128
                .checked_mul(
                    u128::from(proof_domain_size - 1) + u128::from(proof_logrows),
                )
                .ok_or_else(|| "recursive FFT twiddle count overflow".to_string())?,
        )
        .map_err(|_| "recursive FFT twiddle count exceeds u64".to_string())?
    } else {
        0
    };
    let recursive_fft_cached_domain_lengths = if uses_recursive_fft {
        vec![proof_domain_size]
    } else {
        Vec::new()
    };
    let fft_cache_domain = halo2_proofs::poly::EvaluationDomain::<Fr>::new(
        u32::try_from(degree).map_err(|_| "native GAT degree exceeds u32".to_string())?,
        logrows,
    );
    let quotient_piece_count = fft_cache_domain.get_quotient_poly_degree();
    let fft_cache_entry_count = fft_cache_domain.fft_cache_entry_count();
    let fft_cache_entry_capacity = fft_cache_domain.fft_cache_entry_capacity();
    let shplonk_resource_cs = cs
        .clone()
        .with_native_selector_resource_columns(selector_fixed_column_count_upper);
    let shplonk_shape = plonk::native_shplonk_resource_shape::<Fr>(
        &shplonk_resource_cs,
        domain_size,
        usize::try_from(lookup_shape.argument_count)
            .map_err(|_| "native mv-lookup argument count exceeds host index width")?,
        1,
        quotient_piece_count,
    );
    drop(shplonk_resource_cs);
    drop(fft_cache_domain);
    let srs_fft_cache_domain = halo2_proofs::poly::EvaluationDomain::<Fr>::new(1, logrows);
    let srs_fft_cache_entry_count = srs_fft_cache_domain.fft_cache_entry_count();
    let srs_fft_cache_entry_capacity = srs_fft_cache_domain.fft_cache_entry_capacity();
    drop(srs_fft_cache_domain);
    let parallel_fft_lookup_table_elements = if proof_logrows < 10 {
        1u64 << proof_logrows
    } else {
        (1u64 << 10) + (1u64 << (proof_logrows - 10))
    };
    let size_of_fft_data = std::mem::size_of::<
        halo2_proofs::fft::recursive::FFTData<Fr>,
    >();
    let size_of_fft_data_once_lock = std::mem::size_of::<
        std::sync::OnceLock<halo2_proofs::fft::recursive::FFTData<Fr>>,
    >();
    let size_of_fft_data_cache_entry = std::mem::size_of::<(
        usize,
        std::sync::Arc<
            std::sync::OnceLock<halo2_proofs::fft::recursive::FFTData<Fr>>,
        >,
    )>();
    let size_of_fft_data_cache_map = std::mem::size_of::<std::collections::HashMap<
        usize,
        std::sync::Arc<
            std::sync::OnceLock<halo2_proofs::fft::recursive::FFTData<Fr>>,
        >,
    >>();
    let fft_parallel_thread_count = halo2_proofs::fft::parallel_thread_count();
    let device_capacity = halo2_proofs::icicle::device_capacity_identity()
        .map_err(|error| format!("native GAT CUDA memory identity failed: {}", error))?;
    let (
        size_of_mv_lookup_prepared_bytes,
        size_of_mv_lookup_prepared_table_bytes,
        size_of_mv_lookup_committed_bytes,
    ) =
        plonk::native_mv_lookup_resource_layout::<G1Affine>();
    let (
        size_of_mv_lookup_compact_table_bytes,
        size_of_mv_lookup_id_tape_bytes,
        size_of_mv_lookup_source_bytes,
    ) = plonk::native_mv_lookup_compact_resource_layout::<G1Affine>();
    let (
        size_of_shplonk_commitment_extension_bytes,
        size_of_shplonk_rotation_set_extension_bytes,
        size_of_shplonk_fixed_aggregate_cache_group_bytes,
    ) = halo2_proofs::poly::kzg::multiopen::native_shplonk_resource_layout::<G1Affine>();
    let size_of_coeff_polynomial_header_bytes = std::mem::size_of::<
        halo2_proofs::poly::Polynomial<Fr, halo2_proofs::poly::Coeff>,
    >();
    let shplonk_rotation_set_count = shplonk_shape.rotation_set_commitment_counts.len();
    let permutation_grand_product_set_count = u64::try_from(shplonk_shape.permutation_product_set_count)
        .map_err(|_| "native permutation grand-product set count exceeds u64".to_string())?;
    let permutation_grand_product_bank_mapped_bytes_upper = u64::try_from(
        u128::from(permutation_grand_product_set_count)
            .checked_mul(shplonk_shape.proof_instance_count as u128)
            .and_then(|sets| sets.checked_mul(u128::from(domain_size)))
            .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<Fr>() as u128))
            .ok_or_else(|| "native permutation grand-product mapped bank size overflow".to_string())?,
    )
    .map_err(|_| "native permutation grand-product mapped bank exceeds u64".to_string())?;
    let memory_model = serde_json::json!({
        "schema": "ezkl-halo2-static-envelope-v18",
        "evidence_kind": "source_derived_upper_bounds_and_configured_geometry",
        "coverage": "configured_geometry_and_source_upper_bounds_before_native_synthesis; actual_vk_pk_census_is_authenticated_by_setup_receipt",
        "native_target_arch": std::env::consts::ARCH,
        "uses_recursive_fft": uses_recursive_fft,
        "proof_logrows": proof_logrows,
        "proof_domain_size": proof_domain_size,
        "recursive_fft_cached_domain_lengths": recursive_fft_cached_domain_lengths,
        "recursive_fft_twiddle_field_elements": recursive_fft_twiddle_field_elements,
        "fft_cache_supported_length_count": fft_cache_entry_count,
        "fft_cache_entry_capacity": fft_cache_entry_capacity,
        "srs_fft_cache_supported_length_count": srs_fft_cache_entry_count,
        "srs_fft_cache_entry_capacity": srs_fft_cache_entry_capacity,
        "fft_parallel_thread_count": fft_parallel_thread_count,
        "parallel_fft_lookup_table_elements": parallel_fft_lookup_table_elements,
        "size_of_fftdata_bytes": size_of_fft_data,
        "size_of_fftdata_once_lock_bytes": size_of_fft_data_once_lock,
        "size_of_fftdata_cache_entry_bytes": size_of_fft_data_cache_entry,
        "size_of_fftdata_cache_map_bytes": size_of_fft_data_cache_map,
        "size_of_arc_header_bytes": std::mem::size_of::<(usize, usize)>(),
        "advice_column_phase_ids": cs.advice_column_phase(),
        "size_of_fr_bytes": std::mem::size_of::<Fr>(),
        "size_of_assigned_fr_bytes": std::mem::size_of::<plonk::Assigned<Fr>>(),
        "size_of_assigned_event_bytes": plonk::native_assigned_event_bytes::<Fr>(),
        "size_of_option_fr_bytes": std::mem::size_of::<Option<Fr>>(),
        "size_of_batch_invert_entry_bytes": std::mem::size_of::<(Fr, &mut Fr)>(),
        "size_of_usize_bytes": std::mem::size_of::<usize>(),
        "size_of_usize_pair_bytes": std::mem::size_of::<(usize, usize)>(),
        "size_of_permutation_hashmap_header_bytes": std::mem::size_of::<std::collections::HashMap<usize, usize>>(),
        "size_of_g1_affine_bytes": std::mem::size_of::<G1Affine>(),
        "size_of_g1_projective_bytes": std::mem::size_of::<<G1Affine as halo2curves::group::prime::PrimeCurveAffine>::Curve>(),
        "size_of_g2_affine_bytes": std::mem::size_of::<halo2curves::bn256::G2Affine>(),
        "size_of_fft_stage_bytes": std::mem::size_of::<halo2_proofs::fft::recursive::FFTStage>(),
        "size_of_atomic_u64_bytes": std::mem::size_of::<std::sync::atomic::AtomicU64>(),
        "size_of_lookup_map_entry_bytes": plonk::native_mv_lookup_map_entry_bytes(),
        "size_of_lookup_map_header_bytes": plonk::native_mv_lookup_map_header_bytes(),
        "size_of_vec_u8_bytes": std::mem::size_of::<Vec<u8>>(),
        "size_of_lookup_repr_bytes": Fr::default().to_repr().as_ref().len(),
        "size_of_vec_header_bytes": std::mem::size_of::<Vec<Fr>>(),
        "size_of_polynomial_header_bytes": std::mem::size_of::<halo2_proofs::poly::Polynomial<Fr, halo2_proofs::poly::LagrangeCoeff>>(),
        "size_of_fixed_column_source_bytes": plonk::fixed_column_source_size_bytes::<Fr>(),
        "size_of_mv_lookup_prepared_bytes": size_of_mv_lookup_prepared_bytes,
        "size_of_mv_lookup_prepared_table_bytes": size_of_mv_lookup_prepared_table_bytes,
        "size_of_mv_lookup_committed_bytes": size_of_mv_lookup_committed_bytes,
        "size_of_mv_lookup_compact_table_bytes": size_of_mv_lookup_compact_table_bytes,
        "size_of_mv_lookup_id_tape_bytes": size_of_mv_lookup_id_tape_bytes,
        "size_of_mv_lookup_source_bytes": size_of_mv_lookup_source_bytes,
        "size_of_shplonk_commitment_extension_bytes": size_of_shplonk_commitment_extension_bytes,
        "size_of_shplonk_rotation_set_extension_bytes": size_of_shplonk_rotation_set_extension_bytes,
        "size_of_shplonk_fixed_aggregate_cache_group_bytes": size_of_shplonk_fixed_aggregate_cache_group_bytes,
        "size_of_coeff_polynomial_header_bytes": size_of_coeff_polynomial_header_bytes,
        "lookup_table_cardinality_upper_bounds": lookup_shape.table_cardinality_upper_bounds,
        "fixed_value_explicit_zero_seen": fixed_value_explicit_zero_seen_upper,
        "lookup_argument_table_group_indices": lookup_shape.argument_table_group_indices,
        "custom_gate_dependency_columns": evaluation_shape.custom_gate_dependency_columns,
        "lookup_dependency_columns": evaluation_shape.lookup_dependency_columns,
        "lookup_query_polynomial_counts": evaluation_shape.lookup_query_polynomial_counts,
        "shuffle_dependency_columns": evaluation_shape.shuffle_dependency_columns,
        "shplonk_rotation_set_commitment_counts": shplonk_shape.rotation_set_commitment_counts,
        "shplonk_query_polynomial_count": shplonk_shape.query_polynomial_count,
        "shplonk_rotation_set_count": shplonk_rotation_set_count,
        "shplonk_max_rotation_count": shplonk_shape.max_rotation_count,
        "shplonk_unique_opening_point_count": shplonk_shape.unique_opening_point_count,
        "shplonk_max_commitments_per_rotation_set": shplonk_shape.max_commitments_per_rotation_set,
        "shplonk_permutation_product_set_count": shplonk_shape.permutation_product_set_count,
        "permutation_grand_product_bank_mapped_bytes_upper": permutation_grand_product_bank_mapped_bytes_upper,
        "shplonk_advice_query_polynomial_count": shplonk_shape.advice_query_polynomial_count,
        "shplonk_fixed_query_polynomial_count": shplonk_shape.fixed_query_polynomial_count,
        "shplonk_lookup_argument_count": shplonk_shape.lookup_argument_count,
        "shplonk_shuffle_argument_count": shplonk_shape.shuffle_argument_count,
        "shplonk_proof_instance_count": shplonk_shape.proof_instance_count,
        "shplonk_quotient_piece_count": shplonk_shape.quotient_piece_count,
        "graph_witness_scalar_element_count": graph_witness_scalar_element_count,
        "size_of_graph_witness_scalar_bytes": std::mem::size_of::<Fr>(),
        "size_of_gat_valtype_bytes": std::mem::size_of::<crate::tensor::ValType<Fr>>(),
        "gat_transient_witness_bytes_upper": gat_transient_witness_bytes_upper,
        "gat_transient_witness_evidence_kind": "source-derived-conservative-live-tensor-envelope-v1",
        "gat_transient_witness_formula_id": "whole-gat-retained-inputs-plus-max-live-head-phase-v1",
        "advice_event_record_bytes_upper": advice_event_record_bytes_upper,
        "advice_event_decoder_record_bytes_upper": advice_event_decoder_record_bytes_upper,
        "advice_event_assignment_counts_upper": advice_event_assignment_counts_upper,
        "advice_event_count_upper": advice_event_assignment_count_upper,
        "advice_event_reservation_chunk_bytes": 1usize << 20,
        "advice_event_batch_size_upper": advice_event_batch_size_upper,
        "advice_event_decoded_entry_bytes_upper": advice_event_decoded_entry_bytes_upper,
        "advice_event_pending_batch_bytes_upper": advice_event_pending_batch_bytes_upper,
        "advice_event_flush_scratch_bytes_upper": advice_event_flush_scratch_bytes_upper,
        "advice_event_reader_buffer_bytes_upper": advice_event_reader_buffer_bytes_upper,
        "advice_event_read_buffer_bytes_upper": advice_event_read_buffer_bytes_upper,
        "advice_event_column_metadata_bytes_upper": advice_event_column_metadata_bytes_upper,
        "advice_event_writer_buffer_bytes_upper": advice_event_writer_buffer_bytes_upper,
        "advice_event_column_bytes_upper": advice_event_column_bytes_upper,
        "advice_event_latest_element_bytes": std::mem::size_of::<u64>(),
        "advice_event_log_bound_evidence_kind": "configured-domain-upper-per-advice-column-v1",
        "advice_event_phase_materialization_order": "per-phase-encoded-bytes-descending-index-ascending-v1",
        "cuda_uuid": device_capacity.0,
        "cuda_free_bytes_at_admission": device_capacity.1,
        "cuda_total_memory_bytes": device_capacity.2,
        "selector_activation_bitmap_bytes_upper": selector_activation_bitmap_bytes,
        "selector_activation_storage_bytes_upper": selector_activation_storage_bytes_upper,
        "selector_activation_storage_kind": "rle_bitmap_hybrid_v1",
        "selector_cache_encoded_bytes": u64::try_from(
            5u128
                .checked_add(
                    (cs.num_selectors() as u128)
                        .checked_mul((u128::from(domain_size) + 7) / 8 + 5)
                        .ok_or_else(|| "selector cache encoded upper bound overflow".to_string())?,
                )
                .ok_or_else(|| "selector cache encoded upper bound overflow".to_string())?,
        )
        .map_err(|_| "selector cache encoded upper bound exceeds u64".to_string())?,
        "fixed_polynomial_read_mode": "streamed_one_polynomial_chunked_v1",
        "fixed_polynomial_read_chunk_fields_upper": fixed_polynomial_read_chunk_fields_upper,
        "fixed_polynomial_read_raw_chunk_bytes_upper": fixed_polynomial_read_raw_chunk_bytes_upper,
        "fixed_polynomial_read_decoded_chunk_bytes_upper": fixed_polynomial_read_decoded_chunk_bytes_upper,
        "fixed_polynomial_read_heap_scratch_bytes_upper": fixed_polynomial_read_heap_scratch_bytes_upper,
        "native_pk_read_buffer_capacity_bytes": *EZKL_BUF_CAPACITY,
        "native_pk_format_version": 4,
        "native_pk_sigma_wire_encoding": "packed_source_target_indices_lsb0_v1",
        "native_pk_sigma_index_width_formula": "0_if_cells_le_1_else_ceil_log2_cells",
        "native_successor_read_cache_bytes": halo2_proofs::poly::NATIVE_SUCCESSOR_READ_CACHE_BYTES,
        "native_target_bitmap_fixed_cache_bytes": halo2_proofs::poly::NATIVE_TARGET_BITMAP_FIXED_CACHE_BYTES,
    });
    println!(
        "EZKL_NATIVE_PHASE DONE stage=configure_constraint_system dataset={} elapsed_ms={:.3} advice={} fixed={} instance={} selectors={} lookups={} shuffles={} permutation_columns={} blinding_factors={}",
        loaded.dataset,
        configure_started.elapsed().as_secs_f64() * 1000.0,
        cs.num_advice_columns(),
        cs.num_fixed_columns(),
        cs.num_instance_columns(),
        cs.num_selectors(),
        cs.lookups().len(),
        cs.shuffles().len(),
        cs.num_permutation_columns(),
        cs.blinding_factors(),
    );
    settings.save(&settings_path)?;
    let settings_sha256 = get_file_hash(&settings_path)?;
    let relation_identity = native_admission_relation_identity(&loaded)?;
    let admission = serde_json::json!({
        "schema_version": 8,
        "shape_evidence_kind": "configured_geometry_source_upper_bounds_before_native_synthesis_v1",
        "relation_id": "R_GAT-ezkl",
        "dataset": loaded.dataset,
        "model_arch_id": loaded.spec.model.model_arch_id,
        "quant_cfg_id": loaded.spec.quant_cfg_id,
        "resource_manifest_sha256": loaded.resource_manifest_sha256,
        "relation_identity": relation_identity,
        "settings_sha256": settings_sha256,
        "logrows": logrows,
        "domain_size": domain_size,
        "max_public_srs_logrows": crate::graph::MAX_PUBLIC_SRS,
        "num_rows": settings.num_rows,
        "total_assignments": settings.total_assignments,
        "total_const_size": settings.total_const_size,
        "total_dynamic_col_size": settings.dynamic_lookup_params.total_dynamic_col_size,
        "max_dynamic_input_len": settings.dynamic_lookup_params.max_dynamic_input_len,
        "num_dynamic_lookups": settings.dynamic_lookup_params.num_dynamic_lookups,
        "num_shuffles": settings.shuffle_params.num_shuffles,
        "module_sizes": settings.module_sizes,
        "constraint_system": {
            "degree": degree,
            "minimum_rows": cs.minimum_rows(),
            "num_fixed_columns": cs.num_fixed_columns(),
            "num_custom_gate_groups": evaluation_shape.custom_gate_dependency_columns.len(),
            "selector_fixed_column_count_upper": selector_fixed_column_count,
            "fixed_column_count_post_compression": fixed_column_count_post_compression,
            "permutation_copy_call_count_upper": permutation_copy_call_count_upper,
            "permutation_successor_entry_count_upper": permutation_successor_entry_count_upper,
            "num_advice_columns": cs.num_advice_columns(),
            "num_instance_columns": cs.num_instance_columns(),
            "num_selectors": cs.num_selectors(),
            "num_lookup_arguments": lookup_shape.argument_count,
            "num_lookup_query_polynomials": lookup_shape.query_polynomial_count,
            "max_lookup_query_polynomials_per_argument": lookup_shape.max_query_polynomials_per_argument,
            "num_unique_lookup_table_expression_groups": lookup_table_group_count,
            "num_shuffle_arguments": cs.shuffles().len(),
            "num_permutation_columns": cs.num_permutation_columns(),
            "num_blinding_factors": cs.blinding_factors(),
        },
        "scalar_domain_vector_bytes": scalar_domain_bytes,
        "native_memory_model": memory_model,
    });
    info!("native GAT admission: {}", admission);
    std::fs::write(&output_path, serde_json::to_vec_pretty(&admission)?)?;
    Ok(output_path.display().to_string())
}

#[cfg(not(feature = "gpu-accelerated"))]
fn gat_prepare_admission(
    _loaded: crate::gat_adapter::LoadedGat,
    _output_dir: PathBuf,
) -> Result<String, EZKLError> {
    Err("native formal GAT admission requires the authenticated GPU build".into())
}

fn native_admission_relation_identity(loaded: &crate::gat_adapter::LoadedGat) -> Result<JsonValue, EZKLError> {
    let graph = &loaded.spec.graph;
    let canonical_graph_digest = graph
        .comparison_canonical_digest()
        .map_err(EZKLError::from)?;
    Ok(serde_json::json!({
        "schema_version": loaded.spec.schema_version,
        "native_layout_revision": "gat_fixed_topology_routing_v1",
        "dataset": loaded.dataset,
        "resource_manifest_sha256": loaded.resource_manifest_sha256,
        "model_arch_id": loaded.spec.model.model_arch_id,
        "quant_cfg_id": loaded.spec.quant_cfg_id,
        "input_preprocess_id": loaded.input_preprocess_id,
        "batching_rule": loaded.batching_rule,
        "graph": {
            "n_total": graph.n_total,
            "n": graph.n,
            "e": graph.e,
            "g_batch": graph.g_batch,
            "topology_id": graph.topology_id,
            "topology_digest": graph.topology_digest,
            "canonical_graph_digest": canonical_graph_digest,
            "node_ptr": graph.node_ptr,
            "edge_ptr": graph.edge_ptr,
        },
        "model_binding": loaded.spec.model_binding,
        "topology": loaded.spec.topology,
        "quantization": loaded.spec.quantization,
        "quant_cfg_id": loaded.spec.quant_cfg_id,
        "tables": {
            "static_tables_id": loaded.spec.tables.static_tables_id,
            "static_tables_digest": loaded.spec.tables.static_tables_digest,
            "backend_aux_tables_id": loaded.spec.tables.backend_aux_tables_id,
        },
        "ledger_identity": loaded.spec.ledger_identity,
        "ezkl_security_profile_id": loaded.spec.ezkl_security_profile_id,
    }))
}

fn load_native_admission_circuit(
    loaded: &crate::gat_adapter::LoadedGat,
    admission_dir: &Path,
) -> Result<(GraphCircuit, usize), EZKLError> {
    let admission_path = admission_dir.join("admission.json");
    let settings_path = admission_dir.join("settings.json");
    for (path, label) in [
        (&admission_path, "native admission receipt"),
        (&settings_path, "admitted settings"),
    ] {
        require_formal_regular_file(path, label)?;
    }
    let admission: JsonValue = serde_json::from_slice(&std::fs::read(&admission_path)?)?;
    let expected_relation_identity = native_admission_relation_identity(loaded)?;
    if admission.get("schema_version").and_then(JsonValue::as_u64) != Some(8)
        || admission.get("shape_evidence_kind").and_then(JsonValue::as_str)
            != Some("configured_geometry_source_upper_bounds_before_native_synthesis_v1")
        || admission.get("relation_id").and_then(JsonValue::as_str) != Some("R_GAT-ezkl")
        || admission.get("dataset").and_then(JsonValue::as_str) != Some(loaded.dataset.as_str())
        || admission.get("resource_manifest_sha256").and_then(JsonValue::as_str)
            != Some(loaded.resource_manifest_sha256.as_str())
        || admission.get("relation_identity") != Some(&expected_relation_identity)
    {
        return Err("native admission relation/resource identity does not match the current canonical dataset".into());
    }
    if admission.get("settings_sha256").and_then(JsonValue::as_str)
        != Some(get_file_hash(&settings_path)?.as_str())
    {
        return Err("native admission settings digest mismatch".into());
    }

    let settings = GraphSettings::load(&settings_path)?;
    let model = build_model_for_run_args(
        loaded.spec.clone(),
        RoundingMode::NearestTiesAway,
        &settings.run_args,
    )
    .map_err(EZKLError::from)?;
    let circuit = GraphCircuit::new_gat_witnessless_from_settings(
        model,
        settings.clone(),
        CheckMode::UNSAFE,
    );
    assert_native_gat_identity(&circuit, &loaded.spec)?;
    let current_cs = configure_canonical_constraint_system(&circuit);
    let admission_cs = admission
        .get("constraint_system")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| "native admission has no constraint-system counts".to_string())?;
    let admission_count = |name: &str| -> Result<usize, EZKLError> {
        let value = admission_cs
            .get(name)
            .and_then(JsonValue::as_u64)
            .ok_or_else(|| format!("native admission {} is missing or invalid", name))?;
        usize::try_from(value)
            .map_err(|_| format!("native admission {} exceeds host index width", name).into())
    };

    let selector_fixed = admission_count("selector_fixed_column_count_upper")?;
    if selector_fixed > current_cs.num_selectors() {
        return Err("native admission selector-column upper bound is inconsistent".into());
    }
    let (current_evaluation_cs, _) = current_cs
        .clone()
        .directly_convert_selectors_to_fixed(
            vec![Vec::new(); current_cs.num_selectors()],
            false,
        );
    let current_evaluation_shape = cap_native_selector_resource_shape(
        plonk::native_evaluation_resource_shape::<G1Affine>(&current_evaluation_cs),
        &current_cs,
        selector_fixed,
    )?;
    drop(current_evaluation_cs);

    let current_shplonk_domain = halo2_proofs::poly::EvaluationDomain::<Fr>::new(
        u32::try_from(current_cs.degree())
            .map_err(|_| "native admission degree exceeds u32".to_string())?,
        settings.run_args.logrows,
    );
    let current_quotient_piece_count = current_shplonk_domain.get_quotient_poly_degree();
    let current_shplonk_cs = current_cs
        .clone()
        .with_native_selector_resource_columns(selector_fixed);
    let current_shplonk_shape = plonk::native_shplonk_resource_shape::<Fr>(
        &current_shplonk_cs,
        1u64
            .checked_shl(settings.run_args.logrows)
            .ok_or_else(|| "native admission domain size overflow".to_string())?,
        current_cs.lookups().len(),
        1,
        current_quotient_piece_count,
    );
    drop(current_shplonk_cs);
    drop(current_shplonk_domain);

    let precompression_fixed = admission_count("num_fixed_columns")?;
    let fixed_post_compression = admission_count("fixed_column_count_post_compression")?;
    let permutation_copy_call_count_upper = admission_count("permutation_copy_call_count_upper")?;
    let permutation_successor_entry_count_upper =
        admission_count("permutation_successor_entry_count_upper")?;
    if fixed_post_compression
        != precompression_fixed
            .checked_add(selector_fixed)
            .ok_or_else(|| "native admission fixed-column count overflow".to_string())?
    {
        return Err("native admission post-compression fixed-column count is inconsistent".into());
    }
    let admitted_permutation_cells = (current_cs.num_permutation_columns() as u128)
        .checked_mul(1u128 << settings.run_args.logrows)
        .ok_or_else(|| "native admission permutation cell count overflow".to_string())?;
    let admitted_copy_bound = (permutation_copy_call_count_upper as u128)
        .checked_mul(2)
        .ok_or_else(|| "native admission permutation copy bound overflow".to_string())?;
    let expected_successor_entry_count_upper =
        admitted_permutation_cells.min(admitted_copy_bound);
    if permutation_copy_call_count_upper as u128 > admitted_permutation_cells {
        return Err("native admission permutation-copy upper bound is inconsistent".into());
    }
    if permutation_successor_entry_count_upper as u128 != expected_successor_entry_count_upper {
        return Err("native admission compact permutation successor bound is inconsistent".into());
    }
    for (name, current) in [
        ("degree", current_cs.degree()),
        ("minimum_rows", current_cs.minimum_rows()),
        ("num_fixed_columns", current_cs.num_fixed_columns()),
        ("num_advice_columns", current_cs.num_advice_columns()),
        ("num_instance_columns", current_cs.num_instance_columns()),
        ("num_selectors", current_cs.num_selectors()),
        (
            "num_custom_gate_groups",
            current_evaluation_shape.custom_gate_dependency_columns.len(),
        ),
        ("num_lookup_arguments", current_cs.lookups().len()),
        ("num_shuffle_arguments", current_cs.shuffles().len()),
        ("num_permutation_columns", current_cs.num_permutation_columns()),
        ("num_blinding_factors", current_cs.blinding_factors()),
    ] {
        if admission_count(name)? != current {
            return Err(format!("native admission {} disagrees with the current constraint system", name).into());
        }
    }
    let memory_model = admission
        .get("native_memory_model")
        .and_then(JsonValue::as_object)
        .ok_or_else(|| "native admission has no memory model".to_string())?;
    if memory_model.get("schema").and_then(JsonValue::as_str)
        != Some("ezkl-halo2-static-envelope-v18")
        || memory_model.get("fixed_polynomial_read_mode").and_then(JsonValue::as_str)
            != Some("streamed_one_polynomial_chunked_v1")
        || memory_model.get("evidence_kind").and_then(JsonValue::as_str)
            != Some("source_derived_upper_bounds_and_configured_geometry")
        || memory_model.get("coverage").and_then(JsonValue::as_str)
            != Some("configured_geometry_and_source_upper_bounds_before_native_synthesis; actual_vk_pk_census_is_authenticated_by_setup_receipt")
    {
        return Err("native admission memory model version or polynomial reader mode is unsupported".into());
    }
    if memory_model.get("native_pk_format_version").and_then(JsonValue::as_u64) != Some(4)
        || memory_model.get("native_pk_sigma_wire_encoding").and_then(JsonValue::as_str)
            != Some("packed_source_target_indices_lsb0_v1")
        || memory_model.get("native_pk_sigma_index_width_formula").and_then(JsonValue::as_str)
            != Some("0_if_cells_le_1_else_ceil_log2_cells")
    {
        return Err("native admission PK/Sigma wire format does not match the current compact source".into());
    }
    let fixed_value_explicit_zero_seen = memory_model
        .get("fixed_value_explicit_zero_seen")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "native admission fixed-zero census is missing or malformed".to_string())?;
    if fixed_value_explicit_zero_seen.len() != current_cs.num_fixed_columns()
        || fixed_value_explicit_zero_seen.iter().any(|value| value.as_bool().is_none())
    {
        return Err("native admission fixed-zero census disagrees with the current fixed-column shape".into());
    }
    let expected_lookup_argument_table_groups =
        JsonValue::from(native_mv_lookup_argument_table_group_indices(&current_cs)?);
    if memory_model.get("lookup_argument_table_group_indices")
        != Some(&expected_lookup_argument_table_groups)
    {
        return Err("native admission lookup argument/table group mapping disagrees with the current constraint system".into());
    }
    let current_advice_phases = JsonValue::from(current_cs.advice_column_phase());
    if memory_model.get("advice_column_phase_ids") != Some(&current_advice_phases) {
        return Err("native admission advice-column phases disagree with the current constraint system".into());
    }
    let n = 1u128
        .checked_shl(settings.run_args.logrows)
        .ok_or_else(|| "native admission domain size overflow".to_string())?;
    let expected_permutation_grand_product_bank_mapped_bytes_upper = u64::try_from(
        (current_shplonk_shape.permutation_product_set_count as u128)
            .checked_mul(current_shplonk_shape.proof_instance_count as u128)
            .and_then(|sets| sets.checked_mul(n))
            .and_then(|bytes| bytes.checked_mul(std::mem::size_of::<Fr>() as u128))
            .ok_or_else(|| "native admission permutation grand-product bank size overflow".to_string())?,
    )
    .map_err(|_| "native admission permutation grand-product bank exceeds u64".to_string())?;
    let expected_selector_bitmap_bytes = (current_cs.num_selectors() as u128)
        .checked_mul((n + 7) / 8)
        .ok_or_else(|| "native admission selector bitmap size overflow".to_string())?;
    let selector_cache_encoded_bytes = memory_model
        .get("selector_cache_encoded_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| "native admission selector cache byte count is missing or invalid".to_string())?;
    let min_selector_cache_encoded_bytes = 5u128
        .checked_add((current_cs.num_selectors() as u128).checked_mul(4).ok_or_else(|| "native selector cache index byte count overflow".to_string())?)
        .ok_or_else(|| "native selector cache encoded byte count overflow".to_string())?;
    let max_selector_cache_encoded_bytes = 5u128
        .checked_add(
            (current_cs.num_selectors() as u128)
                .checked_mul((n + 7) / 8 + 5)
                .ok_or_else(|| "native selector cache encoded byte ceiling overflow".to_string())?,
        )
        .ok_or_else(|| "native selector cache encoded byte ceiling overflow".to_string())?;
    if u128::from(selector_cache_encoded_bytes) < min_selector_cache_encoded_bytes
        || u128::from(selector_cache_encoded_bytes) > max_selector_cache_encoded_bytes
    {
        return Err("native admission selector cache byte count is outside its canonical encoding bounds".into());
    }
    let selector_activation_storage_bytes_upper = memory_model
        .get("selector_activation_storage_bytes_upper")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| "native admission selector activation storage upper bound is missing or invalid".to_string())?;
    if memory_model
        .get("selector_activation_storage_kind")
        .and_then(JsonValue::as_str)
        != Some("rle_bitmap_hybrid_v1")
    {
        return Err("native admission selector activation storage representation is unsupported".into());
    }
    // Static admission carries a source-derived dense bitmap ceiling and
    // temporary promotion bound. Native keygen records the actual hybrid
    // activation storage peak in its authenticated setup receipt.
    let max_selector_activation_storage_bytes = expected_selector_bitmap_bytes
        .checked_mul(2)
        .and_then(|bytes| {
            bytes.checked_add((current_cs.num_selectors() as u128).checked_mul(64)?)
        })
        .ok_or_else(|| "native admission selector activation storage bound overflow".to_string())?;
    if u128::from(selector_activation_storage_bytes_upper) != max_selector_activation_storage_bytes {
        return Err("native admission selector activation storage upper bound is inconsistent".into());
    }
    let chunk_fields = 16 * 1024usize;
    let serialized_fr_bytes = Fr::default().to_repr().as_ref().len();
    let raw_chunk_bytes = chunk_fields * serialized_fr_bytes;
    let decoded_chunk_bytes = chunk_fields * std::mem::size_of::<Fr>();
    let read_heap_bytes = raw_chunk_bytes + decoded_chunk_bytes;
    let batch_invert_entry_bytes = std::mem::size_of::<(Fr, &mut Fr)>() as u128;
    let (prepared_bytes, prepared_table_bytes, committed_bytes) =
        plonk::native_mv_lookup_resource_layout::<G1Affine>();
    let quotient_extent = n
        .checked_mul(current_cs.degree().saturating_sub(1) as u128)
        .ok_or_else(|| "native admission proof-domain extent overflow".to_string())?;
    let expected_proof_domain_size = quotient_extent
        .checked_next_power_of_two()
        .ok_or_else(|| "native admission proof-domain size overflow".to_string())?;
    let expected_proof_logrows = expected_proof_domain_size.trailing_zeros();
    let uses_recursive_fft = !cfg!(target_arch = "x86_64");
    let expected_twiddle_elements = if uses_recursive_fft {
        2u128
            .checked_mul(
                expected_proof_domain_size
                    .checked_sub(1)
                    .and_then(|value| value.checked_add(u128::from(expected_proof_logrows)))
                    .ok_or_else(|| "native admission recursive FFT twiddle count overflow".to_string())?,
            )
            .ok_or_else(|| "native admission recursive FFT twiddle count overflow".to_string())?
    } else {
        0
    };
    let expected_cached_domain_lengths = if uses_recursive_fft {
        vec![u64::try_from(expected_proof_domain_size)
            .map_err(|_| "native admission proof domain exceeds u64".to_string())?]
    } else {
        Vec::new()
    };
    let expected_parallel_fft_lookup_table_elements = if expected_proof_logrows < 10 {
        1u128 << expected_proof_logrows
    } else {
        (1u128 << 10) + (1u128 << (expected_proof_logrows - 10))
    };
    let cache_shape_domain = halo2_proofs::poly::EvaluationDomain::<Fr>::new(
        u32::try_from(current_cs.degree())
            .map_err(|_| "native admission degree exceeds u32".to_string())?,
        settings.run_args.logrows,
    );
    let expected_cache_supported_length_count = cache_shape_domain.fft_cache_entry_count();
    let expected_cache_entry_capacity = cache_shape_domain.fft_cache_entry_capacity();
    drop(cache_shape_domain);
    let srs_cache_shape_domain = halo2_proofs::poly::EvaluationDomain::<Fr>::new(
        1,
        settings.run_args.logrows,
    );
    let expected_srs_cache_supported_length_count = srs_cache_shape_domain.fft_cache_entry_count();
    let expected_srs_cache_entry_capacity = srs_cache_shape_domain.fft_cache_entry_capacity();
    drop(srs_cache_shape_domain);
    for (name, expected) in [
        ("native_target_arch", JsonValue::from(std::env::consts::ARCH)),
        ("uses_recursive_fft", JsonValue::from(uses_recursive_fft)),
        (
            "recursive_fft_twiddle_field_elements",
            JsonValue::from(u64::try_from(expected_twiddle_elements)
                .map_err(|_| "native admission FFT twiddle count exceeds u64".to_string())?),
        ),
        (
            "fft_cache_supported_length_count",
            JsonValue::from(expected_cache_supported_length_count as u64),
        ),
        (
            "fft_cache_entry_capacity",
            JsonValue::from(expected_cache_entry_capacity as u64),
        ),
        (
            "srs_fft_cache_supported_length_count",
            JsonValue::from(expected_srs_cache_supported_length_count as u64),
        ),
        (
            "srs_fft_cache_entry_capacity",
            JsonValue::from(expected_srs_cache_entry_capacity as u64),
        ),
        (
            "fft_parallel_thread_count",
            JsonValue::from(halo2_proofs::fft::parallel_thread_count() as u64),
        ),
        (
            "parallel_fft_lookup_table_elements",
            JsonValue::from(u64::try_from(expected_parallel_fft_lookup_table_elements)
                .map_err(|_| "native admission parallel FFT table count exceeds u64".to_string())?),
        ),
        (
            "recursive_fft_cached_domain_lengths",
            serde_json::json!(expected_cached_domain_lengths),
        ),
        (
            "custom_gate_dependency_columns",
            serde_json::json!(current_evaluation_shape.custom_gate_dependency_columns),
        ),
        (
            "lookup_dependency_columns",
            serde_json::json!(current_evaluation_shape.lookup_dependency_columns),
        ),
        (
            "lookup_query_polynomial_counts",
            serde_json::json!(current_evaluation_shape.lookup_query_polynomial_counts),
        ),
        (
            "shuffle_dependency_columns",
            serde_json::json!(current_evaluation_shape.shuffle_dependency_columns),
        ),
        (
            "shplonk_rotation_set_commitment_counts",
            serde_json::json!(current_shplonk_shape.rotation_set_commitment_counts),
        ),
        (
            "shplonk_query_polynomial_count",
            JsonValue::from(current_shplonk_shape.query_polynomial_count as u64),
        ),
        (
            "shplonk_rotation_set_count",
            JsonValue::from(current_shplonk_shape.rotation_set_count as u64),
        ),
        (
            "shplonk_max_rotation_count",
            JsonValue::from(current_shplonk_shape.max_rotation_count as u64),
        ),
        (
            "shplonk_unique_opening_point_count",
            JsonValue::from(current_shplonk_shape.unique_opening_point_count as u64),
        ),
        (
            "shplonk_max_commitments_per_rotation_set",
            JsonValue::from(current_shplonk_shape.max_commitments_per_rotation_set as u64),
        ),
        (
            "shplonk_permutation_product_set_count",
            JsonValue::from(current_shplonk_shape.permutation_product_set_count as u64),
        ),
        (
            "permutation_grand_product_bank_mapped_bytes_upper",
            JsonValue::from(expected_permutation_grand_product_bank_mapped_bytes_upper),
        ),
        (
            "shplonk_advice_query_polynomial_count",
            JsonValue::from(current_shplonk_shape.advice_query_polynomial_count as u64),
        ),
        (
            "shplonk_fixed_query_polynomial_count",
            JsonValue::from(current_shplonk_shape.fixed_query_polynomial_count as u64),
        ),
        (
            "shplonk_lookup_argument_count",
            JsonValue::from(current_shplonk_shape.lookup_argument_count as u64),
        ),
        (
            "shplonk_shuffle_argument_count",
            JsonValue::from(current_shplonk_shape.shuffle_argument_count as u64),
        ),
        (
            "shplonk_proof_instance_count",
            JsonValue::from(current_shplonk_shape.proof_instance_count as u64),
        ),
        (
            "shplonk_quotient_piece_count",
            JsonValue::from(current_shplonk_shape.quotient_piece_count as u64),
        ),
    ] {
        if memory_model.get(name) != Some(&expected) {
            return Err(format!("native admission {} disagrees with source-matched evaluation geometry", name).into());
        }
    }
    let expected_memory_sizes = [
        ("proof_logrows", expected_proof_logrows as u128),
        ("proof_domain_size", expected_proof_domain_size),
        (
            "size_of_fftdata_bytes",
            std::mem::size_of::<halo2_proofs::fft::recursive::FFTData<Fr>>() as u128,
        ),
        (
            "size_of_fftdata_once_lock_bytes",
            std::mem::size_of::<std::sync::OnceLock<halo2_proofs::fft::recursive::FFTData<Fr>>>() as u128,
        ),
        (
            "size_of_fftdata_cache_entry_bytes",
            std::mem::size_of::<(
                usize,
                std::sync::Arc<std::sync::OnceLock<halo2_proofs::fft::recursive::FFTData<Fr>>>,
            )>() as u128,
        ),
        (
            "size_of_fftdata_cache_map_bytes",
            std::mem::size_of::<std::collections::HashMap<
                usize,
                std::sync::Arc<std::sync::OnceLock<halo2_proofs::fft::recursive::FFTData<Fr>>>,
            >>() as u128,
        ),
        (
            "size_of_arc_header_bytes",
            std::mem::size_of::<(usize, usize)>() as u128,
        ),
        ("size_of_fr_bytes", std::mem::size_of::<Fr>() as u128),
        (
            "size_of_gat_valtype_bytes",
            std::mem::size_of::<crate::tensor::ValType<Fr>>() as u128,
        ),
        ("size_of_assigned_fr_bytes", std::mem::size_of::<plonk::Assigned<Fr>>() as u128),
        ("size_of_assigned_event_bytes", plonk::native_assigned_event_bytes::<Fr>() as u128),
        ("size_of_option_fr_bytes", std::mem::size_of::<Option<Fr>>() as u128),
        ("size_of_batch_invert_entry_bytes", batch_invert_entry_bytes),
        ("size_of_usize_bytes", std::mem::size_of::<usize>() as u128),
        ("size_of_usize_pair_bytes", std::mem::size_of::<(usize, usize)>() as u128),
        (
            "size_of_permutation_hashmap_header_bytes",
            std::mem::size_of::<std::collections::HashMap<usize, usize>>() as u128,
        ),
        ("size_of_g1_affine_bytes", std::mem::size_of::<G1Affine>() as u128),
        (
            "size_of_g1_projective_bytes",
            std::mem::size_of::<<G1Affine as halo2curves::group::prime::PrimeCurveAffine>::Curve>() as u128,
        ),
        ("size_of_g2_affine_bytes", std::mem::size_of::<halo2curves::bn256::G2Affine>() as u128),
        (
            "size_of_fft_stage_bytes",
            std::mem::size_of::<halo2_proofs::fft::recursive::FFTStage>() as u128,
        ),
        ("size_of_atomic_u64_bytes", std::mem::size_of::<std::sync::atomic::AtomicU64>() as u128),
        ("size_of_lookup_map_entry_bytes", plonk::native_mv_lookup_map_entry_bytes() as u128),
        (
            "size_of_lookup_map_header_bytes",
            plonk::native_mv_lookup_map_header_bytes() as u128,
        ),
        ("size_of_vec_u8_bytes", std::mem::size_of::<Vec<u8>>() as u128),
        ("size_of_lookup_repr_bytes", Fr::default().to_repr().as_ref().len() as u128),
        ("size_of_vec_header_bytes", std::mem::size_of::<Vec<Fr>>() as u128),
        (
            "size_of_polynomial_header_bytes",
            std::mem::size_of::<halo2_proofs::poly::Polynomial<Fr, halo2_proofs::poly::LagrangeCoeff>>() as u128,
        ),
        ("size_of_mv_lookup_prepared_bytes", prepared_bytes as u128),
        ("size_of_mv_lookup_prepared_table_bytes", prepared_table_bytes as u128),
        ("size_of_mv_lookup_committed_bytes", committed_bytes as u128),
        (
            "size_of_mv_lookup_compact_table_bytes",
            plonk::native_mv_lookup_compact_resource_layout::<G1Affine>().0 as u128,
        ),
        (
            "size_of_mv_lookup_id_tape_bytes",
            plonk::native_mv_lookup_compact_resource_layout::<G1Affine>().1 as u128,
        ),
        (
            "size_of_mv_lookup_source_bytes",
            plonk::native_mv_lookup_compact_resource_layout::<G1Affine>().2 as u128,
        ),
        (
            "size_of_shplonk_commitment_extension_bytes",
            halo2_proofs::poly::kzg::multiopen::native_shplonk_resource_layout::<G1Affine>().0 as u128,
        ),
        (
            "size_of_shplonk_rotation_set_extension_bytes",
            halo2_proofs::poly::kzg::multiopen::native_shplonk_resource_layout::<G1Affine>().1 as u128,
        ),
        (
            "size_of_shplonk_fixed_aggregate_cache_group_bytes",
            halo2_proofs::poly::kzg::multiopen::native_shplonk_resource_layout::<G1Affine>().2 as u128,
        ),
        (
            "size_of_coeff_polynomial_header_bytes",
            std::mem::size_of::<
                halo2_proofs::poly::Polynomial<Fr, halo2_proofs::poly::Coeff>,
            >() as u128,
        ),
        (
            "size_of_fixed_column_source_bytes",
            plonk::fixed_column_source_size_bytes::<Fr>() as u128,
        ),
    ];
    for (name, expected) in [
        ("selector_activation_bitmap_bytes_upper", expected_selector_bitmap_bytes),
        ("fixed_polynomial_read_chunk_fields_upper", chunk_fields as u128),
        ("fixed_polynomial_read_raw_chunk_bytes_upper", raw_chunk_bytes as u128),
        ("fixed_polynomial_read_decoded_chunk_bytes_upper", decoded_chunk_bytes as u128),
        ("fixed_polynomial_read_heap_scratch_bytes_upper", read_heap_bytes as u128),
        ("native_pk_read_buffer_capacity_bytes", *EZKL_BUF_CAPACITY as u128),
        (
            "native_successor_read_cache_bytes",
            halo2_proofs::poly::NATIVE_SUCCESSOR_READ_CACHE_BYTES as u128,
        ),
        (
            "native_target_bitmap_fixed_cache_bytes",
            halo2_proofs::poly::NATIVE_TARGET_BITMAP_FIXED_CACHE_BYTES as u128,
        ),
    ]
    .into_iter()
    .chain(expected_memory_sizes)
    {
        if memory_model.get(name).and_then(JsonValue::as_u64) != u64::try_from(expected).ok() {
            return Err(format!("native admission {} disagrees with the source memory bound", name).into());
        }
    }
    let advice_event_record_bytes_upper = 9usize
        .checked_add(Fr::default().to_repr().as_ref().len())
        .ok_or_else(|| "native admission advice event record width overflow".to_string())?;
    let advice_event_decoder_record_bytes_upper = 9usize
        .checked_add(2 * Fr::default().to_repr().as_ref().len())
        .ok_or_else(|| "native admission advice event decoder width overflow".to_string())?;
    let advice_event_batch_size_upper = 8192usize;
    let advice_event_decoded_entry_bytes_upper = plonk::native_assigned_event_bytes::<Fr>();
    let advice_event_pending_batch_bytes_upper = advice_event_batch_size_upper
        .checked_mul(advice_event_decoded_entry_bytes_upper)
        .ok_or_else(|| "native admission advice event pending-batch size overflow".to_string())?;
    let advice_event_flush_scratch_bytes_upper = advice_event_batch_size_upper
        .checked_mul(
            std::mem::size_of::<Option<Fr>>()
                .checked_add(std::mem::size_of::<(Fr, &mut Fr)>())
                .ok_or_else(|| "native admission advice event flush-scratch entry width overflow".to_string())?,
        )
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<Option<Fr>>>()))
        .ok_or_else(|| "native admission advice event flush-scratch size overflow".to_string())?;
    let advice_event_reader_buffer_bytes_upper = 8192usize
        .checked_add(std::mem::size_of::<std::io::BufReader<std::fs::File>>())
        .ok_or_else(|| "native admission advice event reader-buffer size overflow".to_string())?;
    let advice_event_read_buffer_bytes_upper = advice_event_batch_size_upper
        .checked_mul(
            advice_event_decoded_entry_bytes_upper
                .checked_add(std::mem::size_of::<Option<Fr>>())
                .and_then(|bytes| bytes.checked_add(std::mem::size_of::<(Fr, &mut Fr)>()))
                .ok_or_else(|| "native admission advice event read-buffer width overflow".to_string())?,
        )
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<Vec<u8>>()))
        .and_then(|bytes| bytes.checked_add(advice_event_reader_buffer_bytes_upper))
        .and_then(|bytes| bytes.checked_add(advice_event_decoder_record_bytes_upper))
        .ok_or_else(|| "native admission advice event read-buffer size overflow".to_string())?;
    let advice_event_column_metadata_bytes_upper = std::mem::size_of::<std::path::PathBuf>()
        .checked_add(std::mem::size_of::<Option<std::io::BufWriter<std::fs::File>>>())
        .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<u64>()))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<std::sync::Arc<()>>()))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<u8>>()))
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Vec<plonk::Assigned<Fr>>>()))
        .and_then(|bytes| bytes.checked_add(3 * std::mem::size_of::<u64>()))
        .and_then(|bytes| bytes.checked_add(64))
        .ok_or_else(|| "native admission advice event metadata bound overflow".to_string())?;
    let advice_event_writer_buffer_bytes_upper = 8192usize;
    let advice_event_column_bytes_upper = advice_event_column_metadata_bytes_upper
        .checked_add(advice_event_record_bytes_upper)
        .ok_or_else(|| "native admission advice event column bound overflow".to_string())?;
    let advice_event_assignment_counts_upper = memory_model
        .get("advice_event_assignment_counts_upper")
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "native admission advice event assignment counts are missing".to_string())?;
    if advice_event_assignment_counts_upper.len() != current_cs.num_advice_columns() {
        return Err("native admission advice event assignment counts disagree with configured advice columns".into());
    }
    let advice_event_count_upper = advice_event_assignment_counts_upper
        .iter()
        .try_fold(0u64, |total, count| {
            total.checked_add(count.as_u64()?)
        })
        .ok_or_else(|| "native admission advice event assignment count is malformed or overflows".to_string())?;
    if memory_model
        .get("advice_event_count_upper")
        .and_then(JsonValue::as_u64)
        != Some(advice_event_count_upper)
    {
        return Err("native admission advice event count disagrees with its per-column census".into());
    }
    if memory_model
        .get("gat_transient_witness_formula_id")
        .and_then(JsonValue::as_str)
        != Some("whole-gat-retained-inputs-plus-max-live-head-phase-v1")
        || memory_model
            .get("gat_transient_witness_evidence_kind")
            .and_then(JsonValue::as_str)
            != Some("source-derived-conservative-live-tensor-envelope-v1")
        || memory_model
            .get("gat_transient_witness_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(crate::gat_adapter::op::gat_transient_witness_bytes_upper(&loaded.spec)?)
        || memory_model
            .get("advice_event_record_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_record_bytes_upper as u64)
        || memory_model
            .get("advice_event_decoder_record_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_decoder_record_bytes_upper as u64)
        || memory_model
            .get("advice_event_reservation_chunk_bytes")
            .and_then(JsonValue::as_u64)
            != Some((1usize << 20) as u64)
        || memory_model
            .get("advice_event_batch_size_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_batch_size_upper as u64)
        || memory_model
            .get("advice_event_decoded_entry_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_decoded_entry_bytes_upper as u64)
        || memory_model
            .get("advice_event_pending_batch_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_pending_batch_bytes_upper as u64)
        || memory_model
            .get("advice_event_flush_scratch_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_flush_scratch_bytes_upper as u64)
        || memory_model
            .get("advice_event_reader_buffer_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_reader_buffer_bytes_upper as u64)
        || memory_model
            .get("advice_event_read_buffer_bytes_upper")
            .and_then(JsonValue::as_u64)
            != Some(advice_event_read_buffer_bytes_upper as u64)
            || memory_model
                .get("advice_event_column_metadata_bytes_upper")
                .and_then(JsonValue::as_u64)
                != Some(advice_event_column_metadata_bytes_upper as u64)
            || memory_model
                .get("advice_event_writer_buffer_bytes_upper")
                .and_then(JsonValue::as_u64)
                != Some(advice_event_writer_buffer_bytes_upper as u64)
            || memory_model
                .get("advice_event_column_bytes_upper")
                .and_then(JsonValue::as_u64)
                != Some(advice_event_column_bytes_upper as u64)
        || memory_model
            .get("advice_event_latest_element_bytes")
            .and_then(JsonValue::as_u64)
            != Some(std::mem::size_of::<u64>() as u64)
        || memory_model
            .get("advice_event_log_bound_evidence_kind")
            .and_then(JsonValue::as_str)
            != Some("configured-domain-upper-per-advice-column-v1")
        || memory_model
            .get("advice_event_phase_materialization_order")
            .and_then(JsonValue::as_str)
            != Some("per-phase-encoded-bytes-descending-index-ascending-v1")
    {
        return Err("native admission GAT tensor live-bound disagrees with the canonical relation geometry".into());
    }
    let run_args = &settings.run_args;
    if !run_args.input_visibility.is_public()
        || !run_args.output_visibility.is_public()
        || !run_args.param_visibility.is_hashed_public()
        || run_args.input_scale != loaded.spec.scales.s_input
        || run_args.param_scale != loaded.spec.scales.s_w_hidden[0][0]
        || run_args.lookup_range
            != (loaded.spec.tables.lrelu.input_min, loaded.spec.tables.lrelu.input_max)
        || run_args.check_mode != CheckMode::UNSAFE
        || run_args.logrows == 0
        || run_args.logrows > crate::graph::MAX_PUBLIC_SRS
    {
        return Err("native admission circuit does not retain the formal GAT settings".into());
    }
    for (key, expected) in [
        ("logrows", JsonValue::from(settings.run_args.logrows)),
        ("num_rows", JsonValue::from(settings.num_rows as u64)),
        ("total_assignments", JsonValue::from(settings.total_assignments as u64)),
        ("total_const_size", JsonValue::from(settings.total_const_size as u64)),
        (
            "total_dynamic_col_size",
            JsonValue::from(settings.dynamic_lookup_params.total_dynamic_col_size as u64),
        ),
        (
            "max_dynamic_input_len",
            JsonValue::from(settings.dynamic_lookup_params.max_dynamic_input_len as u64),
        ),
        (
            "num_dynamic_lookups",
            JsonValue::from(settings.dynamic_lookup_params.num_dynamic_lookups as u64),
        ),
        ("num_shuffles", JsonValue::from(settings.shuffle_params.num_shuffles as u64)),
    ] {
        if admission.get(key) != Some(&expected) {
            return Err(format!("native admission {} disagrees with its authenticated settings", key).into());
        }
    }
    Ok((circuit, fixed_post_compression))
}

fn native_mem_available_bytes() -> Result<u64, EZKLError> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    for line in meminfo.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() == Some("MemAvailable:") {
            let kib = parts
                .next()
                .ok_or_else(|| "MemAvailable is missing its byte count".to_string())?
                .parse::<u64>()
                .map_err(|_| "MemAvailable contains an invalid byte count".to_string())?;
            return kib
                .checked_mul(1024)
                .ok_or_else(|| "MemAvailable byte count overflow".into());
        }
    }
    Err("/proc/meminfo has no MemAvailable field for the native keygen gate".into())
}

fn native_keygen_preflight_json(preflight: &plonk::NativeKeygenPreflight) -> JsonValue {
    serde_json::json!({
        "logrows": preflight.logrows,
        "domain_size": preflight.domain_size,
        "extended_logrows": preflight.extended_logrows,
        "constraint_degree": preflight.constraint_degree,
        "advice_columns": preflight.advice_columns,
        "fixed_columns": preflight.fixed_columns,
        "instance_columns": preflight.instance_columns,
        "permutation_columns": preflight.permutation_columns,
        "selector_count": preflight.selector_count,
        "lookup_count": preflight.lookup_count,
        "shuffle_count": preflight.shuffle_count,
        "blinding_factors": preflight.blinding_factors,
        "fixed_lagrange_backing_bytes": preflight.fixed_lagrange_backing_bytes,
        "selector_dense_bitmap_ceiling_bytes": preflight.selector_dense_bitmap_ceiling_bytes,
        "selector_activation_container_bytes": preflight.selector_activation_container_bytes,
        "selector_activation_initial_backing_bytes": preflight.selector_activation_initial_backing_bytes,
        "one_selector_decode_bytes": preflight.one_selector_decode_bytes,
        "one_selector_encoded_bytes": preflight.one_selector_encoded_bytes,
        "permutation_logical_cells": preflight.permutation_logical_cells,
        "permutation_data_virtual_bytes": preflight.permutation_data_virtual_bytes,
        "permutation_directory_virtual_bytes": preflight.permutation_directory_virtual_bytes,
        "permutation_allocation_bitmap_bytes": preflight.permutation_allocation_bitmap_bytes,
        "permutation_bounded_cache_bytes": preflight.permutation_bounded_cache_bytes,
    })
}

fn native_keygen_census_json(
    census: &plonk::NativeKeygenCensus,
    lookup_shape: &NativeMvLookupShape,
) -> JsonValue {
    serde_json::json!({
        "preflight": native_keygen_preflight_json(&census.preflight),
        "compressed_selector_columns": census.compressed_selector_columns,
        "selector_activation_storage_current_bytes": census.selector_activation_storage_current_bytes,
        "selector_activation_storage_peak_bytes": census.selector_activation_storage_peak_bytes,
        "selector_activation_counts": census.selector_activation_counts,
        "fixed_assignment_callback_counts": census.fixed_assignment_callback_counts,
        "fixed_value_assignment_counts": census.fixed_value_assignment_counts,
        "fixed_value_explicit_zero_seen": census.fixed_value_explicit_zero_seen,
        "advice_assignment_counts": census.advice_assignment_counts,
        "lookup_table_cardinality_upper_bounds": lookup_shape.table_cardinality_upper_bounds,
        "lookup_argument_table_group_indices": lookup_shape.argument_table_group_indices,
        "permutation_copy_calls": census.permutation_copy_calls,
        "permutation_successor_edges": census.permutation_successor_edges,
        "permutation_nondefault_records": census.permutation_nondefault_records,
        "permutation_data_pages_allocated": census.permutation_data_pages_allocated,
        "permutation_directory_pages_allocated": census.permutation_directory_pages_allocated,
        "permutation_data_allocated_bytes": census.permutation_data_allocated_bytes,
        "permutation_directory_allocated_bytes": census.permutation_directory_allocated_bytes,
        "permutation_cache_bytes": census.permutation_cache_bytes,
        "permutation_allocation_bitmap_bytes": census.permutation_allocation_bitmap_bytes,
    })
}

fn native_keygen_budget_receipt(
    loaded: &crate::gat_adapter::LoadedGat,
    admission_dir: &Path,
    output_dir: &Path,
) -> Result<(JsonValue, String), EZKLError> {
    let path = output_dir.join("native_keygen_budget.json");
    require_formal_regular_file(&path, "native keygen resource budget receipt")?;
    let receipt: JsonValue = serde_json::from_slice(&std::fs::read(&path)?)?;
    let admission_path = admission_dir.join("admission.json");
    let settings_path = admission_dir.join("settings.json");
    let executable = std::env::current_exe()?;
    let session_id = std::env::var("EZKL_FORMAL_SESSION_ID")
        .map_err(|_| "formal native setup lacks the runner-authenticated session id")?;
    let build_receipt_sha256 = std::env::var("EZKL_FORMAL_BUILD_RECEIPT_SHA256")
        .map_err(|_| "formal native setup lacks the authenticated build receipt digest")?;
    let build_environment_sha256 = std::env::var("EZKL_FORMAL_BUILD_ENVIRONMENT_SHA256")
        .map_err(|_| "formal native setup lacks the authenticated build environment digest")?;
    let expected = [
        ("schema", "ezkl.native-keygen-budget.v3".to_string()),
        ("session_id", session_id),
        ("dataset", loaded.dataset.clone()),
        ("admission_sha256", get_file_hash(&admission_path)?),
        ("settings_sha256", get_file_hash(&settings_path)?),
        ("binary_sha256", get_file_hash(&executable)?),
        ("resource_manifest_sha256", loaded.resource_manifest_sha256.clone()),
        ("build_receipt_sha256", build_receipt_sha256),
        ("build_environment_sha256", build_environment_sha256),
    ];
    for (name, value) in expected {
        if receipt.get(name).and_then(JsonValue::as_str) != Some(value.as_str()) {
            return Err(format!("native keygen budget {} does not match this authenticated setup", name).into());
        }
    }
    for name in [
        "host_available_bytes",
        "host_os_diagnostic_reserve_bytes",
        "setup_keygen_future_host_reserve_bytes",
        "device_workspace_peak_bytes",
        "disk_available_bytes",
        "disk_required_known_bytes",
        "proof_spill_reserve_bytes",
        "permutation_successor_backing_file_upper_bytes",
        "native_permutation_backing_budget_bytes",
        "native_successor_read_cache_bytes",
        "native_target_bitmap_fixed_cache_bytes",
        "native_pk_read_buffer_capacity_bytes",
        "native_pk_format_version",
        "gat_transient_witness_bytes_upper",
    ] {
        if receipt.get(name).and_then(JsonValue::as_u64).is_none() {
            return Err(format!("native keygen budget {} is missing or invalid", name).into());
        }
    }
    let gat_transient_witness_bytes_upper =
        crate::gat_adapter::op::gat_transient_witness_bytes_upper(&loaded.spec)?;
    if receipt["gat_transient_witness_bytes_upper"].as_u64()
        != Some(gat_transient_witness_bytes_upper)
    {
        return Err("native keygen budget GAT witness workspace does not match the authenticated relation geometry".into());
    }
    if receipt.get("integrated_device").and_then(JsonValue::as_bool).is_none() {
        return Err("native keygen budget integrated-device state is missing or invalid".into());
    }
    if receipt["native_successor_read_cache_bytes"].as_u64()
        != u64::try_from(halo2_proofs::poly::NATIVE_SUCCESSOR_READ_CACHE_BYTES).ok()
        || receipt["native_target_bitmap_fixed_cache_bytes"].as_u64()
            != Some(halo2_proofs::poly::NATIVE_TARGET_BITMAP_FIXED_CACHE_BYTES)
        || receipt["native_pk_read_buffer_capacity_bytes"].as_u64()
            != u64::try_from(*EZKL_BUF_CAPACITY).ok()
        || receipt["native_pk_format_version"].as_u64() != Some(4)
        || receipt["native_pk_sigma_wire_encoding"].as_str()
            != Some("packed_source_target_indices_lsb0_v1")
    {
        return Err("native keygen budget PK-reader bounds or format version do not match this compiled Halo2 source".into());
    }
    let host_available = receipt["host_available_bytes"].as_u64().unwrap_or(0);
    let disk_available = receipt["disk_available_bytes"].as_u64().unwrap_or(0);
    let disk_required = receipt["disk_required_known_bytes"].as_u64().unwrap_or(u64::MAX);
    if host_available == 0
        || disk_available == 0
        || disk_required > disk_available
        || receipt["proof_spill_reserve_bytes"].as_u64() != Some(0)
        || receipt["disk_budget_scope"].as_str()
            != Some("setup_keygen_peak_and_source_bounded_permutation_backing; proof_disk_checked_after_actual_setup_census")
    {
        return Err("native keygen budget has no admitted host or disk headroom".into());
    }
    Ok((receipt, get_file_hash(&path)?))
}

fn native_keygen_budget_for_preflight(
    preflight: &plonk::NativeKeygenPreflight,
    budget_receipt: &JsonValue,
    model: &Model,
    stage: &str,
) -> Result<plonk::NativeKeygenBudget, std::io::Error> {
    let fail = |message: String| std::io::Error::other(message);
    let admitted_k = budget_receipt
        .get("logrows")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| fail("native keygen budget lacks admitted k".into()))?;
    let admitted_domain = budget_receipt
        .get("domain_size")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| fail("native keygen budget lacks admitted domain".into()))?;
    if u64::from(preflight.logrows) != admitted_k
        || preflight.domain_size != admitted_domain
        || !native_keygen_domains_within_capacity(preflight.logrows, preflight.extended_logrows)
    {
        return Err(fail(format!("{} native keygen domain exceeds admitted GAT/SRS geometry", stage)));
    }
    let integrated = budget_receipt
        .get("integrated_device")
        .and_then(JsonValue::as_bool)
        .ok_or_else(|| fail("native keygen budget has malformed unified-device state".into()))?;
    let device_headroom = if integrated {
        budget_receipt["device_workspace_peak_bytes"].as_u64().unwrap_or(u64::MAX)
    } else {
        0
    };
    let host_snapshot = budget_receipt["host_available_bytes"].as_u64().unwrap_or(0);
    let host_live = native_mem_available_bytes().map_err(|error| fail(error.to_string()))?;
    let host = host_snapshot.min(host_live);
    let os_reserve = budget_receipt["host_os_diagnostic_reserve_bytes"].as_u64().unwrap_or(u64::MAX);
    let input_shapes = model.graph.input_shapes().map_err(|error| fail(error.to_string()))?;
    let param_shapes = model.graph.param_shapes().map_err(|error| fail(error.to_string()))?;
    let output_shapes = model.graph.output_shapes().map_err(|error| fail(error.to_string()))?;
    let witnessless_elements = input_shapes
        .iter()
        .chain(param_shapes.iter())
        .chain(output_shapes.iter())
        .try_fold(0u64, |total, shape| {
            let elements = shape.iter().try_fold(1u64, |count, dimension| {
                count.checked_mul(u64::try_from(*dimension).ok()?)
            })?;
            total.checked_add(elements)
        })
        .ok_or_else(|| fail("witnessless GAT tensor shape byte count overflow".into()))?;
    let witnessless_bytes = witnessless_elements
        .checked_mul(std::mem::size_of::<halo2_proofs::circuit::Value<Fr>>() as u64)
        .ok_or_else(|| fail("witnessless GAT tensor buffer byte count overflow".into()))?;
    let future_keygen_reserve = budget_receipt["setup_keygen_future_host_reserve_bytes"]
        .as_u64()
        .unwrap_or(u64::MAX);
    let gat_transient_reserve = budget_receipt["gat_transient_witness_bytes_upper"]
        .as_u64()
        .unwrap_or(u64::MAX);
    let live_nonselector = preflight
        // This mapped backing is counted as a conservative potential dirty
        // host-residency upper bound, not as measured RSS.
        .fixed_lagrange_backing_bytes
        .checked_add(preflight.permutation_bounded_cache_bytes)
        .and_then(|bytes| bytes.checked_add(preflight.one_selector_decode_bytes))
        .and_then(|bytes| bytes.checked_add(preflight.one_selector_encoded_bytes))
        .and_then(|bytes| bytes.checked_add(witnessless_bytes))
        .and_then(|bytes| bytes.checked_add(future_keygen_reserve))
        .and_then(|bytes| bytes.checked_add(gat_transient_reserve))
        .and_then(|bytes| bytes.checked_add(device_headroom))
        .and_then(|bytes| bytes.checked_add(os_reserve))
        .ok_or_else(|| fail("native keygen preflight resident reserve overflow".into()))?;
    let max_selector_activation_bytes = host.checked_sub(live_nonselector).ok_or_else(|| {
        fail(format!(
            "{} native keygen preflight needs at least {} resident bytes before selector activation, but live budget is {}",
            stage, live_nonselector, host
        ))
    })?;
    if max_selector_activation_bytes < preflight.selector_activation_initial_backing_bytes {
        return Err(fail(format!(
            "{} selector activation initial backing {} exceeds residual native pool {}",
            stage, preflight.selector_activation_initial_backing_bytes, max_selector_activation_bytes
        )));
    }
    let disk_budget = budget_receipt["native_permutation_backing_budget_bytes"]
        .as_u64()
        .unwrap_or(0);
    let successor_upper = budget_receipt["permutation_successor_backing_file_upper_bytes"]
        .as_u64()
        .ok_or_else(|| fail("native keygen budget lacks its successor-file source bound".into()))?;
    let source_backing_upper = preflight
        .permutation_data_virtual_bytes
        .checked_add(preflight.permutation_directory_virtual_bytes)
        .and_then(|bytes| bytes.checked_add(successor_upper))
        .ok_or_else(|| fail("native permutation source backing ceiling overflow".into()))?;
    Ok(plonk::NativeKeygenBudget {
        max_selector_activation_bytes,
        max_permutation_backing_bytes: disk_budget.min(source_backing_upper),
    })
}

/// Validate the separately bounded base SRS and extended quotient domains.
/// The base domain uses the authenticated public SRS cap; the quotient domain
/// is bounded by BN254 Fr's two-adicity, as in native Halo2 keygen.
pub(crate) fn native_keygen_domains_within_capacity(base_logrows: u32, extended_logrows: u32) -> bool {
    base_logrows <= crate::graph::MAX_PUBLIC_SRS && extended_logrows <= Fr::S
}

fn check_native_keygen_census_budget(
    census: &plonk::NativeKeygenCensus,
    budget_receipt: &JsonValue,
    stage: &str,
) -> Result<(), std::io::Error> {
    let fail = |message: String| std::io::Error::other(message);
    let disk_budget = budget_receipt["native_permutation_backing_budget_bytes"]
        .as_u64()
        .unwrap_or(0);
    let successor_bytes = census
        .permutation_successor_edges
        .checked_mul(16)
        .and_then(|bytes| bytes.checked_add(4095))
        .map(|bytes| bytes / 4096 * 4096)
        .ok_or_else(|| fail("native page-rounded successor-store disk bound overflow".into()))?;
    let permutation_backing = census
        .permutation_data_allocated_bytes
        .checked_add(census.permutation_directory_allocated_bytes)
        .and_then(|bytes| bytes.checked_add(successor_bytes))
        .ok_or_else(|| fail("native permutation backing byte count overflow".into()))?;
    if permutation_backing > disk_budget {
        return Err(fail(format!(
            "{} exact native permutation backing {} exceeds authenticated disk budget {}",
            stage, permutation_backing, disk_budget
        )));
    }
    let host_live = native_mem_available_bytes().map_err(|error| fail(error.to_string()))?;
    let host_snapshot = budget_receipt["host_available_bytes"].as_u64().unwrap_or(0);
    let os_reserve = budget_receipt["host_os_diagnostic_reserve_bytes"].as_u64().unwrap_or(u64::MAX);
    let device_headroom = if budget_receipt["integrated_device"].as_bool().unwrap_or(false) {
        budget_receipt["device_workspace_peak_bytes"].as_u64().unwrap_or(u64::MAX)
    } else {
        0
    };
    let future_setup = budget_receipt["setup_keygen_future_host_reserve_bytes"]
        .as_u64()
        .unwrap_or(u64::MAX);
    let gat_transient_reserve = budget_receipt["gat_transient_witness_bytes_upper"]
        .as_u64()
        .unwrap_or(u64::MAX);
    let retained = future_setup
        .checked_add(gat_transient_reserve)
        .and_then(|bytes| bytes.checked_add(device_headroom))
        .and_then(|bytes| bytes.checked_add(os_reserve))
        .ok_or_else(|| fail("native keygen future resident resource count overflow".into()))?;
    if retained > host_live.min(host_snapshot) {
        return Err(fail(format!(
            "{} native keygen future host/device workspace {} exceeds remaining shared physical pool {} (selector, permutation and fixed backing already reflected in current availability)",
            stage, retained, host_live.min(host_snapshot)
        )));
    }
    Ok(())
}

fn write_native_keygen_receipt(
    output_dir: &Path,
    loaded: &crate::gat_adapter::LoadedGat,
    budget_sha256: &str,
    census: &plonk::NativeKeygenCensus,
    lookup_shape: &NativeMvLookupShape,
    stage_budget_caps: &[JsonValue],
    vk_sha256: &str,
    pk_sha256: &str,
) -> Result<(), EZKLError> {
    let path = output_dir.join("native_keygen_receipt.json");
    if std::fs::symlink_metadata(&path).is_ok() {
        return Err("native keygen receipt already exists".into());
    }
    if stage_budget_caps.len() != 2
        || stage_budget_caps[0].get("stage").and_then(JsonValue::as_str) != Some("vk")
        || stage_budget_caps[1].get("stage").and_then(JsonValue::as_str) != Some("pk")
    {
        return Err("native keygen receipt requires separately measured VK and PK allocation caps".into());
    }
    let value = serde_json::json!({
        "schema": "ezkl.native-keygen-receipt.v4",
        "native_pk_format_version": 4,
        "native_pk_sigma_wire_encoding": "packed_source_target_indices_lsb0_v1",
        "session_id": std::env::var("EZKL_FORMAL_SESSION_ID").unwrap_or_default(),
        "dataset": loaded.dataset,
        "relation_identity": native_admission_relation_identity(loaded)?,
        "resource_manifest_sha256": loaded.resource_manifest_sha256,
        "budget_receipt_sha256": budget_sha256,
        "settings_sha256": get_file_hash(&output_dir.join("settings.json"))?,
        "compiled_circuit_sha256": get_file_hash(&output_dir.join("compiled.circuit"))?,
        "vk_sha256": vk_sha256,
        "pk_sha256": pk_sha256,
        "evidence_kind": "mandatory_vk_pk_native_keygen_synthesis_census",
        "coverage": "actual_vk_and_pk_synthesis_census; exact_selector_advice_fixed_permutation_and_lookup_bounds",
        "stage_budget_caps": stage_budget_caps,
        "native_census": native_keygen_census_json(census, lookup_shape),
    });
    let temporary = output_dir.join("native_keygen_receipt.json.tmp");
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec_pretty(&value)?)?;
    file.sync_all()?;
    std::fs::rename(&temporary, &path)?;
    std::fs::File::open(output_dir)?.sync_all()?;
    Ok(())
}

fn gat_prepare_setup(
    loaded: crate::gat_adapter::LoadedGat,
    output_dir: PathBuf,
    srs_path: PathBuf,
    srs_sha256: String,
    gpu: bool,
    admission_dir: Option<PathBuf>,
) -> Result<String, EZKLError> {
    let _ = gpu;
    std::fs::create_dir_all(&output_dir)?;
    for name in [
        "settings.json",
        "compiled.circuit",
        "pk.key",
        "vk.key",
        "native_keygen_receipt.json",
        "gat_static_tables.json",
        "gat_integer_ledger.json",
        "circuit_ref.json",
        "producer_context.json",
        "producer_manifest.json",
        "artifact_manifest.json",
        "statement.json",
        "public_output.json",
        "srs.kzg",
    ] {
        let path = output_dir.join(name);
        if std::fs::symlink_metadata(&path).is_ok() {
            return Err(format!(
                "native GAT setup refuses to reuse an existing output: {}",
                path.display()
            )
            .into());
        }
    }
    let settings_path = output_dir.join("settings.json");
    let compiled_path = output_dir.join("compiled.circuit");
    let circuit = if let Some(admission_dir) = admission_dir.as_deref() {
        let (circuit, _) = load_native_admission_circuit(&loaded, admission_dir)?;
        circuit
    } else {
        build_canonical_gat_circuit(&loaded)?
    };
    // The shape-only admission persists settings, not a second serialized
    // circuit. Construct the setup circuit once from those settings and keep
    // the compiled artifact's witness storage empty.
    circuit.settings().save(&settings_path)?;
    circuit.save(compiled_path.clone())?;
    let (native_budget_receipt, native_budget_sha256) = native_keygen_budget_receipt(
        &loaded,
        admission_dir.as_deref().ok_or_else(|| {
            "formal native GAT setup requires its shape-only admission directory".to_string()
        })?,
        &output_dir,
    )?;
    let pk_path = output_dir.join("pk.key");
    let vk_path = output_dir.join("vk.key");
    let params = load_formal_kzg_params(
        &srs_path,
        circuit.settings().run_args.logrows,
        &srs_sha256,
    )?;
    let _native_spill = halo2_proofs::poly::native_spill_scope(&output_dir)?;
    let mut keygen_stage_budget_caps = Vec::<JsonValue>::new();
    let (pk, native_census) = create_keys_with_native_gate::<
        KZGCommitmentScheme<Bn256>,
        GraphCircuit,
        _,
        _,
    >(
        &circuit,
        &params,
        false,
        |stage, preflight| {
            let budget = native_keygen_budget_for_preflight(
                preflight,
                &native_budget_receipt,
                circuit.model(),
                stage,
            )?;
            keygen_stage_budget_caps.push(serde_json::json!({
                "stage": stage,
                "max_selector_activation_bytes": budget.max_selector_activation_bytes,
                "max_permutation_backing_bytes": budget.max_permutation_backing_bytes,
            }));
            Ok(budget)
        },
        |stage, census| {
            check_native_keygen_census_budget(census, &native_budget_receipt, stage)?;
            Ok(())
        },
    )?;
    let setup_cs = configure_canonical_constraint_system(&circuit);
    let actual_lookup_shape = native_mv_lookup_shape_with_bounds(
        &setup_cs,
        &native_census.selector_activation_counts,
        &native_census.fixed_assignment_callback_counts,
        &native_census.fixed_value_explicit_zero_seen,
        native_census.preflight.domain_size,
    )?;
    let admission_path = admission_dir
        .as_deref()
        .ok_or_else(|| "formal native GAT setup requires its configured-bound admission".to_string())?
        .join("admission.json");
    let admission: JsonValue = serde_json::from_slice(&std::fs::read(&admission_path)?)?;
    let admitted_lookup_bounds = admission
        .get("native_memory_model")
        .and_then(JsonValue::as_object)
        .and_then(|memory| memory.get("lookup_table_cardinality_upper_bounds"))
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "configured admission lacks lookup source bounds".to_string())?;
    let admitted_lookup_groups = admission
        .get("native_memory_model")
        .and_then(JsonValue::as_object)
        .and_then(|memory| memory.get("lookup_argument_table_group_indices"))
        .and_then(JsonValue::as_array)
        .ok_or_else(|| "configured admission lacks lookup group mapping".to_string())?;
    if actual_lookup_shape.argument_count as usize != setup_cs.lookups().len()
        || actual_lookup_shape.argument_table_group_indices != native_mv_lookup_argument_table_group_indices(&setup_cs)?
        || actual_lookup_shape.argument_table_group_indices.len() != admitted_lookup_groups.len()
        || actual_lookup_shape
            .argument_table_group_indices
            .iter()
            .zip(admitted_lookup_groups)
            .any(|(actual, admitted)| admitted.as_u64() != Some(*actual))
        || actual_lookup_shape.table_cardinality_upper_bounds.len() != admitted_lookup_bounds.len()
        || actual_lookup_shape
            .table_cardinality_upper_bounds
            .iter()
            .zip(admitted_lookup_bounds)
            .any(|(actual, admitted)| match admitted.as_u64() {
                Some(upper) => *actual > upper,
                None => true,
            })
    {
        return Err("actual Setup lookup census exceeds its configured source bounds".into());
    }
    drop(setup_cs);
    drop(params);
    drop(circuit);
    let expected_fixed_columns = native_census
        .preflight
        .fixed_columns
        .checked_add(native_census.compressed_selector_columns)
        .ok_or_else(|| "native keygen fixed-column count overflow".to_string())?;
    if pk.get_vk().fixed_commitments().len() != expected_fixed_columns {
        return Err("native proving-key fixed columns disagree with the exact keygen census".into());
    }
    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
    drop(pk);
    drop(_native_spill);
    write_native_keygen_receipt(
        &output_dir,
        &loaded,
        &native_budget_sha256,
        &native_census,
        &actual_lookup_shape,
        &keygen_stage_budget_caps,
        &get_file_hash(&vk_path)?,
        &get_file_hash(&pk_path)?,
    )?;
    write_generated_metadata(&loaded.spec, &output_dir).map_err(EZKLError::from)?;
    let settings_sha = get_file_hash(&settings_path)?;
    let compiled_sha = get_file_hash(&compiled_path)?;
    let vk_sha = get_file_hash(&vk_path)?;
    let circuit_ref = resolved_circuit_ref(
        &loaded,
        settings_sha.clone(),
        compiled_sha.clone(),
        vk_sha.clone(),
    )?;
    let setup_context = BoundCircuitRef::new(circuit_ref);
    setup_context.validate().map_err(EZKLError::from)?;
    let context_path = output_dir.join(DEFAULT_GAT_SETUP_CONTEXT);
    let context_bytes = serde_json::to_vec_pretty(&setup_context)?;
    std::fs::write(&context_path, context_bytes)?;
    let producer_context = expected_producer_context(&loaded, &setup_context);
    std::fs::write(
        output_dir.join("producer_context.json"),
        serde_json::to_vec_pretty(&producer_context)?,
    )?;
    FormalCliContract::write(Path::new(DEFAULT_FORMAL_CONTRACT)).map_err(EZKLError::from)?;
    Ok(output_dir.display().to_string())
}

fn expected_producer_context(
    loaded: &crate::gat_adapter::LoadedGat,
    setup_context: &BoundCircuitRef,
) -> JsonValue {
    serde_json::json!({
        "schema_version": 1,
        "relation_id": "R_GAT-ezkl",
        "dataset": loaded.dataset,
        "model_arch_id": loaded.spec.model.model_arch_id,
        "quant_cfg_id": loaded.spec.quant_cfg_id,
        "quantization": loaded.spec.quantization,
        "static_tables": {
            "id": loaded.spec.tables.static_tables_id,
            "digest": loaded.spec.tables.static_tables_digest,
        },
        "ledger": loaded.spec.ledger_identity,
        "topology": loaded.spec.topology,
        "model": {
            "model_hash_field": loaded.spec.model_binding.hash,
            "model_digest_sha256": loaded.spec.model_binding.canonical_model_digest,
            "hash_domain": loaded.spec.model_binding.hash_domain,
        },
        "circuit": setup_context,
        "ezkl_security_profile_id": loaded.spec.ezkl_security_profile_id,
    })
}

fn baseline_relative_path(root: &Path, path: &Path, label: &str) -> Result<String, EZKLError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    };
    absolute
        .strip_prefix(root)
        .map(|value| value.to_string_lossy().replace('\\', "/"))
        .map_err(|_| format!("{} escapes baseline root: {}", label, path.display()).into())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExternalProducerManifest {
    schema_version: u32,
    producer_id: String,
    producer_version: String,
    independent: bool,
    relation_id: String,
    dataset: String,
    producer_script: String,
    producer_script_sha256: String,
    config_path: String,
    config_sha256: String,
    resource_manifest_path: String,
    resource_manifest_sha256: String,
    checkpoint_manifest_path: String,
    checkpoint_manifest_sha256: String,
    tensor_archive_path: String,
    tensor_archive_sha256: String,
    feature_path: String,
    feature_sha256: String,
    setup_context_path: String,
    setup_context_sha256: String,
    model_arch_id: String,
    model_hash: String,
    hash_domain: String,
    canonical_model_digest: String,
    circuit_ref: BoundCircuitRef,
    input_digest: String,
    output_digest: String,
    public_output_sha256: String,
    statement_sha256: String,
    public_statement_path: String,
    public_output_path: String,
    provenance: JsonValue,
}

fn validate_external_producer_manifest(
    path: &Path,
    loaded: &crate::gat_adapter::LoadedGat,
    producer_context_path: &Path,
    statement_path: &Path,
    public_output_path: &Path,
    statement: &PublicStatement,
    setup_context: &BoundCircuitRef,
) -> Result<(), EZKLError> {
    require_formal_regular_file(path, "independent producer manifest")?;
    require_formal_regular_file(producer_context_path, "producer setup context")?;
    let bytes = std::fs::read(path)?;
    let manifest: ExternalProducerManifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("parse independent producer manifest: {}", error))?;
    let valid_digest = |label: &str, value: &str| -> Result<(), EZKLError> {
        if value.len() != 64
            || value.bytes().any(|byte| !byte.is_ascii_hexdigit())
            || value.bytes().any(|byte| byte.is_ascii_uppercase())
        {
            return Err(format!("{} is not a lowercase SHA-256 digest", label).into());
        }
        Ok(())
    };
    for (label, value) in [
        ("producer_script_sha256", &manifest.producer_script_sha256),
        ("config_sha256", &manifest.config_sha256),
        ("resource_manifest_sha256", &manifest.resource_manifest_sha256),
        ("checkpoint_manifest_sha256", &manifest.checkpoint_manifest_sha256),
        ("tensor_archive_sha256", &manifest.tensor_archive_sha256),
        ("feature_sha256", &manifest.feature_sha256),
        ("setup_context_sha256", &manifest.setup_context_sha256),
        ("input_digest", &manifest.input_digest),
        ("output_digest", &manifest.output_digest),
        ("public_output_sha256", &manifest.public_output_sha256),
        ("statement_sha256", &manifest.statement_sha256),
    ] {
        valid_digest(label, value)?;
    }
    if manifest.schema_version != 1
        || manifest.producer_id != "ezkl-gat-independent-python"
        || manifest.producer_version != "1"
        || !manifest.independent
        || manifest.relation_id != "R_GAT-ezkl"
        || manifest.dataset != loaded.dataset
        || manifest.producer_script != "scripts/produce_public_statement.py"
        || manifest.config_path != format!("resources/canonical/configs/{}_secure.cfg", loaded.dataset)
        || manifest.resource_manifest_path != "resources/resource_manifest.json"
        || manifest.model_arch_id != loaded.spec.model.model_arch_id
        || manifest.model_hash != loaded.spec.model_binding.hash
        || manifest.hash_domain != loaded.spec.model_binding.hash_domain
        || manifest.canonical_model_digest != loaded.spec.model_binding.canonical_model_digest
        || manifest.circuit_ref != *setup_context
        || manifest.setup_context_path
            != baseline_relative_path(
                &loaded.baseline_root,
                producer_context_path,
                "producer setup context",
            )?
        || manifest.public_statement_path
            != baseline_relative_path(
                &loaded.baseline_root,
                statement_path,
                "public statement",
            )?
        || manifest.public_output_path
            != baseline_relative_path(
                &loaded.baseline_root,
                public_output_path,
                "public output",
            )?
    {
        return Err("independent producer manifest identity disagrees with authenticated resources".into());
    }
    let expected_context = expected_producer_context(loaded, setup_context);
    let context: JsonValue = serde_json::from_slice(&std::fs::read(producer_context_path)?)
        .map_err(|error| format!("parse producer setup context: {}", error))?;
    if context != expected_context {
        return Err("producer setup context is stale or disagrees with authenticated resources".into());
    }
    let checkpoint_root = format!(
        "resources/canonical/checkpoints/{}_gat",
        loaded.dataset
    );
    let feature_name = if loaded.dataset == "cora"
        || loaded.dataset == "citeseer"
        || loaded.dataset == "pubmed"
    {
        "features.txt"
    } else {
        "features.npy"
    };
    let expected_checkpoint_path = format!("{}/manifest.json", checkpoint_root);
    let expected_tensor_path = format!("{}/tensors.txt", checkpoint_root);
    let expected_feature_path = format!(
        "resources/canonical/cache/{}/{}",
        loaded.dataset, feature_name
    );
    if manifest.checkpoint_manifest_path != expected_checkpoint_path
        || manifest.tensor_archive_path != expected_tensor_path
        || manifest.feature_path != expected_feature_path
        || manifest.setup_context_sha256 != get_file_hash(producer_context_path)?
    {
        return Err("independent producer resource/setup paths or digests are stale".into());
    }
    let config_path = loaded
        .baseline_root
        .join(format!("resources/canonical/configs/{}_secure.cfg", loaded.dataset));
    let resource_manifest_path = loaded.baseline_root.join("resources/resource_manifest.json");
    let producer_script_path = loaded
        .baseline_root
        .join("scripts/produce_public_statement.py");
    let checkpoint_manifest_path = loaded
        .baseline_root
        .join(&manifest.checkpoint_manifest_path);
    let tensor_archive_path = loaded.baseline_root.join(&manifest.tensor_archive_path);
    let feature_path = loaded.baseline_root.join(&manifest.feature_path);
    if get_file_hash(&config_path)? != manifest.config_sha256
        || get_file_hash(&resource_manifest_path)? != manifest.resource_manifest_sha256
        || get_file_hash(&producer_script_path)? != manifest.producer_script_sha256
        || get_file_hash(&checkpoint_manifest_path)? != manifest.checkpoint_manifest_sha256
        || get_file_hash(&tensor_archive_path)? != manifest.tensor_archive_sha256
        || get_file_hash(&feature_path)? != manifest.feature_sha256
    {
        return Err("independent producer source/config/resource/model digest is stale".into());
    }
    let mut input_bytes = b"ezkl-gat-public-input-v1".to_vec();
    for value in &statement.public_ezkl.feature_table {
        input_bytes.extend_from_slice(&value.to_le_bytes());
    }
    let topology_bytes = hex::decode(&statement.public_ezkl.topology.digest)
        .map_err(|error| format!("decode statement topology digest: {}", error))?;
    input_bytes.extend_from_slice(&topology_bytes);
    let expected_input_digest =
        crate::gat_adapter::schema::sha256_digest(&input_bytes).map_err(EZKLError::from)?;
    if manifest.resource_manifest_sha256 != loaded.resource_manifest_sha256
        || manifest.input_digest != expected_input_digest
        || manifest.output_digest != statement.output_digest
        || manifest.public_output_sha256 != statement.public_output_sha256
        || manifest.public_output_sha256 != get_file_hash(public_output_path)?
        || manifest.statement_sha256 != get_file_hash(statement_path)?
    {
        return Err("independent producer output/statement digest rebinding failed".into());
    }
    let provenance = manifest
        .provenance
        .as_object()
        .ok_or_else(|| "independent producer provenance is not an object".to_string())?;
    if provenance.get("language").and_then(JsonValue::as_str) != Some("python")
        || provenance.get("algorithm").and_then(JsonValue::as_str)
            != Some("canonical_integer_gat_v1")
        || provenance.get("witness_loaded").and_then(JsonValue::as_bool) != Some(false)
        || provenance
            .get("native_ezkl_commands")
            .and_then(JsonValue::as_array)
            .map(|values| !values.is_empty())
            .unwrap_or(true)
        || provenance
            .get("proof_commands")
            .and_then(JsonValue::as_array)
            .map(|values| !values.is_empty())
            .unwrap_or(true)
    {
        return Err("independent producer provenance is not external-only".into());
    }
    Ok(())
}

fn formal_run_relative_path(run_dir: &Path, baseline_root: &Path) -> Result<PathBuf, EZKLError> {
    let lexical_relative = if run_dir.is_absolute() {
        let relative = run_dir.strip_prefix(baseline_root).map_err(|_| {
            format!("formal run directory escapes baseline: {}", run_dir.display())
        })?;
        require_formal_local_path(relative, "formal run directory")?;
        relative.to_path_buf()
    } else {
        require_formal_local_path(run_dir, "formal run directory")?;
        run_dir.to_path_buf()
    };
    if lexical_relative.as_os_str().is_empty() {
        return Err("formal run directory must be below the baseline root".into());
    }
    let canonical_root = baseline_root.canonicalize()?;
    let canonical_run = baseline_root
        .join(&lexical_relative)
        .canonicalize()
        .map_err(|error| {
            format!(
                "cannot resolve formal run directory {}: {}",
                run_dir.display(),
                error
            )
        })?;
    let canonical_relative = canonical_run.strip_prefix(&canonical_root).map_err(|_| {
        format!("formal run directory escapes baseline: {}", run_dir.display())
    })?;
    require_formal_local_path(canonical_relative, "formal run directory")?;
    if canonical_relative.as_os_str().is_empty() {
        return Err("formal run directory must be below the baseline root".into());
    }
    Ok(canonical_relative.to_path_buf())
}

fn validate_formal_producer_run(run_dir: &Path) -> Result<(), EZKLError> {
    let baseline_root = std::env::current_dir()?;
    let run_relative = formal_run_relative_path(run_dir, &baseline_root)?;
    require_formal_local_path(&run_relative, "formal run directory")?;
    let artifact_path = run_relative.join("artifact_manifest.json");
    let statement_path = run_relative.join("statement.json");
    let public_output_path = run_relative.join("public_output.json");
    let producer_context_path = run_relative.join("producer_context.json");
    let producer_manifest_path = run_relative.join("producer_manifest.json");
    for (path, label) in [
        (&artifact_path, "artifact manifest"),
        (&statement_path, "statement"),
        (&public_output_path, "public output"),
        (&producer_context_path, "producer setup context"),
        (&producer_manifest_path, "producer manifest"),
    ] {
        require_formal_regular_file(path, label)?;
    }
    let artifact = ArtifactManifest::load(&artifact_path).map_err(EZKLError::from)?;
    artifact.validate().map_err(EZKLError::from)?;
    if artifact.producer_manifest_sha256 != get_file_hash(&producer_manifest_path)? {
        return Err("artifact producer manifest digest is stale".into());
    }
    let producer_record = artifact
        .files
        .iter()
        .find(|file| file.path == "producer_manifest.json")
        .ok_or_else(|| "artifact manifest has no producer manifest file".to_string())?;
    if producer_record.sha256 != artifact.producer_manifest_sha256 {
        return Err("artifact producer manifest top-level/file digest mismatch".into());
    }
    let statement = PublicStatement::load(&statement_path).map_err(EZKLError::from)?;
    let output = parse_public_output(&public_output_path, &statement.dataset)?;
    if output != statement.public_ezkl.y {
        return Err("formal producer run public output disagrees with statement".into());
    }
    let loaded = load_dataset_with_manifest(
        &baseline_root,
        Path::new("resources/resource_manifest.json"),
        &artifact.dataset,
    )
    .map_err(EZKLError::from)?;
    if artifact.resource_manifest_sha256 != loaded.resource_manifest_sha256 {
        return Err("formal producer run artifact resource identity is stale".into());
    }
    let setup_context = load_setup_circuit_ref(&run_relative.join(DEFAULT_GAT_SETUP_CONTEXT))?;
    if setup_context != artifact.circuit_ref {
        return Err("formal producer setup context disagrees with artifact circuit".into());
    }
    let mut bound_spec = loaded.spec.clone();
    bound_spec.circuit_ref = Some(artifact.circuit_ref.native.clone());
    bound_spec.validate().map_err(EZKLError::from)?;
    validate_statement_against_loaded(&statement, &bound_spec)?;
    validate_external_producer_manifest(
        &producer_manifest_path,
        &loaded,
        &producer_context_path,
        &statement_path,
        &public_output_path,
        &statement,
        &setup_context,
    )
}

fn validate_external_producer_manifest_public(
    path: &Path,
    producer_context_path: &Path,
    statement_path: &Path,
    public_output_path: &Path,
    statement: &PublicStatement,
    circuit_ref: &BoundCircuitRef,
    authenticated_resources: &crate::gat_adapter::AuthenticatedRelationResources,
    baseline_root: &Path,
) -> Result<(), EZKLError> {
    require_formal_regular_file(path, "independent producer manifest")?;
    require_formal_regular_file(producer_context_path, "producer setup context")?;
    let manifest: ExternalProducerManifest =
        serde_json::from_slice(&std::fs::read(path)?).map_err(|error| {
            format!("parse independent producer manifest: {}", error)
        })?;
    let valid_digest = |label: &str, value: &str| -> Result<(), EZKLError> {
        if value.len() != 64
            || value.bytes().any(|byte| !byte.is_ascii_hexdigit())
            || value.bytes().any(|byte| byte.is_ascii_uppercase())
        {
            return Err(format!("{} is not a lowercase SHA-256 digest", label).into());
        }
        Ok(())
    };
    for (label, value) in [
        ("producer_script_sha256", &manifest.producer_script_sha256),
        ("config_sha256", &manifest.config_sha256),
        ("resource_manifest_sha256", &manifest.resource_manifest_sha256),
        ("checkpoint_manifest_sha256", &manifest.checkpoint_manifest_sha256),
        ("tensor_archive_sha256", &manifest.tensor_archive_sha256),
        ("feature_sha256", &manifest.feature_sha256),
        ("setup_context_sha256", &manifest.setup_context_sha256),
        ("input_digest", &manifest.input_digest),
        ("output_digest", &manifest.output_digest),
        ("public_output_sha256", &manifest.public_output_sha256),
        ("statement_sha256", &manifest.statement_sha256),
    ] {
        valid_digest(label, value)?;
    }
    let root = baseline_root;
    let statement_rel = baseline_relative_path(root, statement_path, "public statement")?;
    let output_rel = baseline_relative_path(root, public_output_path, "public output")?;
    let context_rel =
        baseline_relative_path(root, producer_context_path, "producer setup context")?;
    let feature_name = if statement.dataset == "cora"
        || statement.dataset == "citeseer"
        || statement.dataset == "pubmed"
    {
        "features.txt"
    } else {
        "features.npy"
    };
    let checkpoint_root = format!(
        "resources/canonical/checkpoints/{}_gat",
        statement.dataset
    );
    let expected_context = serde_json::json!({
        "schema_version": 1,
        "relation_id": "R_GAT-ezkl",
        "dataset": statement.dataset,
        "model_arch_id": statement.model_arch_id,
        "quant_cfg_id": statement.quant_cfg_id,
        "quantization": statement.public_ezkl.quant,
        "static_tables": statement.public_ezkl.static_tables,
        "ledger": statement.public_ezkl.ledger,
        "topology": statement.public_ezkl.topology,
        "model": {
            "model_hash_field": statement.public_ezkl.model.hash,
            "model_digest_sha256": statement.public_ezkl.model.canonical_model_digest,
            "hash_domain": statement.public_ezkl.model.hash_domain,
        },
        "circuit": statement.public_ezkl.circuit,
        "ezkl_security_profile_id": statement.ezkl_security_profile_id,
    });
    let context: JsonValue = serde_json::from_slice(&std::fs::read(producer_context_path)?)
        .map_err(|error| format!("parse producer setup context: {}", error))?;
    if context != expected_context {
        return Err("producer setup context is not bound to the public statement".into());
    }
    let expected_checkpoint_path = format!("{}/manifest.json", checkpoint_root);
    let expected_tensor_path = format!("{}/tensors.txt", checkpoint_root);
    let expected_feature_path = format!(
        "resources/canonical/cache/{}/{}",
        statement.dataset, feature_name
    );
    if manifest.schema_version != 1
        || manifest.producer_id != "ezkl-gat-independent-python"
        || manifest.producer_version != "1"
        || !manifest.independent
        || manifest.relation_id != "R_GAT-ezkl"
        || manifest.dataset != statement.dataset
        || manifest.producer_script != "scripts/produce_public_statement.py"
        || manifest.config_path
            != format!(
                "resources/canonical/configs/{}_secure.cfg",
                statement.dataset
            )
        || manifest.resource_manifest_path != "resources/resource_manifest.json"
        || manifest.checkpoint_manifest_path != expected_checkpoint_path
        || manifest.tensor_archive_path != expected_tensor_path
        || manifest.feature_path != expected_feature_path
        || manifest.setup_context_path != context_rel
        || manifest.model_arch_id != statement.model_arch_id
        || manifest.model_hash != statement.public_ezkl.model.hash
        || manifest.hash_domain != statement.public_ezkl.model.hash_domain
        || manifest.canonical_model_digest
            != statement.public_ezkl.model.canonical_model_digest
        || manifest.circuit_ref != *circuit_ref
        || manifest.public_statement_path != statement_rel
        || manifest.public_output_path != output_rel
    {
        return Err("public verifier producer manifest identity mismatch".into());
    }
    let config_path = root.join(&manifest.config_path);
    let resource_manifest_path = root.join(&manifest.resource_manifest_path);
    let producer_script_path = root.join(&manifest.producer_script);
    let checkpoint_manifest_path = root.join(&manifest.checkpoint_manifest_path);
    let tensor_archive_path = root.join(&manifest.tensor_archive_path);
    let feature_path = root.join(&manifest.feature_path);
    if manifest.resource_manifest_sha256 != authenticated_resources.resource_manifest_sha256
        || manifest.config_sha256 != get_file_hash(&config_path)?
        || manifest.resource_manifest_sha256 != get_file_hash(&resource_manifest_path)?
        || manifest.producer_script_sha256 != get_file_hash(&producer_script_path)?
        || manifest.checkpoint_manifest_sha256 != get_file_hash(&checkpoint_manifest_path)?
        || manifest.tensor_archive_sha256 != get_file_hash(&tensor_archive_path)?
        || manifest.feature_sha256 != get_file_hash(&feature_path)?
        || manifest.setup_context_sha256 != get_file_hash(producer_context_path)?
        || manifest.statement_sha256 != get_file_hash(statement_path)?
        || manifest.public_output_sha256 != get_file_hash(public_output_path)?
    {
        return Err("public verifier producer resource/setup digest mismatch".into());
    }
    let mut input_bytes = b"ezkl-gat-public-input-v1".to_vec();
    for value in &statement.public_ezkl.feature_table {
        input_bytes.extend_from_slice(&value.to_le_bytes());
    }
    input_bytes.extend_from_slice(
        &hex::decode(&statement.public_ezkl.topology.digest)
            .map_err(|error| format!("decode statement topology digest: {}", error))?,
    );
    let expected_input_digest =
        crate::gat_adapter::schema::sha256_digest(&input_bytes).map_err(EZKLError::from)?;
    if manifest.input_digest != expected_input_digest
        || manifest.output_digest != statement.output_digest
    {
        return Err("public verifier producer input/output digest mismatch".into());
    }
    let provenance = manifest
        .provenance
        .as_object()
        .ok_or_else(|| "independent producer provenance is not an object".to_string())?;
    if provenance.get("language").and_then(JsonValue::as_str) != Some("python")
        || provenance.get("algorithm").and_then(JsonValue::as_str)
            != Some("canonical_integer_gat_v1")
        || provenance.get("witness_loaded").and_then(JsonValue::as_bool) != Some(false)
        || provenance
            .get("native_ezkl_commands")
            .and_then(JsonValue::as_array)
            .map(|values| !values.is_empty())
            .unwrap_or(true)
        || provenance
            .get("proof_commands")
            .and_then(JsonValue::as_array)
            .map(|values| !values.is_empty())
            .unwrap_or(true)
    {
        return Err("public verifier producer provenance is not external-only".into());
    }
    Ok(())
}

fn gat_prepare_final(
    loaded: crate::gat_adapter::LoadedGat,
    output_dir: PathBuf,
    srs_path: PathBuf,
    srs_sha256: String,
    public_statement_path: PathBuf,
    public_output_path: PathBuf,
    producer_manifest_path: PathBuf,
    backend: String,
) -> Result<String, EZKLError> {
    let settings_path = output_dir.join("settings.json");
    let compiled_path = output_dir.join("compiled.circuit");
    let pk_path = output_dir.join("pk.key");
    let vk_path = output_dir.join("vk.key");
    let context_path = output_dir.join(DEFAULT_GAT_SETUP_CONTEXT);
    let producer_context_path = output_dir.join("producer_context.json");
    let artifact_path = output_dir.join("artifact_manifest.json");
    for (path, label) in [
        (&settings_path, "settings"),
        (&compiled_path, "compiled circuit"),
        (&pk_path, "proving key"),
        (&vk_path, "verification key"),
        (&context_path, "setup CircuitRef"),
        (&producer_context_path, "producer setup context"),
        (&producer_manifest_path, "producer manifest"),
    ] {
        require_formal_regular_file(path, label)?;
    }
    if std::fs::symlink_metadata(&artifact_path).is_ok() {
        return Err(format!(
            "final GAT preparation refuses to overwrite an existing artifact manifest: {}",
            artifact_path.display()
        )
        .into());
    }
    let setup_context = load_setup_circuit_ref(&context_path)?;
    let settings_sha = get_file_hash(&settings_path)?;
    let compiled_sha = get_file_hash(&compiled_path)?;
    let vk_sha = get_file_hash(&vk_path)?;
    let pk_sha = get_file_hash(&pk_path)?;
    let circuit_ref = resolved_circuit_ref(
        &loaded,
        settings_sha.clone(),
        compiled_sha.clone(),
        vk_sha.clone(),
    )?;
    if setup_context.native != circuit_ref {
        return Err("setup CircuitRef does not match the native setup artifacts".into());
    }
    let mut bound_spec = loaded.spec.clone();
    bound_spec.circuit_ref = Some(circuit_ref.clone());
    bound_spec.validate().map_err(EZKLError::from)?;

    let supplied_statement =
        PublicStatement::load(&public_statement_path).map_err(EZKLError::from)?;
    if supplied_statement.dataset != loaded.dataset {
        return Err("external public statement dataset disagrees with gat-prepare dataset".into());
    }
    let supplied_output = parse_public_output(&public_output_path, &loaded.dataset)?;
    if supplied_output != supplied_statement.public_ezkl.y {
        return Err(
            "external public output Y disagrees element-by-element with public statement".into(),
        );
    }
    let supplied_output_sha256 = get_file_hash(&public_output_path)?;
    if supplied_output_sha256 != supplied_statement.public_output_sha256 {
        return Err("external public output digest disagrees with public statement".into());
    }
    validate_statement_against_loaded(&supplied_statement, &bound_spec)?;
    if supplied_statement.public_ezkl.circuit.native != circuit_ref {
        return Err(
            "external public statement CircuitRef does not match the setup artifacts".into(),
        );
    }
    validate_external_producer_manifest(
        &producer_manifest_path,
        &loaded,
        &producer_context_path,
        &public_statement_path,
        &public_output_path,
        &supplied_statement,
        &setup_context,
    )?;

    let bound_statement_path = output_dir.join("statement.json");
    let bound_output_path = output_dir.join("public_output.json");
    let bound_srs_path = output_dir.join("srs.kzg");
    copy_external_file(&public_statement_path, &bound_statement_path)?;
    copy_external_file(&public_output_path, &bound_output_path)?;
    materialize_authenticated_srs(&srs_path, &bound_srs_path, &srs_sha256)?;
    let mut files: Vec<ArtifactFile> = [
        (&settings_path, "settings.json"),
        (&compiled_path, "compiled.circuit"),
        (&pk_path, "pk.key"),
        (&vk_path, "vk.key"),
        (&context_path, DEFAULT_GAT_SETUP_CONTEXT),
        (&bound_statement_path, "statement.json"),
        (&bound_output_path, "public_output.json"),
        (&bound_srs_path, "srs.kzg"),
        (&producer_context_path, "producer_context.json"),
        (&producer_manifest_path, "producer_manifest.json"),
        (
            &output_dir.join("gat_static_tables.json"),
            "gat_static_tables.json",
        ),
        (
            &output_dir.join("gat_integer_ledger.json"),
            "gat_integer_ledger.json",
        ),
    ]
    .into_iter()
    .map(|(path, name)| {
        crate::gat_adapter::artifact_file(path, name.to_string()).map_err(EZKLError::from)
    })
    .collect::<Result<_, _>>()?;
    if std::env::var_os("EZKL_FORMAL_SESSION_ID").is_some() {
        for name in [
            "setup_receipt.json",
            "native_keygen_budget.json",
            "native_keygen_receipt.json",
            "native_prove_budget.json",
        ] {
            let path = output_dir.join(name);
            require_formal_regular_file(&path, "formal native resource receipt")?;
            files.push(crate::gat_adapter::artifact_file(&path, name.to_string()).map_err(EZKLError::from)?);
        }
    }
    let binary_path = std::env::current_exe()?;
    let binary_sha = get_file_hash(&binary_path)?;
    let source_sha = source_tree_hash(&loaded.baseline_root)?;
    let statement_sha = get_file_hash(&bound_statement_path)?;
    let srs_provenance = SrsProvenance {
        source: "official_ezkl_ceremony".into(),
        trusted: true,
        reduced: false,
        generated: false,
        ceremony_id: format!("sha256:{}", srs_sha256),
        sha256: srs_sha256.clone(),
    };
    let artifact = ArtifactManifest {
        schema_version: 1,
        baseline: "ezkl".into(),
        relation_id: "R_GAT-ezkl".into(),
        dataset: loaded.dataset.clone(),
        backend,
        proof_scope: "full_gat".into(),
        privacy_policy: "model_and_trace_hidden".into(),
        public_statement_mode: "external".into(),
        output_mode: "public_logits".into(),
        quant_cfg_id: loaded.spec.quant_cfg_id.clone(),
        static_tables_id: loaded.spec.tables.static_tables_id.clone(),
        circuit_ref: BoundCircuitRef::new(circuit_ref),
        compiled_circuit_id: format!("compiled:{}", compiled_sha),
        settings_id: format!("settings:{}", settings_sha),
        vk_id: format!("vk:{}", vk_sha),
        binary_sha256: binary_sha,
        source_tree_sha256: source_sha,
        resource_manifest_sha256: loaded.resource_manifest_sha256.clone(),
        files,
        source_snapshot: loaded.source_snapshot.clone(),
        statement_sha256: statement_sha,
        producer_manifest_sha256: get_file_hash(&producer_manifest_path)?,
        compiled_circuit_sha256: compiled_sha,
        settings_sha256: settings_sha,
        vk_sha256: vk_sha,
        pk_sha256: pk_sha,
        srs_sha256: srs_sha256.clone(),
        srs_provenance,
        ezkl_security_profile_id: loaded.spec.ezkl_security_profile_id.clone(),
    };
    artifact.save(&artifact_path).map_err(EZKLError::from)?;
    FormalCliContract::write(Path::new(DEFAULT_FORMAL_CONTRACT)).map_err(EZKLError::from)?;
    Ok(output_dir.display().to_string())
}

fn load_setup_circuit_ref(path: &Path) -> Result<BoundCircuitRef, EZKLError> {
    let bytes = std::fs::read(path)?;
    let context: BoundCircuitRef = serde_json::from_slice(&bytes)?;
    context.validate().map_err(EZKLError::from)?;
    Ok(context)
}

fn gat_produce(
    dataset: String,
    resource_manifest: PathBuf,
    output_dir: PathBuf,
    gpu: bool,
) -> Result<String, EZKLError> {
    let _ = (dataset, resource_manifest, output_dir, gpu);
    return Err(
        "gat-produce is a disabled legacy command; use the independent Python public-statement producer"
            .into(),
    );
}

fn gat_witness(
    dataset: String,
    resource_manifest: PathBuf,
    compiled_circuit: PathBuf,
    statement_path: PathBuf,
    output: PathBuf,
    gpu: bool,
) -> Result<String, EZKLError> {
    enforce_gat_gpu(gpu)?;
    require_formal_local_path(&resource_manifest, "resource manifest")?;
    require_formal_regular_file(&compiled_circuit, "compiled circuit")?;
    require_formal_regular_file(&statement_path, "statement")?;
    require_formal_local_path(&output, "witness output")?;
    let baseline_root = std::env::current_dir()?;
    let loaded = load_dataset_with_manifest(&baseline_root, &resource_manifest, &dataset)
        .map_err(EZKLError::from)?;
    let statement = PublicStatement::load(&statement_path).map_err(EZKLError::from)?;
    if statement.dataset != loaded.dataset {
        return Err("witness dataset and external statement dataset disagree".into());
    }
    let output_dir = compiled_circuit
        .parent()
        .ok_or_else(|| "compiled circuit has no setup output directory".to_string())?;
    let producer_context_path = output_dir.join("producer_context.json");
    let producer_manifest_path = output_dir.join("producer_manifest.json");
    require_formal_regular_file(&producer_context_path, "producer setup context")?;
    require_formal_regular_file(&producer_manifest_path, "producer manifest")?;
    let setup_context = load_setup_circuit_ref(&output_dir.join(DEFAULT_GAT_SETUP_CONTEXT))?;
    let settings_path = output_dir.join("settings.json");
    let vk_path = output_dir.join("vk.key");
    require_formal_regular_file(&settings_path, "settings")?;
    require_formal_regular_file(&vk_path, "verification key")?;
    let circuit_ref = resolved_circuit_ref(
        &loaded,
        get_file_hash(&settings_path)?,
        get_file_hash(&compiled_circuit)?,
        get_file_hash(&vk_path)?,
    )?;
    if setup_context.native != circuit_ref {
        return Err("setup CircuitRef does not match the native setup artifacts".into());
    }
    if statement.public_ezkl.circuit.native != circuit_ref {
        return Err("external statement CircuitRef does not match the setup artifacts".into());
    }
    let mut bound_spec = loaded.spec.clone();
    bound_spec.circuit_ref = Some(circuit_ref.clone());
    bound_spec.validate().map_err(EZKLError::from)?;
    validate_statement_against_loaded(&statement, &bound_spec)?;
    validate_external_producer_manifest(
        &producer_manifest_path,
        &loaded,
        &producer_context_path,
        &statement_path,
        &output_dir.join("public_output.json"),
        &statement,
        &setup_context,
    )?;
    let mut circuit = GraphCircuit::load(compiled_circuit)?;
    assert_native_gat_identity(&circuit, &loaded.spec)?;
    if !circuit.settings().run_args.input_visibility.is_public()
        || !circuit.settings().run_args.output_visibility.is_public()
        || !circuit
            .settings()
            .run_args
            .param_visibility
            .is_hashed_public()
    {
        return Err("prepared GAT circuit does not use public inputs/outputs and native public Hashed parameters".into());
    }
    let mut native_inputs = gat_input_tensors(&loaded.spec)?;
    let native_params = gat_parameter_tensor(&loaded.spec)?;
    let witness = circuit.forward_with_runtime_params::<KZGCommitmentScheme<Bn256>>(
        &mut native_inputs,
        Some(&native_params),
        None,
        None,
        RegionSettings::all_true(
            circuit.settings().run_args.decomp_base,
            circuit.settings().run_args.decomp_legs,
        ),
    )?;
    let native_hash = witness
        .processed_params
        .as_ref()
        .and_then(|x| x.poseidon_hash.as_ref())
        .and_then(|x| (x.len() == 1).then(|| native_hash_string(x[0])))
        .ok_or_else(|| {
            "native public Poseidon H(W) was not produced by GraphCircuit::forward".to_string()
        })?;
    if native_hash != statement.public_ezkl.model_hash {
        return Err(
            "external statement H(W) disagrees with native GraphCircuit parameter module".into(),
        );
    }
    let native_output = witness
        .outputs
        .first()
        .ok_or_else(|| "native GraphCircuit forward produced no output".to_string())?
        .iter()
        .map(|x| crate::fieldutils::felt_to_integer_rep(*x))
        .collect::<Vec<_>>();
    if native_output != statement.public_ezkl.y {
        return Err("external public Y disagrees with native GraphCircuit output".into());
    }
    circuit.load_graph_witness(&witness)?;
    let native_instances = vec![circuit.prepare_public_inputs(&witness)?];
    let expected_instances = public_instances_from_statement(&mut circuit, &statement)?;
    if native_instances != expected_instances {
        return Err(
            "external statement public boundary disagrees with GraphCircuit::prepare_public_inputs"
                .into(),
        );
    }
    witness.save(output.clone())?;
    Ok(output.display().to_string())
}

pub(crate) fn setup(
    compiled_circuit: PathBuf,
    srs_path: Option<PathBuf>,
    vk_path: PathBuf,
    pk_path: PathBuf,
    witness: Option<PathBuf>,
    disable_selector_compression: bool,
) -> Result<String, EZKLError> {
    // these aren't real values so the sanity checks are mostly meaningless

    let mut circuit = GraphCircuit::load(compiled_circuit)?;

    if let Some(witness) = witness {
        let data = GraphWitness::from_path(witness)?;
        circuit.load_graph_witness(&data)?;
    }

    let logrows = circuit.settings().run_args.logrows;

    let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?;
    let spill_parent = pk_path
        .parent()
        .ok_or_else(|| "proving key path has no parent directory for native scratch".to_string())?;
    let _native_spill = halo2_proofs::poly::native_spill_scope(spill_parent)?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
        &circuit,
        &params,
        disable_selector_compression,
    )?;
    drop(params);
    drop(circuit);

    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
    drop(pk);
    drop(_native_spill);
    Ok(String::new())
}

fn native_cpu_proof_evidence() -> Result<GpuEvidence, EZKLError> {
    Ok(GpuEvidence {
        backend: "CPU".into(),
        gpu_backend: "none".into(),
        backend_loaded: true,
        cpu_fallback: false,
        device_synchronized: true,
        gpu_kernel_invocation_count: 0,
        provenance: None,
        invocation_trace_sha256: crate::gat_adapter::schema::sha256_digest(b"ezkl-native-cpu")
            .map_err(EZKLError::from)?,
    })
}

fn write_native_cpu_proof_evidence(directory: &Path) -> Result<(), EZKLError> {
    let evidence = native_cpu_proof_evidence()?;
    std::fs::write(
        directory.join("gpu_evidence.json"),
        serde_json::to_vec_pretty(&evidence)?,
    )?;
    Ok(())
}

fn validate_native_cpu_proof_evidence(directory: &Path) -> Result<(), EZKLError> {
    let evidence_path = directory.join("gpu_evidence.json");
    if evidence_path.is_symlink() {
        return Err("CPU prover evidence must not be a symlink".into());
    }
    let evidence: GpuEvidence = serde_json::from_slice(&std::fs::read(&evidence_path)?)?;
    if evidence != native_cpu_proof_evidence()? {
        return Err("CPU prover evidence does not match the completed proof stage".into());
    }
    Ok(())
}

fn authenticated_native_prove_backing_budget(
    output_dir: &Path,
    compiled_circuit_path: &Path,
    pk_path: &Path,
) -> Result<u64, EZKLError> {
    let regular = |path: &Path, label: &str| -> Result<(), EZKLError> {
        require_formal_regular_file(path, label)
    };
    let setup_receipt_path = output_dir.join("setup_receipt.json");
    let admission_path = output_dir.join("admission.json");
    let keygen_budget_path = output_dir.join("native_keygen_budget.json");
    let keygen_receipt_path = output_dir.join("native_keygen_receipt.json");
    let prove_budget_path = output_dir.join("native_prove_budget.json");
    let settings_path = output_dir.join("settings.json");
    let vk_path = output_dir.join("vk.key");
    for (path, label) in [
        (setup_receipt_path.as_path(), "formal setup receipt"),
        (admission_path.as_path(), "formal native admission"),
        (keygen_budget_path.as_path(), "native keygen budget receipt"),
        (keygen_receipt_path.as_path(), "native keygen census receipt"),
        (prove_budget_path.as_path(), "native prove backing budget receipt"),
        (settings_path.as_path(), "formal settings"),
        (compiled_circuit_path, "formal compiled circuit"),
        (vk_path.as_path(), "formal verifying key"),
        (pk_path, "formal proving key"),
    ] {
        regular(path, label)?;
    }
    let prove_budget: JsonValue = serde_json::from_slice(&std::fs::read(&prove_budget_path)?)?;
    let setup_receipt: JsonValue = serde_json::from_slice(&std::fs::read(&setup_receipt_path)?)?;
    let admission: JsonValue = serde_json::from_slice(&std::fs::read(&admission_path)?)?;
    let keygen_budget: JsonValue = serde_json::from_slice(&std::fs::read(&keygen_budget_path)?)?;
    let keygen_receipt: JsonValue = serde_json::from_slice(&std::fs::read(&keygen_receipt_path)?)?;
    let setup_sha256 = get_file_hash(&setup_receipt_path)?;
    let keygen_budget_sha256 = get_file_hash(&keygen_budget_path)?;
    let keygen_receipt_sha256 = get_file_hash(&keygen_receipt_path)?;
    let current_exe_sha256 = get_file_hash(&std::env::current_exe()?)?;
    let current_compiled_sha256 = get_file_hash(compiled_circuit_path)?;
    let current_settings_sha256 = get_file_hash(&settings_path)?;
    let current_vk_sha256 = get_file_hash(&vk_path)?;
    let current_pk_sha256 = get_file_hash(pk_path)?;
    let session_id = std::env::var("EZKL_FORMAL_SESSION_ID")
        .map_err(|_| "formal native proving lacks the runner-authenticated session id")?;
    let build_receipt_sha256 = std::env::var("EZKL_FORMAL_BUILD_RECEIPT_SHA256")
        .map_err(|_| "formal native proving lacks the authenticated build receipt digest")?;
    let build_environment_sha256 = std::env::var("EZKL_FORMAL_BUILD_ENVIRONMENT_SHA256")
        .map_err(|_| "formal native proving lacks the authenticated build environment digest")?;
    let expected_budget = [
        ("schema", "ezkl.native-prove-budget.v2"),
        ("session_id", session_id.as_str()),
        ("setup_receipt_sha256", setup_sha256.as_str()),
        ("native_keygen_budget_sha256", keygen_budget_sha256.as_str()),
        ("native_keygen_receipt_sha256", keygen_receipt_sha256.as_str()),
        ("binary_sha256", current_exe_sha256.as_str()),
        ("compiled_circuit_sha256", current_compiled_sha256.as_str()),
        ("settings_sha256", current_settings_sha256.as_str()),
        ("vk_sha256", current_vk_sha256.as_str()),
        ("pk_sha256", current_pk_sha256.as_str()),
        ("build_receipt_sha256", build_receipt_sha256.as_str()),
        ("build_environment_sha256", build_environment_sha256.as_str()),
    ];
    for (name, value) in expected_budget {
        if prove_budget.get(name).and_then(JsonValue::as_str) != Some(value) {
            return Err(format!("native prove budget {} does not match this setup or binary", name).into());
        }
    }
    let stage_caps = keygen_receipt
        .get("stage_budget_caps")
        .and_then(JsonValue::as_array)
        .ok_or("native keygen census receipt lacks stage-specific allocation caps")?;
    if stage_caps.len() != 2 {
        return Err("native keygen census receipt must contain VK and PK allocation caps".into());
    }
    let native_census = keygen_receipt
        .get("native_census")
        .ok_or("native keygen receipt lacks its exact native census")?;
    if keygen_budget.get("admission_sha256") != admission.get("sha256") {
        return Err("native keygen budget does not bind the current admission receipt".into());
    }
    let memory_model = admission
        .get("native_memory_model")
        .ok_or("formal admission lacks its native source memory model")?;
    let admission_domain_size = admission
        .get("domain_size")
        .and_then(JsonValue::as_u64)
        .ok_or("formal admission lacks a valid proof domain size")?;
    let grand_product_sets_per_instance = memory_model
        .get("shplonk_permutation_product_set_count")
        .and_then(JsonValue::as_u64)
        .ok_or("formal admission lacks the source-derived permutation product set count")?;
    let grand_product_instance_count = memory_model
        .get("shplonk_proof_instance_count")
        .and_then(JsonValue::as_u64)
        .ok_or("formal admission lacks the source-derived proof instance count")?;
    let grand_product_set_count = grand_product_sets_per_instance
        .checked_mul(grand_product_instance_count)
        .ok_or("native permutation grand-product total set count overflow")?;
    let grand_product_bank_upper = memory_model
        .get("permutation_grand_product_bank_mapped_bytes_upper")
        .and_then(JsonValue::as_u64)
        .ok_or("formal admission lacks its mapped permutation grand-product bank bound")?;
    let fr_bytes = memory_model
        .get("size_of_fr_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("formal admission lacks its scalar field width")?;
    let grand_product_window = admission_domain_size
        .checked_mul(fr_bytes)
        .ok_or("native permutation grand-product window calculation overflow")?;
    let exact_grand_product_bank = grand_product_set_count
        .checked_mul(grand_product_window)
        .ok_or("native permutation grand-product bank calculation overflow")?;
    let exact_grand_product_commit_workspace = grand_product_window
        .checked_mul(4)
        .ok_or("native permutation grand-product commit workspace calculation overflow")?;
    let budget_grand_product_set_count = prove_budget
        .get("permutation_grand_product_set_count")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks the permutation grand-product set count")?;
    let budget_grand_product_bank = prove_budget
        .get("permutation_grand_product_bank_mapped_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks the mapped permutation grand-product bank")?;
    let budget_grand_product_window = prove_budget
        .get("permutation_grand_product_window_mapped_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks the active permutation grand-product window")?;
    let budget_grand_product_workspace = prove_budget
        .get("permutation_grand_product_commit_workspace_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks the permutation commit working set")?;
    if budget_grand_product_set_count != grand_product_set_count
        || grand_product_bank_upper != exact_grand_product_bank
        || budget_grand_product_bank != exact_grand_product_bank
        || budget_grand_product_window != grand_product_window
        || budget_grand_product_workspace != exact_grand_product_commit_workspace
    {
        return Err("native prove budget permutation grand-product storage disagrees with the admitted Halo2 chunk geometry".into());
    }
    let preflight = native_census
        .get("preflight")
        .ok_or("native keygen census lacks its native preflight geometry")?;
    let selector_initial = preflight
        .get("selector_activation_initial_backing_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native keygen preflight lacks selector activation backing geometry")?;
    let source_backing_upper = preflight
        .get("permutation_data_virtual_bytes")
        .and_then(JsonValue::as_u64)
        .and_then(|bytes| bytes.checked_add(preflight.get("permutation_directory_virtual_bytes")?.as_u64()?))
        .and_then(|bytes| bytes.checked_add(keygen_budget.get("permutation_successor_backing_file_upper_bytes")?.as_u64()?))
        .ok_or("native keygen preflight has an invalid permutation backing ceiling")?;
    let host_snapshot = keygen_budget
        .get("host_available_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native keygen budget lacks its authenticated host snapshot")?;
    let disk_snapshot = keygen_budget
        .get("native_permutation_backing_budget_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native keygen budget lacks its authenticated disk backing budget")?;
    let census_selector_peak = native_census
        .get("selector_activation_storage_peak_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native keygen census lacks selector activation peak bytes")?;
    let census_permutation_backing = native_census
        .get("permutation_data_allocated_bytes")
        .and_then(JsonValue::as_u64)
        .and_then(|bytes| bytes.checked_add(native_census.get("permutation_directory_allocated_bytes")?.as_u64()?))
        .and_then(|bytes| {
            let successor_bytes = native_census
                .get("permutation_successor_edges")?
                .as_u64()?
                .checked_mul(16)?
                .checked_add(4095)?
                / 4096
                * 4096;
            bytes.checked_add(successor_bytes)
        })
        .ok_or("native keygen census has invalid permutation backing bytes")?;
    for (index, stage) in ["vk", "pk"].into_iter().enumerate() {
        let cap = &stage_caps[index];
        let selector_cap = cap.get("max_selector_activation_bytes").and_then(JsonValue::as_u64);
        let permutation_cap = cap.get("max_permutation_backing_bytes").and_then(JsonValue::as_u64);
        if cap.get("stage").and_then(JsonValue::as_str) != Some(stage)
            || selector_cap.map(|value| value < selector_initial || value > host_snapshot).unwrap_or(true)
            || permutation_cap.map(|value| value > disk_snapshot.min(source_backing_upper)).unwrap_or(true)
        {
            return Err(format!("native {} keygen allocation cap exceeds its authenticated source/resource envelope", stage.to_uppercase()).into());
        }
    }
    let selector_cap = stage_caps
        .iter()
        .filter_map(|cap| cap.get("max_selector_activation_bytes").and_then(JsonValue::as_u64))
        .min()
        .ok_or("native keygen receipt has no finite selector activation cap")?;
    let permutation_cap = stage_caps
        .iter()
        .filter_map(|cap| cap.get("max_permutation_backing_bytes").and_then(JsonValue::as_u64))
        .min()
        .ok_or("native keygen receipt has no finite permutation backing cap")?;
    if census_selector_peak > selector_cap || census_permutation_backing > permutation_cap {
        return Err("native VK/PK keygen census exceeds the stricter stage allocation cap".into());
    }
    let dataset = prove_budget
        .get("dataset")
        .and_then(JsonValue::as_str)
        .ok_or("native prove budget lacks its dataset identity")?;
    let relation_dataset = keygen_receipt
        .get("relation_identity")
        .and_then(|value| value.get("dataset"))
        .and_then(JsonValue::as_str);
    let keygen_dataset = keygen_receipt.get("dataset").and_then(JsonValue::as_str);
    if keygen_dataset != Some(dataset)
        || relation_dataset != Some(dataset)
        || setup_receipt.get("dataset").and_then(JsonValue::as_str) != Some(dataset)
        || setup_receipt.get("session_id").and_then(JsonValue::as_str) != Some(session_id.as_str())
        || keygen_receipt.get("session_id").and_then(JsonValue::as_str) != Some(session_id.as_str())
        || keygen_receipt.get("budget_receipt_sha256").and_then(JsonValue::as_str)
            != Some(keygen_budget_sha256.as_str())
        || setup_receipt.get("native_keygen_receipt_sha256").and_then(JsonValue::as_str)
            != Some(keygen_receipt_sha256.as_str())
    {
        return Err("native prove budget dataset/session does not match authenticated setup and keygen receipts".into());
    }
    for (name, current) in [
        ("settings.json", current_settings_sha256.as_str()),
        ("compiled.circuit", current_compiled_sha256.as_str()),
        ("vk.key", current_vk_sha256.as_str()),
        ("pk.key", current_pk_sha256.as_str()),
        ("native_keygen_budget.json", keygen_budget_sha256.as_str()),
        ("native_keygen_receipt.json", keygen_receipt_sha256.as_str()),
    ] {
        if setup_receipt
            .get("setup_files")
            .and_then(|files| files.get(name))
            .and_then(JsonValue::as_str)
            != Some(current)
        {
            return Err(format!("formal setup receipt does not bind current {}", name).into());
        }
    }
    for (name, value) in [
        ("schema", Some("ezkl.native-keygen-receipt.v4")),
        ("settings_sha256", Some(current_settings_sha256.as_str())),
        ("compiled_circuit_sha256", Some(current_compiled_sha256.as_str())),
        ("vk_sha256", Some(current_vk_sha256.as_str())),
        ("pk_sha256", Some(current_pk_sha256.as_str())),
        ("budget_receipt_sha256", Some(keygen_budget_sha256.as_str())),
    ] {
        if keygen_receipt.get(name).and_then(JsonValue::as_str) != value {
            return Err(format!("native keygen census receipt has a stale {}", name).into());
        }
    }
    if keygen_receipt.get("native_pk_format_version").and_then(JsonValue::as_u64) != Some(4)
        || keygen_receipt.get("native_pk_sigma_wire_encoding").and_then(JsonValue::as_str)
            != Some("packed_source_target_indices_lsb0_v1")
    {
        return Err("native keygen receipt does not bind the current PK/Sigma encoding".into());
    }
    let admission_sha256 = keygen_budget
        .get("admission_sha256")
        .and_then(JsonValue::as_str)
        .ok_or("native keygen budget lacks its admission identity")?;
    for (name, expected) in [
        ("admission_sha256", Some(admission_sha256)),
        ("resource_manifest_sha256", keygen_budget.get("resource_manifest_sha256").and_then(JsonValue::as_str)),
        ("build_receipt_sha256", Some(build_receipt_sha256.as_str())),
        ("build_environment_sha256", Some(build_environment_sha256.as_str())),
    ] {
        if prove_budget.get(name).and_then(JsonValue::as_str) != expected {
            return Err(format!("native prove budget has a stale {}", name).into());
        }
    }
    let max_backing = prove_budget
        .get("max_total_backing_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks a finite backing limit")?;
    let disk_available = prove_budget
        .get("disk_available_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its disk snapshot")?;
    let required = prove_budget
        .get("disk_required_known_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its known disk reserve")?;
    let read_backing_required = prove_budget
        .get("native_pk_read_backing_required_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its source-derived PK-read backing requirement")?;
    let read_mapped_peak = prove_budget
        .get("native_pk_read_mapped_peak_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its whole-PK mapped materialization peak")?;
    let read_mapped_extra = prove_budget
        .get("native_pk_read_mapped_extra_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its whole-PK mapped materialization increment")?;
    let grand_product_commit_host_peak = prove_budget
        .get("proof_permutation_grand_product_commit_host_peak_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its permutation grand-product commit host peak")?;
    let read_host_scratch = prove_budget
        .get("native_pk_read_host_scratch_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its PK-read host scratch upper bound")?;
    let materialize_host_scratch = prove_budget
        .get("native_pk_materialize_host_scratch_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its PK materialization host scratch upper bound")?;
    let read_successor_cache = prove_budget
        .get("native_pk_read_successor_cache_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its successor-cache resident bound")?;
    let read_target_cache = prove_budget
        .get("native_pk_read_target_bitmap_fixed_cache_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its target-bitmap fixed-cache bound")?;
    let read_target_bitmap_allocations = prove_budget
        .get("native_pk_read_target_bitmap_allocation_bitmap_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its target-bitmap allocation metadata bound")?;
    let read_validation_buffers = prove_budget
        .get("native_pk_read_validation_buffers_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its bounded validation-buffer bound")?;
    let read_stream_buffer = prove_budget
        .get("native_pk_read_stream_buffer_upper_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its bounded PK stream-reader buffer bound")?;
    let host_available = prove_budget
        .get("host_available_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its shared host-memory snapshot")?;
    let host_required = prove_budget
        .get("host_required_known_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its known shared host-memory requirement")?;
    let successor_edges = native_census
        .get("permutation_successor_edges")
        .and_then(JsonValue::as_u64)
        .ok_or("native keygen census lacks its exact successor count for PK-read admission")?;
    let permutation_columns = admission
        .get("num_permutation_columns")
        .and_then(JsonValue::as_u64)
        .ok_or("formal admission lacks its permutation column count")?;
    let permutation_cells = admission_domain_size
        .checked_mul(permutation_columns)
        .ok_or("native permutation cell count overflow")?;
    if successor_edges > permutation_cells {
        return Err("native keygen successor count exceeds admitted permutation cells".into());
    }
    let target_bitmap_virtual_bytes = permutation_cells
        .checked_add(7)
        .map(|bits| bits / 8)
        .ok_or("native PK-read target bitmap virtual size calculation overflow")?;
    let target_bitmap_page_count = target_bitmap_virtual_bytes
        .checked_add(4095)
        .map(|bytes| bytes / 4096)
        .ok_or("native PK-read target bitmap page count overflow")?;
    let target_bitmap_reserved = successor_edges
        .min(target_bitmap_page_count)
        .checked_mul(4096)
        .ok_or("native PK-read target bitmap backing calculation overflow")?;
    let sigma_index_width = if permutation_cells <= 1 {
        0
    } else {
        u64::BITS - (permutation_cells - 1).leading_zeros()
    };
    let sigma_payload_bits = successor_edges
        .checked_mul(2)
        .and_then(|pairs| pairs.checked_mul(u64::from(sigma_index_width)))
        .ok_or("native packed Sigma size calculation overflow")?;
    let sigma_payload_bytes = sigma_payload_bits
        .checked_add(7)
        .map(|bits| bits / 8)
        .ok_or("native packed Sigma byte calculation overflow")?;
    let prove_sigma_width = prove_budget
        .get("native_pk_sigma_index_width_bits")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its packed Sigma index width")?;
    let prove_sigma_bytes = prove_budget
        .get("native_pk_sigma_packed_payload_bytes")
        .and_then(JsonValue::as_u64)
        .ok_or("native prove budget lacks its packed Sigma payload size")?;
    if keygen_budget.get("native_pk_format_version").and_then(JsonValue::as_u64) != Some(4)
        || keygen_budget.get("native_pk_sigma_wire_encoding").and_then(JsonValue::as_str)
            != Some("packed_source_target_indices_lsb0_v1")
        || prove_budget.get("native_pk_format_version").and_then(JsonValue::as_u64) != Some(4)
        || prove_budget.get("native_pk_sigma_wire_encoding").and_then(JsonValue::as_str)
            != Some("packed_source_target_indices_lsb0_v1")
        || prove_sigma_width != u64::from(sigma_index_width)
        || prove_sigma_bytes != sigma_payload_bytes
    {
        return Err("native PK/Sigma version or packed payload bound does not match the authenticated permutation geometry".into());
    }
    let exact_read_backing = target_bitmap_reserved;
    let exact_target_bitmap_allocation_bytes =
        halo2_proofs::poly::native_target_bitmap_allocation_bitmap_bytes(permutation_cells)
            .ok_or("native PK-read target bitmap allocation metadata calculation overflow")?;
    let exact_read_successor_cache = u64::try_from(halo2_proofs::poly::NATIVE_SUCCESSOR_READ_CACHE_BYTES)
        .map_err(|_| "native successor read-cache size exceeds u64")?;
    let exact_read_target_cache = halo2_proofs::poly::NATIVE_TARGET_BITMAP_FIXED_CACHE_BYTES;
    let exact_read_validation_buffers = 4u64 * 4096 * 16;
    let exact_read_stream_buffer = u64::try_from(*EZKL_BUF_CAPACITY)
        .map_err(|_| "native PK stream-reader buffer size exceeds u64")?;
    let exact_read_host_scratch = exact_read_successor_cache
        .checked_add(exact_read_target_cache)
        .and_then(|bytes| bytes.checked_add(exact_target_bitmap_allocation_bytes))
        .and_then(|bytes| bytes.checked_add(exact_read_validation_buffers))
        .and_then(|bytes| bytes.checked_add(exact_read_stream_buffer))
        .ok_or("native PK-read host scratch calculation overflow")?;
    let total_pk_reader_host_scratch = read_host_scratch
        .checked_add(materialize_host_scratch)
        .ok_or("native PK materialization host scratch calculation overflow")?;
    if disk_available == 0
        || host_available == 0
        || host_required > host_available
        || read_backing_required != exact_read_backing
        || read_mapped_peak < read_backing_required
        || read_mapped_extra > read_mapped_peak
        || read_mapped_peak > required
        || read_mapped_extra > required
        || grand_product_bank_upper > required
        || budget_grand_product_window > host_required
        || grand_product_commit_host_peak > host_required
        || max_backing != exact_read_backing
        || read_host_scratch != exact_read_host_scratch
        || total_pk_reader_host_scratch > host_required
        || read_successor_cache != exact_read_successor_cache
        || read_target_cache != exact_read_target_cache
        || read_target_bitmap_allocations != exact_target_bitmap_allocation_bytes
        || read_validation_buffers != exact_read_validation_buffers
        || read_stream_buffer != exact_read_stream_buffer
        || keygen_budget.get("native_successor_read_cache_bytes").and_then(JsonValue::as_u64)
            != Some(exact_read_successor_cache)
        || keygen_budget.get("native_target_bitmap_fixed_cache_bytes").and_then(JsonValue::as_u64)
            != Some(exact_read_target_cache)
        || keygen_budget.get("native_pk_read_buffer_capacity_bytes").and_then(JsonValue::as_u64)
            != Some(exact_read_stream_buffer)
        || host_required < total_pk_reader_host_scratch
        || required > disk_available
    {
        return Err("native prove host/disk limits do not match the source-derived PK reader resource bounds".into());
    }
    Ok(max_backing)
}

fn write_native_pk_read_timing_receipt(
    output_dir: &Path,
    pk_path: &Path,
    timing: halo2_proofs::plonk::NativePkReadTiming,
) -> Result<PathBuf, EZKLError> {
    let path = output_dir.join("native_pk_read_timing.json");
    let temporary = output_dir.join("native_pk_read_timing.json.tmp");
    if std::fs::symlink_metadata(&path).is_ok() || std::fs::symlink_metadata(&temporary).is_ok() {
        return Err("native PK read timing refuses to replace an existing attempt receipt".into());
    }
    let budget_path = output_dir.join("native_prove_budget.json");
    require_formal_regular_file(&budget_path, "authenticated native prove budget")?;
    let budget: JsonValue = serde_json::from_slice(&std::fs::read(&budget_path)?)?;
    let required_string = |name: &str| -> Result<String, EZKLError> {
        budget
            .get(name)
            .and_then(JsonValue::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .ok_or_else(|| format!("native prove budget lacks its {} identity", name).into())
    };
    let session_id = required_string("session_id")?;
    let dataset = required_string("dataset")?;
    let pk_sha256 = required_string("pk_sha256")?;
    let binary_sha256 = required_string("binary_sha256")?;
    let setup_receipt_sha256 = required_string("setup_receipt_sha256")?;
    let keygen_budget_sha256 = required_string("native_keygen_budget_sha256")?;
    let keygen_receipt_sha256 = required_string("native_keygen_receipt_sha256")?;
    let admission_sha256 = required_string("admission_sha256")?;
    let resource_manifest_sha256 = required_string("resource_manifest_sha256")?;
    let build_receipt_sha256 = required_string("build_receipt_sha256")?;
    let build_environment_sha256 = required_string("build_environment_sha256")?;
    let settings_sha256 = required_string("settings_sha256")?;
    let compiled_circuit_sha256 = required_string("compiled_circuit_sha256")?;
    let vk_sha256 = required_string("vk_sha256")?;
    if budget.get("schema").and_then(JsonValue::as_str) != Some("ezkl.native-prove-budget.v2")
        || pk_path.file_name().and_then(|name| name.to_str()) != Some("pk.key")
        || std::env::var("EZKL_FORMAL_SESSION_ID").ok().as_deref() != Some(session_id.as_str())
        || std::env::var("EZKL_GAT_DATASET").ok().as_deref() != Some(dataset.as_str())
    {
        return Err("native PK read timing does not match the authenticated formal attempt".into());
    }
    for (name, value) in [
        ("admission_sha256", admission_sha256.as_str()),
        ("resource_manifest_sha256", resource_manifest_sha256.as_str()),
        ("build_receipt_sha256", build_receipt_sha256.as_str()),
        ("build_environment_sha256", build_environment_sha256.as_str()),
        ("settings_sha256", settings_sha256.as_str()),
        ("compiled_circuit_sha256", compiled_circuit_sha256.as_str()),
        ("vk_sha256", vk_sha256.as_str()),
        ("pk_sha256", pk_sha256.as_str()),
        ("binary_sha256", binary_sha256.as_str()),
        ("setup_receipt_sha256", setup_receipt_sha256.as_str()),
        ("native_keygen_budget_sha256", keygen_budget_sha256.as_str()),
        ("native_keygen_receipt_sha256", keygen_receipt_sha256.as_str()),
    ] {
        if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(format!("native prove budget {} identity is not SHA-256", name).into());
        }
    }
    let read_ms = timing.artifact_read_validate.as_secs_f64() * 1000.0;
    let materialize_ms = timing.pk_materialize.as_secs_f64() * 1000.0;
    let source_prepare_ms = timing.sigma_deferred_source_prepare.as_secs_f64() * 1000.0;
    if !read_ms.is_finite()
        || !materialize_ms.is_finite()
        || !source_prepare_ms.is_finite()
        || read_ms <= 0.0
        || materialize_ms < 0.0
        || source_prepare_ms < 0.0
    {
        return Err("native PK read timing returned a nonfinite or invalid duration".into());
    }
    let value = serde_json::json!({
        "schema": "ezkl.native-pk-read-timing.v3",
        "operation": "authenticated_compact_native_pk_restore",
        "session_id": session_id,
        "dataset": dataset,
        "native_prove_budget_sha256": get_file_hash(&budget_path)?,
        "admission_sha256": admission_sha256,
        "resource_manifest_sha256": resource_manifest_sha256,
        "setup_receipt_sha256": setup_receipt_sha256,
        "native_keygen_budget_sha256": keygen_budget_sha256,
        "native_keygen_receipt_sha256": keygen_receipt_sha256,
        "build_receipt_sha256": build_receipt_sha256,
        "build_environment_sha256": build_environment_sha256,
        "binary_sha256": binary_sha256,
        "settings_sha256": settings_sha256,
        "compiled_circuit_sha256": compiled_circuit_sha256,
        "vk_sha256": vk_sha256,
        "pk_sha256": pk_sha256,
        "measurement_scope": "artifact_read_decode_digest_and_mapping_validation; fixed_value_decode_coefficient_ifft_and_deterministic_boundary_materialization; deferred_sigma_source_descriptor_preparation",
        "artifact_read_validate_ms": read_ms,
        "pk_materialize_ms": materialize_ms,
        "sigma_deferred_source_prepare_wall_ms": source_prepare_ms,
    });
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    use std::io::Write;
    file.write_all(&serde_json::to_vec_pretty(&value)?)?;
    file.sync_all()?;
    std::fs::rename(&temporary, &path)?;
    std::fs::File::open(output_dir)?.sync_all()?;
    Ok(path)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prove(
    data_path: PathBuf,
    compiled_circuit_path: PathBuf,
    pk_path: PathBuf,
    proof_path: Option<PathBuf>,
    srs_path: Option<PathBuf>,
    check_mode: CheckMode,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let formal_srs_sha256 = if let Ok(run_dir) = std::env::var("EZKL_GAT_RUN_DIR") {
        validate_formal_producer_run(Path::new(&run_dir))?;
        let path = srs_path
            .as_deref()
            .ok_or("formal GAT proving requires an explicit authenticated SRS path")?;
        Some(require_authentic_srs(path)?)
    } else {
        None
    };
    let native_prove_backing_budget = if formal_srs_sha256.is_some() {
        let output_dir = pk_path
            .parent()
            .ok_or_else(|| "formal proving key has no output directory".to_string())?;
        Some(authenticated_native_prove_backing_budget(
            output_dir,
            &compiled_circuit_path,
            &pk_path,
        )?)
    } else {
        None
    };
    let data = GraphWitness::from_path(data_path)?;
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    circuit.load_graph_witness(&data)?;
    if formal_srs_sha256.is_some() {
        circuit.ensure_complete_gat_witness()?;
    }

    let pretty_public_inputs = circuit.pretty_public_inputs(&data)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;

    let circuit_settings = circuit.settings().clone();

    let proof_split_commits: Option<ProofSplitCommit> = data.into();

    let logrows = circuit_settings.run_args.logrows;
    // creates and verifies the proof

    let spill_parent = pk_path
        .parent()
        .ok_or_else(|| "proving key path has no parent directory for native scratch".to_string())?;
    let _native_spill = halo2_proofs::poly::native_spill_scope(spill_parent)?;

    let (pk, native_pk_read_timing) = if let Some(max_backing) = native_prove_backing_budget {
        let (pk, timing) = load_pk_with_native_backing_budget_and_timing::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
            pk_path.clone(),
            circuit.params(),
            max_backing,
        )?;
        (pk, Some(timing))
    } else {
        (load_pk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(pk_path.clone(), circuit.params())?, None)
    };
    let native_pk_materialize_ms = native_pk_read_timing
        .map(|timing| timing.pk_materialize.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    let native_pk_read_timing_path = if let Some(timing) = native_pk_read_timing {
        let output_dir = pk_path
            .parent()
            .ok_or_else(|| "formal proving key has no output directory for its timing receipt".to_string())?;
        Some(write_native_pk_read_timing_receipt(output_dir, &pk_path, timing)?)
    } else {
        None
    };
    if native_pk_read_timing_path.is_some() && proof_path.is_none() {
        return Err("formal compact PK restore timing requires a proof artifact destination".into());
    }

    let params = match (formal_srs_sha256.as_deref(), srs_path.as_deref()) {
        (Some(digest), Some(path)) => load_formal_kzg_params(path, logrows, digest)?,
        (None, _) => load_params_prover::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?,
        _ => return Err("formal GAT proving lost its authenticated SRS path".into()),
    };
    let mut snark = create_proof_circuit::<
        KZGCommitmentScheme<Bn256>,
        _,
        ProverSHPLONK<_>,
        VerifierSHPLONK<_>,
        KZGSingleStrategy<_>,
        _,
        EvmTranscript<_, _, _, _>,
        EvmTranscript<_, _, _, _>,
    >(
        circuit,
        vec![public_inputs],
        &params,
        &pk,
        check_mode,
        proof_split_commits,
        None,
    )?;
    drop(pk);
    drop(params);
    drop(_native_spill);

    snark.pretty_public_inputs = pretty_public_inputs;

    if let Some(proof_path) = proof_path {
        snark.save(&proof_path)?;
        if let Some(timing) = snark.proof_timing.as_ref() {
            let proof_sha256 = get_file_hash(&proof_path)?;
            let prove_time_ms = timing.prove_ms + native_pk_materialize_ms;
            if !prove_time_ms.is_finite() || prove_time_ms <= 0.0 {
                return Err("native composed prove timing is nonfinite or nonpositive".into());
            }
            let metrics = serde_json::json!({
                "schema_version": 3,
                "operation": "native_halo2_initial_instance_and_phase0_advice_commitments",
                "included_in": "native_create_proof",
                "measurement_scope": "pre_challenge_public_instance_and_phase0_advice_kzg_commitments_plus_affine_normalization",
                "prove_included_in": "native_create_proof_plus_authenticated_pk_materialization",
                "prove_measurement_scope": "native_create_proof_wall_plus_authenticated_fixed_value_decode_coefficient_ifft_and_boundary_rebuild",
                "commitment_time_ms": timing.commitment_ms,
                "instance_commitment_count": timing.instance_commitment_count,
                "phase0_advice_commitment_count": timing.phase0_advice_commitment_count,
                "affine_normalization_batch_count": timing.affine_normalization_batch_count,
                "native_create_proof_time_ms": timing.prove_ms,
                "pk_materialize_time_ms": native_pk_materialize_ms,
                "prove_time_ms": prove_time_ms,
                "proof_sha256": proof_sha256,
            });
            std::fs::write(
                proof_path.with_file_name("native_proof_metrics.json"),
                serde_json::to_vec_pretty(&metrics)?,
            )?;
        }
        let backend = std::env::var("EZKL_GAT_BACKEND").unwrap_or_else(|_| "cpu".into());
        if backend != "cpu" && backend != "gpu" {
            return Err(format!("unsupported formal backend: {}", backend).into());
        }
        if backend == "gpu" {
            let directory = std::env::var("EZKL_GAT_RUN_DIR")
                .map(PathBuf::from)
                .map_err(|_| "GPU formal proof requires EZKL_GAT_RUN_DIR")?;
            #[cfg(feature = "gpu-accelerated")]
            crate::gat_adapter::gpu::write_proof_record(&directory, &proof_path)?;
            #[cfg(not(feature = "gpu-accelerated"))]
            return Err("GPU formal proof requires the matching GPU-capable native binary".into());
        } else if formal_srs_sha256.is_some() {
            let directory = std::env::var("EZKL_GAT_RUN_DIR")
                .map(PathBuf::from)
                .map_err(|_| "formal CPU proof requires EZKL_GAT_RUN_DIR")?;
            write_native_cpu_proof_evidence(&directory)?;
        }
        if let Ok(run_dir) = std::env::var("EZKL_GAT_RUN_DIR") {
            let artifact_path = PathBuf::from(run_dir).join("artifact_manifest.json");
            let mut artifact = ArtifactManifest::load(&artifact_path).map_err(EZKLError::from)?;
            artifact.files.retain(|file| file.path != "proof.json");
            artifact.files.push(
                crate::gat_adapter::artifact_file(&proof_path, "proof.json".into())
                    .map_err(EZKLError::from)?,
            );
            if let Some(timing_path) = native_pk_read_timing_path.as_deref() {
                if artifact.files.iter().any(|file| file.path == "native_pk_read_timing.json") {
                    return Err("artifact manifest already contains a native PK timing from another attempt".into());
                }
                artifact.files.push(
                    crate::gat_adapter::artifact_file(timing_path, "native_pk_read_timing.json".into())
                        .map_err(EZKLError::from)?,
                );
            }
            artifact.save(&artifact_path).map_err(EZKLError::from)?;
        }
    }

    Ok(snark)
}

fn require_formal_local_path(path: &Path, label: &str) -> Result<(), EZKLError> {
    use std::path::Component;
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!(
            "{} must be baseline-local and must not contain '..': {}",
            label,
            path.display()
        )
        .into());
    }
    Ok(())
}

fn require_formal_regular_file(path: &Path, label: &str) -> Result<(), EZKLError> {
    require_formal_local_path(path, label)?;
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|e| format!("cannot stat {} {}: {}", label, path.display(), e))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "{} must be a regular non-symlink file: {}",
            label,
            path.display()
        )
        .into());
    }
    Ok(())
}

fn ensure_no_symlink_path(path: &Path, root: &Path) -> Result<(), EZKLError> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| format!("path escapes artifact root: {}", path.display()))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        if let std::path::Component::Normal(part) = component {
            current.push(part);
            if std::fs::symlink_metadata(&current)?
                .file_type()
                .is_symlink()
            {
                return Err(format!("artifact path contains a symlink: {}", path.display()).into());
            }
        }
    }
    Ok(())
}

fn parse_public_output(path: &Path, dataset: &str) -> Result<Vec<i128>, EZKLError> {
    let bytes = std::fs::read(path)?;
    let value: JsonValue = serde_json::from_slice(&bytes)?;
    let JsonValue::Object(object) = value else {
        return Err("external public output must use the canonical {dataset,Y} object".into());
    };
    let actual = object
        .keys()
        .map(String::as_str)
        .collect::<std::collections::BTreeSet<_>>();
    let expected = ["dataset", "Y"]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    if actual != expected {
        return Err("external public output must contain exactly dataset and Y".into());
    }
    match object.get("dataset") {
        Some(JsonValue::String(found)) if found == dataset => {}
        _ => return Err("external public output dataset disagrees with statement".into()),
    }
    let values = object
        .get("Y")
        .cloned()
        .ok_or_else(|| "external public output must contain Y".to_string())?;
    let JsonValue::Array(values) = values else {
        return Err("external public output Y must be a JSON array".into());
    };
    values
        .into_iter()
        .map(|value| match value {
            JsonValue::Number(number) => number
                .as_i64()
                .map(|x| x as i128)
                .or_else(|| number.as_u64().map(|x| x as i128))
                .ok_or_else(|| "external public output contains a non-integer number".into()),
            JsonValue::String(text) => text.parse::<i128>().map_err(|e| {
                format!("external public output contains invalid integer: {}", e).into()
            }),
            _ => Err("external public output contains a non-integer value".into()),
        })
        .collect()
}

fn validate_artifact_files(
    artifact_path: &Path,
    artifact: &ArtifactManifest,
    statement_path: &Path,
    srs_path: &Path,
    settings_path: &Path,
    compiled_path: &Path,
    vk_path: &Path,
    proof_path: &Path,
) -> Result<(), EZKLError> {
    artifact
        .validate_for_verification()
        .map_err(EZKLError::from)?;
    let root = artifact_path
        .parent()
        .ok_or_else(|| "artifact manifest has no parent directory".to_string())?;
    for file in &artifact.files {
        let relative = Path::new(&file.path);
        require_formal_local_path(relative, "artifact file path")?;
        let path = root.join(relative);
        ensure_no_symlink_path(&path, root)?;
        let digest = get_file_hash(&path)?;
        if digest != file.sha256 {
            return Err(format!("artifact file digest mismatch: {}", file.path).into());
        }
        let expected_top_level = match file.path.as_str() {
            "settings.json" => Some(&artifact.settings_sha256),
            "compiled.circuit" => Some(&artifact.compiled_circuit_sha256),
            "pk.key" => Some(&artifact.pk_sha256),
            "vk.key" => Some(&artifact.vk_sha256),
            "statement.json" => Some(&artifact.statement_sha256),
            "producer_manifest.json" => Some(&artifact.producer_manifest_sha256),
            _ => None,
        };
        if let Some(expected) = expected_top_level {
            if &digest != expected {
                return Err(format!("artifact top-level digest mismatch: {}", file.path).into());
            }
        }
    }
    let checks = [
        (statement_path, &artifact.statement_sha256, "statement"),
        (srs_path, &artifact.srs_sha256, "SRS"),
        (settings_path, &artifact.settings_sha256, "settings"),
        (
            compiled_path,
            &artifact.compiled_circuit_sha256,
            "compiled circuit",
        ),
        (vk_path, &artifact.vk_sha256, "verification key"),
    ];
    for (path, expected, label) in checks {
        let actual = get_file_hash(path)?;
        if &actual != expected {
            return Err(format!("artifact {} digest mismatch", label).into());
        }
    }
    let static_tables_path = root.join("gat_static_tables.json");
    let integer_ledger_path = root.join("gat_integer_ledger.json");
    let producer_context_path = root.join("producer_context.json");
    let producer_manifest_path = root.join("producer_manifest.json");
    let required = [
        ("proof.json", proof_path, "proof"),
        ("srs.kzg", srs_path, "SRS"),
        (
            "gat_static_tables.json",
            static_tables_path.as_path(),
            "static tables",
        ),
        (
            "gat_integer_ledger.json",
            integer_ledger_path.as_path(),
            "integer ledger",
        ),
        (
            "producer_context.json",
            producer_context_path.as_path(),
            "producer context",
        ),
        (
            "producer_manifest.json",
            producer_manifest_path.as_path(),
            "producer manifest",
        ),
    ];
    for (name, path, label) in required {
        let record = artifact
            .files
            .iter()
            .find(|file| file.path == name)
            .ok_or_else(|| format!("artifact manifest is missing required {} file", label))?;
        let actual = get_file_hash(path)?;
        if actual != record.sha256 {
            return Err(format!("artifact {} digest mismatch", label).into());
        }
    }
    Ok(())
}

pub(crate) fn swap_proof_commitments_cmd(
    proof_path: PathBuf,
    witness: PathBuf,
) -> Result<Snark<Fr, G1Affine>, EZKLError> {
    let snark = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
    let witness = GraphWitness::from_path(witness)?;
    let commitments = witness.get_polycommitments();

    let snark_new = swap_proof_commitments::<
        KZGCommitmentScheme<Bn256>,
        _,
        EvmTranscript<G1Affine, _, _, _>,
    >(&snark, &commitments)?;

    if snark_new.proof != *snark.proof {
        log::warn!("swap proof has created a different proof");
    }

    snark_new.save(&proof_path)?;
    Ok(snark_new)
}

#[allow(dead_code)]
pub(crate) fn verify(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    reduced_srs: bool,
) -> Result<bool, EZKLError> {
    verify_with_formal_inputs(
        proof_path,
        settings_path,
        vk_path,
        srs_path,
        reduced_srs,
        None,
        None,
        None,
        None,
    )
}

pub(crate) fn verify_with_formal_inputs(
    proof_path: PathBuf,
    settings_path: PathBuf,
    vk_path: PathBuf,
    srs_path: Option<PathBuf>,
    reduced_srs: bool,
    statement_path: Option<PathBuf>,
    public_output_path: Option<PathBuf>,
    artifact_manifest_path: Option<PathBuf>,
    resource_manifest_path: Option<PathBuf>,
) -> Result<bool, EZKLError> {
    let formal_inputs = statement_path.is_some()
        || public_output_path.is_some()
        || artifact_manifest_path.is_some()
        || resource_manifest_path.is_some();
    let mut formal_context: Option<(
        PublicStatement,
        ArtifactManifest,
        Vec<Vec<String>>,
        String,
        String,
        PathBuf,
    )> = None;
    let mut formal_srs_sha256: Option<String> = None;
    if formal_inputs {
        require_formal_regular_file(&proof_path, "proof")?;
        require_formal_regular_file(&settings_path, "settings")?;
        require_formal_regular_file(&vk_path, "verification key")?;
        // Check the Prove-stage evidence as soon as the proof path and run
        // identity are available. This catches a missing/mismatched GPU
        // record (or an accidentally GPU-enabled verifier environment) before
        // authenticating/loading the large SRS or running cryptographic Verify.
        let backend = std::env::var("EZKL_GAT_BACKEND").unwrap_or_else(|_| "cpu".into());
        if backend != "cpu" && backend != "gpu" {
            return Err(format!("unsupported formal backend: {}", backend).into());
        }
        let run_dir = std::env::var("EZKL_GAT_RUN_DIR")
            .map(PathBuf::from)
            .map_err(|_| "formal verify requires EZKL_GAT_RUN_DIR for native attestations")?;
        if backend == "gpu" {
            #[cfg(feature = "gpu-accelerated")]
            { crate::gat_adapter::gpu::read_proof_evidence(&run_dir, &proof_path)?; }
            #[cfg(not(feature = "gpu-accelerated"))]
            { return Err("GPU formal verifier requires the matching GPU-capable native binary".into()); }
        } else {
            validate_native_cpu_proof_evidence(&run_dir)?;
        }
        let authenticated_srs_path = srs_path
            .as_deref()
            .ok_or("formal verify requires an explicit authenticated SRS path")?;
        formal_srs_sha256 = Some(require_authentic_srs(authenticated_srs_path)?);
        let statement_path = statement_path.ok_or_else(|| {
            "formal verify requires --statement, --public-output and --artifact-manifest"
                .to_string()
        })?;
        let public_output_path = public_output_path.ok_or_else(|| {
            "formal verify requires --statement, --public-output and --artifact-manifest"
                .to_string()
        })?;
        let artifact_manifest_path = artifact_manifest_path.ok_or_else(|| {
            "formal verify requires --statement, --public-output and --artifact-manifest"
                .to_string()
        })?;
        let resource_manifest_path = resource_manifest_path
            .ok_or_else(|| "formal verify requires --resource-manifest".to_string())?;
        require_formal_regular_file(&statement_path, "statement")?;
        require_formal_regular_file(&public_output_path, "public output")?;
        require_formal_regular_file(&artifact_manifest_path, "artifact manifest")?;
        require_formal_regular_file(&resource_manifest_path, "resource manifest")?;
        if reduced_srs {
            return Err("formal GAT verification forbids reduced SRS".into());
        }
        let statement = PublicStatement::load(&statement_path).map_err(EZKLError::from)?;
        let artifact = ArtifactManifest::load(&artifact_manifest_path).map_err(EZKLError::from)?;
        artifact
            .validate_for_verification()
            .map_err(EZKLError::from)?;
        let baseline_root = std::env::current_dir()?;
        let authenticated_resources = load_relation_resources_with_manifest(
            &baseline_root,
            &resource_manifest_path,
            &statement.dataset,
        )
        .map_err(EZKLError::from)?;
        if artifact.resource_manifest_sha256 != authenticated_resources.resource_manifest_sha256 {
            return Err(
                "artifact resource_manifest_sha256 does not match the authenticated current resource manifest"
                    .into(),
            );
        }
        if artifact.static_tables_id != authenticated_resources.tables.static_tables_id
            || statement.static_table_id != authenticated_resources.tables.static_tables_id
            || statement.public_ezkl.static_tables.id
                != authenticated_resources.tables.static_tables_id
            || statement.public_ezkl.static_tables.digest
                != authenticated_resources.tables.static_tables_digest
            || statement.quant_cfg_id != authenticated_resources.quantization.id
            || statement.public_ezkl.quant != authenticated_resources.quantization
            || statement.degree_bound_id != authenticated_resources.ledger.ledger_id
            || statement.public_ezkl.ledger.id != authenticated_resources.ledger.ledger_id
            || statement.public_ezkl.ledger.digest != authenticated_resources.ledger.ledger_digest
        {
            return Err(
                "formal statement/artifact table or integer-ledger identity disagrees with the authenticated resource manifest"
                    .into(),
            );
        }
        if statement.public_ezkl.circuit != artifact.circuit_ref
            || statement.dataset != artifact.dataset
            || statement.quant_cfg_id != artifact.quant_cfg_id
            || statement.static_table_id != artifact.static_tables_id
        {
            return Err(
                "external statement and artifact circuit/registry identities disagree".into(),
            );
        }
        let output = parse_public_output(&public_output_path, &statement.dataset)?;
        if output != statement.public_ezkl.y {
            return Err(
                "external public output Y disagrees element-by-element with statement".into(),
            );
        }
        let output_sha = get_file_hash(&public_output_path)?;
        if output_sha != statement.public_output_sha256 {
            return Err("external public output digest disagrees with statement".into());
        }
        let artifact_root = artifact_manifest_path
            .parent()
            .ok_or_else(|| "artifact manifest has no parent directory".to_string())?;
        let producer_context_path = artifact_root.join("producer_context.json");
        let producer_manifest_path = artifact_root.join("producer_manifest.json");
        require_formal_regular_file(&producer_context_path, "producer setup context")?;
        require_formal_regular_file(&producer_manifest_path, "producer manifest")?;
        if artifact.producer_manifest_sha256 != get_file_hash(&producer_manifest_path)? {
            return Err("artifact producer manifest digest does not match producer file".into());
        }
        let producer_record = artifact
            .files
            .iter()
            .find(|file| file.path == "producer_manifest.json")
            .ok_or_else(|| "artifact manifest has no producer manifest file".to_string())?;
        if producer_record.sha256 != artifact.producer_manifest_sha256 {
            return Err("artifact producer manifest file/top-level digest mismatch".into());
        }
        let setup_context = load_setup_circuit_ref(&artifact_root.join(DEFAULT_GAT_SETUP_CONTEXT))?;
        if setup_context != artifact.circuit_ref {
            return Err("producer setup context does not match artifact CircuitRef".into());
        }
        validate_external_producer_manifest_public(
            &producer_manifest_path,
            &producer_context_path,
            &statement_path,
            &public_output_path,
            &statement,
            &artifact.circuit_ref,
            &authenticated_resources,
            &baseline_root,
        )?;
        let srs_for_identity = srs_path
            .as_ref()
            .ok_or_else(|| "formal verify requires an explicit SRS path".to_string())?;
        let compiled_path = artifact_root.join("compiled.circuit");
        validate_artifact_files(
            &artifact_manifest_path,
            &artifact,
            &statement_path,
            srs_for_identity,
            &settings_path,
            &compiled_path,
            &vk_path,
            &proof_path,
        )?;
        let mut compiled_circuit = GraphCircuit::load(compiled_path.clone())?;
        if compiled_circuit.settings().run_args.logrows == 0 {
            return Err("formal verify loaded a circuit with invalid zero logrows".into());
        }
        assert_native_gat_public_identity(&compiled_circuit, &statement)?;
        let proof = Snark::load::<KZGCommitmentScheme<Bn256>>(&proof_path)?;
        let expected_instances =
            public_instances_from_statement(&mut compiled_circuit, &statement)?;
        if proof.instances != expected_instances {
            return Err(
                "native proof public instances disagree with the independently reconstructed GraphCircuit inputs/outputs"
                    .into(),
            );
        }
        let proof_instances = proof
            .instances
            .iter()
            .map(|column| {
                column
                    .iter()
                    .map(crate::pfsys::field_to_string)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let proof_instances_digest =
            crate::gat_adapter::schema::sha256_digest(&serde_json::to_vec(&proof_instances)?)
                .map_err(EZKLError::from)?;
        formal_context = Some((
            statement,
            artifact,
            proof_instances,
            proof_instances_digest,
            output_sha,
            statement_path,
        ));
    }
    let circuit_settings = GraphSettings::load(&settings_path)?;

    let logrows = circuit_settings.run_args.logrows;

    let parameter_k = if reduced_srs {
        // only need G_0 for the verification with shplonk
        1
    } else {
        logrows
    };
    let params: ParamsKZG<Bn256> = if formal_inputs {
        let path = srs_path
            .as_deref()
            .ok_or("formal verify lost its authenticated SRS path")?;
        let digest = formal_srs_sha256
            .as_deref()
            .ok_or("formal verify lost its authenticated SRS digest")?;
        load_formal_kzg_params(path, parameter_k, digest)?
    } else {
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, parameter_k)?
    };

    let (verified, verify_time_ms) = verify_commitment::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<'_, Bn256>,
        _,
        KZGSingleStrategy<_>,
        EvmTranscript<G1Affine, _, _, _>,
        GraphCircuit,
        _,
    >(proof_path.clone(), circuit_settings, vk_path, &params, logrows)?;
    drop(params);
    if let Some((
        statement,
        artifact,
        proof_instances,
        proof_instances_digest,
        output_sha,
        statement_path,
    )) = formal_context
    {
        let run_dir = std::env::var("EZKL_GAT_RUN_DIR")
            .map(PathBuf::from)
            .map_err(|_| {
                "formal verify requires EZKL_GAT_RUN_DIR for native attestations".to_string()
            })?;
        std::fs::create_dir_all(&run_dir)?;
        let backend = std::env::var("EZKL_GAT_BACKEND").unwrap_or_else(|_| "cpu".into());
        let metrics = serde_json::json!({
            "schema_version": 1,
            "operation": "native_halo2_independent_verifier",
            "included_in": "native_verify_proof",
            "verify_time_ms": verify_time_ms,
            "proof_sha256": get_file_hash(&proof_path)?,
        });
        std::fs::write(
            run_dir.join("native_verify_metrics.json"),
            serde_json::to_vec_pretty(&metrics)?,
        )?;
        let core = CoreRunManifest {
            schema_version: 1,
            relation_id: "R_GAT-ezkl".into(),
            dataset: statement.dataset,
            backend: backend.clone(),
            proof_scope: "full_gat".into(),
            privacy_policy: "model_and_trace_hidden".into(),
            public_statement_mode: "external".into(),
            output_mode: "public_logits".into(),
            native_verify_result: verified,
            native_public_instance_binding: true,
            verifier_reads_witness: false,
            proof_contains_private_witness: false,
            public_statement_sha256: get_file_hash(&statement_path)?,
            public_output_sha256: output_sha,
            proof_instances_sha256: proof_instances_digest,
            proof_instance_count: proof_instances.iter().map(Vec::len).sum(),
            circuit_ref: artifact.circuit_ref,
        };
        std::fs::write(
            run_dir.join("core_run_manifest.json"),
            serde_json::to_vec_pretty(&core)?,
        )?;
    }
    Ok(verified)
}

fn verify_commitment<
    'a,
    Scheme: CommitmentScheme,
    V: Verifier<'a, Scheme>,
    E: EncodedChallenge<Scheme::Curve>,
    Strategy: VerificationStrategy<'a, Scheme, V>,
    TR: TranscriptReadBuffer<Cursor<Vec<u8>>, Scheme::Curve, E>,
    C: Circuit<<Scheme as CommitmentScheme>::Scalar, Params = Params>,
    Params,
>(
    proof_path: PathBuf,
    settings: Params,
    vk_path: PathBuf,
    params: &'a Scheme::ParamsVerifier,
    logrows: u32,
) -> Result<(bool, f64), EZKLError>
where
    Scheme::Scalar: FromUniformBytes<64>
        + SerdeObject
        + Serialize
        + DeserializeOwned
        + WithSmallOrderMulGroup<3>,
    Scheme::Curve: SerdeObject + Serialize + DeserializeOwned,
    Scheme::ParamsVerifier: 'a,
{
    let proof_bytes = std::fs::read(&proof_path)
        .map_err(|e| crate::pfsys::PfsysError::LoadProof(format!("{}", e)))?;

    let strategy = Strategy::new(params);
    let vk = load_vk::<Scheme, C>(vk_path, settings)?;
    let now = Instant::now();

    let proof = Snark::<Scheme::Scalar, Scheme::Curve>::decode_json_bytes(&proof_bytes)?;
    let result =
        verify_proof_circuit::<V, _, _, _, TR>(&proof, params, &vk, strategy, 1 << logrows);

    let elapsed = now.elapsed();
    info!(
        "verify took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    info!("verified: {}", result.is_ok());
    result
        .map_err(|e: plonk::Error| e.into())
        .map(|_| (true, elapsed.as_secs_f64() * 1000.0))
}

/// helper function for load_params
pub(crate) fn load_params_verifier<Scheme: CommitmentScheme>(
    srs_path: Option<PathBuf>,
    logrows: u32,
) -> Result<Scheme::ParamsVerifier, EZKLError> {
    let srs_path = get_srs_path(logrows, srs_path);
    let mut params = load_srs_verifier::<Scheme>(srs_path)?;
    if logrows < params.k() {
        info!("downsizing params to {} logrows", logrows);
        params.downsize(logrows);
    }
    Ok(params)
}

/// helper function for load_params
pub(crate) fn load_params_prover<Scheme: CommitmentScheme>(
    srs_path: Option<PathBuf>,
    logrows: u32,
) -> Result<Scheme::ParamsProver, EZKLError> {
    let srs_path = get_srs_path(logrows, srs_path);
    let mut params = load_srs_prover::<Scheme>(srs_path)?;
    if logrows < params.k() {
        info!("downsizing params to {} logrows", logrows);
        params.downsize(logrows);
    }
    Ok(params)
}

#[cfg(all(test, feature = "gpu-accelerated"))]
mod native_geometry_diagnostics {
    use super::*;

    #[test]
    fn tiny_fixture_lookup_memory_bounds_match_native_census() {
        use rand::rngs::OsRng;

        let spec = crate::gat_adapter::geometry::tiny_multigraph_spec_for_tests();
        let mut run_args = RunArgs::default();
        run_args.input_scale = spec.scales.s_input;
        run_args.param_scale = spec.scales.s_w_hidden[0][0];
        run_args.param_visibility = Visibility::Hashed {
            hash_is_public: true,
            outlets: Vec::new(),
        };
        run_args.num_inner_cols = 2;
        run_args.decomp_base = 16_384;
        run_args.decomp_legs = 2;
        run_args.ignore_range_check_inputs_outputs = false;
        run_args.logrows = 15;
        let model = crate::gat_adapter::build_model_for_run_args(
            spec.clone(),
            RoundingMode::NearestTiesAway,
            &run_args,
        )
        .expect("build in-memory tiny canonical GAT model");
        let settings = crate::gat_adapter::gat_settings_for_run_args(&spec, &run_args)
            .expect("derive in-memory tiny canonical GAT settings");
        let k = settings.run_args.logrows;
        let circuit = GraphCircuit::new_gat_witnessless_from_settings(
            model,
            settings,
            run_args.check_mode,
        );
        assert!(k >= 15, "14-bit range table requires at least k=15");
        assert!(!circuit.settings().run_args.ignore_range_check_inputs_outputs);
        assert!(circuit.settings().required_range_checks.contains(&(0, 16_383)));

        // Exercise the actual GraphCircuit -> HybridOp::Output layout through
        // Halo2's ordinary VK and PK synthesis. This must configure the native
        // 14-bit decomposition table and produce the census used by the v4
        // Setup receipt without a separate shape-only floorplan pass.
        let spill = tempfile::Builder::new()
            .prefix("ezkl-output-range-keygen")
            .tempdir()
            .expect("create native keygen spill directory");
        let _spill_scope = halo2_proofs::poly::native_spill_scope(spill.path())
            .expect("configure native keygen spill directory");
        let params = ParamsKZG::<Bn256>::setup(k, OsRng);
        let mut before_stages = Vec::new();
        let mut after_stages = Vec::new();
        let (pk, census) = create_keys_with_native_gate::<
            KZGCommitmentScheme<Bn256>,
            GraphCircuit,
            _,
            _,
        >(
            &circuit,
            &params,
            false,
            |stage, preflight| {
                assert_eq!(preflight.logrows, k);
                before_stages.push(stage.to_owned());
                let selector_cap = preflight
                    .selector_activation_container_bytes
                    .checked_add(preflight.selector_activation_initial_backing_bytes)
                    .and_then(|bytes| bytes.checked_add(preflight.selector_dense_bitmap_ceiling_bytes))
                    .expect("finite selector cap");
                let permutation_cap = preflight
                    .permutation_data_virtual_bytes
                    .checked_add(preflight.permutation_directory_virtual_bytes)
                    .and_then(|bytes| bytes.checked_add(preflight.permutation_allocation_bitmap_bytes))
                    .expect("finite permutation cap");
                Ok(plonk::NativeKeygenBudget {
                    max_selector_activation_bytes: selector_cap,
                    max_permutation_backing_bytes: permutation_cap,
                })
            },
            |stage, observed| {
                after_stages.push(stage.to_owned());
                assert_eq!(observed.preflight.logrows, k);
                Ok(())
            },
        )
        .expect("real tiny GraphCircuit VK/PK keygen with native 14-bit range table");
        drop(pk);
        assert_eq!(before_stages, vec!["vk".to_owned(), "pk".to_owned()]);
        assert_eq!(after_stages, vec!["vk".to_owned(), "pk".to_owned()]);

        let cs = configure_canonical_constraint_system(&circuit);
        let domain_size = 1u64 << k;
        let configured_lookup_shape = native_mv_lookup_shape_with_bounds(
            &cs,
            &vec![domain_size; cs.num_selectors()],
            &vec![domain_size; cs.num_fixed_columns()],
            &vec![false; cs.num_fixed_columns()],
            domain_size,
        )
        .expect("derive configured lookup upper bound");
        let actual_lookup_shape = native_mv_lookup_shape_with_bounds(
            &cs,
            &census.selector_activation_counts,
            &census.fixed_assignment_callback_counts,
            &census.fixed_value_explicit_zero_seen,
            domain_size,
        )
        .expect("derive lookup shape from actual Setup census");
        assert_eq!(actual_lookup_shape.argument_count as usize, cs.lookups().len());
        assert!(actual_lookup_shape.argument_count > 0);
        assert_eq!(
            configured_lookup_shape.argument_table_group_indices,
            actual_lookup_shape.argument_table_group_indices
        );
        assert!(configured_lookup_shape
            .table_cardinality_upper_bounds
            .iter()
            .zip(&actual_lookup_shape.table_cardinality_upper_bounds)
            .all(|(configured, actual)| actual <= configured));

        let receipt_census = native_keygen_census_json(&census, &actual_lookup_shape);
        assert_eq!(receipt_census["preflight"]["logrows"], k);
        assert_eq!(
            receipt_census["selector_activation_counts"],
            serde_json::json!(census.selector_activation_counts)
        );
        assert_eq!(
            receipt_census["fixed_assignment_callback_counts"],
            serde_json::json!(census.fixed_assignment_callback_counts)
        );
        assert_eq!(
            receipt_census["advice_assignment_counts"],
            serde_json::json!(census.advice_assignment_counts)
        );
        assert_eq!(
            receipt_census["lookup_table_cardinality_upper_bounds"],
            serde_json::json!(actual_lookup_shape.table_cardinality_upper_bounds)
        );
    }

    /// Explicitly invoked diagnostic for current-source geometry. This does
    /// not create a formal session, keys, proof, or READY marker.
    #[test]
    #[ignore = "runs the full canonical Cora model layout for source-bound geometry evidence"]
    fn full_canonical_cora_constraint_system_shape() {
        let baseline_root = std::env::current_dir().expect("baseline root");
        let manifest_path = PathBuf::from("resources/resource_manifest.json");
        let srs_path = PathBuf::from("resources/native_srs/kzg26.srs");
        let srs_sha256 = require_authentic_srs(&srs_path).expect("authenticate official full-size SRS");
        let loaded = load_dataset_with_manifest(&baseline_root, &manifest_path, "cora")
            .expect("load full canonical Cora resources");
        let circuit = build_canonical_gat_circuit(&loaded).expect("build canonical Cora circuit");
        let settings = circuit.settings();
        let configure_started = Instant::now();
        println!(
            "EZKL_NATIVE_PHASE START stage=configure_constraint_system dataset=cora k={} domain={}",
            settings.run_args.logrows,
            1u64 << settings.run_args.logrows,
        );
        let cs = configure_canonical_constraint_system(&circuit);
        let configure_elapsed_ms = configure_started.elapsed().as_secs_f64() * 1000.0;
        println!(
            "EZKL_NATIVE_PHASE DONE stage=configure_constraint_system dataset=cora elapsed_ms={:.3} advice={} fixed={} instance={} selectors={} lookups={} shuffles={} permutation_columns={} blinding_factors={}",
            configure_elapsed_ms,
            cs.num_advice_columns(),
            cs.num_fixed_columns(),
            cs.num_instance_columns(),
            cs.num_selectors(),
            cs.lookups().len(),
            cs.shuffles().len(),
            cs.num_permutation_columns(),
            cs.blinding_factors(),
        );

        let k = settings.run_args.logrows;
        let domain_size = 1u64.checked_shl(k).expect("Cora domain size");
        let degree = cs.degree();
        let quotient_extent = domain_size
            .checked_mul(u64::try_from(degree.checked_sub(1).expect("degree >= 2")).unwrap())
            .expect("Cora quotient extent");
        let proof_domain_size = quotient_extent
            .checked_next_power_of_two()
            .expect("Cora proof domain");
        let report = serde_json::json!({
            "schema": "ezkl-native-geometry-diagnostic-v1",
            "diagnostic_only": true,
            "formal_gpu_admission": false,
            "dataset": "cora",
            "resource_manifest_sha256": loaded.resource_manifest_sha256,
            "srs_sha256": srs_sha256,
            "relation_identity": native_admission_relation_identity(&loaded).expect("Cora relation identity"),
            "k": k,
            "domain_size": domain_size,
            "settings": {
                "num_rows": settings.num_rows,
                "total_assignments": settings.total_assignments,
                "total_const_size": settings.total_const_size,
                "total_dynamic_col_size": settings.dynamic_lookup_params.total_dynamic_col_size,
                "max_dynamic_input_len": settings.dynamic_lookup_params.max_dynamic_input_len,
                "num_dynamic_lookups": settings.dynamic_lookup_params.num_dynamic_lookups,
                "num_shuffles": settings.shuffle_params.num_shuffles,
            },
            "constraint_system": {
                "degree": degree,
                "num_fixed_columns": cs.num_fixed_columns(),
                "num_advice_columns": cs.num_advice_columns(),
                "num_instance_columns": cs.num_instance_columns(),
                "num_selectors": cs.num_selectors(),
                "num_lookup_arguments": cs.lookups().len(),
                "num_shuffle_arguments": cs.shuffles().len(),
                "num_permutation_columns": cs.num_permutation_columns(),
                "num_blinding_factors": cs.blinding_factors(),
                "advice_column_phase_ids": cs.advice_column_phase(),
            },
            "proof_domain_size": proof_domain_size,
            "proof_logrows": proof_domain_size.trailing_zeros(),
            "rust_type_bytes": {
                "fr": std::mem::size_of::<Fr>(),
                "assigned_fr": std::mem::size_of::<plonk::Assigned<Fr>>(),
                "option_fr": std::mem::size_of::<Option<Fr>>(),
                "usize": std::mem::size_of::<usize>(),
                "usize_pair": std::mem::size_of::<(usize, usize)>(),
                "g1_affine": std::mem::size_of::<G1Affine>(),
            },
            "configure_elapsed_ms": configure_elapsed_ms,
            "proof_executed": false,
            "keys_generated": false,
        });
        println!("EZKL_NATIVE_CORA_GEOMETRY_JSON={}", report);
    }
}
