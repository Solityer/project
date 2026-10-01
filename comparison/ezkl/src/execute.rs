use crate::circuit::region::RegionSettings;
use crate::circuit::CheckMode;
use crate::commands::CalibrationTarget;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::deploy_contract_via_solidity;
#[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
use crate::eth::register_vka_via_rv;
#[allow(unused_imports)]
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use crate::eth::{get_contract_artifacts, verify_proof_via_solidity};
use crate::gat_adapter::{
    build_model_for_run_args, load_dataset_with_manifest, load_relation_resources_with_manifest,
    write_generated_metadata, ArtifactFile, ArtifactManifest, BoundCircuitRef, CoreRunManifest,
    FormalCliContract, GatOp, GpuEvidence, PublicStatement, RoundingMode, SrsProvenance,
};
use crate::graph::input::GraphData;
use crate::graph::modules::ModuleForwardResult;
use crate::graph::{GraphCircuit, GraphSettings, GraphWitness, Model, SupportedOp, Visibility};
use crate::pfsys::{
    create_keys, load_pk, load_vk, save_params, save_pk, swap_proof_commitments, Snark,
};
use crate::pfsys::{create_proof_circuit, verify_proof_circuit, ProofSplitCommit};
use crate::pfsys::{
    canonical_fr_to_string, canonical_string_to_fr, encode_calldata, save_vk, srs::*,
};
use crate::tensor::TensorError;
use crate::RunArgs;
use crate::EZKL_BUF_CAPACITY;
use crate::{commands::*, EZKLError};
use colored::Colorize;
#[cfg(unix)]
use gag::Gag;
use halo2_proofs::dev::VerifyFailure;
#[cfg(feature = "gpu-accelerated")]
use halo2_proofs::icicle::try_load_and_set_backend_device;
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
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use halo2_solidity_verifier;
use halo2curves::bn256::{Bn256, Fr, G1Affine};
use halo2curves::ff::{FromUniformBytes, WithSmallOrderMulGroup};
use halo2curves::serde::SerdeObject;
#[cfg(feature = "gpu-accelerated")]
use icicle_runtime::{stream::IcicleStream, warmup};
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
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use std::fs::File;
use std::io::BufWriter;
use std::io::Cursor;
#[cfg(all(feature = "eth", not(target_arch = "wasm32")))]
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
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
pub fn set_device() {
    if std::env::var("ICICLE_BACKEND_INSTALL_DIR").is_ok() {
        info!("Running with ICICLE GPU");
        try_load_and_set_backend_device("CUDA");
        match warmup(&IcicleStream::default()) {
            Ok(_) => info!("GPU warmed :)"),
            Err(e) => log::error!("GPU warmup failed: {:?}", e),
        }
    } else {
        info!("Running with CPU: 'ICICLE_BACKEND_INSTALL_DIR' not set");
        try_load_and_set_backend_device("CPU");
    }
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
    set_device();
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
        #[cfg(all(
            feature = "eth",
            feature = "reusable-verifier",
            not(target_arch = "wasm32")
        ))]
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
        #[cfg(feature = "reusable-verifier")]
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
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut reader = std::io::BufReader::new(file);
    let mut buffer = vec![];
    let bytes_read = reader.read_to_end(&mut buffer)?;
    info!(
        "read {} bytes from file (vector of len = {})",
        bytes_read,
        buffer.len()
    );

    let hash = sha256::digest(buffer);
    info!("file hash: {}", hash);

    Ok(hash)
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

#[cfg(all(feature = "reusable-verifier", not(target_arch = "wasm32")))]
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
    {
        let backend = std::env::var("ICICLE_BACKEND_INSTALL_DIR")
            .map_err(|_| "GPU formal run requires ICICLE_BACKEND_INSTALL_DIR".to_string())?;
        let backend_path = Path::new(&backend);
        if !backend_path.is_dir() {
            return Err(format!(
                "GPU formal run requires an ICICLE backend directory: {}",
                backend
            )
            .into());
        }
        let device = std::env::var("CUDA_VISIBLE_DEVICES")
            .or_else(|_| std::env::var("ICICLE_DEVICE_ID"))
            .map_err(|_| {
                "GPU formal run requires CUDA_VISIBLE_DEVICES or ICICLE_DEVICE_ID".to_string()
            })?;
        if device.trim().is_empty() {
            return Err("GPU formal run requires a non-empty CUDA device selection".into());
        }
        let _ = (backend_path, device);
        return Err(
            "GPU formal blocked: this official snapshot exposes no reliable native kernel-invocation counter and device-synchronization attestation hook; refusing to label a run GPU".into(),
        );
    }
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
    let bytes = std::fs::read(path)
        .map_err(|e| format!("cannot read official SRS {}: {}", path.display(), e))?;
    crate::gat_adapter::schema::sha256_digest(&bytes).map_err(EZKLError::from)
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
    let bytes = std::fs::read(source)?;
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
        std::fs::write(destination, &bytes)?;
    }
    let rebound = std::fs::read(destination)?;
    if rebound != bytes {
        return Err(format!("external artifact copy changed: {}", destination.display()).into());
    }
    Ok(())
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
        GatPreparePhase::Setup => gat_prepare_setup(loaded, output_dir, srs_path, gpu),
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

fn gat_prepare_setup(
    loaded: crate::gat_adapter::LoadedGat,
    output_dir: PathBuf,
    srs_path: PathBuf,
    gpu: bool,
) -> Result<String, EZKLError> {
    let _ = gpu;
    std::fs::create_dir_all(&output_dir)?;
    for name in [
        "settings.json",
        "compiled.circuit",
        "pk.key",
        "vk.key",
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
    let required_logrows = GraphSettings::required_logrows_for_model(&model, &run_args)?;
    run_args.logrows = run_args.logrows.max(required_logrows);
    run_args.validate().map_err(EZKLError::from)?;
    if run_args.logrows > crate::graph::MAX_PUBLIC_SRS {
        return Err(format!(
            "canonical GAT setup requires logrows {}, exceeding official SRS maximum {}",
            run_args.logrows,
            crate::graph::MAX_PUBLIC_SRS
        )
        .into());
    }
    let formal_inputs = gat_input_tensors(&loaded.spec)?;
    let formal_params = gat_parameter_tensor(&loaded.spec)?;
    let input_shapes = model.graph.input_shapes()?;
    let output_shapes = model.graph.output_shapes()?;
    let param_shapes = model.graph.param_shapes()?;
    let input_scales = model.graph.get_input_scales();
    let param_scales = model
        .graph
        .param_inputs
        .iter()
        .map(|index| {
            model
                .graph
                .nodes
                .get(index)
                .and_then(|node| node.out_scales().first().copied())
                .ok_or_else(|| EZKLError::from(format!("formal GAT parameter input {} has no scale metadata", index)))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_scales = model.graph.get_output_scales()?;
    GatOp::validate_pre_synthesis(
        &loaded.spec,
        &input_shapes,
        &output_shapes,
        &param_shapes,
        &input_scales,
        &param_scales,
        &output_scales,
        &formal_inputs,
        &formal_params,
        run_args.input_scale,
        run_args.param_scale,
    )
    .map_err(EZKLError::from)?;
    info!("canonical GAT setup logrows: {}", run_args.logrows);
    let circuit = GraphCircuit::new(model, &run_args)?;
    let settings_path = output_dir.join("settings.json");
    let compiled_path = output_dir.join("compiled.circuit");
    let pk_path = output_dir.join("pk.key");
    let vk_path = output_dir.join("vk.key");
    circuit.settings().save(&settings_path)?;
    circuit.save(compiled_path.clone())?;
    let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(
        Some(srs_path.clone()),
        circuit.settings().run_args.logrows,
    )?;
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, GraphCircuit>(&circuit, &params, false)?;
    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
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
    copy_external_file(&srs_path, &bound_srs_path)?;
    let files: Vec<ArtifactFile> = [
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
    let pk = create_keys::<KZGCommitmentScheme<Bn256>, GraphCircuit>(
        &circuit,
        &params,
        disable_selector_compression,
    )?;

    save_vk::<G1Affine>(&vk_path, pk.get_vk())?;
    save_pk::<G1Affine>(&pk_path, &pk)?;
    Ok(String::new())
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
    if let Ok(run_dir) = std::env::var("EZKL_GAT_RUN_DIR") {
        validate_formal_producer_run(Path::new(&run_dir))?;
    }
    let data = GraphWitness::from_path(data_path)?;
    let mut circuit = GraphCircuit::load(compiled_circuit_path)?;

    circuit.load_graph_witness(&data)?;

    let pretty_public_inputs = circuit.pretty_public_inputs(&data)?;
    let public_inputs = circuit.prepare_public_inputs(&data)?;

    let circuit_settings = circuit.settings().clone();

    let proof_split_commits: Option<ProofSplitCommit> = data.into();

    let logrows = circuit_settings.run_args.logrows;
    // creates and verifies the proof

    let pk = load_pk::<KZGCommitmentScheme<Bn256>, GraphCircuit>(pk_path, circuit.params())?;

    let params = load_params_prover::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?;
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

    snark.pretty_public_inputs = pretty_public_inputs;

    if let Some(proof_path) = proof_path {
        snark.save(&proof_path)?;
        if let Ok(run_dir) = std::env::var("EZKL_GAT_RUN_DIR") {
            let artifact_path = PathBuf::from(run_dir).join("artifact_manifest.json");
            let mut artifact = ArtifactManifest::load(&artifact_path).map_err(EZKLError::from)?;
            artifact.files.retain(|file| file.path != "proof.json");
            artifact.files.push(
                crate::gat_adapter::artifact_file(&proof_path, "proof.json".into())
                    .map_err(EZKLError::from)?,
            );
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
    if formal_inputs {
        require_formal_regular_file(&proof_path, "proof")?;
        require_formal_regular_file(&settings_path, "settings")?;
        require_formal_regular_file(&vk_path, "verification key")?;
        if let Some(path) = srs_path.as_ref() {
            require_authentic_srs(path)?;
        }
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

    let params: ParamsKZG<Bn256> = if reduced_srs {
        // only need G_0 for the verification with shplonk
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, 1)?
    } else {
        load_params_verifier::<KZGCommitmentScheme<Bn256>>(srs_path, logrows)?
    };

    let verified = verify_commitment::<
        KZGCommitmentScheme<Bn256>,
        VerifierSHPLONK<'_, Bn256>,
        _,
        KZGSingleStrategy<_>,
        EvmTranscript<G1Affine, _, _, _>,
        GraphCircuit,
        _,
    >(proof_path, circuit_settings, vk_path, &params, logrows)?;
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
        if backend == "gpu" {
            enforce_gat_gpu(true)?;
        }
        if backend != "cpu" && backend != "gpu" {
            return Err(format!("unsupported formal backend: {}", backend).into());
        }
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
        let trace = if backend == "cpu" {
            &b"ezkl-native-cpu"[..]
        } else {
            &b"gpu-blocked"[..]
        };
        let evidence = GpuEvidence {
            backend: if backend == "gpu" {
                "CUDA".into()
            } else {
                "CPU".into()
            },
            gpu_backend: if backend == "gpu" {
                "icicle-cuda".into()
            } else {
                "none".into()
            },
            backend_loaded: backend == "cpu",
            cpu_fallback: false,
            device_synchronized: backend == "cpu",
            gpu_kernel_invocation_count: 0,
            provenance: None,
            invocation_trace_sha256: crate::gat_adapter::schema::sha256_digest(trace)
                .map_err(EZKLError::from)?,
        };
        std::fs::write(
            run_dir.join("gpu_evidence.json"),
            serde_json::to_vec_pretty(&evidence)?,
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
) -> Result<bool, EZKLError>
where
    Scheme::Scalar: FromUniformBytes<64>
        + SerdeObject
        + Serialize
        + DeserializeOwned
        + WithSmallOrderMulGroup<3>,
    Scheme::Curve: SerdeObject + Serialize + DeserializeOwned,
    Scheme::ParamsVerifier: 'a,
{
    let proof = Snark::load::<Scheme>(&proof_path)?;

    let strategy = Strategy::new(params);
    let vk = load_vk::<Scheme, C>(vk_path, settings)?;
    let now = Instant::now();

    let result =
        verify_proof_circuit::<V, _, _, _, TR>(&proof, params, &vk, strategy, 1 << logrows);

    let elapsed = now.elapsed();
    info!(
        "verify took {}.{}",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    info!("verified: {}", result.is_ok());
    result.map_err(|e: plonk::Error| e.into()).map(|_| true)
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
