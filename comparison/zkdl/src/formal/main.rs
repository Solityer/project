mod algebra;
mod compiler;
#[cfg(feature = "gpu")]
mod gpu;
mod hyrax;
mod protocol;
mod prover_backend;
mod registry;
mod sumcheck;
mod transcript;
use anyhow::{ensure, Result};
use clap::{Parser, Subcommand};
use hyrax::CommitmentParameters;
use protocol::*;
use registry::WitnessPackage;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "zkdl-formal",
    version,
    about = "Formal zkDL-style GAT prover and independent verifier"
)]
struct Cli {
    /// Required even when the process is launched from the baseline itself.
    /// The native binary refuses aliases and sibling project roots.
    #[arg(long, global = true, value_name = "BASELINE_ROOT")]
    project_root: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Setup {
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value = "artifacts/setup/setup_manifest.json")]
        manifest: PathBuf,
        #[arg(long)]
        max_vector_len: usize,
        #[arg(long, default_value = "zkDL-GAT-transparent-PP-v1")]
        seed_label: String,
    },
    PpInfo {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long, default_value = "artifacts/setup/pp_info.json")]
        manifest: PathBuf,
    },
    RegisterModel {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        model_registry: PathBuf,
        #[arg(long)]
        model_secret: PathBuf,
        #[arg(long)]
        certified_bounds_id: String,
        #[arg(
            long,
            default_value = "artifacts/model_registry/registration_manifest.json"
        )]
        manifest: PathBuf,
    },
    Prove {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        model_registry: PathBuf,
        #[arg(long)]
        model_secret: PathBuf,
        #[arg(long)]
        witness: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        device: String,
    },
    Verify {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        model_registry: PathBuf,
        #[arg(long)]
        proof: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        device: Option<String>,
    },
    CheckStatement {
        #[arg(long)]
        pp: PathBuf,
        #[arg(long)]
        statement: PathBuf,
    },
    Compile {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        statement: PathBuf,
        #[arg(long)]
        witness: PathBuf,
    },
}
fn sha_file(path: &Path) -> Result<String> {
    let b = read_bytes(path)?;
    let mut h = Sha256::new();
    h.update(b);
    Ok(hex::encode(h.finalize()))
}
fn write_manifest(path: &Path, value: serde_json::Value) -> Result<()> {
    write_bytes_atomic(path, &serde_json::to_vec_pretty(&value)?, false)
}

fn checked_cli_path(path: &Path, root: CliPathRoot, output: bool) -> Result<PathBuf> {
    validate_cli_path(path, root, output)
}

/// Private witness/model-opening files are accepted only below the
/// baseline-local private namespace.  Their parent must be a locked-down
/// per-session/private directory and existing files must be owner-only.
fn checked_private_path(path: &Path, output: bool) -> Result<PathBuf> {
    let checked = checked_cli_path(path, CliPathRoot::Artifacts, output)?;
    let relative = relative_baseline_path(&checked)?;
    ensure!(
        relative.starts_with("artifacts/private/"),
        "private input must stay below artifacts/private/"
    );
    let parent = checked
        .parent()
        .ok_or_else(|| anyhow::anyhow!("private path has no parent"))?;
    let parent_meta = fs::symlink_metadata(parent)?;
    ensure!(
        !parent_meta.file_type().is_symlink() && parent_meta.is_dir(),
        "private parent must be a real directory"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            parent_meta.permissions().mode() & 0o777 == 0o700,
            "private parent directory must have mode 0700"
        );
    }
    if !output {
        let metadata = fs::symlink_metadata(&checked)?;
        ensure!(
            !metadata.file_type().is_symlink() && metadata.is_file(),
            "private input must be a regular file"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            ensure!(
                metadata.nlink() == 1,
                "private input must not be hard-linked"
            );
            ensure!(
                metadata.permissions().mode() & 0o777 == 0o600,
                "private input file must have mode 0600"
            );
        }
    }
    Ok(checked)
}

fn cleanup_private_file(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            ensure!(
                !metadata.file_type().is_symlink() && metadata.is_file(),
                "private cleanup target must be a regular file"
            );
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                ensure!(
                    metadata.nlink() == 1,
                    "private cleanup target must not be hard-linked"
                );
            }
            fs::remove_file(path)?;
            Ok(())
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

fn seal_manifest(mut value: serde_json::Value, schema: &str) -> Result<serde_json::Value> {
    {
        let object = value
            .as_object_mut()
            .ok_or_else(|| anyhow::anyhow!("formal manifest must be a JSON object"))?;
        object.insert(
            "manifest_schema".into(),
            serde_json::Value::String(schema.into()),
        );
    }
    let bytes = serde_json::to_vec(&value)?;
    let mut digest = Sha256::new();
    digest.update(b"zkDL-GAT-cli-manifest-v1\0");
    digest.update(bytes);
    value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("formal manifest must be a JSON object"))?
        .insert(
            "manifest_id".into(),
            serde_json::Value::String(format!("sha256:{}", hex::encode(digest.finalize()))),
        );
    Ok(value)
}

fn pp_manifest_fields(pp_path: &Path, pp: &CommitmentParameters) -> Result<serde_json::Value> {
    pp.validate()?;
    let pp_path = checked_cli_path(pp_path, CliPathRoot::Artifacts, false)?;
    let size = fs::metadata(&pp_path)?.len();
    ensure!(size > 0, "PP file must not be empty");
    let pp_sha256 = format!("sha256:{}", sha_file(&pp_path)?);
    Ok(serde_json::json!({
        // `pp_*` is the canonical runner_contract spelling.
        "pp_path": relative_baseline_path(&pp_path)?,
        "pp_sha256": pp_sha256.clone(),
        "pp_size_bytes": size,
        "pp_schema": pp.schema.clone(),
        "pp_id": pp.pp_id.clone(),
        "pp_max_vector_len": pp.max_vector_len,
        "pp_setup_id": pp.setup_id.clone(),
        "pp_security_id": pp.security_id.clone(),
        "pp_protocol": pp.protocol.clone(),
        "pp_security_bits": pp.security_bits,
        // Native pp-info aliases remain available to setup_resources.py.
        "schema": pp.schema.clone(),
        "setup_id": pp.setup_id.clone(),
        "security_id": pp.security_id.clone(),
        "protocol": pp.protocol.clone(),
        "security_bits": pp.security_bits,
        "max_vector_len": pp.max_vector_len,
        "sha256": pp_sha256,
        "size_bytes": size,
        "digest": pp.digest.clone(),
        "pp_digest": pp.digest.clone(),
    }))
}

fn merge_pp_fields(
    mut value: serde_json::Value,
    pp_path: &Path,
    pp: &CommitmentParameters,
) -> Result<serde_json::Value> {
    let pp_value = pp_manifest_fields(pp_path, pp)?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("formal manifest must be a JSON object"))?;
    let pp_object = pp_value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("PP identity must be a JSON object"))?;
    for (key, item) in pp_object {
        object.insert(key.clone(), item.clone());
    }
    Ok(value)
}

fn canonical_path_string(path: &Path) -> Result<String> {
    relative_baseline_path(path)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    validate_project_root(&cli.project_root)?;
    match cli.command {
        Command::Setup {
            output,
            manifest,
            max_vector_len,
            seed_label,
        } => {
            let output = checked_cli_path(&output, CliPathRoot::Artifacts, true)?;
            let manifest = checked_cli_path(&manifest, CliPathRoot::Artifacts, true)?;
            ensure!(
                output != manifest,
                "setup PP and manifest paths must differ"
            );
            ensure!(
                max_vector_len.is_power_of_two(),
                "max-vector-len must be power of two"
            );
            let mut h = Sha256::new();
            h.update(b"zkDL-GAT-PP-seed-v1\0");
            h.update(seed_label.as_bytes());
            let pp = CommitmentParameters::new(h.finalize().into(), max_vector_len)?;
            write_bin(&output, &pp, false)?;
            let manifest_value = seal_manifest(
                merge_pp_fields(
                    serde_json::json!({
                        "protocol_id": "GAT-ZKML-zkDL",
                        "setup": true,
                        "output_path": canonical_path_string(&output)?,
                    }),
                    &output,
                    &pp,
                )?,
                "zkDL_gat_setup_manifest_v1",
            )?;
            write_manifest(&manifest, manifest_value)?;
            println!(
                "SETUP_OK pp_id={} path={} manifest={}",
                pp.parameters_id,
                output.display(),
                manifest.display()
            );
        }
        Command::PpInfo { pp, manifest } => {
            let pp = checked_cli_path(&pp, CliPathRoot::Artifacts, false)?;
            let manifest = checked_cli_path(&manifest, CliPathRoot::Artifacts, true)?;
            let value: CommitmentParameters = read_bin(&pp)?;
            value.validate()?;
            let info = merge_pp_fields(
                serde_json::json!({"protocol_id":"GAT-ZKML-zkDL"}),
                &pp,
                &value,
            )?;
            let manifest_value = seal_manifest(info.clone(), "zkDL_gat_pp_info_manifest_v1")?;
            write_manifest(&manifest, manifest_value)?;
            println!("{}", serde_json::to_string(&info)?);
        }
        Command::RegisterModel {
            pp,
            statement,
            witness,
            model_registry,
            model_secret,
            certified_bounds_id,
            manifest,
        } => {
            let pp = checked_cli_path(&pp, CliPathRoot::Artifacts, false)?;
            let statement = checked_cli_path(&statement, CliPathRoot::RunsTmp, false)?;
            let witness = checked_private_path(&witness, false)?;
            let model_registry = checked_cli_path(&model_registry, CliPathRoot::Artifacts, true)?;
            let model_secret = checked_private_path(&model_secret, true)?;
            let manifest = checked_cli_path(&manifest, CliPathRoot::Artifacts, true)?;
            ensure!(
                model_registry != model_secret && model_registry != manifest,
                "registration destinations must be distinct"
            );
            let ppv: CommitmentParameters = read_bin(&pp)?;
            let s: PublicStatement = read_bin(&statement)?;
            let w: WitnessPackage = read_bin(&witness)?;
            let (r, secret) = register_model(&ppv, &s, &w, certified_bounds_id)?;
            write_model_registration_atomic(&model_registry, &r, &model_secret, &secret)?;
            let registration_manifest = seal_manifest(
                merge_pp_fields(
                    serde_json::json!({
                        "protocol_id": "GAT-ZKML-zkDL",
                        "registered_model": true,
                        "registry_path": canonical_path_string(&model_registry)?,
                        "registry_object_id": r.registry_object_id.clone(),
                        "statement_id": s.statement_id.clone(),
                        "model_registry_id": r.registry_object_id.clone(),
                        "dataset": s.dataset,
                        "input_preprocess_id": s.input_preprocess_id,
                        "data_identity": s.data_identity,
                        "source_input_id": s.source_input_id,
                        "quant_cfg_id": s.quant_cfg_id,
                        "checkpoint_manifest_id": s.checkpoint_manifest_id,
                        "model_shape_id": s.model_shape_id,
                        "model_arch_id": s.model_arch_id,
                        "model_param_id": s.model_param_id,
                        "graph_features_digest": s.graph_features_digest,
                        "topology_digest": s.topology_digest,
                    }),
                    &pp,
                    &ppv,
                )?,
                "zkDL_gat_registration_manifest_v1",
            )?;
            write_manifest(&manifest, registration_manifest)?;
            println!(
                "REGISTER_MODEL_OK registry_id={} manifest={}",
                r.registry_object_id,
                manifest.display()
            );
        }
        Command::Prove {
            pp,
            statement,
            model_registry,
            model_secret,
            witness,
            proof,
            manifest,
            device,
        } => {
            let pp = checked_cli_path(&pp, CliPathRoot::Artifacts, false)?;
            let statement = checked_cli_path(&statement, CliPathRoot::RunsTmp, false)?;
            let model_registry = checked_cli_path(&model_registry, CliPathRoot::Artifacts, false)?;
            let model_secret = checked_private_path(&model_secret, false)?;
            let witness = checked_private_path(&witness, false)?;
            let proof = checked_cli_path(&proof, CliPathRoot::RunsTmp, true)?;
            let manifest = checked_cli_path(&manifest, CliPathRoot::RunsTmp, true)?;
            ensure!(proof != manifest, "proof and manifest paths must differ");
            let ppv: CommitmentParameters = read_bin(&pp)?;
            let s: PublicStatement = read_bin(&statement)?;
            let m = read_bin(&model_registry)?;
            let prove_result = (|| -> Result<ProofEnvelope> {
                let secret = read_bin(&model_secret)?;
                let w = read_bin(&witness)?;
                prove(&ppv, &s, &m, &secret, &w, &device)
            })();
            let cleanup_result = cleanup_private_file(&witness);
            let p = match (prove_result, cleanup_result) {
                (Ok(value), Ok(())) => value,
                (Err(error), Ok(())) => return Err(error),
                (Ok(_), Err(error)) => return Err(error),
                (Err(error), Err(_cleanup_error)) => return Err(error),
            };
            write_bin(&proof, &p, false)?;
            let size = fs::metadata(&proof)?.len();
            let proof_manifest = merge_pp_fields(
                serde_json::json!({"protocol_id":"GAT-ZKML-zkDL","proof_format":p.format,"proof_generated":true,"verified":false,"statement_id":p.statement_id,"model_registry_id":p.model_registry_id,"dataset":s.dataset,"input_preprocess_id":s.input_preprocess_id,"data_identity":s.data_identity,"source_input_id":s.source_input_id,"quant_cfg_id":s.quant_cfg_id,"checkpoint_manifest_id":s.checkpoint_manifest_id,"model_shape_id":s.model_shape_id,"model_arch_id":s.model_arch_id,"model_param_id":s.model_param_id,"graph_features_digest":s.graph_features_digest,"topology_digest":s.topology_digest,"device":p.device,"gpu_telemetry":p.gpu_telemetry,"proof_path":canonical_path_string(&proof)?,"proof_sha256":sha_file(&proof)?,"proof_size_bytes":size,"commitment_time_ms":p.commitment_time_ms,"prove_time_ms":p.prove_time_ms,"fresh_dynamic_commitments":p.dynamic_commitments.len(),"fresh_sumcheck_masks":p.operations.len(),"model_session_commitments":p.model_session_commitments.len(),"model_rerandomization_proofs":p.model_rerandomization.len(),"model_session_rerandomization":"pedersen_blinder_schnorr_v1"}),
                &pp,
                &ppv,
            )?;
            write_manifest(
                &manifest,
                seal_manifest(proof_manifest, "zkDL_gat_proof_manifest_v1")?,
            )?;
            println!("PROOF_OK bytes={} commitment_ms={:.3} prove_ms={:.3}", size, p.commitment_time_ms, p.prove_time_ms);
        }
        Command::Verify {
            pp,
            statement,
            model_registry,
            proof,
            manifest,
            device,
        } => {
            let pp = checked_cli_path(&pp, CliPathRoot::Artifacts, false)?;
            let statement = checked_cli_path(&statement, CliPathRoot::RunsTmp, false)?;
            let model_registry = checked_cli_path(&model_registry, CliPathRoot::Artifacts, false)?;
            let proof = checked_cli_path(&proof, CliPathRoot::RunsTmp, false)?;
            let manifest = checked_cli_path(&manifest, CliPathRoot::RunsTmp, true)?;
            let ppv = read_bin(&pp)?;
            let s = read_bin(&statement)?;
            let m = read_bin(&model_registry)?;
            let p: ProofEnvelope = read_bin(&proof)?;
            let start = Instant::now();
            verify(&ppv, &s, &m, &p, device.as_deref())?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            let verifier_manifest = merge_pp_fields(
                serde_json::json!({"protocol_id":"GAT-ZKML-zkDL","proof_format":p.format,"verified":true,"independent_verifier":true,"verifier_reads_witness":false,"statement_id":p.statement_id,"model_registry_id":p.model_registry_id,"dataset":s.dataset,"input_preprocess_id":s.input_preprocess_id,"data_identity":s.data_identity,"source_input_id":s.source_input_id,"quant_cfg_id":s.quant_cfg_id,"checkpoint_manifest_id":s.checkpoint_manifest_id,"model_shape_id":s.model_shape_id,"model_arch_id":s.model_arch_id,"model_param_id":s.model_param_id,"graph_features_digest":s.graph_features_digest,"topology_digest":s.topology_digest,"device":p.device,"gpu_telemetry_validated":p.gpu_telemetry.is_some() || p.device == "cpu","gpu_telemetry":p.gpu_telemetry,"commitment_time_ms":p.commitment_time_ms,"prove_time_ms":p.prove_time_ms,"model_session_commitments":p.model_session_commitments.len(),"model_rerandomization_proofs":p.model_rerandomization.len(),"model_rerandomization_verified":true,"model_session_rerandomization":"pedersen_blinder_schnorr_v1","proof_path":canonical_path_string(&proof)?,"proof_sha256":sha_file(&proof)?,"proof_size_bytes":fs::metadata(&proof)?.len(),"verify_time_ms":elapsed}),
                &pp,
                &ppv,
            )?;
            write_manifest(
                &manifest,
                seal_manifest(verifier_manifest, "zkDL_gat_verifier_manifest_v1")?,
            )?;
            println!("VERIFY_OK verify_ms={:.3}", elapsed);
        }
        Command::CheckStatement { pp, statement } => {
            let pp = checked_cli_path(&pp, CliPathRoot::Artifacts, false)?;
            let statement = checked_cli_path(&statement, CliPathRoot::RunsTmp, false)?;
            let ppv: CommitmentParameters = read_bin(&pp)?;
            let s: PublicStatement = read_bin(&statement)?;
            ppv.validate()?;
            s.validate()?;
            ensure!(s.pp_id == ppv.parameters_id, "statement PP mismatch");
            println!("STATEMENT_OK id={}", s.statement_id);
        }
        Command::Compile {
            input,
            statement,
            witness,
        } => {
            let input = checked_cli_path(&input, CliPathRoot::RunsTmp, false)?;
            let statement = checked_cli_path(&statement, CliPathRoot::RunsTmp, true)?;
            let witness = checked_private_path(&witness, true)?;
            ensure!(
                statement != witness,
                "compile statement and witness paths must differ"
            );
            let raw: compiler::RelationCompilerInput =
                serde_json::from_slice(&read_bytes(&input)?)?;
            let (s, w) = compiler::compile(raw)?;
            write_bin(&statement, &s, false)?;
            write_bin(&witness, &w, true)?;
            println!(
                "COMPILE_OK statement_id={} registry_id={}",
                s.statement_id, s.registry.registry_id
            );
        }
    }
    Ok(())
}
