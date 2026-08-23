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
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Setup {
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        max_vector_len: usize,
        #[arg(long, default_value = "zkDL-GAT-transparent-PP-v1")]
        seed_label: String,
    },
    PpInfo {
        #[arg(long)]
        pp: PathBuf,
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
    let b = fs::read(path)?;
    let mut h = Sha256::new();
    h.update(b);
    Ok(hex::encode(h.finalize()))
}
fn write_manifest(path: &Path, value: serde_json::Value) -> Result<()> {
    if let Some(p) = path.parent() {
        fs::create_dir_all(p)?;
    }
    fs::write(path, serde_json::to_vec_pretty(&value)?)?;
    Ok(())
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Setup {
            output,
            max_vector_len,
            seed_label,
        } => {
            ensure!(
                max_vector_len.is_power_of_two(),
                "max-vector-len must be power of two"
            );
            let mut h = Sha256::new();
            h.update(b"zkDL-GAT-PP-seed-v1\0");
            h.update(seed_label.as_bytes());
            let pp = CommitmentParameters::new(h.finalize().into(), max_vector_len)?;
            write_bin(&output, &pp, false)?;
            println!(
                "SETUP_OK pp_id={} path={}",
                pp.parameters_id,
                output.display()
            );
        }
        Command::PpInfo { pp } => {
            let value: CommitmentParameters = read_bin(&pp)?;
            value.validate()?;
            println!(
                "{}",
                serde_json::to_string(
                    &serde_json::json!({"schema":"zkDL_gat_pp_info_v1","pp_id":value.parameters_id,"max_vector_len":value.max_vector_len})
                )?
            );
        }
        Command::RegisterModel {
            pp,
            statement,
            witness,
            model_registry,
            model_secret,
            certified_bounds_id,
        } => {
            let ppv: CommitmentParameters = read_bin(&pp)?;
            let s: PublicStatement = read_bin(&statement)?;
            let w: WitnessPackage = read_bin(&witness)?;
            let (r, secret) = register_model(&ppv, &s, &w, certified_bounds_id)?;
            write_bin(&model_registry, &r, false)?;
            write_bin(&model_secret, &secret, true)?;
            println!("REGISTER_MODEL_OK registry_id={}", r.registry_object_id);
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
            let ppv = read_bin(&pp)?;
            let s = read_bin(&statement)?;
            let m = read_bin(&model_registry)?;
            let secret = read_bin(&model_secret)?;
            let w = read_bin(&witness)?;
            let start = Instant::now();
            let p = prove(&ppv, &s, &m, &secret, &w, &device)?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            write_bin(&proof, &p, false)?;
            let size = fs::metadata(&proof)?.len();
            write_manifest(
                &manifest,
                serde_json::json!({"protocol_id":"GAT-ZKML-zkDL","proof_generated":true,"verified":false,"statement_id":p.statement_id,"model_registry_id":p.model_registry_id,"device":p.device,"gpu_telemetry":p.gpu_telemetry,"proof_path":proof,"proof_sha256":sha_file(&proof)?,"proof_size_bytes":size,"prove_time_ms":elapsed,"fresh_dynamic_commitments":p.dynamic_commitments.len(),"fresh_sumcheck_masks":p.operations.len()}),
            )?;
            println!("PROOF_OK bytes={} prove_ms={:.3}", size, elapsed);
        }
        Command::Verify {
            pp,
            statement,
            model_registry,
            proof,
            manifest,
            device,
        } => {
            let ppv = read_bin(&pp)?;
            let s = read_bin(&statement)?;
            let m = read_bin(&model_registry)?;
            let p: ProofEnvelope = read_bin(&proof)?;
            let start = Instant::now();
            verify(&ppv, &s, &m, &p, device.as_deref())?;
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            write_manifest(
                &manifest,
                serde_json::json!({"protocol_id":"GAT-ZKML-zkDL","verified":true,"independent_verifier":true,"verifier_reads_witness":false,"statement_id":p.statement_id,"model_registry_id":p.model_registry_id,"device":p.device,"gpu_telemetry_validated":p.gpu_telemetry.is_some(),"gpu_telemetry":p.gpu_telemetry,"proof_path":proof,"proof_sha256":sha_file(&proof)?,"proof_size_bytes":fs::metadata(&proof)?.len(),"verify_time_ms":elapsed}),
            )?;
            println!("VERIFY_OK verify_ms={:.3}", elapsed);
        }
        Command::CheckStatement { pp, statement } => {
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
            let raw: compiler::RelationCompilerInput = serde_json::from_slice(&fs::read(&input)?)?;
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
