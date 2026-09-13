//! Rust-owned formal CLI contract generation.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Component, Path};

/// Baseline-local location of the generated contract consumed by formal
/// runners.  The build step creates the directory before invoking this
/// command, so contract generation never depends on a pre-existing file.
pub const DEFAULT_FORMAL_CONTRACT: &str = "artifacts/gat_adapter/formal_cli.json";

/// The machine-readable contract consumed by the baseline formal runners.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct FormalCliContract {
    pub schema_version: u32,
    pub baseline: String,
    pub contract_id: String,
    pub generated_by: String,
    pub adapter_version: String,
    pub ready: bool,
    pub relation_id: String,
    pub proof_scope: String,
    pub public_statement_mode: String,
    pub output_mode: String,
    pub privacy_policy: String,
    pub native_verify_required: bool,
    pub gpu_evidence_required: bool,
    pub proof_instance_binding: String,
    pub official_chain: Vec<String>,
    pub prepare_command: String,
    pub producer_command: String,
    pub witness_command: String,
    pub prove_command: String,
    pub verify_command: String,
    pub required_flags: RequiredFlags,
    pub native_backend: String,
    pub gpu_backend: String,
}

/// Required flags for each formal command.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct RequiredFlags {
    pub prepare: Vec<String>,
    pub producer: Vec<String>,
    pub witness: Vec<String>,
    pub prove: Vec<String>,
    pub verify: Vec<String>,
}

impl FormalCliContract {
    /// Construct the canonical contract. The JSON file is never hand-written.
    pub fn canonical() -> Self {
        Self {
            schema_version: 1,
            baseline: "ezkl".into(),
            contract_id: "ezkl-gat-native-v1".into(),
            generated_by: "ezkl::gat_adapter".into(),
            adapter_version: "gat-adapter-rust-v1".into(),
            ready: true,
            relation_id: "R_GAT-ezkl".into(),
            proof_scope: "full_gat".into(),
            public_statement_mode: "external".into(),
            output_mode: "public_logits".into(),
            privacy_policy: "model_and_trace_hidden".into(),
            native_verify_required: true,
            gpu_evidence_required: true,
            proof_instance_binding: "native_public_instances".into(),
            official_chain: vec![
                "gat-prepare".into(),
                "gat-produce".into(),
                "gat-witness".into(),
                "prove".into(),
                "verify".into(),
            ],
            prepare_command: "gat-prepare".into(),
            producer_command: "gat-produce".into(),
            witness_command: "gat-witness".into(),
            prove_command: "prove".into(),
            verify_command: "verify".into(),
            required_flags: RequiredFlags {
                prepare: vec![
                    "--dataset".into(),
                    "--phase".into(),
                    "--resource-manifest".into(),
                    "--output-dir".into(),
                    "--srs-path".into(),
                    "--public-statement".into(),
                    "--public-output".into(),
                    "--gpu".into(),
                ],
                producer: vec![
                    "--dataset".into(),
                    "--resource-manifest".into(),
                    "--output-dir".into(),
                    "--gpu".into(),
                ],
                witness: vec![
                    "--dataset".into(),
                    "--resource-manifest".into(),
                    "--compiled-circuit".into(),
                    "--statement".into(),
                    "--output".into(),
                    "--gpu".into(),
                ],
                prove: vec![
                    "--witness".into(),
                    "--compiled-circuit".into(),
                    "--pk-path".into(),
                    "--proof-path".into(),
                    "--srs-path".into(),
                    "--check-mode".into(),
                ],
                verify: vec![
                    "--settings-path".into(),
                    "--proof-path".into(),
                    "--vk-path".into(),
                    "--srs-path".into(),
                    "--statement".into(),
                    "--public-output".into(),
                    "--artifact-manifest".into(),
                    "--resource-manifest".into(),
                ],
            },
            native_backend: "halo2-kzg-shplonk".into(),
            gpu_backend: "icicle-cuda".into(),
        }
    }

    /// Validate the constants required by formal_cli_contract.schema.json.
    pub fn validate(&self) -> Result<(), String> {
        let expected = Self::canonical();
        if self != &expected {
            return Err("formal CLI contract does not match canonical native ezkl contract".into());
        }
        Ok(())
    }

    /// Write a generated contract and validate it before publication.
    pub fn write(path: &Path) -> Result<(), String> {
        validate_contract_path(path)?;
        let contract = Self::canonical();
        contract.validate()?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("create {}: {}", parent.display(), e))?;
        }
        let bytes = serde_json::to_vec_pretty(&contract)
            .map_err(|e| format!("serialize formal CLI contract: {}", e))?;
        fs::write(path, bytes).map_err(|e| format!("write formal CLI contract: {}", e))
    }
}

fn validate_contract_path(path: &Path) -> Result<(), String> {
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!(
            "formal CLI contract path must be baseline-local and must not contain '..': {}",
            path.display()
        ));
    }
    if path.as_os_str().is_empty() {
        return Err("formal CLI contract path must not be empty".into());
    }
    Ok(())
}
