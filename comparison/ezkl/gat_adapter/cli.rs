//! Rust-owned formal CLI contract generation.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Component, Path, PathBuf};

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
            official_chain: vec!["gat-prepare".into(), "produce-public-statement".into(), "gat-witness".into(), "prove".into(), "verify".into(),],
            prepare_command: "gat-prepare".into(),
            producer_command: "produce-public-statement".into(),
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
                    "--producer-manifest".into(),
                    "--gpu".into(),
                ],
                producer: vec![
                    "--dataset".into(),
                    "--resource-manifest".into(),
                    "--setup-context".into(),
                    "--output-dir".into(),
                    "--config".into(),
                    "--statement".into(),
                    "--public-output".into(),
                    "--producer-manifest".into(),
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
        let resolved_path = resolve_contract_path(path)?;
        let contract = Self::canonical();
        contract.validate()?;
        if let Some(parent) = resolved_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("create {}: {}", parent.display(), e))?;
        }
        let bytes = serde_json::to_vec_pretty(&contract).map_err(|e| format!("serialize formal CLI contract: {}", e))?;
        fs::write(&resolved_path, bytes).map_err(|e| format!("write formal CLI contract: {}", e))
    }
}

/// Write the native runtime source identity through the same baseline-local
/// path checks used for the generated formal CLI contract.
pub fn write_source_tree_sha256(path: &Path, digest: &str) -> Result<(), String> {
    validate_contract_path(path)?;
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("native source-tree SHA-256 must be exactly 64 lowercase hexadecimal characters".into());
    }
    let resolved_path = resolve_contract_path(path)?;
    if let Some(parent) = resolved_path.parent() {
        fs::create_dir_all(parent).map_err(|error| format!("create {}: {}", parent.display(), error))?;
    }
    fs::write(&resolved_path, digest.as_bytes())
        .map_err(|error| format!("write native source-tree SHA-256 {}: {}", resolved_path.display(), error))
}

fn validate_contract_path(path: &Path) -> Result<(), String> {
    if path.is_absolute() || path.components().any(|component| { matches!(component, Component::ParentDir | Component::RootDir | Component::Prefix(_)) }){
        return Err(format!("formal CLI contract path must be baseline-local and must not contain '..': {}", path.display()));
    }
    if path.as_os_str().is_empty() {
        return Err("formal CLI contract path must not be empty".into());
    }
    Ok(())
}

/// Resolve a validated relative contract path only at the IO boundary.
///
/// The formal contract path is intentionally validated in its original
/// baseline-relative form.  Existing path components are then checked for
/// symlinks and the canonical parent/target must remain below the canonical
/// current working directory (the active baseline root).
fn resolve_contract_path(path: &Path) -> Result<PathBuf, String> {
    let baseline_root = std::env::current_dir().map_err(|e| format!("resolve formal CLI baseline root: {}", e))?;
    let canonical_root = baseline_root.canonicalize().map_err(|e| format!("canonicalize formal CLI baseline root: {}", e))?;
    let resolved = baseline_root.join(path);
    ensure_no_symlink_components(&resolved, &baseline_root)?;
    if let Some(parent) = resolved.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| format!("create formal CLI parent {}: {}", parent.display(), e))?;
            ensure_no_symlink_components(&resolved, &baseline_root)?;
        }
        let canonical_parent = parent.canonicalize().map_err(|e| format!("canonicalize formal CLI parent {}: {}", parent.display(), e))?;
        if !canonical_parent.starts_with(&canonical_root) {
            return Err(format!("formal CLI contract path escapes baseline root: {}", path.display()));
        }
    }
    if resolved.exists() {
        let canonical_target = resolved.canonicalize().map_err(|e| format!("canonicalize formal CLI contract {}: {}", resolved.display(), e))?;
        if !canonical_target.starts_with(&canonical_root) {
            return Err(format!("formal CLI contract path escapes baseline root: {}", path.display()));
        }
    }
    Ok(resolved)
}

fn ensure_no_symlink_components(path: &Path, root: &Path) -> Result<(), String> {
    let relative = path.strip_prefix(root).map_err(|_| format!("formal CLI path escapes baseline root: {}", path.display()))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            continue;
        };
        current.push(part);
        match fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!("formal CLI contract path contains a symlink: {}", path.display()));
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => break,
            Err(error) => {
                return Err(format!("inspect formal CLI contract path {}: {}", current.display(), error));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_contract_path, DEFAULT_FORMAL_CONTRACT};
    use std::path::Path;

    #[test]
    fn formal_contract_default_is_baseline_relative() {
        assert!(validate_contract_path(Path::new(DEFAULT_FORMAL_CONTRACT)).is_ok());
    }

    #[test]
    fn formal_contract_rejects_absolute_and_parent_paths() {
        assert!(validate_contract_path(Path::new("/tmp/formal_cli.json")).is_err());
        assert!(validate_contract_path(Path::new("../formal_cli.json")).is_err());
    }
}
