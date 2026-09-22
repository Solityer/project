//! Authenticated, baseline-local resource and artifact manifests.
//!
//! This module intentionally does not reuse the comparison runners' Python
//! manifest parser.  The Rust adapter authenticates every file before it is
//! parsed and treats paths as untrusted input: absolute paths, parent
//! components, symlinks and paths escaping the declared resource root are
//! rejected.

use crate::gat_adapter::schema::{sha256_digest, CircuitRef};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};

/// A source snapshot recorded by the packaging step.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceSourceSnapshot {
    /// Source repository root at packaging time.
    pub root: String,
    /// Source commit.
    pub git_commit: String,
    /// Source tree digest.
    pub git_tree: String,
}

/// One authenticated resource file.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceFile {
    /// Path relative to the baseline checkout (the checked-in manifest uses
    /// `resources/canonical/...`).
    pub path: String,
    /// Exact byte length.
    pub size: u64,
    /// SHA-256 digest of the copied file.
    pub sha256: String,
    /// Original producer path, retained for provenance only.
    pub source_path: String,
    /// Digest of the original producer file.
    pub source_sha256: String,
    /// Dataset or `all`.
    pub dataset: String,
    /// Canonical role.
    pub role: String,
}

/// The checked-in resource manifest.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ResourceManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Baseline name.
    pub baseline: String,
    /// Declared resource root relative to the baseline checkout.
    pub resource_root: String,
    /// Upstream source snapshot.
    pub source_snapshot: ResourceSourceSnapshot,
    /// Files included in the canonical package.
    pub files: Vec<ResourceFile>,
}

/// A digest and size binding for a generated artifact.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactFile {
    /// Baseline-relative artifact path.
    pub path: String,
    /// Byte length.
    pub size: u64,
    /// SHA-256 digest.
    pub sha256: String,
}

/// Circuit identity carried by the strict artifact and statement schemas.
/// The relation id is explicit so Python and Rust cannot silently bind an
/// artifact from another baseline or relation.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct BoundCircuitRef {
    /// Relation this native circuit proves.
    pub relation_id: String,
    /// Native circuit/settings/VK identity and their byte digests.
    #[serde(flatten)]
    pub native: CircuitRef,
}

impl BoundCircuitRef {
    /// Construct a relation-bound native circuit identity.
    pub fn new(native: CircuitRef) -> Self {
        Self {
            relation_id: "R_GAT-ezkl".into(),
            native,
        }
    }

    /// Validate the relation and native identities.
    pub fn validate(&self) -> Result<(), String> {
        if self.relation_id != "R_GAT-ezkl" {
            return Err("circuit_ref relation_id is not R_GAT-ezkl".into());
        }
        self.native.validate()?;
        for (label, value) in [
            ("compiled_circuit_id", &self.native.compiled_circuit_id),
            ("settings_id", &self.native.settings_id),
            ("vk_id", &self.native.vk_id),
            ("model_graph_id", &self.native.model_graph_id),
        ] {
            let lowered = value.to_ascii_lowercase();
            if ["unbound", "pending", "placeholder", "dummy", "fake"]
                .iter()
                .any(|marker| lowered.contains(marker))
            {
                return Err(format!(
                    "circuit_ref {} contains an unbound/placeholder identity",
                    label
                ));
            }
        }
        Ok(())
    }
}

/// Operator-authenticated SRS provenance.  The digest is computed from the
/// selected file and must also be checked against the operator-supplied
/// digest by the formal runner.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SrsProvenance {
    /// Provenance declaration for the official ceremony material.
    pub source: String,
    /// Whether the operator authenticated the full SRS.
    pub trusted: bool,
    /// Reduced SRS is forbidden for formal runs.
    pub reduced: bool,
    /// Generated SRS is forbidden for formal runs.
    pub generated: bool,
    /// Stable operator-supplied ceremony identity.
    pub ceremony_id: String,
    /// SHA-256 of the selected SRS bytes.
    pub sha256: String,
}

/// Native artifact/core manifest emitted by `gat-prepare`.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactManifest {
    /// Manifest schema version.
    pub schema_version: u32,
    /// Baseline identifier.
    pub baseline: String,
    /// Relation identifier.
    pub relation_id: String,
    /// Dataset identity.
    pub dataset: String,
    /// Backend identity for this artifact package.
    pub backend: String,
    /// Proof relation scope.
    pub proof_scope: String,
    /// Privacy boundary.
    pub privacy_policy: String,
    /// Statement acquisition mode.
    pub public_statement_mode: String,
    /// Public output mode.
    pub output_mode: String,
    /// Quantization registry identity.
    pub quant_cfg_id: String,
    /// Static-table registry identity.
    pub static_tables_id: String,
    /// Native relation-bound circuit identity.
    pub circuit_ref: BoundCircuitRef,
    /// Individual native identities mirrored for strict cross-language checks.
    pub compiled_circuit_id: String,
    /// Native settings identity mirrored for strict cross-language checks.
    pub settings_id: String,
    /// Native verification-key identity mirrored for strict cross-language checks.
    pub vk_id: String,
    /// Running official binary digest.
    pub binary_sha256: String,
    /// Current source tree digest.
    pub source_tree_sha256: String,
    /// Canonical resource-manifest digest.
    pub resource_manifest_sha256: String,
    /// Authenticated artifact files.
    pub files: Vec<ArtifactFile>,
    /// Source snapshot identity.
    pub source_snapshot: ResourceSourceSnapshot,
    /// Canonical statement identity.
    pub statement_sha256: String,
    /// Independent producer provenance/identity manifest digest.
    pub producer_manifest_sha256: String,
    /// Circuit/settings/VK identity from the native setup.
    pub compiled_circuit_sha256: String,
    /// Settings digest.
    pub settings_sha256: String,
    /// Verification-key digest.
    pub vk_sha256: String,
    /// Proving-key digest.
    pub pk_sha256: String,
    /// Native SRS digest, never a generated dummy SRS.
    pub srs_sha256: String,
    /// Operator-authenticated SRS provenance.
    pub srs_provenance: SrsProvenance,
    /// Adapter security profile.
    pub ezkl_security_profile_id: String,
}

/// Native verification attestation emitted by the official verify command.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct CoreRunManifest {
    pub schema_version: u32,
    pub relation_id: String,
    pub dataset: String,
    pub backend: String,
    pub proof_scope: String,
    pub privacy_policy: String,
    pub public_statement_mode: String,
    pub output_mode: String,
    pub native_verify_result: bool,
    pub native_public_instance_binding: bool,
    pub verifier_reads_witness: bool,
    pub proof_contains_private_witness: bool,
    pub public_statement_sha256: String,
    pub public_output_sha256: String,
    pub proof_instances_sha256: String,
    pub proof_instance_count: usize,
    pub circuit_ref: BoundCircuitRef,
}

/// Backend provenance emitted next to a formal run.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct GpuProvenance {
    pub frontend_commit: String,
    pub abi: String,
    pub device: String,
    pub architecture: String,
    pub runtime: String,
    pub manifest_sha256: String,
}

/// Strict CPU/GPU execution evidence.  GPU values are never synthesized: the
/// formal path rejects GPU requests when the native counter/synchronization
/// hook is unavailable.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
#[allow(missing_docs)]
pub struct GpuEvidence {
    pub backend: String,
    pub gpu_backend: String,
    pub backend_loaded: bool,
    pub cpu_fallback: bool,
    pub device_synchronized: bool,
    pub gpu_kernel_invocation_count: u64,
    pub provenance: Option<GpuProvenance>,
    pub invocation_trace_sha256: String,
}

/// A fully resolved, authenticated resource file.
#[derive(Clone, Debug)]
pub struct AuthenticatedResource {
    /// Manifest record.
    pub record: ResourceFile,
    /// Resolved path under the declared resource root.
    pub path: PathBuf,
}

fn valid_digest(label: &str, digest: &str) -> Result<(), String> {
    if digest.len() != 64 || digest.bytes().any(|b| !b.is_ascii_hexdigit()) {
        return Err(format!("{} must be a 32-byte hexadecimal digest", label));
    }
    if digest.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err(format!("{} must use lowercase hexadecimal", label));
    }
    Ok(())
}

fn safe_relative(path: &str) -> Result<PathBuf, String> {
    let candidate = Path::new(path);
    if candidate.as_os_str().is_empty() || candidate.is_absolute() {
        return Err(format!("resource path must be relative: {}", path));
    }
    for component in candidate.components() {
        match component {
            Component::Normal(_) => {}
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(format!("resource path contains unsafe component: {}", path));
            }
        }
    }
    Ok(candidate.to_path_buf())
}

fn ensure_no_symlink(path: &Path, root: &Path) -> Result<(), String> {
    let relative = path
        .strip_prefix(root)
        .map_err(|_| format!("path escapes resource root: {}", path.display()))?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let Component::Normal(part) = component else {
            continue;
        };
        current.push(part);
        let metadata = fs::symlink_metadata(&current)
            .map_err(|e| format!("cannot inspect {}: {}", current.display(), e))?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "symlink resource path is forbidden: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<(u64, String), String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut size = 0_u64;
    let mut chunk = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut chunk)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        if read == 0 {
            break;
        }
        size = size
            .checked_add(read as u64)
            .ok_or_else(|| format!("size overflow while reading {}", path.display()))?;
    }
    let digest =
        sha256::digest_file(path).map_err(|e| format!("digest {}: {}", path.display(), e))?;
    Ok((size, digest))
}

impl ResourceManifest {
    /// Parse a strict manifest from disk without authenticating its files.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes =
            fs::read(path).map_err(|e| format!("read manifest {}: {}", path.display(), e))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse manifest {}: {}", path.display(), e))
    }

    /// Load a manifest only when it is a non-symlink path relative to the
    /// baseline checkout.  CLI callers must not be able to redirect resource
    /// authentication outside the packaged baseline.
    pub fn load_from_baseline(baseline_root: &Path, path: &Path) -> Result<Self, String> {
        if path.is_absolute()
            || path.components().any(|component| {
                matches!(
                    component,
                    Component::ParentDir | Component::RootDir | Component::Prefix(_)
                )
            })
        {
            return Err(format!(
                "resource manifest path must be baseline-relative: {}",
                path.display()
            ));
        }
        let resolved = baseline_root.join(path);
        ensure_no_symlink(&resolved, baseline_root)?;
        let canonical_root = baseline_root
            .canonicalize()
            .map_err(|e| format!("canonicalize baseline root: {}", e))?;
        let canonical = resolved
            .canonicalize()
            .map_err(|e| format!("canonicalize manifest {}: {}", resolved.display(), e))?;
        if !canonical.starts_with(&canonical_root) {
            return Err(format!(
                "manifest path escapes baseline root: {}",
                path.display()
            ));
        }
        Self::load(&resolved)
    }

    /// Canonical JSON digest used to bind a run artifact to this manifest.
    pub fn digest(&self) -> Result<String, String> {
        let bytes =
            serde_json::to_vec(self).map_err(|e| format!("serialize resource manifest: {}", e))?;
        sha256_digest(&bytes)
    }

    /// Validate manifest fields and all copied files below `baseline_root`.
    pub fn authenticate(&self, baseline_root: &Path) -> Result<Vec<AuthenticatedResource>, String> {
        if self.schema_version != 1 || self.baseline != "ezkl" {
            return Err("unsupported ezkl resource manifest identity".into());
        }
        let root_rel = safe_relative(&self.resource_root)?;
        let resource_root = baseline_root.join(&root_rel);
        ensure_no_symlink(&resource_root, baseline_root)?;
        if !resource_root.is_dir() {
            return Err(format!(
                "resource root is not a directory: {}",
                resource_root.display()
            ));
        }
        if self.files.is_empty() {
            return Err("resource manifest contains no files".into());
        }
        let mut out = Vec::with_capacity(self.files.len());
        let canonical_root = baseline_root
            .canonicalize()
            .map_err(|e| format!("canonicalize baseline root: {}", e))?;
        for record in &self.files {
            valid_digest("resource sha256", &record.sha256)?;
            valid_digest("source sha256", &record.source_sha256)?;
            let source = Path::new(&record.source_path);
            if !source.is_absolute() {
                return Err(format!(
                    "source path is not absolute: {}",
                    record.source_path
                ));
            }
            ensure_no_symlink(source, baseline_root)?;
            let canonical_source = source
                .canonicalize()
                .map_err(|e| format!("canonicalize source {}: {}", source.display(), e))?;
            if !canonical_source.starts_with(&canonical_root) {
                return Err(format!(
                    "source path escapes baseline root: {}",
                    record.source_path
                ));
            }
            if !fs::metadata(source)
                .map_err(|e| format!("stat source {}: {}", source.display(), e))?
                .is_file()
            {
                return Err(format!(
                    "source path is not a regular file: {}",
                    source.display()
                ));
            }
            let (_, source_digest) = hash_file(source)?;
            if source_digest != record.source_sha256 {
                return Err(format!("source digest mismatch: {}", source.display()));
            }
            let path_rel = safe_relative(&record.path)?;
            let path = baseline_root.join(&path_rel);
            ensure_no_symlink(&path, baseline_root)?;
            let canonical = path
                .canonicalize()
                .map_err(|e| format!("canonicalize {}: {}", path.display(), e))?;
            if !canonical.starts_with(&resource_root) {
                return Err(format!(
                    "resource path escapes declared root: {}",
                    record.path
                ));
            }
            let metadata =
                fs::metadata(&path).map_err(|e| format!("stat {}: {}", path.display(), e))?;
            if !metadata.is_file() {
                return Err(format!("resource is not a regular file: {}", record.path));
            }
            if metadata.len() != record.size {
                return Err(format!("size mismatch for {}", record.path));
            }
            let (size, digest) = hash_file(&path)?;
            if size != record.size || digest != record.sha256 {
                return Err(format!("digest mismatch for {}", record.path));
            }
            out.push(AuthenticatedResource {
                record: record.clone(),
                path,
            });
        }
        Ok(out)
    }

    /// Resolve and authenticate one named file without opening any path not
    /// present in the manifest.
    pub fn authenticate_file(
        &self,
        baseline_root: &Path,
        requested: &str,
    ) -> Result<AuthenticatedResource, String> {
        let path = safe_relative(requested)?;
        let record = self
            .files
            .iter()
            .find(|record| Path::new(&record.path) == path)
            .ok_or_else(|| format!("file is not listed in resource manifest: {}", requested))?;
        let resource_root = baseline_root.join(safe_relative(&self.resource_root)?);
        let resolved = baseline_root.join(safe_relative(&record.path)?);
        ensure_no_symlink(&resource_root, baseline_root)?;
        ensure_no_symlink(&resolved, baseline_root)?;
        if !resolved
            .canonicalize()
            .map_err(|e| e.to_string())?
            .starts_with(&resource_root)
        {
            return Err(format!(
                "resource path escapes declared root: {}",
                requested
            ));
        }
        let metadata =
            fs::metadata(&resolved).map_err(|e| format!("stat {}: {}", resolved.display(), e))?;
        if metadata.len() != record.size {
            return Err(format!("size mismatch for {}", requested));
        }
        let (_, digest) = hash_file(&resolved)?;
        if digest != record.sha256 {
            return Err(format!("digest mismatch for {}", requested));
        }
        Ok(AuthenticatedResource {
            record: record.clone(),
            path: resolved,
        })
    }
}

impl ArtifactManifest {
    /// Save a generated artifact manifest atomically enough for local runs.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        self.validate()?;
        let parent = path.parent().unwrap_or_else(|| Path::new("."));
        fs::create_dir_all(parent).map_err(|e| format!("create {}: {}", parent.display(), e))?;
        let bytes =
            serde_json::to_vec_pretty(self).map_err(|e| format!("serialize artifact: {}", e))?;
        let mut file =
            File::create(path).map_err(|e| format!("create {}: {}", path.display(), e))?;
        file.write_all(&bytes)
            .map_err(|e| format!("write {}: {}", path.display(), e))
    }

    /// Load a strict artifact manifest without accepting compatibility fields.
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = fs::read(path)
            .map_err(|e| format!("read artifact manifest {}: {}", path.display(), e))?;
        let raw: Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse artifact manifest {}: {}", path.display(), e))?;
        validate_artifact_json_shape(&raw)?;
        serde_json::from_value(raw)
            .map_err(|e| format!("parse artifact manifest {}: {}", path.display(), e))
    }

    /// Validate schema-level identities before native verification.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != 1
            || self.baseline != "ezkl"
            || self.relation_id != "R_GAT-ezkl"
            || self.proof_scope != "full_gat"
            || self.privacy_policy != "model_and_trace_hidden"
            || self.public_statement_mode != "external"
            || self.output_mode != "public_logits"
        {
            return Err("artifact manifest has unsupported relation/schema identity".into());
        }
        if self.dataset.is_empty()
            || self.backend.is_empty()
            || self.quant_cfg_id.is_empty()
            || self.static_tables_id.is_empty()
            || self.ezkl_security_profile_id.is_empty()
        {
            return Err("artifact manifest has an empty identity field".into());
        }
        let backend = self.backend.to_ascii_lowercase();
        if backend != "cpu" && backend != "gpu" {
            return Err(format!(
                "artifact manifest has unsupported backend: {}",
                self.backend
            ));
        }
        if [
            self.dataset.as_str(),
            self.quant_cfg_id.as_str(),
            self.static_tables_id.as_str(),
            self.ezkl_security_profile_id.as_str(),
        ]
        .iter()
        .any(|value| {
            let lowered = value.to_ascii_lowercase();
            ["unbound", "pending", "placeholder", "dummy", "fake"]
                .iter()
                .any(|marker| lowered.contains(marker))
        }) {
            return Err("artifact manifest contains an unbound/placeholder identity".into());
        }
        self.circuit_ref.validate()?;
        if self.compiled_circuit_id != self.circuit_ref.native.compiled_circuit_id
            || self.settings_id != self.circuit_ref.native.settings_id
            || self.vk_id != self.circuit_ref.native.vk_id
        {
            return Err("artifact manifest circuit_ref ids disagree with top-level ids".into());
        }
        for (label, digest) in [
            ("binary_sha256", &self.binary_sha256),
            ("source_tree_sha256", &self.source_tree_sha256),
            ("resource_manifest_sha256", &self.resource_manifest_sha256),
            ("statement_sha256", &self.statement_sha256),
            ("producer_manifest_sha256", &self.producer_manifest_sha256),
            ("compiled_circuit_sha256", &self.compiled_circuit_sha256),
            ("settings_sha256", &self.settings_sha256),
            ("vk_sha256", &self.vk_sha256),
            ("pk_sha256", &self.pk_sha256),
            ("srs_sha256", &self.srs_sha256),
        ] {
            valid_digest(label, digest)?;
        }
        if self.srs_provenance.source != "official_ezkl_ceremony"
            || !self.srs_provenance.trusted
            || self.srs_provenance.reduced
            || self.srs_provenance.generated
            || self.srs_provenance.ceremony_id.is_empty()
            || self.srs_provenance.sha256 != self.srs_sha256
        {
            return Err(
                "artifact SRS provenance is not trusted full-size official material".into(),
            );
        }
        if !self.srs_provenance.ceremony_id.starts_with("sha256:")
            || self.srs_provenance.ceremony_id != format!("sha256:{}", self.srs_sha256)
        {
            return Err("artifact SRS ceremony_id does not bind the selected SRS digest".into());
        }
        if self.files.is_empty() {
            return Err("artifact manifest contains no authenticated files".into());
        }
        let mut paths = std::collections::HashSet::new();
        for file in &self.files {
            let relative = safe_relative(&file.path)?;
            if !paths.insert(relative) {
                return Err(format!(
                    "artifact manifest contains duplicate file: {}",
                    file.path
                ));
            }
            if file.size == 0 {
                return Err(format!(
                    "artifact manifest contains an empty file: {}",
                    file.path
                ));
            }
            valid_digest("artifact file sha256", &file.sha256)?;
        }
        self.validate_required_files(false)?;
        for (label, value) in [
            ("source_snapshot.root", &self.source_snapshot.root),
            (
                "source_snapshot.git_commit",
                &self.source_snapshot.git_commit,
            ),
            ("source_snapshot.git_tree", &self.source_snapshot.git_tree),
        ] {
            if value.is_empty() {
                return Err(format!("{} must not be empty", label));
            }
        }
        Ok(())
    }

    /// Validate the file set required before proving.  The proof itself is
    /// intentionally optional until the official prove command has produced
    /// it; verification promotes the same manifest with `require_proof=true`.
    pub fn validate_required_files(&self, require_proof: bool) -> Result<(), String> {
        let mut names = std::collections::HashSet::new();
        for file in &self.files {
            names.insert(file.path.as_str());
        }
        let required = [
            "statement.json",
            "public_output.json",
            "producer_context.json",
            "producer_manifest.json",
            "settings.json",
            "compiled.circuit",
            "srs.kzg",
            "pk.key",
            "vk.key",
            "gat_static_tables.json",
            "gat_integer_ledger.json",
        ];
        for path in required {
            if !names.contains(path) {
                return Err(format!(
                    "artifact manifest is missing required file: {}",
                    path
                ));
            }
        }
        if require_proof && !names.contains("proof.json") {
            return Err("artifact manifest is missing required file: proof.json".into());
        }
        Ok(())
    }

    /// Validate the complete verifier-facing artifact after official proving.
    pub fn validate_for_verification(&self) -> Result<(), String> {
        self.validate()?;
        self.validate_required_files(true)
    }
}

fn validate_artifact_json_shape(value: &Value) -> Result<(), String> {
    ensure_json_keys(
        value,
        &[
            "schema_version",
            "baseline",
            "relation_id",
            "dataset",
            "backend",
            "proof_scope",
            "privacy_policy",
            "public_statement_mode",
            "output_mode",
            "quant_cfg_id",
            "static_tables_id",
            "circuit_ref",
            "compiled_circuit_id",
            "settings_id",
            "vk_id",
            "binary_sha256",
            "source_tree_sha256",
            "resource_manifest_sha256",
            "files",
            "source_snapshot",
            "statement_sha256",
            "producer_manifest_sha256",
            "compiled_circuit_sha256",
            "settings_sha256",
            "vk_sha256",
            "pk_sha256",
            "srs_sha256",
            "srs_provenance",
            "ezkl_security_profile_id",
        ],
        "artifact manifest",
    )?;
    let object = value
        .as_object()
        .ok_or_else(|| "artifact manifest must be a JSON object".to_string())?;
    ensure_json_keys(
        object
            .get("circuit_ref")
            .ok_or_else(|| "artifact circuit_ref is missing".to_string())?,
        &[
            "relation_id",
            "compiled_circuit_id",
            "compiled_circuit_digest",
            "settings_id",
            "settings_digest",
            "vk_id",
            "vk_digest",
            "model_graph_id",
            "model_graph_digest",
            "ezkl_security_profile_id",
        ],
        "artifact circuit_ref",
    )?;
    ensure_json_keys(
        object
            .get("source_snapshot")
            .ok_or_else(|| "artifact source_snapshot is missing".to_string())?,
        &["root", "git_commit", "git_tree"],
        "artifact source_snapshot",
    )?;
    ensure_json_keys(
        object
            .get("srs_provenance")
            .ok_or_else(|| "artifact srs_provenance is missing".to_string())?,
        &[
            "source",
            "trusted",
            "reduced",
            "generated",
            "ceremony_id",
            "sha256",
        ],
        "artifact srs_provenance",
    )?;
    let files = object
        .get("files")
        .and_then(Value::as_array)
        .ok_or_else(|| "artifact files must be an array".to_string())?;
    for (index, file) in files.iter().enumerate() {
        ensure_json_keys(
            file,
            &["path", "size", "sha256"],
            &format!("artifact files[{}]", index),
        )?;
    }
    Ok(())
}

fn ensure_json_keys(value: &Value, expected: &[&str], label: &str) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("{} must be a JSON object", label))?;
    let actual = object
        .keys()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let wanted = expected
        .iter()
        .map(|key| (*key).to_string())
        .collect::<std::collections::BTreeSet<_>>();
    if actual != wanted {
        return Err(format!(
            "{} schema mismatch; missing={:?}, extra={:?}",
            label,
            wanted.difference(&actual).collect::<Vec<_>>(),
            actual.difference(&wanted).collect::<Vec<_>>()
        ));
    }
    Ok(())
}

/// Authenticate an artifact file and return its size/digest.
pub fn artifact_file(path: &Path, display_path: String) -> Result<ArtifactFile, String> {
    if fs::symlink_metadata(path)
        .map_err(|e| format!("stat {}: {}", path.display(), e))?
        .file_type()
        .is_symlink()
    {
        return Err(format!(
            "artifact file must not be a symlink: {}",
            path.display()
        ));
    }
    let metadata = fs::metadata(path).map_err(|e| format!("stat {}: {}", path.display(), e))?;
    if !metadata.is_file() {
        return Err(format!(
            "artifact path is not a regular file: {}",
            path.display()
        ));
    }
    let (size, sha256) = hash_file(path)?;
    if metadata.len() != size {
        return Err(format!("file changed while hashing {}", path.display()));
    }
    Ok(ArtifactFile {
        path: display_path,
        size,
        sha256,
    })
}
