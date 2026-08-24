//! Local placeholder for builds that do not enable EVM verifier generation.
//!
//! The native CSR GAT baseline is built with `--no-default-features --features ezkl,no-banner`,
//! so this crate is resolved but not compiled into the CLI path used by the experiment.

/// Placeholder batch-open scheme enum.
#[derive(Clone, Copy, Debug)]
pub enum BatchOpenScheme {
    /// Placeholder variant matching the upstream API name used by ezkl.
    Bdfg21,
}

/// Placeholder Solidity generator.
#[derive(Clone, Debug, Default)]
pub struct SolidityGenerator;

impl SolidityGenerator {
    /// Constructs a placeholder generator and refuses runtime use.
    pub fn new<T, U>(_params: T, _vk: U, _scheme: BatchOpenScheme) -> Self {
        Self
    }

    /// Keep the optional EVM command path type-compatible in the native
    /// comparison build.  The renderer is intentionally not linked here;
    /// requesting Solidity output returns an explicit error instead of
    /// producing an unverifiable artifact.
    pub fn render(&self) -> Result<String, String> {
        Err("halo2_solidity_verifier renderer is not linked in this build".to_string())
    }

    pub fn render_separately(&self) -> Result<(String, String), String> {
        Err("halo2_solidity_verifier renderer is not linked in this build".to_string())
    }

    pub fn render_separately_vka_words(&self) -> Result<(String, Vec<[u8; 32]>), String> {
        Err("halo2_solidity_verifier renderer is not linked in this build".to_string())
    }
}

/// Placeholder calldata encoder for reusable-verifier paths.
pub fn encode_register_vk_calldata<T>(_vk: T) -> Result<Vec<u8>, String> {
    Err("halo2_solidity_verifier stub is unavailable in native CSR build".to_string())
}
