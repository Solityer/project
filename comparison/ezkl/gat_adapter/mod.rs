//! Canonical GAT adapter for the official ezkl graph/circuit pipeline.
//!
//! The adapter is baseline-local.  The only upstream changes are the small
//! `SupportedOp` and graph-construction hooks required to serialize and lay
//! out [`op::GatOp`] through the native `GraphCircuit` path.

/// Native graph/model construction helpers.
pub mod builder;
/// Rust-generated formal runner contract.
pub mod cli;
/// Strict CLI/contract-facing loader and resource authentication.
pub mod loader;
/// Authenticated resource and native artifact manifests.
pub mod manifest;
/// Native ezkl circuit operation.
pub mod op;
/// Independent integer reference relation and witness semantics.
pub mod relation;
/// Canonical input and manifest schema.
pub mod schema;
/// Strict verifier-facing public statement.
pub mod statement;

pub use builder::{build_model, build_model_for_run_args, PUBLIC_INPUT_NAMES};
pub use cli::{FormalCliContract, RequiredFlags};
pub use loader::{
    load_dataset, load_dataset_with_manifest, load_relation_resources_with_manifest, write_generated_metadata, AuthenticatedRelationResources, LoadedGat,
};
pub use manifest::{
    artifact_file, ArtifactFile, ArtifactManifest, BoundCircuitRef, CoreRunManifest, GpuEvidence, GpuProvenance, ResourceFile, ResourceManifest, SrsProvenance,
};
pub use op::GatOp;
pub use relation::{evaluate, evaluate_signed_division, AggregationTrace, GatTrace, HiddenTrace, RoundingMode, SignedDivision,};
pub use schema::{
    multiplier_to_scale_exponent, scale_exponent_to_multiplier, BoundFamily, CanonicalGat, CanonicalGraph, CanonicalModel, CircuitRef, DigestIdentity,
    ExactTable, HeadWeights, IntegerBoundLedger, ModelBinding, PublicInstanceBinding, ScaleChain, StaticTables, GAT_SCHEMA_VERSION,
};
pub use statement::{output_digest, PublicEzkl, PublicStatement};

/// Pinned native CUDA backend initialization and proof execution evidence.
#[cfg(feature = "gpu-accelerated")]
pub mod gpu;
