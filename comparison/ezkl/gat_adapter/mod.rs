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
/// Compact native byte-slack bounded-range gadget used by formal GAT.
pub(crate) mod range_gadget;
/// Shape-only geometry for the canonical GAT relation.
pub mod geometry;
/// Native einsum geometry metadata shared with Halo2 layout analysis.
pub mod geometry_primitives;
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
pub use geometry::gat_settings_for_run_args;
/// Internal row-reserve constant used by the formal GAT relation fixtures.
#[doc(hidden)]
pub use range_gadget::GAT_RANGE_BLINDING_FACTOR_RESERVE;
pub use relation::{evaluate, evaluate_output, evaluate_signed_division, AggregationTrace, GatTrace, HiddenTrace, RoundingMode, SignedDivision,};
pub use schema::{
    multiplier_to_scale_exponent, scale_exponent_to_multiplier, BoundFamily, CanonicalGat, CanonicalGraph, CanonicalModel, CircuitRef, DigestIdentity,
    ExactTable, HeadWeights, IntegerBoundLedger, ModelBinding, PublicInstanceBinding, ScaleChain, StaticTables, GAT_SCHEMA_VERSION,
};
pub use statement::{output_digest, PublicEzkl, PublicStatement};

/// Run the shared, bounded geometry and MockProver regressions used by the
/// root-crate formal integration test without duplicating their fixtures.
#[cfg(test)]
pub(crate) fn run_geometry_and_mock_regressions() {
    geometry::run_tiny_settings_parity_fixture();
    geometry_primitives::run_dummy_duplication_boundary_regression();
    op::run_gat_bounded_mock_regressions();
    op::run_nonzero_attention_route_equivalence();
}

/// Pinned native CUDA backend initialization and proof execution evidence.
#[cfg(feature = "gpu-accelerated")]
pub mod gpu;
