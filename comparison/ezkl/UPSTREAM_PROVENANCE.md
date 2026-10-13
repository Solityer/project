# ezkl upstream provenance

This file records source provenance for the baseline-local official ezkl/Halo2
tree. It is an audit record, not a replacement for the upstream repository or
for a dependency lockfile.

## Source identity

- Official repository identified by the preserved upstream README: <https://github.com/zkonduit/ezkl>.
- Local project repository: `git@github.com:Solityer/project.git`.
- Local import commit: `616f5192dfe742704bd451021c06cd633b94b980` (2026-09-09 21:50:37 +0800). This commit created the complete `comparison/ezkl-main/` official Rust/Cargo tree in the local project.
- Local rename commit: `07d35a2edf69085bc2047dcf138bf1768ba8e671` (2026-09-09 22:01:47 +0800). It renamed `comparison/ezkl-main/` to `comparison/ezkl/` without changing the imported files.
- Earlier local baseline history includes `f4a2fe34fea870e7f78e47b1c627fdb529e37914` and `85f767aefb5c6ff7f7fa8da2c289c53cda23d1a9`; those commits contain prior comparison adaptation material and are not asserted to be pristine upstream identities.
- Rust toolchain: `nightly-2025-12-01` from `rust-toolchain`.

The official upstream commit/tag corresponding to the imported tree is **unresolved**:
the local Git history contains the import and rename commits above, but no
upstream ezkl commit, release tag, archive digest, or upstream Git object was
recorded for the import. This is a real provenance blocker. It must not be
replaced by a guessed commit or a fabricated digest. Formal release remains
blocked until the operator records the exact upstream source identity and
locks the resolved dependency graph.

## Current source and canonical resource snapshots

The source digest below is a historical snapshot from this record's earlier
state; it is stale after the GAT geometry, native Halo2 storage, setup/prove
gates, and runner receipt changes listed below. Do not use it to authenticate a
current artifact. `scripts/formal_common.py::source_tree_hash()` computes the
current baseline-local identity at build/run time from canonical relative
paths and file SHA-256, including `src/`, `gat_adapter/`, `.cargo/`, `vendor/`,
`scripts/*.py`, `scripts/*.sh`, `Cargo.toml`, `Cargo.lock`, `build.rs`, and
`rust-toolchain`. The resource digest below is the last recorded manifest
snapshot and must be compared against the current run receipt.

| object | identity |
| --- | --- |
| source tree hash | historical only; stale pending the next source-matched build receipt |
| resource manifest hash | `82f905232afc6f1690b4f30df7ab212ffe74602b79080a305a6af378674d80fc` |
| canonical resource files | `59` |
| canonical resource bytes | `202007204` |
| canonical manifest source snapshot | local project commit `07d35a2edf69085bc2047dcf138bf1768ba8e671`, tree `3fc5e857e8973076da3120b4695ad87267f68878` |

The canonical resource set is baseline-local and is copied from the main
project only as a read-only staging source. It contains the five dataset/cache
sets, five checkpoint bundles, and the shared quantization configuration. No
SRS, test asset, dummy/generated SRS, reduced SRS, placeholder hash, or
`qat_best_logits` file is accepted as a canonical resource.

## Cargo dependency provenance

The following identities are taken from the current official `Cargo.toml`:

- `halo2_gadgets` and `halo2_proofs`: `https://github.com/zkonduit/halo2`, branch `ac/conditional-compilation-icicle2`.
- `halo2curves`: `https://github.com/privacy-scaling-explorations/halo2curves`, revision `b753a832e92d5c86c5c997327a9cf9de86a18851`.
- `halo2_wrong_ecc` (`ecc`): `https://github.com/zkonduit/halo2wrong`, branch `ac/chunked-mv-lookup`.
- `snark-verifier`: `https://github.com/zkonduit/snark-verifier`, branch `ac/chunked-mv-lookup`.
- `halo2_solidity_verifier`: `https://github.com/zkonduit/ezkl-verifier`, branch `main`.
- `icicle-runtime` (GPU feature): `https://github.com/ingonyama-zk/icicle`, branch `emir/gate_eval_2`.
- `tract-onnx`: `https://github.com/sonos/tract/`, revision `37132e0397d0a73e5bd3a8615d932dabe44f6736`.
- `pyo3-async-runtimes`: `https://github.com/PyO3/pyo3-async-runtimes`, version `0.27.0` with the declared Git source.

Several dependencies are branch-based. A baseline-local `Cargo.lock` is now
present and is included in the dynamic source-tree identity; the branch names
remain declared source selectors, while the lockfile and source-matched build
receipt identify the resolved graph used by a particular run. CPU and GPU
builds must use the same resolved graph. The build receipt also binds the
adapter contract, source-tree hash, and resource-manifest hash. This lockfile
does not resolve the missing pristine ezkl upstream commit/tag/archive identity
described above.

## Tracked GAT/native adaptation areas

The current completion pass adds no upstream version claim and keeps GAT logic
inside the adapter boundary where possible. Its source areas are:

- `gat_adapter/geometry.rs`, `gat_adapter/geometry_primitives.rs`, and
  `gat_adapter/mod.rs`: deterministic shape-only GAT settings and geometry
  primitives used to avoid a second full dummy synthesis during admission.
- `gat_adapter/op.rs`: source-equivalent node-shared attention scoring,
  edge-routed aggregation, nonzero multi-graph relation regression, and a
  checked upper envelope for live GAT tensor buffers used by resource gates.
- `src/graph/mod.rs` and `src/tensor/var.rs`: witnessless GAT keygen circuit
  construction, complete-witness proof guard, and source-equivalent dummy
  duplicate-offset semantics.
- `vendor/halo2/halo2_proofs/src/plonk/keygen.rs`, `plonk.rs`,
  `plonk/permutation.rs`, and `poly/spill.rs`: native keygen preflight/census
  gates, bounded selector activation storage/serialization, sparse
  permutation spill backing, compact formal PK serialization, authenticated
  bounded PK restoration, and its non-overlapping read/σ-materialization
  timing breakdown.
- `src/execute.rs`, `src/pfsys/mod.rs`, `scripts/run_hybrid_formal.py`,
  `scripts/run_single_dataset.py`, `scripts/formal_common.py`,
  `scripts/collect_results.py`, and `scripts/finalization_capacity.py`:
  authenticated per-stage resource gates, compact setup PK output, bounded
  proving-key reload, session-bound timing and setup/prove receipts,
  source-derived GAT and native advice-spill envelopes, serial result
  checkpoints, and final-result validation of the receipt chain.

The dynamic source-tree digest is the source identity for new builds and
results. Existing keys, proofs, or receipts made before these changes are not
compatible with the changed source identity.

## Adapter contract boundary

The only accepted formal contract is the strict schema in
`scripts/formal_cli_contract.schema.json`. It must be generated by the Rust
adapter (`generated_by = ezkl::gat_adapter`) after the real adapter is linked
into the official CLI. The contract requires:

`gat-prepare` (adapter settings/compile/setup), `gat-witness` (adapter witness
generation), then the official `prove` and `verify` commands, native public
instance binding, full-GAT scope, external public statement and public logits,
and ICICLE CUDA provenance for GPU runs.

An absent, stale, hand-written, or non-adapter contract is a hard failure. The
build script deliberately checks resources and compiles before looking for the
contract, then checks the real adapter-generated contract before writing a
backend build receipt. This avoids a pre-build deadlock without treating a
missing adapter wire-up as success.

## Open blockers

1. Exact official ezkl upstream commit/tag/archive identity is unresolved.
2. Several Cargo dependencies use branch selectors. A baseline-local
   `Cargo.lock` is present and included in the dynamic source identity; each
   formal build receipt must still identify the exact resolved graph used.
3. This checkout currently contains `resources/native_srs/kzg26.srs` and a
   five-dataset manifest whose entries claim the native source SHA-256 was
   verified. This provenance note does not independently rehash that 8 GiB
   file; a fresh formal session reauthenticates it against the pinned native
   hash source and refreshes/provisions only through `provision_srs.py` if
   needed. Generated, dummy, reduced, and fabricated SRS material remains
   rejected.
4. A formal run requires the adapter-generated contract, identity-bound
   settings/compiled circuit/PK/VK artifacts, external public statement/Y, and
   real GPU ICICLE/CUDA evidence. Static capacity and native keygen receipts do
   not establish proof generation or independent verification; only five
   verified result cells and the final validator establish experiment success.
