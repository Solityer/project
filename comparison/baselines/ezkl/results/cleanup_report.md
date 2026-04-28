# Cleanup Report

## Metadata

- Cleanup time: 2026-04-28 Asia/Shanghai
- Cleanup base commit: `6ffe0a8`
- Scope: `/home/pzh/project/comparison/baselines/ezkl`
- Main repository paths outside `comparison/baselines/ezkl/`: not modified

## Deleted Files And Directories

- Build products: `target/`, `build_gat_adapter/`, `scripts/__pycache__/`, `tmp/`
- Stale run directory: `runs/tiny_native_backend/`
- Old or redundant reports:
  - `results/implementation_plan.md`
  - `results/segment_softmax_analysis_summary.md`
  - `results/native_csr_segment_attention_report.md`
  - `results/native_compressed_csr_attention_report.md`
- Reproducible large per-run payloads from each formal dataset run directory:
  - `proof.json`
  - `pk.key`
  - `vk.key`
  - `pk.srs`
  - `vk.srs`
  - `witness.json`
  - `model.compiled`
  - `formal_tables.bin`
  - `csr_segment_layout.bin`
  - `node_features.f64.bin`
  - generated ONNX/native marker files
  - stdout/stderr/time dumps
  - intermediate manifests such as `compressed_row_estimate.json`, `metadata.json`, `input.json`, and tensor payload indexes

## Retained Core Files

- Source/config/scripts:
  - `Cargo.toml`, `Cargo.lock`, `src/`
  - `cpp/gat_ezkl_adapter/`
  - `configs/full_datasets.json`
  - `configs/ezkl_routes.json`
  - `scripts/run_gat_ezkl_baseline.py`
  - `scripts/collect_gat_ezkl_results.py`
  - `scripts/validate_final_results.py`
- Final result files:
  - `results/gat_ezkl_full_results.json`
  - `results/gat_ezkl_full_results.csv`
  - `results/main_vs_ezkl_full_comparison.json`
  - `results/main_vs_ezkl_full_comparison.csv`
  - `results/summary.md`
  - `results/control_variable_audit.md`
  - `results/large_public_output_policy_review.md`
  - `results/native_full_experiment_report.md`
  - `results/native_constraint_debug_report.md`
  - `results/code_cleanliness_audit.md`
  - `results/cleanup_plan.md`
  - `results/cleanup_report.md`
- Per-dataset audit artifacts retained under each `runs/<dataset>/`:
  - `run_manifest.json`
  - `model_info.json`
  - `graph_attention_spec.json`
  - `final_logits_binding.json`
  - `formal_tables.json`
  - `formal_table_hashes.json`
  - `native_full_witness_preflight.json`
  - `keygen_memory_profile.json`
  - `prove_memory_profile.json`

## Moved Or Archived Files

- None. Stale/failed/debug artifacts were deleted rather than archived to keep the final result surface unambiguous.

## .gitignore Updates

- Added/normalized ignores for `target/`, `build_gat_adapter/`, CMake products, object/archive/shared-library files, `*.tmp`, `*.bak`, `*.old`, `*.orig`, `core*`, `perf.data`, `nohup.out`, `tiny_*`, `scratch_*`, `debug_*`, and `*.log`.
- Kept large proof/setup/witness payloads ignored.
- Documented that minimal `runs/<dataset>/*.json` audit artifacts are intentionally force-added because the project-root `.gitignore` ignores all `runs/` directories.

## Large File Scan

Command:

```bash
find . -type f -size +50M -printf '%s %p\n' | sort -nr
```

Remaining large files are ignored local toolchain/cache files only:

- `.rustup/toolchains/.../libLLVM.so.21.1-rust-1.93.0-nightly`
- `.rustup/toolchains/.../librustc_driver-769d1396d4f2b9af.so`
- `.rustup/toolchains/.../libcore-f07b8928441ca275.rlib`
- `.cargo_local/git/.../halo2...pack`
- `.cargo_local/git/.../libicicle_backend_cuda_curve_bn254.so`

No large proof, key, SRS, witness, compiled circuit, feature tensor, or formal table binary remains in the final run artifact surface.

## Grep Cleanliness Check

Command:

```bash
grep -R "TODO\\|FIXME\\|HACK\\|XXX\\|stub\\|dummy\\|fake\\|placeholder\\|temporary\\|debug-only\\|not implemented\\|unimplemented!\\|todo!\\|panic!(\"TODO" -n src cpp scripts configs || true
```

Remaining hits are upstream ezkl generic infrastructure only and are listed in `results/code_cleanliness_audit.md`. There are no hits in `cpp/gat_ezkl_adapter/`, `scripts/`, `configs/`, or the strict native GAT route implementation.

## Result Validation

- `python3 scripts/validate_final_results.py`: `final_result_validation=OK`
- `results/gat_ezkl_full_results.json`: 5 rows
- All rows: `status=verified`
- All rows: `route_id=ezkl_gat_native_compressed_csr_attention_full`
- All rows: `strict_public_input_equivalent=true`
- All rows: `full_gat_forward_in_ezkl_proof=true`
- All rows: `final_logits_derivation_complete=true`
- All rows: `no_precomputed_logits_bypass=true`
- All four metrics are present for every dataset

## Build And Test Validation

- Rust test:
  - Command: `cargo test native_gat --no-default-features --features ezkl,no-banner`
  - Toolchain used: local `.rustup/toolchains/nightly-2025-12-01-aarch64-unknown-linux-gnu`
  - Result: passed, 23 tests passed
- Rust build:
  - Command: `cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner`
  - Toolchain used: local `.rustup/toolchains/nightly-2025-12-01-aarch64-unknown-linux-gnu`
  - Result: passed
- C++ build:
  - Command: `cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release && cmake --build build_gat_adapter -j`
  - Result: passed
- Python syntax:
  - Command: `python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py`
  - Result: passed
- Build outputs created during validation were deleted afterward.

## Control Variable Audit

- `results/control_variable_audit.md` remains present and aligned to the final route/results.
- No toy, smoke, subset, sample, shard, partial, debug dataset, scaled graph, output-binding replacement, digest-only output, folding, identity route, or precomputed-logits bypass appears in the final verified route.

## Repository Pollution Check

- All cleanup edits are inside `comparison/baselines/ezkl/`.
- No files under `/home/pzh/project/src`, `/home/pzh/project/include`, `/home/pzh/project/configs`, `/home/pzh/project/data`, `/home/pzh/project/artifacts`, `/home/pzh/project/runs`, `/home/pzh/project/scripts`, `/home/pzh/project/README.md`, or `/home/pzh/project/GAT-ZKML-多层多头.md` were modified.

## Git Status Before Final Commit

All dirty paths are expected cleanup changes under `comparison/baselines/ezkl/`. The final cleanup commit is expected to make `git status --short` empty.
