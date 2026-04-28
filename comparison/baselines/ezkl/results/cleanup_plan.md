# Cleanup Plan

Created: 2026-04-28

## Current Directory Tree Summary

- Source and build metadata: `Cargo.toml`, `Cargo.lock`, `src/`, `build.rs`, `tests/`, `benches/`, `examples/`, upstream ezkl docs/support files.
- GAT comparison adapter: `cpp/gat_ezkl_adapter/` with CMake, headers, and C++ source for the strict native CSR route.
- Experiment scripts/configs: `scripts/run_gat_ezkl_baseline.py`, `scripts/collect_gat_ezkl_results.py`, `configs/ezkl_routes.json`, `configs/full_datasets.json`.
- Final result reports: `results/*.json`, `results/*.csv`, `results/*.md`.
- Run artifacts: `runs/{cora_full,citeseer_full,pubmed_full,ppi_full_formal,ogbn_arxiv_full}/` plus stale `runs/tiny_native_backend/`.
- Local generated outputs/caches: `target/`, `build_gat_adapter/`, `scripts/__pycache__/`, `tmp/`, `.cargo_local/`, `.rustup/`.

## Must Keep

- Rust project source and lock/config files needed to build the `ezkl` binary.
- `src/native_gat.rs` and original ezkl source needed by the strict native GAT route.
- `cpp/gat_ezkl_adapter/` source and CMake files needed to export full-dataset graph/checkpoint/formal-table metadata.
- `configs/full_datasets.json` and a final-only `configs/ezkl_routes.json`.
- `scripts/run_gat_ezkl_baseline.py`, `scripts/collect_gat_ezkl_results.py`, and a new formal `scripts/validate_final_results.py`.
- Final reports:
  - `results/gat_ezkl_full_results.json`
  - `results/gat_ezkl_full_results.csv`
  - `results/main_vs_ezkl_full_comparison.json`
  - `results/main_vs_ezkl_full_comparison.csv`
  - `results/summary.md`
  - `results/control_variable_audit.md`
  - `results/large_public_output_policy_review.md`
  - `results/native_full_experiment_report.md`
  - `results/native_constraint_debug_report.md` after stale wording is removed
  - `results/cleanup_plan.md`
  - `results/code_cleanliness_audit.md`
  - `results/cleanup_report.md`
- Per-dataset minimal audit artifacts for the five formal datasets:
  - `run_manifest.json`
  - `model_info.json`
  - `graph_attention_spec.json`
  - `final_logits_binding.json`
  - `formal_tables.json`
  - `formal_table_hashes.json`
  - `native_full_witness_preflight.json`
  - `keygen_memory_profile.json`
  - `prove_memory_profile.json`

## Can Delete

- Build products and caches: `target/`, `build_gat_adapter/`, `scripts/__pycache__/`, `tmp/`.
- Stale diagnostic run: `runs/tiny_native_backend/`.
- Large reproducible proof/setup/witness payloads inside formal run dirs:
  - `proof.json`, `pk.key`, `vk.key`, `pk.srs`, `vk.srs`, `witness.json`, `model.compiled`, `formal_tables.bin`, `node_features.f64.bin`, `csr_segment_layout.bin`, `input.json`, `tensor_payloads.json`, and generated ONNX/native marker files.
- Ad hoc stdout/stderr/time artifacts: `exporter.stdout.txt`, `exporter.stderr.txt`, `*.time.txt`, `prove_memory_profile_internal.json`.
- Old or redundant reports that are not final-result sources:
  - `results/implementation_plan.md`
  - `results/segment_softmax_analysis_summary.md`
  - `results/native_csr_segment_attention_report.md`
  - `results/native_compressed_csr_attention_report.md`

## Move/Archive

- No archive move is planned. Old failed/blocker artifacts will be deleted rather than archived to avoid misleading final-result readers.

## .gitignore Updates

- Keep ignoring generated proof/build payloads:
  - `target/`, `build_gat_adapter/`, CMake products, object/archive/shared-library outputs, `*.tmp`, `*.bak`, `*.old`, `*.orig`, `core*`, `perf.data`, `nohup.out`, `tiny_*`, `scratch_*`, `debug_*`, `*.log`.
- Stop ignoring the minimal formal run audit JSON files listed above so they can be committed and reviewed.
- Continue ignoring large reproducible `runs/*` payloads such as keys, proofs, SRS files, witnesses, compiled circuits, binary tables, feature tensors, and generated stdout/stderr/time dumps.

## Cleanup Reasons

- Remove old failed-route evidence from the default result surface now that all five formal datasets are verified.
- Avoid committing or retaining huge reproducible proof/key/witness artifacts.
- Keep enough small audit metadata to independently inspect strictness, table policy, public-instance binding, memory profile, and final metrics.
- Make future reviewer entry points clear: README, final results, control-variable audit, and per-dataset manifests.

## Reproduction Impact

- Source/config/script reproduction is unaffected.
- Deleted proof/key/witness/build artifacts are reproducible by running the documented build and run commands.
- Minimal run manifests and audit JSON files remain available without rerunning large proofs.

## Final Result Impact

- Final JSON/CSV/Markdown result tables are retained.
- Summary, comparison, control-variable audit, large-public-output review, and native full experiment report remain aligned to the five verified datasets.

## Strict Route Impact

- No route semantics are changed by the cleanup.
- The final route remains `ezkl_gat_native_compressed_csr_attention_full`.
- No output-binding, digest, folding, identity, subset, shard, sample, or precomputed-logits route is retained as a final benchmark route.
