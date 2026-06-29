# Optimization Profile Report

Generated: 2026-06-20

## Retained Main Optimization

| field | value |
|---|---|
| patch_name | fused_native_same_domain_opening_fold |
| selected_high_impact_route | OPENING_WITNESS_REUSE |
| touched_hot_path_files | `src/crypto/kzg.cpp` |
| target_stage | `KZG::open_batch` evaluation-basis opening witness generation |
| affected_datasets | cora, citeseer, pubmed, ppi, ogbn-arxiv |
| affected_backends | CPU and GPU rows, because KZG opening witness is CPU-backed in both current backends |
| before_benchmark_path | `results/performance_engineering/benchmarks/kzg_opening_prepatch_20260620T123254Z.csv` |
| after_benchmark_path | `results/performance_engineering/benchmarks/kzg_opening_afterpatch_fused_native_20260620T124100Z.csv` |
| before_stage_time_ms | 6495.171 / 9787.320 / 27226.775 |
| after_stage_time_ms | 5224.400 / 8516.558 / 26448.595 |
| stage_speedup | 1.243x / 1.149x / 1.029x |
| keep_or_revert_decision | keep |
| protocol_equivalence_status | unchanged labels, challenges, commitments, openings, proof schema, SRS, verifier equation |
| security_validation_status | CPU ctest passed; CUDA ctest passed; static scan passed; full security regression passed |

The optimization changes how the folded opening witness scalars are computed. It does not change what is opened or verified. Same-domain evaluation-basis terms are collected first; the per-row numerator and denominator scaling are then fused into the quotient scalar generation pass using native field operations.

## Retained Auxiliary Optimization

| field | value |
|---|---|
| patch_name | t_edge_eval_direct_backend |
| selected_high_impact_route | FUSED_QUOTIENT_COMMIT_PIPELINE |
| touched_hot_path_files | `src/protocol/quotients.cpp` |
| target_stage | `evaluate_t_edge` inside quotient materialization and verifier recomputation |
| before_benchmark_path | `results/performance_engineering/benchmarks/t_edge_eval_prepatch_sample8192_20260620T122644Z.csv` |
| after_benchmark_path | `results/performance_engineering/benchmarks/t_edge_eval_afterpatch_sample8192_20260620T122746Z.csv` |
| result | 3-4 percent sampled stage reduction across five modeled dataset shapes |
| keep_or_revert_decision | keep as auxiliary only |

## Verification

- `cmake --build build -j 12`: passed.
- `cmake --build build_cuda -j 12`: passed.
- `ctest --test-dir build --output-on-failure`: passed.
- `ctest --test-dir build_cuda --output-on-failure`: passed.
- `python3 tools/security/static_scan_no_tau.py`: passed.
- `GATZK_RUN_FULL_SECURITY_TESTS=1 ./build/gatzk_tests`: passed.

## Final Export Status

The retained optimization was validated, but final export is blocked by the OGBN LReLU static-table capacity issue documented in `results/performance_engineering/failed_runs/ogbn_lrelu_static_table_blocker_20260620.md`.

The failed `cpu/ogbn-arxiv` canary reached the official `gatzk_run` path and failed with `pair lookup query escaped static table for P_h0_L`. A formal raw range diagnostic shows that OGBN needs a contiguous LReLU table span of at least `152646788` rows under the current trace encoding, which cannot fit the current `edge=2^23` table domain and cannot be fixed by a runner/export/report change.

## Matrix Abort 20260627T042622Z

- cpu/cora: dry-run completed VERIFY_OK row exceeded near_2_2x budget

## 2026-06-27 Retained Engineering Patches

| field | value |
|---|---|
| patch_name | native_term_cached_folded_opening |
| touched_hot_path_files | `src/crypto/kzg.cpp` |
| target_stage | `KZG::open_batch` folded same-domain opening witness generation |
| before_benchmark_path | `results/performance_engineering/benchmarks/kzg_opening_before_native_terms_20260627T2012.csv` |
| after_benchmark_path | `results/performance_engineering/benchmarks/kzg_opening_after_native_terms_20260627T2012.csv` |
| before_stage_time_ms | 5223.681 / 8559.215 / 25589.918 |
| after_stage_time_ms | 5203.098 / 8489.074 / 25553.790 |
| stage_speedup | 1.004x / 1.008x / 1.001x |
| protocol_equivalence_status | unchanged labels, challenges, commitments, openings, proof schema, SRS, verifier equation |
| validation_status | CPU build passed; CUDA build passed; CPU ctest passed; CUDA ctest passed; static scan passed |

| field | value |
|---|---|
| patch_name | configured_activation_table_bound_and_same_size_srs_alias |
| touched_hot_path_files | `src/protocol/prover.cpp`; `src/crypto/kzg.cpp`; secure config files |
| target_stage | activation-table/domain planning and evaluation-basis commitment SRS selection |
| evidence | Restores old main-experiment Planetoid `range_bits=12` table policy and permits same-size Lagrange material from the same SRS manifest without generating or loading trapdoor material. |
| protocol_equivalence_status | proof relation, transcript order, commitment/opening labels, VK/SRS identity, and KZG semantics unchanged |
| validation_status | CPU build passed; CUDA build passed; CPU ctest passed; CUDA ctest passed; static scan passed |

Official timing was not run for these patches while process `2586636` was consuming CPU/GPU resources. The run remains excluded from `latest.*` until a clean full final matrix is completed.

## 2026-06-27 Fixed-Point Scale Repair

| field | value |
|---|---|
| patch_name | fixed_point_scale16_trace_binding_repair |
| touched_hot_path_files | `src/protocol/trace.cpp` |
| target_stage | fixed-point projection/attention witness materialization and binding trace construction |
| before_diagnostic_path | `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_after_table_policy_20260627.csv` |
| after_diagnostic_path | `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_current_trace_after_scale_fix_20260627.csv` |
| before_ogbn_hidden_span_raw | 112700982 |
| after_ogbn_hidden_span_raw | 440240 |
| before_ogbn_output_span_raw | 65299878 |
| after_ogbn_output_span_raw | 255080 |
| protocol_equivalence_status | verifier equation, quotient formula, proof schema, transcript order, commitment/opening labels unchanged |
| validation_status | CPU build passed; CUDA build passed; CPU ctest passed; CUDA ctest passed; static scan passed; full security regression passed |

This repair keeps fixed-point dot products at the trace scale used by the activation tables. The projection/attention binding accumulators keep the existing quotient equation and absorb deterministic rounding residuals through private `b_*` binding witness values.

## Matrix Abort 20260627T131913Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260627T132322Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T050032Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T053720Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T054051Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T055148Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T055504Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T060210Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T060729Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T061702Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T062025Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T062341Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T062836Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T063519Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T063835Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T064143Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T064458Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T100932Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T102127Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T102948Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T104004Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T104759Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T105644Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T110616Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260628T112210Z

- cpu/cora: row exited with code 1

## Matrix Abort 20260628T112823Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T060142Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T060754Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T061735Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T062905Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T063907Z

- cpu/cora: row exited with code 124

## Matrix Abort 20260629T065542Z

- cpu/cora: row exited with code 124
