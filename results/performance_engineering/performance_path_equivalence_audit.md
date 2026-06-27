# Performance Path Equivalence Audit

Generated: 2026-06-27

## Scope

Only `/home/pzh/project` local files were used. `comparison/**` was ignored.

## Build And Test State

| item | finding |
| --- | --- |
| CPU build | `cmake --build build -j 12` passed. |
| CUDA build | `cmake --build build_cuda -j 12` passed with existing NVCC warnings. |
| CPU ctest | `ctest --test-dir build --output-on-failure` passed. |
| CUDA ctest | `ctest --test-dir build_cuda --output-on-failure` passed. |
| static scan | `python3 tools/security/static_scan_no_tau.py` passed. |
| full security regression | `GATZK_RUN_FULL_SECURITY_TESTS=1 build/gatzk_tests` passed. |

## Frozen Hashes

- source_tree_hash: `29d554b6d8bcad44dc41566ef8f9c32fcbb846f804eb2ec2ac2fa85f6dd33aa8`
- cpu_binary_hash: `f5305f31937ad8e977b9fe05a3b462a6eac1c458d2f14f88d26d360613ac9990`
- gpu_binary_hash: `d3b1b2979d630a538c8a98f1944886b158a167d26638397a6af51fe97810a592`
- runner_hash: `b901308870f8a860fbb99b36e52761d6e749a8f481829bdb6aa89c25ee5c09e3`
- exporter_hash: `c24e4fb066c61214825ca204da6ed84447eba1f3fccc7016ec57e0824465b579`

## Path Findings

- Release/CUDA Release builds are present.
- Audit/reference/tamper timings are recorded separately from proof-core components.
- Runner now enforces per-row timeout and completed-over-budget abort using `baseline_budget_2x.json`.
- Existing Cora/Citeseer/PubMed/PPI diagnostics are full secure paths but far over near-2.2x and are not final.
- GPU diagnostics record CUDA hotspot acceleration only; KZG MSM, pairing, quotient materialization, and opening witness remain CPU fallback stages.
- OGBN full-dataset LReLU diagnostic shows fixed contiguous table span `152646788`, requiring edge/table domain `2^28`, while fixed SRS edge domain is `2^23`.
- Diagnostic-only fixed-point rescale probe (`results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_rescale_products_20260627T095943Z.csv`) reduces OGBN raw spans to `440240` hidden and `255080` output, which fits the current large-graph table policy. This was not retained because it would require changing binding quotient/verifier equations.

## Protocol Equivalence

No retained change in this pass modifies the GAT model, zkGAT relation, proof scheme, verifier equation, proof schema, transcript order, commitment/opening label order, public metadata statement fields, public/private boundary, or KZG semantics. Budget fields are experiment-control metadata. The fixed-point rescale probe is diagnostic-only and is excluded from the proving path.

## Matrix Decision

Final matrix must not start. Preflight status is `DO_NOT_ENTER_FINAL_MATRIX`; see `results/performance_engineering/final_matrix_preflight_budget_check.json`.

## 2026-06-27 Continuation

- Retained table/domain policy repair: `src/protocol/prover.cpp` now sizes non-large activation tables from configured `range_bits` instead of a fixed `16384` lower bound, restoring the old main-experiment Planetoid table/domain scale.
- Retained same-identity SRS material reuse: `src/crypto/kzg.cpp` can use Lagrange material with the same domain size from the loaded SRS manifest when a semantic domain alias is missing. This does not introduce tau/runtime SRS material and keeps the same VK/SRS identity.
- Secure configs restored to old main-experiment range policy where applicable: `configs/cora_secure.cfg`, `configs/citeseer_secure.cfg`, `configs/pubmed_secure.cfg`, and `configs/ogbn_arxiv_secure.cfg` use `range_bits=12`; PPI remains `range_bits=14`, matching its full formal config.
- CPU build passed: `cmake --build build -j 12 --target gatzk_run gatzk_tests gatzk_lrelu_range_diag`.
- CUDA build passed: `cmake --build build_cuda -j 12 --target gatzk_run gatzk_tests gatzk_lrelu_range_diag`.
- CPU ctest passed: `ctest --test-dir build --output-on-failure`.
- CUDA ctest passed: `ctest --test-dir build_cuda --output-on-failure`.
- Static scan passed: `python3 tools/security/static_scan_no_tau.py`.
- OGBN raw diagnostic remains over the fixed contiguous LReLU table span: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_after_table_policy_20260627.csv`.
- OGBN scale-consistent diagnostic remains within current large-graph table capacity: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_rescale_products_after_table_policy_20260627.csv`.
- Official timing/final row execution paused: process `2586636` (`.venv/bin/python -B run_token_ablation_matrix.py ...`) was observed at about `88.8%` CPU and `346 MiB` GPU memory, so official timing would be resource-polluted under the main invariants.

## 2026-06-27 Scale Fix Validation

- Retained fixed-point scale repair: `src/protocol/trace.cpp` now rescales each raw fixed-point dot product back to scale 16 and closes projection/attention binding traces by absorbing deterministic rounding residuals into private `b_*` binding witness columns.
- Verifier equation, quotient formula, proof schema, transcript order, and commitment/opening labels were not changed by this scale repair.
- OGBN current-trace range evidence after the scale repair: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_current_trace_after_scale_fix_20260627.csv`, with hidden span `440240` and output span `255080`.
- CPU build passed after scale repair: `cmake --build build -j 12 --target gatzk_run gatzk_tests gatzk_lrelu_range_diag`.
- CUDA build passed after scale repair: `cmake --build build_cuda -j 12 --target gatzk_run gatzk_tests gatzk_lrelu_range_diag`.
- CPU ctest passed after scale repair: `ctest --test-dir build --output-on-failure`.
- CUDA ctest passed after scale repair: `ctest --test-dir build_cuda --output-on-failure`.
- Static scan passed after scale repair: `python3 tools/security/static_scan_no_tau.py`.
- Full security regression passed after scale repair: `GATZK_RUN_FULL_SECURITY_TESTS=1 build/gatzk_tests`.
- Current binary hashes after scale repair: CPU `190fd2ad856c3a8b1e34311f448d09b67662348ad5c067f3d630556d4b7e9196`; CUDA `4c3f4b2d5e7a8f3ba3e1fa14b5553edbc55f776a8afa80c4ee7080271dd54937`.
- Official timing/final matrix remains paused because process `2586636` continued to consume about `89%` CPU and `346 MiB` GPU memory after validation.
