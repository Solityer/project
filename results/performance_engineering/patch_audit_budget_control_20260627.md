# Patch Audit: Budget Control And Formal Conflict Evidence

patch_id: `budget_control_and_preflight_conflict_20260627`

## Touched Files

- `scripts/run_secure_main_experiments.py`
- `scripts/export_paper_secure.py`
- `results/performance_engineering/baseline_budget_2x.json`
- `results/performance_engineering/baseline_budget_2x.md`
- `results/performance_engineering/dataset_integrity_audit.json`
- `results/performance_engineering/dataset_integrity_audit.md`
- `results/performance_engineering/srs_pk_vk_integrity_audit.json`
- `results/performance_engineering/srs_pk_vk_integrity_audit.md`
- `results/performance_engineering/final_matrix_preflight_budget_check.json`
- `results/performance_engineering/final_matrix_preflight_budget_check.md`
- `results/performance_engineering/failed_runs/runner_budget_enforcement_dry_run.md`
- `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_20260627.csv`

## Route

selected_high_impact_route: `RUNNER_BUDGET_ENFORCEMENT_AND_FORMAL_CONFLICT_AUDIT`

This patch is control-plane only. The retained shared hot-path optimization already present in the worktree is `OPENING_WITNESS_REUSE`; this patch prevents invalid final-matrix entry and stale/over-budget latest export.

## Hashes

- before source_tree_hash: `1a6d9fd48088d357ba4b2811a7cf8a5417c231706a677159b48ee353e893e303`
- after source_tree_hash: `29d554b6d8bcad44dc41566ef8f9c32fcbb846f804eb2ec2ac2fa85f6dd33aa8`
- before CPU binary_hash: `f5305f31937ad8e977b9fe05a3b462a6eac1c458d2f14f88d26d360613ac9990`
- after CPU binary_hash: `f5305f31937ad8e977b9fe05a3b462a6eac1c458d2f14f88d26d360613ac9990`
- before GPU binary_hash: `41046a21a056c60af3d4edbac6a8a2f6aade21c74852bd6b6fbdd884047f297a`
- after GPU binary_hash: `d3b1b2979d630a538c8a98f1944886b158a167d26638397a6af51fe97810a592`
- before runner_hash: `56671945d5e71f2c123cae16a5ea9f89f2c96e0e8816a42c355ebe3e615c7905`
- after runner_hash: `b901308870f8a860fbb99b36e52761d6e749a8f481829bdb6aa89c25ee5c09e3`
- before exporter_hash: `6eb51f9d4a0e79fba006cde87fe9237216bc584fef8c40e3cbc138e75d2bec50`
- after exporter_hash: `c24e4fb066c61214825ca204da6ed84447eba1f3fccc7016ec57e0824465b579`

## Before/After Benchmark Paths

- before benchmark path: `results/performance_engineering/benchmarks/kzg_opening_prepatch_20260620T123254Z.csv`
- after benchmark path: `results/performance_engineering/benchmarks/kzg_opening_afterpatch_fused_native_20260620T124100Z.csv`
- runner dry-run path: `results/performance_engineering/failed_runs/runner_budget_enforcement_dry_run.md`

## Speedup / Gate Effect

- shared hot-path speedup evidence remains the existing KZG opening benchmark evidence.
- runner gate effect: timeout row abort passed; completed-over-budget abort passed.
- final-matrix speedup claim: none; final matrix is forbidden by preflight.

## Proof-Equivalence Audit

| item | changed | note |
| --- | --- | --- |
| relation | no | runner/exporter/report changes only |
| scheme | no | KZG/protocol code not changed in this patch |
| verifier equation | no | verifier code not changed in this patch |
| proof schema | no | schema code not changed in this patch |
| transcript/order | no | transcript/challenge code not changed in this patch |
| label/order | no | label-order code not changed in this patch |
| metadata statement fields | no | budget fields are runner/export metadata, not proof public metadata |
| public/private boundary | no | no witness/public boundary code changed |
| KZG semantics | no | no KZG code changed in this patch |
| proof_bytes_changed | no final proof generated | final matrix not run |
| proof_bytes_changed_reason | n/a | no final proof generated |
| proof_size before/after | n/a | no final proof generated |
| verifier result | tests passed | see validation logs in command output and ctest |
| tamper result | full security regression passed | `GATZK_RUN_FULL_SECURITY_TESTS=1 build/gatzk_tests` |
| reference consistency result | full security regression passed | same command |
| proof payload audit result | full security regression passed | same command |

## Keep/Revert Decision

keep_or_revert_decision: `keep`

Reason: the patch adds required hard budget enforcement and prevents invalid latest export. It does not change proof semantics.
