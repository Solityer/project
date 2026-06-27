# Initial Checkpoint

Created: 2026-06-27T12:17:58+08:00

Scope: zkGAT secure main experiment performance optimization. `comparison/**` is intentionally ignored for this task.

## Git State

HEAD:

```text
aa263e8284d4b0b680ac8fb8725252c59b44a915
```

Status before this task's edits:

```text
## main...origin/main
 M CMakeLists.txt
 M results/paper_secure/freeze_identity.json
 M scripts/run_secure_main_experiments.py
 M src/algebra/cuda_backend.cu
 M src/crypto/kzg.cpp
 M src/model/gat.cpp
 M src/protocol/prover.cpp
 M src/protocol/quotients.cpp
?? results/paper_secure/baseline_comparison_report.md
?? results/paper_secure/cleanup_audit_report.md
?? results/paper_secure/gpu_backend_report.md
?? results/paper_secure/negative_optimization_report.md
?? results/paper_secure/optimization_profile_report.md
?? results/performance_engineering/benchmarks/
?? results/performance_engineering/diagnostics/
?? results/performance_engineering/failed_runs/
?? results/performance_engineering/five_dataset_bottleneck_matrix.md
?? results/performance_engineering/performance_path_equivalence_audit.md
?? tools/benchmarks/
```

Diff stat before this task's edits:

```text
 CMakeLists.txt                            |    16 +
 results/paper_secure/freeze_identity.json |    14 +-
 scripts/run_secure_main_experiments.py    |     3 +-
 src/algebra/cuda_backend.cu               |  1396 ++--
 src/crypto/kzg.cpp                        |    52 +-
 src/model/gat.cpp                         |    12 +-
 src/protocol/prover.cpp                   | 10769 +++++++++++++++-------------
 src/protocol/quotients.cpp                |    19 +-
 8 files changed, 6618 insertions(+), 5663 deletions(-)
```

## Hashes

Source tree hash, limited to source/script/config/doc/test files and excluding `.git`, `comparison`, build outputs, virtualenv, `data`, `runs`, `results`, and large artifacts:

```text
source_tree_hash_source_docs_configs_only 1a6d9fd48088d357ba4b2811a7cf8a5417c231706a677159b48ee353e893e303
source_file_count 92
```

Binary and runner hashes:

```text
f5305f31937ad8e977b9fe05a3b462a6eac1c458d2f14f88d26d360613ac9990  build/gatzk_run
dc93766f39a6b44588ece674c84adff61abd1304905916df54c0eb882987de49  build/gatzk_tests
41046a21a056c60af3d4edbac6a8a2f6aade21c74852bd6b6fbdd884047f297a  build_cuda/gatzk_run
b5d3cf8782dce9c8a089d6782b34bff0e197a59be8c5a33b8e039ff0e7df3d61  build_cuda/gatzk_tests
56671945d5e71f2c123cae16a5ea9f89f2c96e0e8816a42c355ebe3e615c7905  scripts/run_secure_main_experiments.py
6eb51f9d4a0e79fba006cde87fe9237216bc584fef8c40e3cbc138e75d2bec50  scripts/export_paper_secure.py
```

SRS manifest and VK hashes:

```text
f5ea7dff25203f129796a2665aa02b2ba33bebb651447c5d5cccb9363a318871  artifacts/srs/zkgat_secure_main_srs_v1/srs_manifest.json
5af8f41e59c03b80adf53de1fb603c69e24aaa9abcd50ecfea20d69227eb5b10  artifacts/srs/zkgat_secure_main_srs_v1/vk.bin
```

## Existing Final Exports

```text
results/paper_secure/latest.csv size 6482 sha256 b507a46d524f3a20986615850be627e27370cff7ccd85bdd30cd1d4e7ceb2ba5
results/paper_secure/latest.json size 15507 sha256 0fa946f0b4870bd99bf550bbf60aa927ffcce3007f879e6a9c4289b03c12d5bb
latest.json top-level type: list
```

## Existing Performance Engineering Files

```text
results/performance_engineering/benchmarks/kzg_opening_afterpatch_20260620T123640Z.csv size 698
results/performance_engineering/benchmarks/kzg_opening_afterpatch_20260620T123640Z.meta size 360
results/performance_engineering/benchmarks/kzg_opening_afterpatch_fused_native_20260620T124100Z.csv size 698
results/performance_engineering/benchmarks/kzg_opening_afterpatch_fused_native_20260620T124100Z.meta size 360
results/performance_engineering/benchmarks/kzg_opening_prepatch_20260620T123254Z.csv size 698
results/performance_engineering/benchmarks/kzg_opening_prepatch_20260620T123254Z.meta size 338
results/performance_engineering/benchmarks/msm_stage_prepatch_cpu_route_bench_20260620T121449Z.csv size 428
results/performance_engineering/benchmarks/msm_stage_prepatch_cpu_route_bench_20260620T121449Z.meta size 26
results/performance_engineering/benchmarks/t_edge_eval_afterpatch_sample8192_20260620T122746Z.csv size 900
results/performance_engineering/benchmarks/t_edge_eval_afterpatch_sample8192_20260620T122746Z.meta size 357
results/performance_engineering/benchmarks/t_edge_eval_prepatch_sample8192_20260620T122644Z.csv size 902
results/performance_engineering/benchmarks/t_edge_eval_prepatch_sample8192_20260620T122644Z.meta size 327
results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_after_table_failure_20260620.csv size 193
results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_20260620.csv size 222
results/performance_engineering/failed_runs/ogbn_cpu_initial_final_runner_lrelu_failure_20260620T172559Z.log size 493
results/performance_engineering/failed_runs/ogbn_cpu_lrelu_table_after_negative_rebalance_20260620T175945Z.log size 353
results/performance_engineering/failed_runs/ogbn_lrelu_range_diag_formal_raw_after_failure_20260620.csv size 222
results/performance_engineering/failed_runs/ogbn_lrelu_static_table_blocker_20260620.md size 3617
results/performance_engineering/failed_runs/t_edge_eval_prepatch_20260620T122150Z.interrupted.csv size 0
results/performance_engineering/failed_runs/t_edge_eval_prepatch_20260620T122150Z.interrupted.meta size 260
results/performance_engineering/failed_runs/t_edge_eval_prepatch_20260620T122150Z.interrupted.reason size 115
results/performance_engineering/five_dataset_bottleneck_matrix.md size 4556
results/performance_engineering/gpu_backend_gap_report.md size 3389
results/performance_engineering/order_of_magnitude_redesign_plan.md size 14953
results/performance_engineering/performance_budget_report.md size 11673
results/performance_engineering/performance_path_equivalence_audit.md size 4167
results/performance_engineering/performance_readiness_report.md size 5503
results/performance_engineering/performance_redesign_plan.md size 5722
```

## Visible Build, Run, Result State

Build trees exist at `build/` and `build_cuda/`; both contain `gatzk_run`, `gatzk_tests`, `gatzk_srs_setup`, stage benchmark binaries, and CMake metadata.

Recent secure run directories visible before this task:

```text
runs/cora_secure/paper_secure_final
runs/citeseer_secure/paper_secure_final
runs/pubmed_secure/paper_secure_final
runs/ppi_secure/paper_secure_final
runs/secure_main_runner/archives
runs/secure_main_runner/logs
```

SRS directory summary for `artifacts/srs/zkgat_secure_main_srs_v1`:

```text
srs_manifest.json 4120 bytes
vk.bin 220 bytes
pk_coef.bin 9663676440 bytes
pk_lagrange_FH.bin 4831838232 bytes
pk_lagrange_edge.bin 1207959576 bytes
```

Dataset file summary:

```text
cora Planetoid files present
citeseer Planetoid files present
pubmed Planetoid files present
ppi JSON/NPY/walk files present
ogbn-arxiv raw/processed/split directories present
```

## Comparison Ignored Status

`comparison/**` was not read for task decisions and is excluded from subsequent source-tree hashes and file operations.

## Checkpoint Notes

The worktree was already dirty before this task began. These existing changes are treated as user/local state and must not be reverted unless they directly conflict with this task and the revert scope is explicitly attributable to this task's own edits.

## Current Session Checkpoint

Updated: 2026-06-27T17:25:30+08:00

Required invariant read:

```text
MAIN_EXPERIMENT_INVARIANTS.md read: yes
MAIN_EXPERIMENT_INVARIANTS.md sha256: 184aee9817dbd1523a1c76de7310f292fadc8faf636a5d1035b4d1825872eeb2
```

Current HEAD:

```text
aa263e8284d4b0b680ac8fb8725252c59b44a915
```

Current git status includes local main-experiment edits plus unrelated `comparison/**` changes already present in the worktree. `comparison/**` remains ignored and untouched for this main-experiment pass.

Current non-comparison diff stat:

```text
 CMakeLists.txt                            |    16 +
 results/paper_secure/freeze_identity.json |    14 +-
 scripts/export_paper_secure.py            |    39 +-
 scripts/run_secure_main_experiments.py    |   268 +-
 src/algebra/cuda_backend.cu               |  1396 ++--
 src/crypto/kzg.cpp                        |    52 +-
 src/model/gat.cpp                         |    12 +-
 src/protocol/prover.cpp                   | 10769 +++++++++++++++-------------
 src/protocol/quotients.cpp                |    19 +-
 9 files changed, 6869 insertions(+), 5716 deletions(-)
```

Current source and runner hashes:

```text
source_tree_hash: 29d554b6d8bcad44dc41566ef8f9c32fcbb846f804eb2ec2ac2fa85f6dd33aa8
source_file_count: 62
runner_hash: b901308870f8a860fbb99b36e52761d6e749a8f481829bdb6aa89c25ee5c09e3
cpu_binary_hash: f5305f31937ad8e977b9fe05a3b462a6eac1c458d2f14f88d26d360613ac9990
gpu_binary_hash: d3b1b2979d630a538c8a98f1944886b158a167d26638397a6af51fe97810a592
```

Existing final export hashes:

```text
results/paper_secure/latest.csv sha256: b507a46d524f3a20986615850be627e27370cff7ccd85bdd30cd1d4e7ceb2ba5
results/paper_secure/latest.json sha256: 0fa946f0b4870bd99bf550bbf60aa927ffcce3007f879e6a9c4289b03c12d5bb
latest.csv row count observed: 10
latest.json top-level list length observed: 10
```

Existing `results/performance_engineering/**` state was inventoried by path, size, and selected report contents only. Prior OGBN LReLU/domain failure reports exist, but are not treated as a final stop condition for this pass.

Process isolation:

```text
comparison/** ignored and untouched status: yes
comparison process detected: no
comparison process left untouched: yes
heavy non-comparison CPU process detected: yes
official timing permitted under current resources: no
```

Resource note: a separate high-CPU Python process and editor/Codex processes were visible at checkpoint time. Official final-matrix timing must wait until this interference clears or be recorded only as diagnostic.

## Current Session Checkpoint - 2026-06-27T20:12:35+08:00

Required invariant read:

```text
MAIN_EXPERIMENT_INVARIANTS.md read: yes
MAIN_EXPERIMENT_INVARIANTS.md sha256: 184aee9817dbd1523a1c76de7310f292fadc8faf636a5d1035b4d1825872eeb2
```

Current HEAD:

```text
aa263e8284d4b0b680ac8fb8725252c59b44a915
```

Current non-comparison git status:

```text
 M CMakeLists.txt
 M results/paper_secure/freeze_identity.json
 M scripts/export_paper_secure.py
 M scripts/run_secure_main_experiments.py
 M src/algebra/cuda_backend.cu
 M src/crypto/kzg.cpp
 M src/model/gat.cpp
 M src/protocol/prover.cpp
 M src/protocol/quotients.cpp
?? MAIN_EXPERIMENT_INVARIANTS.md
?? results/paper_secure/baseline_comparison_report.md
?? results/paper_secure/cleanup_audit_report.md
?? results/paper_secure/gpu_backend_report.md
?? results/paper_secure/negative_optimization_report.md
?? results/paper_secure/optimization_profile_report.md
?? results/performance_engineering/baseline_budget_2x.json
?? results/performance_engineering/baseline_budget_2x.md
?? results/performance_engineering/benchmarks/
?? results/performance_engineering/dataset_integrity_audit.json
?? results/performance_engineering/dataset_integrity_audit.md
?? results/performance_engineering/diagnostics/
?? results/performance_engineering/failed_runs/
?? results/performance_engineering/final_matrix_preflight_budget_check.json
?? results/performance_engineering/final_matrix_preflight_budget_check.md
?? results/performance_engineering/five_dataset_bottleneck_matrix.md
?? results/performance_engineering/initial_checkpoint.md
?? results/performance_engineering/patch_audit_budget_control_20260627.md
?? results/performance_engineering/performance_path_equivalence_audit.md
?? results/performance_engineering/srs_pk_vk_integrity_audit.json
?? results/performance_engineering/srs_pk_vk_integrity_audit.md
?? tools/benchmarks/
```

Current non-comparison diff stat:

```text
 CMakeLists.txt                            |    16 +
 results/paper_secure/freeze_identity.json |    14 +-
 scripts/export_paper_secure.py            |    39 +-
 scripts/run_secure_main_experiments.py    |   268 +-
 src/algebra/cuda_backend.cu               |  1396 ++--
 src/crypto/kzg.cpp                        |    52 +-
 src/model/gat.cpp                         |    12 +-
 src/protocol/prover.cpp                   | 10769 +++++++++++++++-------------
 src/protocol/quotients.cpp                |    19 +-
 9 files changed, 6869 insertions(+), 5716 deletions(-)
```

Current source, runner, binary, and existing export hashes:

```text
source_tree_hash: e5f5552a4f2ca43959c05de7ac3bb8c99eaebbe1dc3941da1b7c61982225a337
source_file_count: 83
runner_hash: b901308870f8a860fbb99b36e52761d6e749a8f481829bdb6aa89c25ee5c09e3
exporter_hash: c24e4fb066c61214825ca204da6ed84447eba1f3fccc7016ec57e0824465b579
cpu_binary_hash: f5305f31937ad8e977b9fe05a3b462a6eac1c458d2f14f88d26d360613ac9990
cpu_tests_hash: dc93766f39a6b44588ece674c84adff61abd1304905916df54c0eb882987de49
gpu_binary_hash: d3b1b2979d630a538c8a98f1944886b158a167d26638397a6af51fe97810a592
gpu_tests_hash: 274e28e0d13c31dd95404a75d01bdfd742eb346be1230d41bd48a405abc5d0d9
results/paper_secure/latest.csv sha256: b507a46d524f3a20986615850be627e27370cff7ccd85bdd30cd1d4e7ceb2ba5
results/paper_secure/latest.json sha256: 0fa946f0b4870bd99bf550bbf60aa927ffcce3007f879e6a9c4289b03c12d5bb
```

Existing `results/performance_engineering/**`: present and inventoried earlier in this file. This pass will append/update required main-experiment reports without treating previous diagnostics as final results.

Process isolation:

```text
comparison/** untouched status: yes
comparison process detected: no
comparison process left untouched: yes
heavy non-comparison CPU process detected: yes
official timing permitted under current resources: no
```

Resource note: a non-main-experiment `run_token_ablation_matrix.py` Python process was using high CPU at this checkpoint. Official final-matrix rows must wait for resource interference to clear; code inspection, builds, focused tests, and non-official benchmarks may proceed.
