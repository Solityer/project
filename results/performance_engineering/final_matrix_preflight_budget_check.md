# Final Matrix Preflight Budget Check

Status: `DO_NOT_ENTER_FINAL_MATRIX`

The 10-row final matrix is forbidden. Existing same-row diagnostics are far over the near-2.2x budget for Cora/Citeseer/PubMed/PPI, GPU evidence is CUDA-hotspot only with KZG/quotient/opening fallback, and OGBN has a fixed-domain LReLU table conflict.

| order | backend | dataset | expected_core_ms | near_2_2x_ms | status | allowed | reason |
| ---: | --- | --- | ---: | ---: | --- | --- | --- |
| 1 | cpu | cora | 959300.554 | 4924.205 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 2 | gpu | cora | 959082.874 | 3604.242 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 3 | cpu | citeseer | 2576829.793 | 4365.242 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 4 | gpu | citeseer | 2568287.975 | 4396.911 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 5 | cpu | pubmed | 3139857.140 | 19447.606 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 6 | gpu | pubmed | 3148782.811 | 9530.147 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 7 | cpu | ppi | 2503629.240 | 24805.277 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 8 | gpu | ppi | 2467419.978 | 14202.613 | over_near_2_2x | no | current proof_core_total_ms exceeds near_2_2x and/or evidence hash is not frozen-current |
| 9 | cpu | ogbn-arxiv | 0.000 | 185703.137 | formal_constraint_conflict | no | required contiguous LReLU table span needs 2^28 edge-domain/SRS while fixed edge SRS is 2^23 |
| 10 | gpu | ogbn-arxiv | 0.000 | 121032.263 | formal_constraint_conflict | no | required contiguous LReLU table span needs 2^28 edge-domain/SRS while fixed edge SRS is 2^23 |

## Frozen Hash Check

- current_source_tree_hash: `29d554b6d8bcad44dc41566ef8f9c32fcbb846f804eb2ec2ac2fa85f6dd33aa8`
- current_cpu_binary_hash: `f5305f31937ad8e977b9fe05a3b462a6eac1c458d2f14f88d26d360613ac9990`
- current_gpu_binary_hash: `d3b1b2979d630a538c8a98f1944886b158a167d26638397a6af51fe97810a592`
- current_runner_hash: `b901308870f8a860fbb99b36e52761d6e749a8f481829bdb6aa89c25ee5c09e3`

## Evidence Paths

- cpu/cora: `runs/cora_secure/paper_secure_final/cpu/warm/run_manifest.json` (same-row official diagnostic)
- gpu/cora: `runs/cora_secure/paper_secure_final/gpu/warm/run_manifest.json` (same-row official diagnostic)
- cpu/citeseer: `runs/citeseer_secure/paper_secure_final/cpu/warm/run_manifest.json` (same-row official diagnostic)
- gpu/citeseer: `runs/citeseer_secure/paper_secure_final/gpu/warm/run_manifest.json` (same-row official diagnostic)
- cpu/pubmed: `runs/pubmed_secure/paper_secure_final/cpu/warm/run_manifest.json` (same-row official diagnostic)
- gpu/pubmed: `runs/pubmed_secure/paper_secure_final/gpu/warm/run_manifest.json` (same-row official diagnostic)
- cpu/ppi: `runs/ppi_secure/paper_secure_final/cpu/warm/run_manifest.json` (same-row official diagnostic)
- gpu/ppi: `runs/ppi_secure/paper_secure_final/gpu/warm/run_manifest.json` (same-row official diagnostic)
- cpu/ogbn-arxiv: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_20260627.csv` (formal raw full-dataset diagnostic plus failed official row logs)
- gpu/ogbn-arxiv: `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_formal_raw_20260627.csv` (formal raw full-dataset diagnostic plus failed official row logs)

## Gate Result

- all 10 rows allowed_to_enter_final_matrix: no
- all 10 rows expected <= 2.2x: no
- high-risk rows had real diagnostic/full-stage evidence: no for OGBN final row; diagnostic proves conflict instead
- no projection-only high-risk row: yes
- preflight diagnostic cumulative cap respected: yes; no final matrix was started
