# zkDL Full GAT Baseline Summary

Generated at: `2026-05-05T08:22:53.443197+00:00`

Correct baseline directory: `comparison/baselines/zkdl`.

The full-dataset statement is attempted for `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, and `ogbn_arxiv_full`. No row is marked verified unless a real zkDL proof and verifier acceptance path succeeds.

## CPU

| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | --- | --- | --- |
| cora_full | blocked |  |  |  |  |
| citeseer_full | blocked |  |  |  |  |
| pubmed_full | blocked |  |  |  |  |
| ppi_full_formal | blocked |  |  |  |  |
| ogbn_arxiv_full | blocked |  |  |  |  |

## GPU

| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | gpu_backend_active | cpu_fallback_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora_full | blocked |  |  |  |  | True | False |
| citeseer_full | blocked |  |  |  |  | True | False |
| pubmed_full | blocked |  |  |  |  | True | False |
| ppi_full_formal | blocked |  |  |  |  | True | False |
| ogbn_arxiv_full | blocked |  |  |  |  | True | False |

## Blocked Items

| dataset | route_id | failure_step | block_reason |
| --- | --- | --- | --- |
| cora_full | zkdl_gat_full_forward_cpu | zkdl_cpu_backend_capability_check | zkdl_cpu_backend_unavailable |
| citeseer_full | zkdl_gat_full_forward_cpu | zkdl_cpu_backend_capability_check | zkdl_cpu_backend_unavailable |
| pubmed_full | zkdl_gat_full_forward_cpu | zkdl_cpu_backend_capability_check | zkdl_cpu_backend_unavailable |
| ppi_full_formal | zkdl_gat_full_forward_cpu | zkdl_cpu_backend_capability_check | zkdl_cpu_backend_unavailable |
| ogbn_arxiv_full | zkdl_gat_full_forward_cpu | zkdl_cpu_backend_capability_check | zkdl_cpu_backend_unavailable |
| cora_full | zkdl_gat_full_forward_gpu | zkdl_gat_full_forward_capability_check | gat_forward_constraint_not_representable_in_current_zkdl |
| citeseer_full | zkdl_gat_full_forward_gpu | zkdl_gat_full_forward_capability_check | gat_forward_constraint_not_representable_in_current_zkdl |
| pubmed_full | zkdl_gat_full_forward_gpu | zkdl_gat_full_forward_capability_check | gat_forward_constraint_not_representable_in_current_zkdl |
| ppi_full_formal | zkdl_gat_full_forward_gpu | zkdl_gat_full_forward_capability_check | gat_forward_constraint_not_representable_in_current_zkdl |
| ogbn_arxiv_full | zkdl_gat_full_forward_gpu | zkdl_gat_full_forward_capability_check | gat_forward_constraint_not_representable_in_current_zkdl |
