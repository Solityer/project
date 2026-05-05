# zkDL Full GAT Baseline Summary

Generated at: `2026-05-05T08:52:45.977186+00:00`

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
| cora_full | zkdl_gat_full_forward_cpu | zkdl_commitment | zkdl_cpu_hyrax_tensor_commitment_backend_missing |
| citeseer_full | zkdl_gat_full_forward_cpu | zkdl_commitment | zkdl_cpu_hyrax_tensor_commitment_backend_missing |
| pubmed_full | zkdl_gat_full_forward_cpu | zkdl_commitment | zkdl_cpu_hyrax_tensor_commitment_backend_missing |
| ppi_full_formal | zkdl_gat_full_forward_cpu | zkdl_commitment | zkdl_cpu_hyrax_tensor_commitment_backend_missing |
| ogbn_arxiv_full | zkdl_gat_full_forward_cpu | zkdl_commitment | zkdl_cpu_hyrax_tensor_commitment_backend_missing |
| cora_full | zkdl_gat_full_forward_gpu | proof_statement_build | zkdl_missing_dst_group_max_inverse_lookup_rescale_fac4dnn_and_final_logits_binding |
| citeseer_full | zkdl_gat_full_forward_gpu | proof_statement_build | zkdl_missing_dst_group_max_inverse_lookup_rescale_fac4dnn_and_final_logits_binding |
| pubmed_full | zkdl_gat_full_forward_gpu | proof_statement_build | zkdl_missing_dst_group_max_inverse_lookup_rescale_fac4dnn_and_final_logits_binding |
| ppi_full_formal | zkdl_gat_full_forward_gpu | proof_statement_build | zkdl_missing_dst_group_max_inverse_lookup_rescale_fac4dnn_and_final_logits_binding |
| ogbn_arxiv_full | zkdl_gat_full_forward_gpu | proof_statement_build | zkdl_missing_dst_group_max_inverse_lookup_rescale_fac4dnn_and_final_logits_binding |
