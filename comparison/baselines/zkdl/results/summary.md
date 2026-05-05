# zkDL Full GAT Baseline Summary

Generated at: `2026-05-05T09:08:51.368190+00:00`

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
| cora_full | zkdl_gat_full_forward_cpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cpu_reached_full_dataset_hidden_projection_prefix_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| citeseer_full | zkdl_gat_full_forward_cpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cpu_reached_full_dataset_hidden_projection_prefix_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| pubmed_full | zkdl_gat_full_forward_cpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cpu_reached_full_dataset_hidden_projection_prefix_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| ppi_full_formal | zkdl_gat_full_forward_cpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cpu_reached_full_dataset_hidden_projection_prefix_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| ogbn_arxiv_full | zkdl_gat_full_forward_cpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cpu_reached_full_dataset_hidden_projection_prefix_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| cora_full | zkdl_gat_full_forward_gpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cuda_reached_full_dataset_hidden_projection_prefix_and_primitive_probe_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| citeseer_full | zkdl_gat_full_forward_gpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cuda_reached_full_dataset_hidden_projection_prefix_and_primitive_probe_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| pubmed_full | zkdl_gat_full_forward_gpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cuda_reached_full_dataset_hidden_projection_prefix_and_primitive_probe_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| ppi_full_formal | zkdl_gat_full_forward_gpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cuda_reached_full_dataset_hidden_projection_prefix_and_primitive_probe_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
| ogbn_arxiv_full | zkdl_gat_full_forward_gpu | hidden_layer_0_head_0_sparse_route_sumcheck | zkdl_cuda_reached_full_dataset_hidden_projection_prefix_and_primitive_probe_but_public_sparse_route_sumcheck_for_R_src_R_dst_is_missing |
