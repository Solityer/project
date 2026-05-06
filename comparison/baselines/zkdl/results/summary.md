# zkDL Full GAT Baseline Summary

Generated at: `2026-05-06T04:27:22.581185+00:00`

Correct baseline directory: `comparison/baselines/zkdl`.

The full-dataset statement is attempted for `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, and `ogbn_arxiv_full`. No row is marked verified unless a real zkDL proof and verifier acceptance path succeeds.

`semantic_precheck_passed=true` only means the baseline recomputed the full-dataset forward and local semantic checker accepted it. It is not a zkDL proof and never contributes to `proof_size_bytes`.

## CPU

| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | --- | --- | --- |
| cora_full | blocked | 146.662 | 512.534 | 0.001 | 58533 |
| citeseer_full | blocked | 414.401 | 1406.316 | 0.001 | 58537 |
| pubmed_full | blocked | 472.798 | 926.239 | 0.001 | 58811 |
| ppi_full_formal | blocked | 1141.193 | 1429.471 | 0.001 | 14529 |
| ogbn_arxiv_full | blocked | 1474.416 | 2166.763 | 0.001 | 14595 |

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
| cora_full | zkdl_gat_full_forward_cpu | dense_matmul_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.projection,residual=0x11e64285603d2e58 |
| citeseer_full | zkdl_gat_full_forward_cpu | dense_matmul_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.projection,residual=0x004325289b33db6c |
| pubmed_full | zkdl_gat_full_forward_cpu | dense_matmul_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.projection,residual=0x04d57e25cf7768ff |
| ppi_full_formal | zkdl_gat_full_forward_cpu | dense_matmul_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.projection,residual=0x03aa07f8285d11d8 |
| ogbn_arxiv_full | zkdl_gat_full_forward_cpu | dense_matmul_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.projection,residual=0x01f6e28ff47d3519 |
| cora_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| citeseer_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| pubmed_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ppi_full_formal | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ogbn_arxiv_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
