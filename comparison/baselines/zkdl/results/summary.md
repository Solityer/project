# zkDL Full GAT Baseline Summary

Generated at: `2026-05-06T04:04:33.321208+00:00`

Correct baseline directory: `comparison/baselines/zkdl`.

The full-dataset statement is attempted for `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, and `ogbn_arxiv_full`. No row is marked verified unless a real zkDL proof and verifier acceptance path succeeds.

`semantic_precheck_passed=true` only means the baseline recomputed the full-dataset forward and local semantic checker accepted it. It is not a zkDL proof and never contributes to `proof_size_bytes`.

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
| cora_full | zkdl_gat_full_forward_cpu | zkdl_full_gat_prover_backend | semantic_precheck_passed_but_no_real_zkdl_composed_full_gat_forward_prover_verifier_or_transferable_proof_object_is_available |
| citeseer_full | zkdl_gat_full_forward_cpu | zkdl_full_gat_prover_backend | semantic_precheck_passed_but_no_real_zkdl_composed_full_gat_forward_prover_verifier_or_transferable_proof_object_is_available |
| pubmed_full | zkdl_gat_full_forward_cpu | zkdl_full_gat_prover_backend | semantic_precheck_passed_but_no_real_zkdl_composed_full_gat_forward_prover_verifier_or_transferable_proof_object_is_available |
| ppi_full_formal | zkdl_gat_full_forward_cpu | zkdl_full_gat_prover_backend | semantic_precheck_passed_but_no_real_zkdl_composed_full_gat_forward_prover_verifier_or_transferable_proof_object_is_available |
| ogbn_arxiv_full | zkdl_gat_full_forward_cpu | zkdl_full_gat_prover_backend | semantic_precheck_passed_but_no_real_zkdl_composed_full_gat_forward_prover_verifier_or_transferable_proof_object_is_available |
| cora_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| citeseer_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| pubmed_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ppi_full_formal | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ogbn_arxiv_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
