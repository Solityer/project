# zkDL Full GAT Baseline Summary

Generated at: `2026-05-06T05:01:27.352366+00:00`

Correct baseline directory: `comparison/baselines/zkdl`.

The full-dataset statement is attempted for `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, and `ogbn_arxiv_full`. No row is marked verified unless a real zkDL proof and verifier acceptance path succeeds.

`semantic_precheck_passed=true` only means the baseline recomputed the full-dataset forward and local semantic checker accepted it. It is not a zkDL proof and never contributes to `proof_size_bytes`.

Formal commitment/prove/verify/proof-size metrics are populated only for `status=verified`. Failed proof-attempt timings are debug fields in each run manifest and are excluded from the comparison tables.

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
| cora_full | zkdl_gat_full_forward_cpu | softmax_denominator_inverse_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.softmax_inverse,residual=0x1cc6b127a1cae79d |
| citeseer_full | zkdl_gat_full_forward_cpu | softmax_denominator_inverse_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.softmax_inverse,residual=0x18ab80b8cc3fd87e |
| pubmed_full | zkdl_gat_full_forward_cpu | softmax_denominator_inverse_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.softmax_inverse,residual=0x05afcbfe734e4a25 |
| ppi_full_formal | zkdl_gat_full_forward_cpu | softmax_denominator_inverse_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.softmax_inverse,residual=0x095540416daff023 |
| ogbn_arxiv_full | zkdl_gat_full_forward_cpu | softmax_denominator_inverse_sumcheck_verify | zkdl_sumcheck_residual_nonzero:section=hidden.layer0.head0.softmax_inverse,residual=0x0ed0ae1819dc4550 |
| cora_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| citeseer_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| pubmed_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ppi_full_formal | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ogbn_arxiv_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
