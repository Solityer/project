# Native CSR Segment Attention Report

- Legacy report retained for continuity; rows below reflect the latest native route manifest for each dataset.
- Layout: CSR/ragged dst segments over `src[E]`, `dst[E]`, `segment_ptr[N+1]`, `edge_ptr[G+1]`, and `node_ptr[G+1]`.
- Exact inverse attempt: native witness inverse column with `Sum_i * Inv_i = 1`; no epsilon reciprocal is used.

| Dataset | route_id | primitive | status | block_reason | failure_step | rows | required_logrows | max_logrows |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Cora | ezkl_gat_native_compressed_csr_attention_full | GraphSegmentAttentionCompressedCSR | blocked | native_full_gat_forward_constraints_incomplete | prove_time_ms | 2117846 | 22 | 26 |
| Citeseer | ezkl_gat_native_compressed_csr_attention_full |  | blocked | not_run_because_cora_full_blocked_at_native_public_output_binding | not_started |  |  |  |
| Pubmed | ezkl_gat_native_compressed_csr_attention_full |  | blocked | not_run_because_cora_full_blocked_at_native_public_output_binding | not_started |  |  |  |
| PPI | ezkl_gat_native_compressed_csr_attention_full |  | blocked | not_run_because_cora_full_blocked_at_native_public_output_binding | not_started |  |  |  |
| ogbn-arxiv | ezkl_gat_native_compressed_csr_attention_full |  | blocked | not_run_because_cora_full_blocked_at_native_public_output_binding | not_started |  |  |  |

## Artifact Contract

- Cora: spec=True, csr=True, tensor_payloads=True
- Citeseer: spec=True, csr=True, tensor_payloads=True
- Pubmed: spec=True, csr=True, tensor_payloads=True
- PPI: spec=True, csr=True, tensor_payloads=True
- ogbn-arxiv: spec=True, csr=True, tensor_payloads=True
