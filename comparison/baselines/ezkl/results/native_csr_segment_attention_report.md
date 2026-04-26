# Native CSR Segment Attention Report

- Legacy report retained for continuity; rows below reflect the latest native route manifest for each dataset.
- Layout: CSR/ragged dst segments over `src[E]`, `dst[E]`, `segment_ptr[N+1]`, `edge_ptr[G+1]`, and `node_ptr[G+1]`.
- Exact inverse attempt: native witness inverse column with `Sum_i * Inv_i = 1`; no epsilon reciprocal is used.

| Dataset | route_id | primitive | status | block_reason | failure_step | rows | required_logrows | max_logrows |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: |
| Cora | ezkl_gat_native_compressed_csr_attention_full | GraphSegmentAttentionCompressedCSR | blocked | ezkl_compressed_csr_backend_unavailable | setup_ms | 2117846 | 22 | 26 |
| Citeseer | ezkl_gat_native_compressed_csr_attention_full | GraphSegmentAttentionCompressedCSR | blocked | ezkl_compressed_csr_backend_unavailable | setup_ms | 2218463 | 22 | 26 |
| Pubmed | ezkl_gat_native_compressed_csr_attention_full | GraphSegmentAttentionCompressedCSR | blocked | ezkl_compressed_csr_backend_unavailable | setup_ms | 16143874 | 24 | 26 |
| PPI | ezkl_gat_native_compressed_csr_attention_full | GraphSegmentAttentionCompressedCSR | blocked | ezkl_compressed_csr_backend_unavailable | setup_ms | 43372416 | 26 | 26 |
| ogbn-arxiv | ezkl_gat_native_compressed_csr_attention_full | GraphSegmentAttentionCompressedCSR | blocked | ezkl_compressed_csr_backend_unavailable | setup_ms | 59159176 | 26 | 26 |

## Artifact Contract

- Cora: spec=True, csr=True, tensor_payloads=True
- Citeseer: spec=True, csr=True, tensor_payloads=True
- Pubmed: spec=True, csr=True, tensor_payloads=True
- PPI: spec=True, csr=True, tensor_payloads=True
- ogbn-arxiv: spec=True, csr=True, tensor_payloads=True
