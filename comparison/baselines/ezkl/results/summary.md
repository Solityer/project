# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Verified ezkl full-dataset rows: 0/5.
- Strict public input equivalent is true only for rows with a verified full native GAT proof.
- No precomputed logits, identity, reshape, digest, folding, subset, shard, or scaled route was used.

## Dataset Status

| Dataset | status | route_id | block_reason |
| --- | --- | --- | --- |
| Cora | blocked | ezkl_gat_native_compressed_csr_attention_full | ezkl_compressed_csr_backend_unavailable |
| Citeseer | blocked | ezkl_gat_native_compressed_csr_attention_full | ezkl_compressed_csr_backend_unavailable |
| Pubmed | blocked | ezkl_gat_native_compressed_csr_attention_full | ezkl_compressed_csr_backend_unavailable |
| PPI | blocked | ezkl_gat_native_compressed_csr_attention_full | ezkl_compressed_csr_backend_unavailable |
| ogbn-arxiv | blocked | ezkl_gat_native_compressed_csr_attention_full | ezkl_compressed_csr_backend_unavailable |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 1.562 | 1207.818 | blocked | 153.379 | blocked | 90755 | blocked |
| Citeseer | 453.773 | 2.204 | 1418.744 | blocked | 153.837 | blocked | 90763 | blocked |
| Pubmed | 2962.933 | 2.320 | 5604.074 | blocked | 175.454 | blocked | 90760 | blocked |
| PPI | 3563.973 | 1.851 | 7111.447 | blocked | 424.710 | blocked | 21360 | blocked |
| ogbn-arxiv | 18071.562 | 1.467 | 66540.600 | blocked | 417.380 | blocked | 21361 | blocked |

## Blocked/Failed Reasons

- Cora: blocked / ezkl_compressed_csr_backend_unavailable
- Citeseer: blocked / ezkl_compressed_csr_backend_unavailable
- Pubmed: blocked / ezkl_compressed_csr_backend_unavailable
- PPI: blocked / ezkl_compressed_csr_backend_unavailable
- ogbn-arxiv: blocked / ezkl_compressed_csr_backend_unavailable
