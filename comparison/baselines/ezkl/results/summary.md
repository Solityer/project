# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Strict public input equivalent: false for all completed rows because no full GAT ezkl proof verified.
- Full GAT forward entered ezkl proof: false for all rows.
- No precomputed logits, identity, reshape, digest, folding, subset, shard, or scaled route was used.

## Dataset Status

| Dataset | status | route_id | block_reason |
| --- | --- | --- | --- |
| Cora | blocked | ezkl_gat_native_csr_segment_attention_full | ezkl_native_csr_compile_resource_boundary |
| Citeseer | blocked | ezkl_gat_native_csr_segment_attention_full | ezkl_native_csr_compile_resource_boundary |
| Pubmed | blocked | ezkl_gat_native_csr_segment_attention_full | ezkl_native_csr_compile_resource_boundary |
| PPI | blocked | ezkl_gat_native_csr_segment_attention_full | ezkl_native_csr_compile_resource_boundary |
| ogbn-arxiv | blocked | ezkl_gat_native_csr_segment_attention_full | ezkl_native_csr_compile_resource_boundary |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | blocked | 1207.818 | blocked | 153.379 | blocked | 90755 | blocked |
| Citeseer | 453.773 | blocked | 1418.744 | blocked | 153.837 | blocked | 90763 | blocked |
| Pubmed | 2962.933 | blocked | 5604.074 | blocked | 175.454 | blocked | 90760 | blocked |
| PPI | 3563.973 | blocked | 7111.447 | blocked | 424.710 | blocked | 21360 | blocked |
| ogbn-arxiv | 18071.562 | blocked | 66540.600 | blocked | 417.380 | blocked | 21361 | blocked |

## Blocked/Failed Reasons

- Cora: blocked / ezkl_native_csr_compile_resource_boundary
- Citeseer: blocked / ezkl_native_csr_compile_resource_boundary
- Pubmed: blocked / ezkl_native_csr_compile_resource_boundary
- PPI: blocked / ezkl_native_csr_compile_resource_boundary
- ogbn-arxiv: blocked / ezkl_native_csr_compile_resource_boundary
