# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Verified ezkl full-dataset rows: 3/5.
- Strict public input equivalent is true only for rows with a verified full native GAT proof.
- No output-binding workaround, precomputed logits, identity, digest, folding, subset, shard, or sample route was used.

## Dataset Status

| Dataset | status | route_id | strict | full GAT forward in proof | final logits derivation | block_reason |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| PPI | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | True | native_keygen_resource_failure |
| ogbn-arxiv | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | True | native_keygen_resource_failure |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 34249.261 | 1207.818 | 20726.156 | 153.379 | 566.251 | 90755 | 1280198 |
| Citeseer | 453.773 | 35124.103 | 1418.744 | 21365.887 | 153.837 | 676.547 | 90763 | 1347596 |
| Pubmed | 2962.933 | 69588.681 | 5604.074 | 43774.539 | 175.454 | 1303.938 | 90760 | 3973278 |
| PPI | 3563.973 | 1001891.381 | 7111.447 | blocked | 424.710 | blocked | 21360 | blocked |
| ogbn-arxiv | 18071.562 | 1734496.290 | 66540.600 | blocked | 417.380 | blocked | 21361 | blocked |

## Blocked/Failed Reasons

- Cora: verified / 
- Citeseer: verified / 
- Pubmed: verified / 
- PPI: blocked / native_keygen_resource_failure
- ogbn-arxiv: blocked / native_keygen_resource_failure
