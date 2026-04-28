# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Verified ezkl full-dataset rows: 5/5.
- Strict public input equivalent is true only for rows with a verified full native GAT proof.
- No output-binding workaround, precomputed logits, identity, digest, folding, subset, shard, or sample route was used.

## Dataset Status

| Dataset | status | route_id | strict | full GAT forward in proof | final logits derivation | block_reason |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| PPI | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| ogbn-arxiv | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 39418.157 | 1207.818 | 20340.017 | 153.379 | 372.952 | 90755 | 1280306 |
| Citeseer | 453.773 | 39775.686 | 1418.744 | 20593.018 | 153.837 | 489.516 | 90763 | 1347736 |
| Pubmed | 2962.933 | 79890.208 | 5604.074 | 43471.667 | 175.454 | 966.513 | 90760 | 3973400 |
| PPI | 3563.973 | 1289180 | 7111.447 | 1921540 | 424.710 | 23240 | 21360 | 461655303 |
| ogbn-arxiv | 18071.562 | 1290660 | 66540.600 | 1860150 | 417.380 | 24660 | 21361 | 453849446 |

## Blocked/Failed Reasons

- Cora: verified /
- Citeseer: verified /
- Pubmed: verified /
- PPI: verified /
- ogbn-arxiv: verified /
