# GAT-ZKML 与 ezkl CPU 严格全量数据集结果

- 已验证全量数据集行数：5/5。
- 路线：`ezkl_gat_native_compressed_csr_attention_full`。
- `strict_public_input_equivalent` 只在完整 native GAT proof 已通过验证时为 `true`。
- 未使用 output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。

## 数据集状态

| 数据集 | status | route_id | strict | full GAT forward in proof | final logits derivation |
| --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| PPI | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| ogbn-arxiv | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |

## CPU 四项指标

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 39418.157 | 20340.017 | 372.952 | 1280306 |
| Citeseer | 39775.686 | 20593.018 | 489.516 | 1347736 |
| Pubmed | 79890.208 | 43471.667 | 966.513 | 3973400 |
| PPI | 1289180 | 1921540 | 23240 | 461655303 |
| ogbn-arxiv | 1290660 | 1860150 | 24660 | 453849446 |

## 与主仓库基准对比

| 数据集 | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 39418.157 | 1207.818 | 20340.017 | 153.379 | 372.952 | 90755 | 1280306 |
| Citeseer | 453.773 | 39775.686 | 1418.744 | 20593.018 | 153.837 | 489.516 | 90763 | 1347736 |
| Pubmed | 2962.933 | 79890.208 | 5604.074 | 43471.667 | 175.454 | 966.513 | 90760 | 3973400 |
| PPI | 3563.973 | 1289180 | 7111.447 | 1921540 | 424.710 | 23240 | 21360 | 461655303 |
| ogbn-arxiv | 18071.562 | 1290660 | 66540.600 | 1860150 | 417.380 | 24660 | 21361 | 453849446 |
