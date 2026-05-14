# GAT-ZKML 与 ezkl CPU 严格全量数据集结果

- 已验证全量数据集行数：4/5。
- 路线：`ezkl_gat_native_compressed_csr_attention_full`。
- 结果只来自 clean run directory 中完整自动执行链生成的 manifest。
- 未使用 GPU、output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。

## 数据集状态

| 数据集 | status | route_id | strict | full GAT forward in proof | final logits derivation | manifest strict check |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |
| PPI | blocked | ezkl_gat_native_compressed_csr_attention_full | false | false | true | blocked |
| ogbn-arxiv | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |

## CPU 四项指标

| 数据集 | status | final benchmark | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | verified | true | 39075.786 | 19686.690 | 362.892 | 1280277 |
| Citeseer | verified | true | 39120.864 | 20777.446 | 453.699 | 1347710 |
| Pubmed | verified | true | 79577.150 | 44588.471 | 893.838 | 3973380 |
| PPI | blocked | false | 1322255.020 | 1384312.235 |  |  |
| ogbn-arxiv | verified | true | 1276740.830 | 1941818.225 | 24279.105 | 453849473 |

## 与主仓库基准对比

| 数据集 | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 39075.786 | 1207.818 | 19686.690 | 153.379 | 362.892 | 90755 | 1280277 |
| Citeseer | 453.773 | 39120.864 | 1418.744 | 20777.446 | 153.837 | 453.699 | 90763 | 1347710 |
| Pubmed | 2962.933 | 79577.150 | 5604.074 | 44588.471 | 175.454 | 893.838 | 90760 | 3973380 |
| PPI | 3563.973 |  | 7111.447 |  | 424.710 |  | 21360 |  |
| ogbn-arxiv | 18071.562 | 1276740.830 | 66540.600 | 1941818.225 | 417.380 | 24279.105 | 21361 | 453849473 |
