# GAT-ZKML 与 ezkl CPU 严格全量数据集结果

- 已验证全量数据集行数：5/5。
- 路线：`ezkl_gat_native_compressed_csr_attention_full`。
- 结果只来自 clean run directory 中完整自动执行链生成的 manifest。
- 未使用 GPU、output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。

## 数据集状态

| 数据集 | status | route_id | strict | full GAT forward in proof | final logits derivation | manifest strict check |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |
| PPI | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |
| ogbn-arxiv | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true | passed |

## CPU 四项指标

| 数据集 | status | final benchmark | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | verified | true | 39610.144 | 19551.087 | 399.098 | 1280317 |
| Citeseer | verified | true | 39191.259 | 20785.288 | 492.913 | 1347633 |
| Pubmed | verified | true | 80007.432 | 42373.141 | 989.291 | 3973375 |
| PPI | verified | true | 1280185.848 | 1956645.759 | 24571.380 | 461655281 |
| ogbn-arxiv | verified | true | 1287653.770 | 1947900.981 | 25975.184 | 453849491 |

## 与主仓库基准对比

| 数据集 | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 39610.144 | 1207.818 | 19551.087 | 153.379 | 399.098 | 90755 | 1280317 |
| Citeseer | 453.773 | 39191.259 | 1418.744 | 20785.288 | 153.837 | 492.913 | 90763 | 1347633 |
| Pubmed | 2962.933 | 80007.432 | 5604.074 | 42373.141 | 175.454 | 989.291 | 90760 | 3973375 |
| PPI | 3563.973 | 1280185.848 | 7111.447 | 1956645.759 | 424.710 | 24571.380 | 21360 | 461655281 |
| ogbn-arxiv | 18071.562 | 1287653.770 | 66540.600 | 1947900.981 | 417.380 | 25975.184 | 21361 | 453849491 |
