# GAT-ZKML 与 ezkl 全量数据集 CPU 汇总

- 已验证全量数据集行数：5/5。
- 路线：`ezkl_gat_native_compressed_csr_attention_full`。
- `strict_public_input_equivalent` 只在完整 native GAT proof 已通过验证时为 `true`。
- 未使用 output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。

## 数据集状态

| 数据集 | status | route_id | backend | strict | full GAT forward in proof | final logits derivation | block_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | cpu | True | True | True |  |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | cpu | True | True | True |  |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | cpu | True | True | True |  |
| PPI | verified | ezkl_gat_native_compressed_csr_attention_full | cpu | True | True | True |  |
| ogbn-arxiv | verified | ezkl_gat_native_compressed_csr_attention_full | cpu | True | True | True |  |

## 四项指标对比

| 数据集 | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 39418.157 | 1207.818 | 20340.017 | 153.379 | 372.952 | 90755 | 1280306 |
| Citeseer | 453.773 | 39775.686 | 1418.744 | 20593.018 | 153.837 | 489.516 | 90763 | 1347736 |
| Pubmed | 2962.933 | 79890.208 | 5604.074 | 43471.667 | 175.454 | 966.513 | 90760 | 3973400 |
| PPI | 3563.973 | 1289180 | 7111.447 | 1921540 | 424.710 | 23240 | 21360 | 461655303 |
| ogbn-arxiv | 18071.562 | 1290660 | 66540.600 | 1860150 | 417.380 | 24660 | 21361 | 453849446 |

## 阻塞或失败原因

- Cora: verified /
- Citeseer: verified /
- Pubmed: verified /
- PPI: verified /
- ogbn-arxiv: verified /
