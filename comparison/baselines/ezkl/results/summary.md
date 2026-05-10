# GAT-ZKML 与 ezkl CPU 数据集结果

- 已验证全量数据集行数：5/5。
- 路线：`ezkl_gat_native_compressed_csr_attention_full`。
- `strict_public_input_equivalent` 只在完整 native GAT proof 已通过验证时为 `true`。

## 数据集状态

| 数据集 | status | route_id | strict | full GAT forward in proof | final logits derivation |
| --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| PPI | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |
| ogbn-arxiv | verified | ezkl_gat_native_compressed_csr_attention_full | true | true | true |

## CPU 结果

| 数据集 | 承诺时间（ms） | 证明时间（ms） | 验证时间（ms） | proof（byte） |
| --- | ---: | ---: | ---: | ---: |
| Cora | 39418.157 | 20340.017 | 372.952 | 1280306 |
| Citeseer | 39775.686 | 20593.018 | 489.516 | 1347736 |
| Pubmed | 79890.208 | 43471.667 | 966.513 | 3973400 |
| PPI | 1289180 | 1921540 | 23240 | 461655303 |
| ogbn-arxiv | 1290660 | 1860150 | 24660 | 453849446 |

## 基准对比

| 数据集 | main 承诺时间（ms） | ezkl 承诺时间（ms） | main 证明时间（ms） | ezkl 证明时间（ms） | main 验证时间（ms） | ezkl 验证时间（ms） | main proof（byte） | ezkl proof（byte） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 39418.157 | 1207.818 | 20340.017 | 153.379 | 372.952 | 90755 | 1280306 |
| Citeseer | 453.773 | 39775.686 | 1418.744 | 20593.018 | 153.837 | 489.516 | 90763 | 1347736 |
| Pubmed | 2962.933 | 79890.208 | 5604.074 | 43471.667 | 175.454 | 966.513 | 90760 | 3973400 |
| PPI | 3563.973 | 1289180 | 7111.447 | 1921540 | 424.710 | 23240 | 21360 | 461655303 |
| ogbn-arxiv | 18071.562 | 1290660 | 66540.600 | 1860150 | 417.380 | 24660 | 21361 | 453849446 |
