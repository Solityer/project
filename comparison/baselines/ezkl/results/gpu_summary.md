# GAT-ZKML 与 ezkl 全量数据集 GPU 汇总

- 已验证全量数据集行数：0/5。
- 路线：`ezkl_gat_native_compressed_csr_attention_full_gpu`。
- `strict_public_input_equivalent` 只在完整 native GAT proof 已通过验证时为 `true`。
- 未使用 output-binding workaround、precomputed logits、identity、digest、folding、subset、shard 或 sample route。

## 数据集状态

| 数据集 | status | route_id | backend | strict | full GAT forward in proof | final logits derivation | block_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | blocked | ezkl_gat_native_compressed_csr_attention_full_gpu | gpu | False | False | True | gpu_backend_unavailable |
| Citeseer | blocked | ezkl_gat_native_compressed_csr_attention_full_gpu | gpu | False | False | True | gpu_backend_unavailable |
| Pubmed | blocked | ezkl_gat_native_compressed_csr_attention_full_gpu | gpu | False | False | True | gpu_backend_unavailable |
| PPI | blocked | ezkl_gat_native_compressed_csr_attention_full_gpu | gpu | False | False | True | gpu_backend_unavailable |
| ogbn-arxiv | blocked | ezkl_gat_native_compressed_csr_attention_full_gpu | gpu | False | False | True | gpu_backend_unavailable |

## 四项指标对比

| 数据集 | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | blocked | 1207.818 | blocked | 153.379 | blocked | 90755 | blocked |
| Citeseer | 453.773 | blocked | 1418.744 | blocked | 153.837 | blocked | 90763 | blocked |
| Pubmed | 2962.933 | blocked | 5604.074 | blocked | 175.454 | blocked | 90760 | blocked |
| PPI | 3563.973 | blocked | 7111.447 | blocked | 424.710 | blocked | 21360 | blocked |
| ogbn-arxiv | 18071.562 | blocked | 66540.600 | blocked | 417.380 | blocked | 21361 | blocked |

## 阻塞或失败原因

- Cora: blocked / gpu_backend_unavailable
- Citeseer: blocked / gpu_backend_unavailable
- Pubmed: blocked / gpu_backend_unavailable
- PPI: blocked / gpu_backend_unavailable
- ogbn-arxiv: blocked / gpu_backend_unavailable
