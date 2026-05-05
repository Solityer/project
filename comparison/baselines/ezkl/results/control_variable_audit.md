# 控制变量审计

## 路线范围

唯一正式路线：

- `ezkl_gat_native_compressed_csr_attention_full`

该路线属于证明语句等价组：

- `gat_native_compressed_csr_attention_full_v1`

本目录以主仓库作为 formal 语义真源。除证明系统与 ezkl/Halo2 必要适配外，数据集、checkpoint、图预处理、量化配置、最终 logits 语义和公开输出策略均与主仓库对齐。

## 全量数据集覆盖

| 数据集 | manifest |
| --- | --- |
| `cora_full` | `runs/cora_full/run_manifest.json` |
| `citeseer_full` | `runs/citeseer_full/run_manifest.json` |
| `pubmed_full` | `runs/pubmed_full/run_manifest.json` |
| `ppi_full_formal` | `runs/ppi_full_formal/run_manifest.json` |
| `ogbn_arxiv_full` | `runs/ogbn_arxiv_full/run_manifest.json` |

最终结果未使用 toy、smoke、subset、sample、shard、partial、debug 数据集或缩放图。

## 保持不变的对象

| 控制变量 | 审计结论 |
| --- | --- |
| checkpoint bundle | 复用主仓库全量数据集配置导出的 checkpoint 元数据。 |
| `model_arch_id` | 由 adapter 导出的 `model_info.json` 与 manifest 绑定。 |
| `model_param_id` | 由 checkpoint 摘要与 manifest 绑定。 |
| `quant_cfg_id` | 保持 `fp32_bundle_export`，逆元查找表策略来自公开量化和表元数据。 |
| `hidden_profile` | 由同一 checkpoint manifest 固定。 |
| `d_in_profile` | 由同一 checkpoint manifest 固定。 |
| `K_out` | 由同一 checkpoint manifest 固定。 |
| `output_average_rule` | 保持“每个输出头加偏置后算术平均”的 formal 规则。 |
| 图预处理 | 保持全量图、自环、去重、稳定排序与 compressed CSR 输入。 |
| 最终 logits 语义 | 保持 row-major node-by-class final logits public instances。 |

## 允许的工程适配

| 适配 | 对证明语句的影响 |
| --- | --- |
| native compressed CSR attention | 只改变稀疏图在 `ezkl/Halo2` 中的承载方式。 |
| shape-only setup | setup 阶段只消费公开实例形状，prove 与 verify 仍绑定完整公开 logits。 |
| chunked public instance columns | 只拆分同一 row-major public logits 序列。 |
| deterministic inverse table range policy | 查找表范围来自公开量化、度数和表边界，不来自 witness。 |
| tagged lookup table columns | 复用查找表列，不改变 inverse、LReLU、Exp、ELU 语义。 |
| prove-time payload deduplication | 只减少重复主机内存，不改变约束或公开实例。 |

## 严格性检查

| 数据集 | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete |
| --- | --- | --- | --- | --- |
| `cora_full` | `verified` | `true` | `true` | `true` |
| `citeseer_full` | `verified` | `true` | `true` | `true` |
| `pubmed_full` | `verified` | `true` | `true` | `true` |
| `ppi_full_formal` | `verified` | `true` | `true` | `true` |
| `ogbn_arxiv_full` | `verified` | `true` | `true` | `true` |

## 禁用项确认

正式结果未启用：

- output-binding replacement
- digest-only output
- folding
- identity route
- precomputed logits bypass
- subset
- shard
- sample
- toy、smoke、debug 数据集
