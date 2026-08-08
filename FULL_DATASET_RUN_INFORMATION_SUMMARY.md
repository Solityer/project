# 五数据集 × CPU/GPU 全量运行信息汇总

> 用途：将 5 个数据集 × CPU/GPU 共 10 组 full-run 日志中的运行信息、正式计时和日志聚合数据集中整理，减少直接阅读原始长日志的需要。  
> 实验批次启动日期：2026-08-02；其中 OGBN-Arxiv GPU 运行于 2026-08-03 00:18:06 结束。  
> 运行入口：`gatzk_run`。  
> 十组运行最终状态均为 `VERIFY_OK`、`exit_code=0`。  
> 本文件只汇总运行事实、日志字段及其直接算术整理，不对后续优化方向、修改方案、性能原因或优先级作判断。

---

## 1. 运行对象与日志索引

| 数据集 | 后端 | 原始日志 |
|---|---|---|
| Cora | CPU | `cpu_cora_full_20260802_133115_810065175.log` |
| Cora | GPU | `gpu_cora_full_20260802_185345_591037366.log` |
| Citeseer | CPU | `cpu_citeseer_full_20260802_134010_709600775.log` |
| Citeseer | GPU | `gpu_citeseer_full_20260802_190148_114509453.log` |
| Pubmed | CPU | `cpu_pubmed_full_20260802_141258_202025151.log` |
| Pubmed | GPU | `gpu_pubmed_full_20260802_193229_853151035.log` |
| PPI | CPU | `cpu_ppi_full_20260802_144939_227306531.log` |
| PPI | GPU | `gpu_ppi_full_20260802_200719_968895716.log` |
| OGBN-Arxiv | CPU | `cpu_ogbn_arxiv_full_20260802_162227_017652071.log` |
| OGBN-Arxiv | GPU | `gpu_ogbn_arxiv_full_20260802_214228_120595691.log` |

对应配置：

| 数据集 | 配置 |
|---|---|
| Cora | `configs/cora_secure.cfg` |
| Citeseer | `configs/citeseer_secure.cfg` |
| Pubmed | `configs/pubmed_secure.cfg` |
| PPI | `configs/ppi_secure.cfg` |
| OGBN-Arxiv | `configs/ogbn_arxiv_secure.cfg` |

后端设置：

- CPU：`compute_backend=cpu`
- GPU：`compute_backend=cuda_hotspots`

---

## 2. 正式运行结果

定义：

`formal_total = commitment_time_ms + prove_time_ms + verify_time_ms`

计时统一换算为秒。`Proof size` 保留日志中的原始 byte 值。`formal_total` 和差值由原始毫秒字段直接计算，显示值存在毫秒级四舍五入。

| 数据集 | 后端 | Commitment | Prove | Verify | Formal total | Proof size (bytes) |
|---|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 35.042 s | 118.970 s | 0.643 s | 154.655 s | 1,462,620 |
| Cora | GPU | 35.388 s | 78.161 s | 0.551 s | 114.100 s | 1,462,631 |
| Citeseer | CPU | 81.478 s | 261.968 s | 1.116 s | 344.562 s | 1,493,237 |
| Citeseer | GPU | 81.510 s | 257.850 s | 1.084 s | 340.443 s | 1,493,241 |
| Pubmed | CPU | 111.724 s | 467.240 s | 1.670 s | 580.634 s | 2,743,664 |
| Pubmed | GPU | 110.757 s | 463.866 s | 1.444 s | 576.068 s | 2,743,659 |
| PPI | CPU | 683.748 s | 3996.545 s | 7.033 s | 4687.327 s | 220,865,254 |
| PPI | GPU | 674.127 s | 4149.348 s | 5.545 s | 4829.019 s | 220,865,255 |
| OGBN-Arxiv | CPU | 1190.055 s | 4507.622 s | 9.437 s | 5707.114 s | 217,003,699 |
| OGBN-Arxiv | GPU | 1198.349 s | 4666.998 s | 64.929 s | 5930.276 s | 217,003,688 |

### 2.1 GPU 与 CPU 的数值差

定义：

`Δ = GPU - CPU`

因此：

- `Δ < 0`：GPU 数值更低；
- `Δ > 0`：GPU 数值更高。

| 数据集 | Commitment Δ | Prove Δ | Verify Δ | Formal total Δ | Formal total 相对变化 |
|---|---:|---:|---:|---:|---:|
| Cora | +0.346 s | -40.810 s | -0.092 s | -40.555 s | -26.22% |
| Citeseer | +0.032 s | -4.118 s | -0.033 s | -4.118 s | -1.20% |
| Pubmed | -0.967 s | -3.374 s | -0.225 s | -4.567 s | -0.79% |
| PPI | -9.621 s | +152.802 s | -1.489 s | +141.693 s | +3.02% |
| OGBN-Arxiv | +8.294 s | +159.376 s | +55.492 s | +223.162 s | +3.91% |

---

## 3. Trace / Dynamic Commitment 聚合数据

以下字段来自日志中的 `warm:trace:done` 和 `warm:trace:profile` 聚合记录。

> 这些字段之间存在父子包含关系，不能直接把同一行各列相加作为总时间。

### 3.1 Trace 总量与 commitment 相关聚合字段

除特别注明外，计时单位均为秒。

| 数据集 | 后端 | Trace total | Commit dynamic | Materialization | MSM | Derived point |
|---|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 75.799 | 53.097 | 31.549 | 20.446 | 3.712 |
| Cora | GPU | 35.215 | 22.138 | 0.456 | 20.653 | 3.739 |
| Citeseer | CPU | 60.889 | 39.093 | 0.821 | 33.130 | 4.692 |
| Citeseer | GPU | 59.220 | 37.679 | 0.731 | 33.124 | 4.608 |
| Pubmed | CPU | 127.231 | 58.509 | 1.240 | 52.555 | 16.000 |
| Pubmed | GPU | 126.682 | 57.998 | 1.272 | 51.873 | 15.970 |
| PPI | CPU | 1337.271 | 780.575 | 4.054 | 570.562 | 81.334 |
| PPI | GPU | 1341.394 | 782.973 | 3.887 | 563.472 | 79.979 |
| OGBN-Arxiv | CPU | 1786.484 | 1182.120 | 4.599 | 1005.095 | 83.905 |
| OGBN-Arxiv | GPU | 1796.309 | 1187.557 | 4.932 | 1008.723 | 82.778 |

### 3.2 Trace generation / lookup 聚合字段

除 `Finalize` 外，计时单位均为秒；`Finalize` 保留日志中的毫秒值。

| 数据集 | 后端 | Trace generation | Forward | Lookup inverse | Lookup accumulator | Lookup pack | Lookup nested total | Hidden stage | Output stage | Trace other | Finalize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 22.187 | 0.482 | 1.862 | 0.381 | 0.312 | 2.555 | 12.907 | 6.838 | 2.443 | 0.028 |
| Cora | GPU | 12.559 | 0.487 | 1.948 | 0.384 | 0.312 | 2.644 | 3.033 | 6.917 | 2.609 | 0.026 |
| Citeseer | CPU | 20.884 | 0.838 | 4.162 | 0.810 | 0.423 | 5.396 | 5.081 | 6.912 | 8.891 | 0.027 |
| Citeseer | GPU | 20.564 | 0.896 | 3.971 | 0.789 | 0.429 | 5.189 | 5.072 | 6.958 | 8.534 | 0.025 |
| Pubmed | CPU | 65.870 | 2.678 | 7.805 | 1.630 | 1.717 | 11.152 | 14.196 | 42.916 | 8.758 | 0.028 |
| Pubmed | GPU | 65.916 | 2.669 | 7.852 | 1.664 | 1.705 | 11.220 | 14.163 | 42.913 | 8.839 | 0.029 |
| PPI | CPU | 526.118 | 28.356 | 43.651 | 9.684 | 17.716 | 71.051 | 40.348 | 472.155 | 13.615 | 0.032 |
| PPI | GPU | 527.816 | 28.224 | 43.300 | 9.413 | 22.434 | 75.147 | 38.848 | 475.139 | 13.830 | 0.030 |
| OGBN-Arxiv | CPU | 577.513 | 25.300 | 86.292 | 24.263 | 22.417 | 132.972 | 61.834 | 486.894 | 28.785 | 0.032 |
| OGBN-Arxiv | GPU | 581.682 | 24.771 | 85.763 | 24.144 | 28.894 | 138.802 | 64.300 | 489.941 | 27.440 | 0.032 |

---

## 4. Quotient 聚合阶段计时

下表保留日志中 `t_edge`、`t_FH`、`t_N` 的 `coset_prewarm`、`evaluate`、`coefficients` 以及 quotient `commit_batch` 聚合时间。

| 数据集 | 后端 | `t_edge` pre/eval/coef | `t_FH` pre/eval/coef | `t_N` pre/eval/coef | Quotient commit batch |
|---|---:|---:|---:|---:|---:|
| Cora | CPU | 13.305 / 6.121 / 0.026 s | 5.847 / 3.008 / 1.240 s | 4.011 / 4.319 / 0.008 s | 10.884 s |
| Cora | GPU | 13.933 / 6.005 / 0.023 s | 5.891 / 3.130 / 1.129 s | 3.979 / 4.454 / 0.005 s | 10.996 s |
| Citeseer | CPU | 11.687 / 6.796 / 0.021 s | 62.012 / 18.736 / 5.600 s | 3.528 / 4.245 / 0.005 s | 43.656 s |
| Citeseer | GPU | 11.167 / 6.667 / 0.021 s | 62.159 / 18.631 / 4.711 s | 3.481 / 4.381 / 0.008 s | 43.778 s |
| Pubmed | CPU | 76.299 / 56.060 / 0.147 s | 61.562 / 49.251 / 4.907 s | 22.958 / 8.431 / 0.023 s | 43.169 s |
| Pubmed | GPU | 76.287 / 55.467 / 0.131 s | 61.421 / 49.345 / 4.905 s | 20.320 / 8.677 / 0.027 s | 42.914 s |
| PPI | CPU | 1361.891 / 593.757 / 2.628 s | 5.139 / 19.686 / 1.069 s | 43.283 / 7.067 / 0.037 s | 31.852 s |
| PPI | GPU | 1489.358 / 603.695 / 2.712 s | 5.271 / 19.628 / 1.067 s | 43.044 / 7.519 / 0.040 s | 30.676 s |
| OGBN-Arxiv | CPU | 1178.451 / 514.316 / 2.380 s | 130.713 / 597.995 / 11.237 s | 44.402 / 7.612 / 0.136 s | 101.055 s |
| OGBN-Arxiv | GPU | 1290.944 / 515.549 / 2.370 s | 146.160 / 563.394 / 9.673 s | 44.679 / 9.263 / 0.138 s | 106.848 s |

---

## 5. Coset Storage 中出现 file-backed 文件的聚合记录

以下字段来自日志中的 `coset_storage:tiered` 统计。本表采用固定筛选条件：**只列出 `files > 0` 的记录**；`files=0` 的记录不在本表展开。

| 数据集 | 后端 | 域 | 文件数 | File-backed bytes | Lazy blinder views | Avoided materialization (bytes) | Evaluation block |
|---|---:|---|---:|---:|---:|---:|---:|
| Citeseer | CPU | `t_FH` | 2 | 4,294,967,296 | 0 | 0 | 32768 |
| Citeseer | GPU | `t_FH` | 2 | 4,294,967,296 | 0 | 0 | 32768 |
| Pubmed | CPU | `t_edge` | 1010 | 33,889,976,320 | 1626 | 54,559,506,432 | 32768 |
| Pubmed | GPU | `t_edge` | 1010 | 33,889,976,320 | 1626 | 54,559,506,432 | 32768 |
| Pubmed | CPU | `t_FH` | 2 | 4,294,967,296 | 0 | 0 | 32768 |
| Pubmed | GPU | `t_FH` | 2 | 4,294,967,296 | 0 | 0 | 32768 |
| PPI | CPU | `t_edge` | 610 | 654,982,512,640 | 102 | 109,521,666,048 | 32768 |
| PPI | GPU | `t_edge` | 610 | 654,982,512,640 | 103 | 110,595,407,872 | 32768 |
| OGBN-Arxiv | CPU | `t_edge` | 521 | 559,419,490,304 | 87 | 93,415,538,688 | 32768 |
| OGBN-Arxiv | GPU | `t_edge` | 521 | 559,419,490,304 | 87 | 93,415,538,688 | 32768 |
| OGBN-Arxiv | CPU | `t_FH` | 4 | 17,179,869,184 | 0 | 0 | 32768 |
| OGBN-Arxiv | GPU | `t_FH` | 4 | 17,179,869,184 | 0 | 0 | 32768 |

说明：

- `bytes` 与 `lazy_blinder_bytes_avoided` 保留日志中的整数 byte 值，不再合并 CPU/GPU 数值或改写为范围；
- 这些值是日志中的累计/逻辑统计字段；
- 不表示同等数量数据同时常驻内存；
- 不等同于实际物理磁盘读写量。

---

## 6. Domain Opening 聚合数据

各域 opening 存在并行关系，因此 `edge`、`FH`、`N` 等子项不能相加得到 total。

| 数据集 | 后端 | Domain opening total | Edge | FH | N |
|---|---:|---:|---:|---:|---:|
| Cora | CPU | 3.572 s | 3.192 s | 2.907 s | 0.125 s |
| Cora | GPU | 3.022 s | 0.574 s | 2.649 s | 0.065 s |
| Citeseer | CPU | 13.725 s | 3.030 s | 13.026 s | 0.105 s |
| Citeseer | GPU | 14.644 s | 1.550 s | 13.169 s | 0.070 s |
| Pubmed | CPU | 13.652 s | 10.536 s | 12.625 s | 0.608 s |
| Pubmed | GPU | 11.357 s | 2.061 s | 9.551 s | 1.264 s |
| PPI | CPU | 673.082 s | 672.341 s | 4.002 s | 1.467 s |
| PPI | GPU | 670.079 s | 669.331 s | 2.483 s | 0.075 s |
| OGBN-Arxiv | CPU | 589.248 s | 586.173 s | 19.670 s | 2.598 s |
| OGBN-Arxiv | GPU | 614.558 s | 596.187 s | 18.851 s | 3.080 s |

### 6.1 Domain opening total 的 CPU/GPU 数值差

| 数据集 | CPU | GPU | GPU - CPU |
|---|---:|---:|---:|
| Cora | 3.572 s | 3.022 s | -0.550 s |
| Citeseer | 13.725 s | 14.644 s | +0.919 s |
| Pubmed | 13.652 s | 11.357 s | -2.295 s |
| PPI | 673.082 s | 670.079 s | -3.004 s |
| OGBN-Arxiv | 589.248 s | 614.558 s | +25.310 s |

---

## 7. External Opening / KZG Witness MSM 规模信息

原始日志中存在大量逐 opening 的 `kzg:opening_witness_msm` 记录。为控制篇幅，本节采用固定筛选条件：只归纳 `input ≥ 1,000,000` 的 dense witness MSM，不计算 `external_opening` 总耗时。

| 数据集 | 日志中 input ≥ 1,000,000 的 dense witness MSM 规模 |
|---|---|
| Cora | `FH input=4,194,304` |
| Citeseer | `FH input=16,777,216` |
| Pubmed | `FH input=16,777,216` |
| PPI | `edge input=8,388,608`；`FH input=4,194,304` |
| OGBN-Arxiv | `edge input=8,388,608`；`FH input=33,554,432` |

PPI 和 OGBN-Arxiv 的 edge opening 日志中均存在多条 `input=8,388,608` 的 dense witness MSM。

现有日志没有一个可直接使用的完整 `external_opening` 聚合总时间，因此本文件不通过逐条 MSM 记录自行推导该阶段总耗时。

---

## 8. 未逐行保留的重复日志类型

为减少篇幅，以下重复内容不逐行复制：

- 仅表示流程推进的 `start` / `done` 记录；
- public/static polynomial 的逐对象构造记录；
- 权重列、attention 参数、bias 列等逐项重复记录；
- SRS 各域无聚合计时的进度记录；
- warmup / light-warmup 中与正式测量部分重复的对象列表；
- `commit_batch` 内各对象的纯进度记录；
- `t_edge:coset_prewarm:derived:*` 的逐对象进度记录；
- domain opening 中重复的小域进度行；
- 大量逐 opening KZG witness MSM 明细；
- export 等只有流程状态、没有新增聚合运行数据的记录。

这些原始明细仍保留在对应 full-run log 中；本文件只负责汇总，不替代原始日志。
