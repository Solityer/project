# 五数据集 × CPU/GPU 全量运行信息汇总

> 用途：将 5 个数据集 × CPU/GPU 共 10 组 full-run 日志中与后续运行复核、阶段定位和性能分析有关的客观运行信息集中整理，尽量减少重新逐行阅读约 6.6 万行原始日志的需要。  
> 实验批次运行区间：2026-08-08T17:21:00+08:00 至 2026-08-09T03:40:44+08:00。  
> 运行入口：`gatzk_run`。  
> 十组运行最终状态均为 `VERIFY_OK`、`exit_code=0`。  
> 本文件只汇总日志直接记录的事实、日志字段以及可由这些字段直接计算的数值；不在本文件中判断性能原因、优化方向、修改方案或优先级。  
> 对存在父子包含、并行执行或可能重叠的阶段，本文件明确保留原始口径，不把子项简单相加冒充阶段总时间。

---

## 1. 运行对象、时间与日志索引

| 数据集 | 后端 | 原始日志 | Warmup | Start | End | 进程墙钟 | 日志行数 | 日志大小 |
|---|---|---|---|---|---|---:|---:|---:|
| Cora | CPU | `cpu_cora_full_20260808_172100_826915218.log` | `light-warmup` | `2026-08-08T17:21:00+08:00` | `2026-08-08T17:29:07+08:00` | 487 s | 6,959 | 1,071,783 B |
| Cora | GPU | `gpu_cora_full_20260808_222856_228087510.log` | `light-warmup` | `2026-08-08T22:28:56+08:00` | `2026-08-08T22:36:56+08:00` | 480 s | 6,959 | 1,071,776 B |
| Citeseer | CPU | `cpu_citeseer_full_20260808_172907_036383333.log` | `light-warmup` | `2026-08-08T17:29:07+08:00` | `2026-08-08T18:00:28+08:00` | 1881 s | 7,289 | 1,087,623 B |
| Citeseer | GPU | `gpu_citeseer_full_20260808_223656_471575355.log` | `light-warmup` | `2026-08-08T22:36:56+08:00` | `2026-08-08T23:08:10+08:00` | 1874 s | 7,289 | 1,087,606 B |
| Pubmed | CPU | `cpu_pubmed_full_20260808_180028_552808000.log` | `light-warmup` | `2026-08-08T18:00:28+08:00` | `2026-08-08T18:37:25+08:00` | 2217 s | 7,271 | 1,090,705 B |
| Pubmed | GPU | `gpu_pubmed_full_20260808_230810_737176837.log` | `light-warmup` | `2026-08-08T23:08:10+08:00` | `2026-08-08T23:44:59+08:00` | 2209 s | 7,271 | 1,090,664 B |
| PPI | CPU | `cpu_ppi_full_20260808_183725_070481685.log` | `warmup` | `2026-08-08T18:37:25+08:00` | `2026-08-08T20:02:18+08:00` | 5093 s | 6,175 | 561,092 B |
| PPI | GPU | `gpu_ppi_full_20260808_234459_338580084.log` | `warmup` | `2026-08-08T23:44:59+08:00` | `2026-08-09T01:12:27+08:00` | 5248 s | 6,175 | 561,137 B |
| OGBN-Arxiv | CPU | `cpu_ogbn_arxiv_full_20260808_200218_918190902.log` | `light-warmup` | `2026-08-08T20:02:18+08:00` | `2026-08-08T22:28:56+08:00` | 8798 s | 5,119 | 478,983 B |
| OGBN-Arxiv | GPU | `gpu_ogbn_arxiv_full_20260809_011227_531159550.log` | `light-warmup` | `2026-08-09T01:12:27+08:00` | `2026-08-09T03:40:44+08:00` | 8897 s | 5,119 | 478,992 B |

对应配置与后端请求：

| 数据集 | 配置 | CPU 请求 | GPU 请求 |
|---|---|---|---|
| Cora | `configs/cora_secure.cfg` | `cpu` | `cuda_hotspots` |
| Citeseer | `configs/citeseer_secure.cfg` | `cpu` | `cuda_hotspots` |
| Pubmed | `configs/pubmed_secure.cfg` | `cpu` | `cuda_hotspots` |
| PPI | `configs/ppi_secure.cfg` | `cpu` | `cuda_hotspots` |
| OGBN-Arxiv | `configs/ogbn_arxiv_secure.cfg` | `cpu` | `cuda_hotspots` |

每个日志还记录了唯一 `export_tag`：

- Cora CPU: `cpu_cora_full_20260808_172100_826915218`
- Cora GPU: `gpu_cora_full_20260808_222856_228087510`
- Citeseer CPU: `cpu_citeseer_full_20260808_172907_036383333`
- Citeseer GPU: `gpu_citeseer_full_20260808_223656_471575355`
- Pubmed CPU: `cpu_pubmed_full_20260808_180028_552808000`
- Pubmed GPU: `gpu_pubmed_full_20260808_230810_737176837`
- PPI CPU: `cpu_ppi_full_20260808_183725_070481685`
- PPI GPU: `gpu_ppi_full_20260808_234459_338580084`
- OGBN-Arxiv CPU: `cpu_ogbn_arxiv_full_20260808_200218_918190902`
- OGBN-Arxiv GPU: `gpu_ogbn_arxiv_full_20260809_011227_531159550`


### 1.1 Full-run 顶层日志顺序

十组日志的正式运行部分均保留相同的顶层 progress 顺序；预热标签因数据集而异，Cora/Citeseer/Pubmed/OGBN-Arxiv 使用 `light-warmup`，PPI 使用 `warmup`：

```text
{light-warmup|warmup}:context
        ↓
warm:context
        ↓
warm:trace
        ↓
warm:prove
        ├── prove:quotient
        ├── prove:domain_opening
        └── prove:external_opening
        ↓
warm:verify
        ↓
status=VERIFY_OK 正式结果行
        ↓
warm:export
        ↓
exit_code=0
```

`warm:prove` 内部的三段顶层顺序在本批次日志中为 `quotient → domain_opening → external_opening`。上述顺序只描述 console log 中的顶层事件边界，不表示这些父阶段内部没有并行执行。

### 1.2 日志事件解析注意事项

本批次日志存在少量两个事件文本粘连到同一物理行的情况，因此对数值事件做机器统计时不能只依赖物理行首的 `^[progress]`。例如 Citeseer CPU/GPU 各有 1 条 `kzg:opening_witness_msm` 数值事件因前一事件文本直接连接而没有独立的 `[progress]` 行首；本文件对 KZG witness MSM 的统计按事件字段模式识别，已将这两条记录计入。

### 1.3 Context progress-event 结构

下表只统计两个 Context 父阶段边界内部、事件文本以 `:start` 结尾的子 progress 事件。`子 start 事件` 以及各类别列均为**事件数量**，不是时间。成对数值统一按 `初始 warmup / warm:context` 顺序书写。`Public`、`Static`、`Public alias`、`FH coeff` 和 `KZG` 分别按 `context:public:*:start`、`context:static:*:start`、`context:public_alias:*:start`、`context:public_fh_coeff:*:start` 和 `kzg:*:start` 直接计数；未落入这些类别的 Context 顶层子事件仍包含在 `子 start 事件` 总数中。事件数量不能直接解释为 Context 耗时。

| 数据集 | 后端 | 初始 Context 标签 | 子 start 事件（初始 / warm） | Public（初始 / warm） | Static（初始 / warm） | Public alias（初始 / warm） | FH coeff（初始 / warm） | KZG（初始 / warm） |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Cora | CPU | `light-warmup` | 177 / 0 | 58 / 0 | 98 / 0 | 4 / 0 | 4 / 0 | 9 / 0 |
| Cora | GPU | `light-warmup` | 177 / 0 | 58 / 0 | 98 / 0 | 4 / 0 | 4 / 0 | 9 / 0 |
| Citeseer | CPU | `light-warmup` | 176 / 167 | 58 / 58 | 97 / 97 | 4 / 4 | 4 / 4 | 9 / 0 |
| Citeseer | GPU | `light-warmup` | 176 / 167 | 58 / 58 | 97 / 97 | 4 / 4 | 4 / 4 | 9 / 0 |
| Pubmed | CPU | `light-warmup` | 173 / 164 | 58 / 58 | 94 / 94 | 4 / 4 | 4 / 4 | 9 / 0 |
| Pubmed | GPU | `light-warmup` | 173 / 164 | 58 / 58 | 94 / 94 | 4 / 4 | 4 / 4 | 9 / 0 |
| PPI | CPU | `warmup` | 214 / 205 | 51 / 51 | 142 / 142 | 4 / 4 | 4 / 4 | 9 / 0 |
| PPI | GPU | `warmup` | 214 / 205 | 51 / 51 | 142 / 142 | 4 / 4 | 4 / 4 | 9 / 0 |
| OGBN-Arxiv | CPU | `light-warmup` | 133 / 124 | 51 / 51 | 61 / 61 | 4 / 4 | 4 / 4 | 9 / 0 |
| OGBN-Arxiv | GPU | `light-warmup` | 133 / 124 | 51 / 51 | 61 / 61 | 4 / 4 | 4 / 4 | 9 / 0 |

本批次 Cora CPU/GPU 的 `warm:context:start` 与 `warm:context:done` 之间没有记录任何子 `:start` progress 事件，因此相应 warm 侧计数为 0；其余数据集的 warm Context 子事件按日志原样计数。该差异仅作为执行日志结构事实保留，本文件不据此推断原因。

---

## 2. 本批次正式运行结果

定义：`formal_total = commitment_time_ms + prove_time_ms + verify_time_ms`。

`formal_total` 仅由最终结果行的三项正式计时直接相加，不等同于从 `start_time` 到 `end_time` 的整个进程墙钟时间；后者还包含 warmup/light-warmup、Context 构建、正式计时区间外流程和 export 等。
`Process wall` 由日志中秒级 `start_time` 与 `end_time` 直接相减，时间分辨率为 1 s；因此 `Wall - formal` 虽按直接算术差显示到毫秒位，但其整体测量精度仍受 `Process wall` 的 1 s 分辨率约束，不应按毫秒级墙钟精度解释。

| 数据集 | 后端 | Commitment | Prove | Verify | Formal total | Process wall | Wall - formal | Proof size (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 34.833 s | 78.629 s | 0.621 s | 114.083 s | 487.000 s | 372.917 s | 1,462,641 |
| Cora | GPU | 34.701 s | 76.678 s | 0.553 s | 111.932 s | 480.000 s | 368.068 s | 1,462,638 |
| Citeseer | CPU | 81.658 s | 257.255 s | 1.411 s | 340.324 s | 1881.000 s | 1540.676 s | 1,493,253 |
| Citeseer | GPU | 80.743 s | 254.685 s | 1.086 s | 336.514 s | 1874.000 s | 1537.486 s | 1,493,224 |
| Pubmed | CPU | 123.208 s | 487.602 s | 1.737 s | 612.547 s | 2217.000 s | 1604.453 s | 2,743,624 |
| Pubmed | GPU | 122.822 s | 488.309 s | 1.950 s | 613.080 s | 2209.000 s | 1595.920 s | 2,743,585 |
| PPI | CPU | 672.196 s | 3540.018 s | 11.456 s | 4223.670 s | 5093.000 s | 869.330 s | 220,865,254 |
| PPI | GPU | 679.228 s | 3664.583 s | 5.550 s | 4349.361 s | 5248.000 s | 898.639 s | 220,865,259 |
| OGBN-Arxiv | CPU | 1196.393 s | 4108.522 s | 10.176 s | 5315.090 s | 8798.000 s | 3482.910 s | 217,003,705 |
| OGBN-Arxiv | GPU | 1194.991 s | 4280.775 s | 17.427 s | 5493.193 s | 8897.000 s | 3403.807 s | 217,003,698 |

### 2.1 GPU 与 CPU 的直接数值差

定义：`Δ = GPU - CPU`；负值表示 GPU 数值更低，正值表示 GPU 数值更高。

| 数据集 | Commitment Δ | Prove Δ | Verify Δ | Formal total Δ | Formal total 相对变化 | Process wall Δ | Proof size Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | -0.131 s | -1.951 s | -0.068 s | -2.151 s | -1.89% | -7 s | -3 B |
| Citeseer | -0.916 s | -2.570 s | -0.324 s | -3.810 s | -1.12% | -7 s | -29 B |
| Pubmed | -0.386 s | +0.707 s | +0.212 s | +0.533 s | +0.09% | -8 s | -39 B |
| PPI | +7.032 s | +124.565 s | -5.906 s | +125.691 s | +2.98% | +155 s | +5 B |
| OGBN-Arxiv | -1.402 s | +172.254 s | +7.251 s | +178.103 s | +3.35% | +99 s | -7 B |

---

## 3. Trace / Dynamic Commitment 聚合数据

以下字段来自 `warm:trace:done` 和 `warm:trace:profile`。字段之间存在父子包含关系，不能把同一行各列相加作为 Trace 总时间。

### 3.1 Trace 总量与聚合 profile

| 数据集 | 后端 | Trace total | Commit dynamic | Materialization | MSM | Derived point | Trace generation | Forward | Lookup inverse | Lookup accumulator | Lookup pack | Lookup nested total | Hidden stage | Output stage | Trace other | Finalize (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 35.168 | 22.077 | 0.445 | 20.598 | 3.737 | 12.580 | 0.489 | 1.949 | 0.388 | 0.312 | 2.649 | 3.149 | 6.843 | 2.588 | 0.026 |
| Cora | GPU | 35.202 | 21.935 | 0.455 | 20.440 | 3.754 | 12.746 | 0.488 | 1.991 | 0.393 | 0.324 | 2.708 | 3.136 | 6.972 | 2.638 | 0.025 |
| Citeseer | CPU | 59.088 | 37.490 | 0.768 | 32.978 | 4.683 | 20.655 | 0.862 | 3.940 | 0.785 | 0.416 | 5.141 | 5.014 | 6.950 | 8.691 | 0.025 |
| Citeseer | GPU | 60.053 | 37.911 | 0.806 | 33.195 | 4.702 | 21.226 | 0.832 | 4.221 | 0.814 | 0.437 | 5.472 | 5.055 | 6.950 | 9.221 | 0.027 |
| Pubmed | CPU | 138.172 | 67.269 | 0.975 | 61.437 | 17.675 | 68.033 | 2.721 | 7.910 | 1.679 | 1.733 | 11.323 | 14.465 | 44.790 | 8.778 | 0.028 |
| Pubmed | GPU | 132.969 | 62.481 | 1.010 | 56.607 | 17.306 | 67.555 | 2.775 | 7.873 | 1.672 | 1.724 | 11.268 | 14.330 | 44.451 | 8.775 | 0.026 |
| PPI | CPU | 1334.953 | 771.497 | 4.130 | 560.333 | 80.892 | 532.596 | 28.374 | 44.888 | 10.019 | 17.772 | 72.679 | 42.433 | 476.736 | 13.427 | 0.033 |
| PPI | GPU | 1353.073 | 790.823 | 4.111 | 567.245 | 80.286 | 531.561 | 28.307 | 44.011 | 9.680 | 22.910 | 76.601 | 41.793 | 475.914 | 13.853 | 0.035 |
| OGBN-Arxiv | CPU | 1794.504 | 1186.534 | 5.137 | 1008.206 | 83.421 | 581.383 | 24.804 | 86.866 | 24.207 | 22.259 | 133.331 | 61.878 | 490.178 | 29.327 | 0.037 |
| OGBN-Arxiv | GPU | 1806.732 | 1192.542 | 5.186 | 1008.860 | 84.588 | 588.860 | 23.619 | 86.984 | 24.414 | 29.493 | 140.891 | 65.367 | 495.084 | 28.409 | 0.034 |

### 3.2 Trace 聚合字段的 GPU - CPU 差值

| 数据集 | Trace total Δ | Commit dynamic Δ | Materialization Δ | MSM Δ | Derived point Δ | Trace generation Δ | Lookup nested Δ | Hidden stage Δ | Output stage Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | +0.034 s | -0.141 s | +0.010 s | -0.158 s | +0.017 s | +0.166 s | +0.059 s | -0.013 s | +0.129 s |
| Citeseer | +0.965 s | +0.421 s | +0.038 s | +0.218 s | +0.019 s | +0.571 s | +0.331 s | +0.041 s | +0.001 s |
| Pubmed | -5.204 s | -4.788 s | +0.035 s | -4.829 s | -0.369 s | -0.477 s | -0.054 s | -0.135 s | -0.339 s |
| PPI | +18.120 s | +19.326 s | -0.019 s | +6.912 s | -0.606 s | -1.036 s | +3.922 s | -0.640 s | -0.821 s |
| OGBN-Arxiv | +12.228 s | +6.008 s | +0.049 s | +0.653 s | +1.167 s | +7.477 s | +7.560 s | +3.488 s | +4.906 s |

### 3.3 Trace `commit_batch` MSM profile 统计

日志包含大量 `trace:commit_batch:...:msm_profile` 记录。下表同时保留 profile 记录数量、日志字段 `total_ms` 的直接分布统计，以及每条 profile 的 `objects=` 字段统计。`objects` 是该 profile 日志直接记录的实际对象数，不要求等于外层 `commit_batch:size`。`Σ logged profile` 只是这些 profile 行的 `total_ms` 数值之和，不等同于 `commit_dynamic_ms`，不能跨父子/并行关系当作正式阶段总时间。

| 数据集 | 后端 | msm_profile 记录数 | Σ logged profile | 平均 | P95 | 最大 | Σ MSM objects | 平均 objects/profile | 最大 objects/profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 898 | 20.598 s | 22.938 ms | 72.764 ms | 3276.603 ms | 3,456 | 3.849 | 64 |
| Cora | GPU | 898 | 20.440 s | 22.761 ms | 69.727 ms | 3268.101 ms | 3,456 | 3.849 | 64 |
| Citeseer | CPU | 898 | 32.978 s | 36.723 ms | 71.954 ms | 11033.292 ms | 3,451 | 3.843 | 64 |
| Citeseer | GPU | 898 | 33.195 s | 36.966 ms | 71.939 ms | 11090.631 ms | 3,451 | 3.843 | 64 |
| Pubmed | CPU | 898 | 61.437 s | 68.415 ms | 354.891 ms | 11164.927 ms | 3,436 | 3.826 | 64 |
| Pubmed | GPU | 898 | 56.607 s | 63.037 ms | 343.913 ms | 9410.107 ms | 3,436 | 3.826 | 64 |
| PPI | CPU | 733 | 560.333 s | 764.438 ms | 5822.834 ms | 6102.079 ms | 1,422 | 1.940 | 48 |
| PPI | GPU | 733 | 567.245 s | 773.868 ms | 5895.949 ms | 6392.600 ms | 1,422 | 1.940 | 48 |
| OGBN-Arxiv | CPU | 643 | 1008.206 s | 1567.973 ms | 7131.074 ms | 19370.784 ms | 912 | 1.418 | 12 |
| OGBN-Arxiv | GPU | 643 | 1008.860 s | 1568.988 ms | 7129.208 ms | 20038.853 ms | 912 | 1.418 | 12 |

### 3.4 Trace `commit_batch` 调用结构

`trace:commit_batch` 日志中的 `start`、`materialized` 和 `msm_profile` 事件计数并不总相等。下表按事件标记直接计数，并保留 `Start - Materialized` 与 `Materialized - MSM profile` 两个直接算术差。本批次每组运行中 `msm:start` 的计数均与 `msm_profile` 相等。两列差值只表示对应事件数量之差，不预设其具体内部调用语义；如需确定单次调用路径，应回到相应原始日志顺序。

| 数据集 | 后端 | `start` 事件 | `materialized` | Start - Materialized | `linear_derived` 行 | Σ linear-derived | Σ msm_inputs | `msm:start` | `msm_profile` | Materialized - MSM profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 903 | 901 | 2 | 74 | 154 | 362 | 898 | 898 | 3 |
| Cora | GPU | 903 | 901 | 2 | 74 | 154 | 362 | 898 | 898 | 3 |
| Citeseer | CPU | 903 | 901 | 2 | 74 | 154 | 362 | 898 | 898 | 3 |
| Citeseer | GPU | 903 | 901 | 2 | 74 | 154 | 362 | 898 | 898 | 3 |
| Pubmed | CPU | 903 | 901 | 2 | 74 | 154 | 362 | 898 | 898 | 3 |
| Pubmed | GPU | 903 | 901 | 2 | 74 | 154 | 362 | 898 | 898 | 3 |
| PPI | CPU | 970 | 762 | 208 | 33 | 35 | 15 | 733 | 733 | 29 |
| PPI | GPU | 970 | 762 | 208 | 33 | 35 | 15 | 733 | 733 | 29 |
| OGBN-Arxiv | CPU | 850 | 672 | 178 | 33 | 35 | 6 | 643 | 643 | 29 |
| OGBN-Arxiv | GPU | 850 | 672 | 178 | 33 | 35 | 6 | 643 | 643 | 29 |

这些计数是事件结构统计，不是耗时；各事件计数以及两列直接差值不能解释成相互独立的证明阶段。具体父子调用关系需要结合原始日志中的相邻事件顺序判断。

### 3.5 Trace `commit_batch` 按 batch size 的事件与 MSM profile 聚合

下表按日志字段 `size=` 分组，保留各 size 的 start 数、materialized 数、MSM profile 数、profile `total_ms` 直接求和以及 linear-derived / MSM-input 计数。若某个 size 的 `Start events > Materialized`，只表示该 size 下两类事件计数存在差值；具体对应何种内部调用路径需要结合原始日志顺序判断。`Σ profile ms` 仍只是 profile 行的数值和，不是独立墙钟阶段时间。

| 数据集 | 后端 | Batch size | Start events | Materialized | MSM profiles | Σ profile ms | Linear-derived rows | Σ linear-derived | Σ msm_inputs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 1 | 40 | 40 | 37 | 3617.816 | 3 | 3 | 0 |
| Cora | CPU | 2 | 29 | 29 | 29 | 238.063 | 0 | 0 | 0 |
| Cora | CPU | 3 | 403 | 401 | 401 | 10570.226 | 53 | 106 | 53 |
| Cora | CPU | 4 | 346 | 346 | 346 | 4775.680 | 0 | 0 | 0 |
| Cora | CPU | 5 | 8 | 8 | 8 | 192.593 | 0 | 0 | 0 |
| Cora | CPU | 6 | 18 | 18 | 18 | 162.864 | 0 | 0 | 0 |
| Cora | CPU | 7 | 5 | 5 | 5 | 9.665 | 0 | 0 | 0 |
| Cora | CPU | 8 | 24 | 24 | 24 | 198.387 | 0 | 0 | 0 |
| Cora | CPU | 9 | 1 | 1 | 1 | 25.594 | 1 | 3 | 6 |
| Cora | CPU | 12 | 11 | 11 | 11 | 328.522 | 0 | 0 | 0 |
| Cora | CPU | 17 | 1 | 1 | 1 | 22.116 | 1 | 2 | 15 |
| Cora | CPU | 18 | 8 | 8 | 8 | 246.548 | 8 | 16 | 128 |
| Cora | CPU | 23 | 8 | 8 | 8 | 190.382 | 8 | 24 | 160 |
| Cora | CPU | 64 | 1 | 1 | 1 | 19.706 | 0 | 0 | 0 |
| Cora | GPU | 1 | 40 | 40 | 37 | 3609.386 | 3 | 3 | 0 |
| Cora | GPU | 2 | 29 | 29 | 29 | 230.540 | 0 | 0 | 0 |
| Cora | GPU | 3 | 403 | 401 | 401 | 10570.248 | 53 | 106 | 53 |
| Cora | GPU | 4 | 346 | 346 | 346 | 4684.719 | 0 | 0 | 0 |
| Cora | GPU | 5 | 8 | 8 | 8 | 177.081 | 0 | 0 | 0 |
| Cora | GPU | 6 | 18 | 18 | 18 | 171.458 | 0 | 0 | 0 |
| Cora | GPU | 7 | 5 | 5 | 5 | 8.859 | 0 | 0 | 0 |
| Cora | GPU | 8 | 24 | 24 | 24 | 206.055 | 0 | 0 | 0 |
| Cora | GPU | 9 | 1 | 1 | 1 | 26.178 | 1 | 3 | 6 |
| Cora | GPU | 12 | 11 | 11 | 11 | 300.981 | 0 | 0 | 0 |
| Cora | GPU | 17 | 1 | 1 | 1 | 21.809 | 1 | 2 | 15 |
| Cora | GPU | 18 | 8 | 8 | 8 | 221.964 | 8 | 16 | 128 |
| Cora | GPU | 23 | 8 | 8 | 8 | 191.067 | 8 | 24 | 160 |
| Cora | GPU | 64 | 1 | 1 | 1 | 19.329 | 0 | 0 | 0 |
| Citeseer | CPU | 1 | 40 | 40 | 37 | 12766.669 | 3 | 3 | 0 |
| Citeseer | CPU | 2 | 29 | 29 | 29 | 30.960 | 0 | 0 | 0 |
| Citeseer | CPU | 3 | 403 | 401 | 401 | 15759.787 | 53 | 106 | 53 |
| Citeseer | CPU | 4 | 346 | 346 | 346 | 4062.478 | 0 | 0 | 0 |
| Citeseer | CPU | 5 | 8 | 8 | 8 | 30.206 | 0 | 0 | 0 |
| Citeseer | CPU | 6 | 23 | 23 | 23 | 50.236 | 0 | 0 | 0 |
| Citeseer | CPU | 8 | 24 | 24 | 24 | 42.333 | 0 | 0 | 0 |
| Citeseer | CPU | 9 | 1 | 1 | 1 | 9.461 | 1 | 3 | 6 |
| Citeseer | CPU | 12 | 11 | 11 | 11 | 64.922 | 0 | 0 | 0 |
| Citeseer | CPU | 17 | 1 | 1 | 1 | 2.309 | 1 | 2 | 15 |
| Citeseer | CPU | 18 | 8 | 8 | 8 | 26.974 | 8 | 16 | 128 |
| Citeseer | CPU | 23 | 8 | 8 | 8 | 111.702 | 8 | 24 | 160 |
| Citeseer | CPU | 64 | 1 | 1 | 1 | 19.586 | 0 | 0 | 0 |
| Citeseer | GPU | 1 | 40 | 40 | 37 | 12875.632 | 3 | 3 | 0 |
| Citeseer | GPU | 2 | 29 | 29 | 29 | 27.626 | 0 | 0 | 0 |
| Citeseer | GPU | 3 | 403 | 401 | 401 | 15800.545 | 53 | 106 | 53 |
| Citeseer | GPU | 4 | 346 | 346 | 346 | 4075.062 | 0 | 0 | 0 |
| Citeseer | GPU | 5 | 8 | 8 | 8 | 48.250 | 0 | 0 | 0 |
| Citeseer | GPU | 6 | 23 | 23 | 23 | 49.067 | 0 | 0 | 0 |
| Citeseer | GPU | 8 | 24 | 24 | 24 | 39.587 | 0 | 0 | 0 |
| Citeseer | GPU | 9 | 1 | 1 | 1 | 9.140 | 1 | 3 | 6 |
| Citeseer | GPU | 12 | 11 | 11 | 11 | 95.646 | 0 | 0 | 0 |
| Citeseer | GPU | 17 | 1 | 1 | 1 | 2.102 | 1 | 2 | 15 |
| Citeseer | GPU | 18 | 8 | 8 | 8 | 36.815 | 8 | 16 | 128 |
| Citeseer | GPU | 23 | 8 | 8 | 8 | 116.293 | 8 | 24 | 160 |
| Citeseer | GPU | 64 | 1 | 1 | 1 | 19.460 | 0 | 0 | 0 |
| Pubmed | CPU | 1 | 40 | 40 | 37 | 12818.654 | 3 | 3 | 0 |
| Pubmed | CPU | 2 | 29 | 29 | 29 | 121.024 | 0 | 0 | 0 |
| Pubmed | CPU | 3 | 408 | 406 | 406 | 26269.541 | 53 | 106 | 53 |
| Pubmed | CPU | 4 | 346 | 346 | 346 | 20651.678 | 0 | 0 | 0 |
| Pubmed | CPU | 5 | 8 | 8 | 8 | 199.768 | 0 | 0 | 0 |
| Pubmed | CPU | 6 | 18 | 18 | 18 | 143.692 | 0 | 0 | 0 |
| Pubmed | CPU | 8 | 24 | 24 | 24 | 96.065 | 0 | 0 | 0 |
| Pubmed | CPU | 9 | 1 | 1 | 1 | 63.682 | 1 | 3 | 6 |
| Pubmed | CPU | 12 | 11 | 11 | 11 | 332.551 | 0 | 0 | 0 |
| Pubmed | CPU | 17 | 1 | 1 | 1 | 7.646 | 1 | 2 | 15 |
| Pubmed | CPU | 18 | 8 | 8 | 8 | 165.296 | 8 | 16 | 128 |
| Pubmed | CPU | 23 | 8 | 8 | 8 | 527.761 | 8 | 24 | 160 |
| Pubmed | CPU | 64 | 1 | 1 | 1 | 39.434 | 0 | 0 | 0 |
| Pubmed | GPU | 1 | 40 | 40 | 37 | 10940.505 | 3 | 3 | 0 |
| Pubmed | GPU | 2 | 29 | 29 | 29 | 117.021 | 0 | 0 | 0 |
| Pubmed | GPU | 3 | 408 | 406 | 406 | 25021.600 | 53 | 106 | 53 |
| Pubmed | GPU | 4 | 346 | 346 | 346 | 19016.693 | 0 | 0 | 0 |
| Pubmed | GPU | 5 | 8 | 8 | 8 | 199.439 | 0 | 0 | 0 |
| Pubmed | GPU | 6 | 18 | 18 | 18 | 144.948 | 0 | 0 | 0 |
| Pubmed | GPU | 8 | 24 | 24 | 24 | 93.286 | 0 | 0 | 0 |
| Pubmed | GPU | 9 | 1 | 1 | 1 | 63.769 | 1 | 3 | 6 |
| Pubmed | GPU | 12 | 11 | 11 | 11 | 284.327 | 0 | 0 | 0 |
| Pubmed | GPU | 17 | 1 | 1 | 1 | 7.455 | 1 | 2 | 15 |
| Pubmed | GPU | 18 | 8 | 8 | 8 | 148.590 | 8 | 16 | 128 |
| Pubmed | GPU | 23 | 8 | 8 | 8 | 529.940 | 8 | 24 | 160 |
| Pubmed | GPU | 64 | 1 | 1 | 1 | 39.870 | 0 | 0 | 0 |
| PPI | CPU | 1 | 721 | 721 | 694 | 551348.460 | 27 | 27 | 0 |
| PPI | CPU | 2 | 13 | 7 | 5 | 1383.392 | 2 | 4 | 0 |
| PPI | CPU | 3 | 102 | 2 | 2 | 1126.336 | 0 | 0 | 0 |
| PPI | CPU | 4 | 90 | 4 | 4 | 1640.984 | 2 | 2 | 6 |
| PPI | CPU | 5 | 4 | 3 | 3 | 1709.297 | 1 | 1 | 4 |
| PPI | CPU | 6 | 7 | 4 | 4 | 1224.992 | 1 | 1 | 5 |
| PPI | CPU | 8 | 4 | 4 | 4 | 440.145 | 0 | 0 | 0 |
| PPI | CPU | 9 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | CPU | 12 | 5 | 2 | 2 | 28.003 | 0 | 0 | 0 |
| PPI | CPU | 17 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | CPU | 18 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | CPU | 23 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | CPU | 25 | 5 | 5 | 5 | 331.075 | 0 | 0 | 0 |
| PPI | CPU | 48 | 10 | 10 | 10 | 1100.729 | 0 | 0 | 0 |
| PPI | CPU | 121 | 5 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | GPU | 1 | 721 | 721 | 694 | 557902.493 | 27 | 27 | 0 |
| PPI | GPU | 2 | 13 | 7 | 5 | 1478.300 | 2 | 4 | 0 |
| PPI | GPU | 3 | 102 | 2 | 2 | 1365.688 | 0 | 0 | 0 |
| PPI | GPU | 4 | 90 | 4 | 4 | 1656.006 | 2 | 2 | 6 |
| PPI | GPU | 5 | 4 | 3 | 3 | 1702.898 | 1 | 1 | 4 |
| PPI | GPU | 6 | 7 | 4 | 4 | 1242.274 | 1 | 1 | 5 |
| PPI | GPU | 8 | 4 | 4 | 4 | 436.045 | 0 | 0 | 0 |
| PPI | GPU | 9 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | GPU | 12 | 5 | 2 | 2 | 20.623 | 0 | 0 | 0 |
| PPI | GPU | 17 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | GPU | 18 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | GPU | 23 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| PPI | GPU | 25 | 5 | 5 | 5 | 333.855 | 0 | 0 | 0 |
| PPI | GPU | 48 | 10 | 10 | 10 | 1107.059 | 0 | 0 | 0 |
| PPI | GPU | 121 | 5 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 1 | 618 | 618 | 591 | 993129.849 | 27 | 27 | 0 |
| OGBN-Arxiv | CPU | 2 | 23 | 17 | 15 | 8628.138 | 4 | 6 | 2 |
| OGBN-Arxiv | CPU | 3 | 92 | 7 | 7 | 2923.177 | 2 | 2 | 4 |
| OGBN-Arxiv | CPU | 4 | 79 | 8 | 8 | 1452.536 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 5 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 6 | 4 | 1 | 1 | 2.144 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 8 | 4 | 4 | 4 | 633.207 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 9 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 12 | 20 | 17 | 17 | 1437.417 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 17 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 18 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 23 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | CPU | 40 | 5 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 1 | 618 | 618 | 591 | 993753.550 | 27 | 27 | 0 |
| OGBN-Arxiv | GPU | 2 | 23 | 17 | 15 | 8595.680 | 4 | 6 | 2 |
| OGBN-Arxiv | GPU | 3 | 92 | 7 | 7 | 2925.538 | 2 | 2 | 4 |
| OGBN-Arxiv | GPU | 4 | 79 | 8 | 8 | 1454.347 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 5 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 6 | 4 | 1 | 1 | 2.203 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 8 | 4 | 4 | 4 | 682.938 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 9 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 12 | 20 | 17 | 17 | 1445.322 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 17 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 18 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 23 | 1 | 0 | 0 | 0.000 | 0 | 0 | 0 |
| OGBN-Arxiv | GPU | 40 | 5 | 0 | 0 | 0.000 | 0 | 0 | 0 |

### 3.6 每组运行耗时最高的 10 条 Trace MSM profile

`Top object` 保留对应 profile 行 `top=` 中排在第一位的对象，包括域、表示、长度、非零数和路径标签。
`top=` 中各对象自己的 `ms=` 保留日志原始对象级计时口径；这些对象级 `ms` 不能简单相加作为该 `msm_profile` 的 `total_ms`。本文件在 profile 聚合统计中只使用日志直接给出的 `total_ms`。

| 数据集 | 后端 | Rank | Batch total | Batch size | Labels | Top object |
|---|---:|---:|---:|---:|---|---|
| Cora | CPU | 1 | 3.277 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=4194304/nz=3880564/ms=3257.401/mt` |
| Cora | CPU | 2 | 0.234 s | 1 | `P_m_feat` | `P_m_feat@FH/evaluation/len=4194304/nz=3880564/ms=214.764/sparse/mt` |
| Cora | CPU | 3 | 0.085 s | 4 | `P_h3_J_src_node,P_h3_J_src_edge,P_h3_R_src_node,P_h3_R_src` | `P_h3_R_src@edge/evaluation/len=32768/nz=13264/ms=53.706/tail/mt` |
| Cora | CPU | 4 | 0.081 s | 4 | `P_h1_AggBind_src_J_node,P_h1_AggBind_src_J_edge,P_h1_AggBind_src_R_node,P_h1_AggBind_src_R_edge` | `P_h1_AggBind_src_J_edge@edge/evaluation/len=32768/nz=13264/ms=61.212/tail/mt` |
| Cora | CPU | 5 | 0.078 s | 4 | `P_h3_J_t_node,P_h3_J_t_edge,P_h3_R_t_node,P_h3_R_t` | `P_h3_R_t@edge/evaluation/len=32768/nz=13264/ms=53.077/tail/mt` |
| Cora | CPU | 6 | 0.077 s | 4 | `P_h0_AggBind_dst_J_node,P_h0_AggBind_dst_J_edge,P_h0_AggBind_dst_R_node,P_h0_AggBind_dst_R_edge` | `P_h0_AggBind_dst_R_edge@edge/evaluation/len=32768/nz=13264/ms=53.385/tail/mt` |
| Cora | CPU | 7 | 0.077 s | 4 | `P_h0_J_t_node,P_h0_J_t_edge,P_h0_R_t_node,P_h0_R_t` | `P_h0_R_t@edge/evaluation/len=32768/nz=13264/ms=52.495/tail/mt` |
| Cora | CPU | 8 | 0.077 s | 4 | `P_out0_AggBind_src_J_node,P_out0_AggBind_src_J_edge,P_out0_AggBind_src_R_node,P_out0_AggBind_src_R_edge` | `P_out0_AggBind_src_J_edge@edge/evaluation/len=32768/nz=13264/ms=53.465/tail/mt` |
| Cora | CPU | 9 | 0.077 s | 4 | `P_h6_AggBind_src_J_node,P_h6_AggBind_src_J_edge,P_h6_AggBind_src_R_node,P_h6_AggBind_src_R_edge` | `P_h6_AggBind_src_R_edge@edge/evaluation/len=32768/nz=13264/ms=51.028/tail/mt` |
| Cora | CPU | 10 | 0.077 s | 4 | `P_h6_J_dst_node,P_h6_J_dst_edge,P_h6_R_dst_node,P_h6_R_dst` | `P_h6_J_dst_edge@edge/evaluation/len=32768/nz=13264/ms=53.501/tail/mt` |
| Cora | GPU | 1 | 3.268 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=4194304/nz=3880564/ms=3248.650/mt` |
| Cora | GPU | 2 | 0.234 s | 1 | `P_m_feat` | `P_m_feat@FH/evaluation/len=4194304/nz=3880564/ms=214.814/sparse/mt` |
| Cora | GPU | 3 | 0.080 s | 4 | `P_h5_J_src_node,P_h5_J_src_edge,P_h5_R_src_node,P_h5_R_src` | `P_h5_R_src@edge/evaluation/len=32768/nz=13264/ms=48.273/tail/mt` |
| Cora | GPU | 4 | 0.080 s | 4 | `P_h3_J_t_node,P_h3_J_t_edge,P_h3_R_t_node,P_h3_R_t` | `P_h3_R_t@edge/evaluation/len=32768/nz=13264/ms=56.252/tail/mt` |
| Cora | GPU | 5 | 0.080 s | 4 | `P_out0_J_t_node,P_out0_J_t_edge,P_out0_R_t_node,P_out0_R_t` | `P_out0_R_t@edge/evaluation/len=32768/nz=13264/ms=59.784/tail/mt` |
| Cora | GPU | 6 | 0.079 s | 4 | `P_out0_AggBind_dst_J_node,P_out0_AggBind_dst_J_edge,P_out0_AggBind_dst_R_node,P_out0_AggBind_dst_R_edge` | `P_out0_AggBind_dst_R_edge@edge/evaluation/len=32768/nz=13264/ms=60.066/tail/mt` |
| Cora | GPU | 7 | 0.079 s | 4 | `P_h2_J_src_node,P_h2_J_src_edge,P_h2_R_src_node,P_h2_R_src` | `P_h2_R_src@edge/evaluation/len=32768/nz=13264/ms=52.376/tail/mt` |
| Cora | GPU | 8 | 0.077 s | 4 | `P_h6_AggBind_src_J_node,P_h6_AggBind_src_J_edge,P_h6_AggBind_src_R_node,P_h6_AggBind_src_R_edge` | `P_h6_AggBind_src_R_edge@edge/evaluation/len=32768/nz=13264/ms=52.308/tail/mt` |
| Cora | GPU | 9 | 0.077 s | 4 | `P_h3_AggBind_src_J_node,P_h3_AggBind_src_J_edge,P_h3_AggBind_src_R_node,P_h3_AggBind_src_R_edge` | `P_h3_AggBind_src_R_edge@edge/evaluation/len=32768/nz=13264/ms=54.003/tail/mt` |
| Cora | GPU | 10 | 0.077 s | 4 | `P_h5_AggBind_src_J_node,P_h5_AggBind_src_J_edge,P_h5_AggBind_src_R_node,P_h5_AggBind_src_R_edge` | `P_h5_AggBind_src_R_edge@edge/evaluation/len=32768/nz=13264/ms=55.052/tail/mt` |
| Citeseer | CPU | 1 | 11.033 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=16777216/nz=12319881/ms=10663.024/tail/mt` |
| Citeseer | CPU | 2 | 1.524 s | 1 | `P_m_feat` | `P_m_feat@FH/evaluation/len=16777216/nz=12319881/ms=1178.744/sparse/tail/mt` |
| Citeseer | CPU | 3 | 0.197 s | 1 | `P_R_feat` | `P_R_feat@FH/evaluation/len=16777216/nz=0/ms=96.322/sparse/tail` |
| Citeseer | CPU | 4 | 0.077 s | 4 | `P_h5_J_dst_node,P_h5_J_dst_edge,P_h5_R_dst_node,P_h5_R_dst` | `P_h5_R_dst@edge/evaluation/len=32768/nz=12431/ms=54.887/tail/mt` |
| Citeseer | CPU | 5 | 0.077 s | 4 | `P_h3_J_t_node,P_h3_J_t_edge,P_h3_R_t_node,P_h3_R_t` | `P_h3_R_t@edge/evaluation/len=32768/nz=12431/ms=52.226/tail/mt` |
| Citeseer | CPU | 6 | 0.076 s | 4 | `P_h1_J_t_node,P_h1_J_t_edge,P_h1_R_t_node,P_h1_R_t` | `P_h1_R_t@edge/evaluation/len=32768/nz=12431/ms=52.853/tail/mt` |
| Citeseer | CPU | 7 | 0.076 s | 4 | `P_out0_J_src_node,P_out0_J_src_edge,P_out0_R_src_node,P_out0_R_src` | `P_out0_R_src@edge/evaluation/len=32768/nz=12431/ms=52.168/tail/mt` |
| Citeseer | CPU | 8 | 0.075 s | 4 | `P_h1_J_dst_node,P_h1_J_dst_edge,P_h1_R_dst_node,P_h1_R_dst` | `P_h1_R_dst@edge/evaluation/len=32768/nz=12431/ms=52.048/tail/mt` |
| Citeseer | CPU | 9 | 0.075 s | 4 | `P_h0_AggBind_dst_J_node,P_h0_AggBind_dst_J_edge,P_h0_AggBind_dst_R_node,P_h0_AggBind_dst_R_edge` | `P_h0_AggBind_dst_R_edge@edge/evaluation/len=32768/nz=12431/ms=52.566/tail/mt` |
| Citeseer | CPU | 10 | 0.075 s | 4 | `P_h2_J_t_node,P_h2_J_t_edge,P_h2_R_t_node,P_h2_R_t` | `P_h2_R_t@edge/evaluation/len=32768/nz=12431/ms=54.055/tail/mt` |
| Citeseer | GPU | 1 | 11.091 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=16777216/nz=12319881/ms=10719.590/tail/mt` |
| Citeseer | GPU | 2 | 1.550 s | 1 | `P_m_feat` | `P_m_feat@FH/evaluation/len=16777216/nz=12319881/ms=1196.557/sparse/tail/mt` |
| Citeseer | GPU | 3 | 0.222 s | 1 | `P_R_feat` | `P_R_feat@FH/evaluation/len=16777216/nz=0/ms=111.331/sparse/tail` |
| Citeseer | GPU | 4 | 0.078 s | 3 | `P_h6_Div_Range_Q_5_J_tbl,P_h6_Div_Range_Q_5_J_qry,P_h6_Div_Range_Q_5_R` | `P_h6_Div_Range_Q_5_R@edge/evaluation/len=32768/nz=26615/ms=76.709/tail/mt` |
| Citeseer | GPU | 5 | 0.077 s | 4 | `P_h2_J_t_node,P_h2_J_t_edge,P_h2_R_t_node,P_h2_R_t` | `P_h2_R_t@edge/evaluation/len=32768/nz=12431/ms=54.371/tail/mt` |
| Citeseer | GPU | 6 | 0.076 s | 4 | `P_h1_J_t_node,P_h1_J_t_edge,P_h1_R_t_node,P_h1_R_t` | `P_h1_R_t@edge/evaluation/len=32768/nz=12431/ms=52.951/tail/mt` |
| Citeseer | GPU | 7 | 0.076 s | 4 | `P_h6_J_t_node,P_h6_J_t_edge,P_h6_R_t_node,P_h6_R_t` | `P_h6_R_t@edge/evaluation/len=32768/nz=12431/ms=54.198/tail/mt` |
| Citeseer | GPU | 8 | 0.075 s | 4 | `P_out0_J_t_node,P_out0_J_t_edge,P_out0_R_t_node,P_out0_R_t` | `P_out0_R_t@edge/evaluation/len=32768/nz=12431/ms=50.851/tail/mt` |
| Citeseer | GPU | 9 | 0.075 s | 4 | `P_h0_J_dst_node,P_h0_J_dst_edge,P_h0_R_dst_node,P_h0_R_dst` | `P_h0_R_dst@edge/evaluation/len=32768/nz=12431/ms=51.712/tail/mt` |
| Citeseer | GPU | 10 | 0.075 s | 4 | `P_h6_J_src_node,P_h6_J_src_edge,P_h6_R_src_node,P_h6_R_src` | `P_h6_R_src@edge/evaluation/len=32768/nz=12431/ms=52.620/tail/mt` |
| Pubmed | CPU | 1 | 11.165 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=16777216/nz=9858500/ms=10850.384/tail/mt` |
| Pubmed | CPU | 2 | 1.411 s | 1 | `P_m_feat` | `P_m_feat@FH/evaluation/len=16777216/nz=9858500/ms=1109.402/sparse/tail/mt` |
| Pubmed | CPU | 3 | 0.530 s | 4 | `P_out0_J_src_node,P_out0_J_src_edge,P_out0_R_src_node,P_out0_R_src` | `P_out0_J_src_edge@edge/evaluation/len=262144/nz=108365/ms=524.322/tail/mt` |
| Pubmed | CPU | 4 | 0.519 s | 4 | `P_out0_J_t_node,P_out0_J_t_edge,P_out0_R_t_node,P_out0_R_t` | `P_out0_J_t_edge@edge/evaluation/len=262144/nz=108365/ms=513.161/tail/mt` |
| Pubmed | CPU | 5 | 0.509 s | 4 | `P_h7_J_dst_node,P_h7_J_dst_edge,P_h7_R_dst_node,P_h7_R_dst` | `P_h7_J_dst_edge@edge/evaluation/len=262144/nz=108365/ms=503.241/tail/mt` |
| Pubmed | CPU | 6 | 0.503 s | 4 | `P_h0_J_dst_node,P_h0_J_dst_edge,P_h0_R_dst_node,P_h0_R_dst` | `P_h0_J_dst_edge@edge/evaluation/len=262144/nz=108365/ms=496.871/tail/mt` |
| Pubmed | CPU | 7 | 0.489 s | 4 | `P_h0_J_t_node,P_h0_J_t_edge,P_h0_R_t_node,P_h0_R_t` | `P_h0_J_t_edge@edge/evaluation/len=262144/nz=108365/ms=483.310/tail/mt` |
| Pubmed | CPU | 8 | 0.486 s | 4 | `P_h2_J_dst_node,P_h2_J_dst_edge,P_h2_R_dst_node,P_h2_R_dst` | `P_h2_J_dst_edge@edge/evaluation/len=262144/nz=108365/ms=479.456/tail/mt` |
| Pubmed | CPU | 9 | 0.464 s | 4 | `P_h0_J_src_node,P_h0_J_src_edge,P_h0_R_src_node,P_h0_R_src` | `P_h0_J_src_edge@edge/evaluation/len=262144/nz=108365/ms=457.750/tail/mt` |
| Pubmed | CPU | 10 | 0.439 s | 4 | `P_h1_J_dst_node,P_h1_J_dst_edge,P_h1_R_dst_node,P_h1_R_dst` | `P_h1_J_dst_edge@edge/evaluation/len=262144/nz=108365/ms=432.632/tail/mt` |
| Pubmed | GPU | 1 | 9.410 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=16777216/nz=9858500/ms=9097.428/tail/mt` |
| Pubmed | GPU | 2 | 1.284 s | 1 | `P_m_feat` | `P_m_feat@FH/evaluation/len=16777216/nz=9858500/ms=978.651/sparse/tail/mt` |
| Pubmed | GPU | 3 | 0.499 s | 4 | `P_h7_J_src_node,P_h7_J_src_edge,P_h7_R_src_node,P_h7_R_src` | `P_h7_J_src_edge@edge/evaluation/len=262144/nz=108365/ms=492.840/tail/mt` |
| Pubmed | GPU | 4 | 0.467 s | 4 | `P_h7_J_t_node,P_h7_J_t_edge,P_h7_R_t_node,P_h7_R_t` | `P_h7_J_t_edge@edge/evaluation/len=262144/nz=108365/ms=460.219/tail/mt` |
| Pubmed | GPU | 5 | 0.456 s | 4 | `P_h7_J_dst_node,P_h7_J_dst_edge,P_h7_R_dst_node,P_h7_R_dst` | `P_h7_J_dst_edge@edge/evaluation/len=262144/nz=108365/ms=449.394/tail/mt` |
| Pubmed | GPU | 6 | 0.416 s | 4 | `P_h0_AggBind_src_J_node,P_h0_AggBind_src_J_edge,P_h0_AggBind_src_R_node,P_h0_AggBind_src_R_edge` | `P_h0_AggBind_src_J_edge@edge/evaluation/len=262144/nz=108365/ms=409.605/tail/mt` |
| Pubmed | GPU | 7 | 0.415 s | 4 | `P_out0_J_dst_node,P_out0_J_dst_edge,P_out0_R_dst_node,P_out0_R_dst` | `P_out0_J_dst_edge@edge/evaluation/len=262144/nz=108365/ms=408.477/tail/mt` |
| Pubmed | GPU | 8 | 0.413 s | 4 | `P_out0_J_t_node,P_out0_J_t_edge,P_out0_R_t_node,P_out0_R_t` | `P_out0_J_t_edge@edge/evaluation/len=262144/nz=108365/ms=406.603/tail/mt` |
| Pubmed | GPU | 9 | 0.405 s | 4 | `P_h1_AggBind_dst_J_node,P_h1_AggBind_dst_J_edge,P_h1_AggBind_dst_R_node,P_h1_AggBind_dst_R_edge` | `P_h1_AggBind_dst_J_edge@edge/evaluation/len=262144/nz=108365/ms=398.454/tail/mt` |
| Pubmed | GPU | 10 | 0.402 s | 4 | `P_h4_AggBind_dst_J_node,P_h4_AggBind_dst_J_edge,P_h4_AggBind_dst_R_node,P_h4_AggBind_dst_R_edge` | `P_h4_AggBind_dst_J_edge@edge/evaluation/len=262144/nz=108365/ms=396.276/tail/mt` |
| PPI | CPU | 1 | 6.102 s | 1 | `P_out0_BiasView_R_qry` | `P_out0_BiasView_R_qry@edge/evaluation/len=8388608/nz=6890224/ms=6065.326/tail/mt` |
| PPI | CPU | 2 | 6.039 s | 1 | `P_avg_Bind_Acc` | `P_avg_Bind_Acc@edge/evaluation/len=8388608/nz=6890224/ms=6001.719/tail/mt` |
| PPI | CPU | 3 | 6.035 s | 1 | `P_avg_Div_Range_A_0_R` | `P_avg_Div_Range_A_0_R@edge/evaluation/len=8388608/nz=6890223/ms=5996.980/tail/mt` |
| PPI | CPU | 4 | 6.033 s | 1 | `P_avg_Div_Range_Q_1_R` | `P_avg_Div_Range_Q_1_R@edge/evaluation/len=8388608/nz=6890223/ms=5994.237/tail/mt` |
| PPI | CPU | 5 | 6.031 s | 1 | `P_out0_Bind_Acc` | `P_out0_Bind_Acc@edge/evaluation/len=8388608/nz=6890224/ms=5993.137/tail/mt` |
| PPI | CPU | 6 | 6.025 s | 1 | `P_avg_Div_Range_DMinus1_4_R` | `P_avg_Div_Range_DMinus1_4_R@edge/evaluation/len=8388608/nz=6890223/ms=5986.154/tail/mt` |
| PPI | CPU | 7 | 6.024 s | 1 | `P_avg_Div_Range_Q_0_R` | `P_avg_Div_Range_Q_0_R@edge/evaluation/len=8388608/nz=6890223/ms=5984.323/tail/mt` |
| PPI | CPU | 8 | 6.019 s | 1 | `P_out0_Div_Range_Delta_0_R` | `P_out0_Div_Range_Delta_0_R@edge/evaluation/len=8388608/nz=6890223/ms=5980.271/tail/mt` |
| PPI | CPU | 9 | 6.017 s | 1 | `P_out0_Bind_b` | `P_out0_Bind_b@edge/evaluation/len=8388608/nz=6890224/ms=5977.749/tail/mt` |
| PPI | CPU | 10 | 6.016 s | 1 | `P_out0_Div_Range_DMinus1_0_R` | `P_out0_Div_Range_DMinus1_0_R@edge/evaluation/len=8388608/nz=6890223/ms=5976.411/tail/mt` |
| PPI | GPU | 1 | 6.393 s | 1 | `P_out0_Div_Range_Q_2_R` | `P_out0_Div_Range_Q_2_R@edge/evaluation/len=8388608/nz=6890223/ms=6354.105/tail/mt` |
| PPI | GPU | 2 | 6.178 s | 1 | `P_out0_Div_Range_A_4_R` | `P_out0_Div_Range_A_4_R@edge/evaluation/len=8388608/nz=6890223/ms=6138.974/tail/mt` |
| PPI | GPU | 3 | 6.161 s | 1 | `P_out0_Div_Range_A_3_R` | `P_out0_Div_Range_A_3_R@edge/evaluation/len=8388608/nz=6890223/ms=6122.023/tail/mt` |
| PPI | GPU | 4 | 6.115 s | 1 | `P_out0_Div_Range_DMinus1_0_R` | `P_out0_Div_Range_DMinus1_0_R@edge/evaluation/len=8388608/nz=6890223/ms=6076.050/tail/mt` |
| PPI | GPU | 5 | 6.101 s | 1 | `P_out0_Div_Range_A_1_R` | `P_out0_Div_Range_A_1_R@edge/evaluation/len=8388608/nz=6890223/ms=6061.099/tail/mt` |
| PPI | GPU | 6 | 6.097 s | 1 | `P_out0_Div_Range_DMinus1_2_R` | `P_out0_Div_Range_DMinus1_2_R@edge/evaluation/len=8388608/nz=6890223/ms=6057.917/tail/mt` |
| PPI | GPU | 7 | 6.084 s | 1 | `P_out0_Div_Range_A_2_R` | `P_out0_Div_Range_A_2_R@edge/evaluation/len=8388608/nz=6890223/ms=6044.776/tail/mt` |
| PPI | GPU | 8 | 6.051 s | 1 | `P_out0_Div_Range_DMinus1_1_R` | `P_out0_Div_Range_DMinus1_1_R@edge/evaluation/len=8388608/nz=6890223/ms=6011.305/tail/mt` |
| PPI | GPU | 9 | 6.040 s | 1 | `P_avg_Div_Range_Q_4_R` | `P_avg_Div_Range_Q_4_R@edge/evaluation/len=8388608/nz=6890223/ms=6000.592/tail/mt` |
| PPI | GPU | 10 | 6.039 s | 1 | `P_avg_Div_Range_R_3_R` | `P_avg_Div_Range_R_3_R@edge/evaluation/len=8388608/nz=6890223/ms=5999.353/tail/mt` |
| OGBN-Arxiv | CPU | 1 | 19.371 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=33554432/nz=21675904/ms=18689.400/tail/mt` |
| OGBN-Arxiv | CPU | 2 | 7.298 s | 1 | `P_out0_Div_Range_Delta_3_J_tbl` | `P_out0_Div_Range_Delta_3_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7260.743/mt` |
| OGBN-Arxiv | CPU | 3 | 7.288 s | 1 | `P_out0_Div_Range_Q_0_J_tbl` | `P_out0_Div_Range_Q_0_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7251.284/mt` |
| OGBN-Arxiv | CPU | 4 | 7.288 s | 1 | `P_out0_Div_Range_A_0_J_tbl` | `P_out0_Div_Range_A_0_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7251.502/mt` |
| OGBN-Arxiv | CPU | 5 | 7.285 s | 1 | `P_avg_Div_Range_R_1_J_tbl` | `P_avg_Div_Range_R_1_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7248.498/mt` |
| OGBN-Arxiv | CPU | 6 | 7.280 s | 1 | `P_avg_Div_Range_A_1_J_tbl` | `P_avg_Div_Range_A_1_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7243.170/mt` |
| OGBN-Arxiv | CPU | 7 | 7.279 s | 1 | `P_out0_Div_Range_DMinus1_0_J_tbl` | `P_out0_Div_Range_DMinus1_0_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7242.359/mt` |
| OGBN-Arxiv | CPU | 8 | 7.279 s | 1 | `P_avg_Div_Range_Delta_0_J_tbl` | `P_avg_Div_Range_Delta_0_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7241.875/mt` |
| OGBN-Arxiv | CPU | 9 | 7.277 s | 1 | `P_out0_Div_Range_DMinus1_3_J_tbl` | `P_out0_Div_Range_DMinus1_3_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7240.293/mt` |
| OGBN-Arxiv | CPU | 10 | 7.275 s | 1 | `P_h0_Div_Range_Delta_2_J_tbl` | `P_h0_Div_Range_Delta_2_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7239.630/mt` |
| OGBN-Arxiv | GPU | 1 | 20.039 s | 1 | `P_J_tbl_feat` | `P_J_tbl_feat@FH/evaluation/len=33554432/nz=21675904/ms=19365.794/tail/mt` |
| OGBN-Arxiv | GPU | 2 | 7.461 s | 1 | `P_h0_Div_Range_Q_2_J_tbl` | `P_h0_Div_Range_Q_2_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7424.720/mt` |
| OGBN-Arxiv | GPU | 3 | 7.305 s | 1 | `P_h0_Div_Range_DMinus1_1_J_tbl` | `P_h0_Div_Range_DMinus1_1_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7268.287/mt` |
| OGBN-Arxiv | GPU | 4 | 7.294 s | 1 | `P_h0_Div_Range_A_1_J_tbl` | `P_h0_Div_Range_A_1_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7258.398/mt` |
| OGBN-Arxiv | GPU | 5 | 7.287 s | 1 | `P_h0_Div_Range_Q_1_J_tbl` | `P_h0_Div_Range_Q_1_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7249.942/mt` |
| OGBN-Arxiv | GPU | 6 | 7.286 s | 1 | `P_h0_Div_Range_Q_3_J_tbl` | `P_h0_Div_Range_Q_3_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7248.615/mt` |
| OGBN-Arxiv | GPU | 7 | 7.284 s | 1 | `P_h0_Div_Range_R_3_J_tbl` | `P_h0_Div_Range_R_3_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7246.851/mt` |
| OGBN-Arxiv | GPU | 8 | 7.280 s | 1 | `P_out0_Div_Range_Q_2_J_tbl` | `P_out0_Div_Range_Q_2_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7243.056/mt` |
| OGBN-Arxiv | GPU | 9 | 7.280 s | 1 | `P_h0_Div_Range_R_1_J_tbl` | `P_h0_Div_Range_R_1_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7242.825/mt` |
| OGBN-Arxiv | GPU | 10 | 7.279 s | 1 | `P_avg_Div_Range_Q_3_J_tbl` | `P_avg_Div_Range_Q_3_J_tbl@edge/evaluation/len=8388608/nz=8388606/ms=7241.413/mt` |

全部 `msm_profile:total_ms` 的逐行和与 `warm:trace:profile:msm_ms` 在十组运行中均仅有小于 0.012 ms 的文本舍入差，说明本节已经覆盖本批次日志中的全部 Trace MSM profile 数值行。

### 3.7 Trace MSM profile 的归一化 batch family 聚合

为保留“重复出现但单次不一定进入 Top-10”的 batch family，本表将 batch labels 中的隐藏头编号、列编号和 SignedRoundDiv limb 编号归一化为 `*`，然后对相同 family 的日志 `total_ms` 做直接计数与求和。这里只是日志数值聚合，不把这些和当作独立阶段总时间。

| 数据集 | 后端 | Rank | Normalized batch family | Calls | Σ profile | Avg | Max |
|---|---:|---:|---|---:|---:|---:|---:|
| Cora | CPU | 1 | `P_J_tbl_feat` | 1 | 3.277 s | 3276.603 ms | 3276.603 ms |
| Cora | CPU | 2 | `P_h*_Div_Range_Delta_*_J_tbl,P_h*_Div_Range_Delta_*_J_qry,P_h*_Div_Range_Delta_*_R` | 48 | 1.650 s | 34.379 ms | 67.559 ms |
| Cora | CPU | 3 | `P_h*_Div_Range_R_*_J_tbl,P_h*_Div_Range_R_*_J_qry,P_h*_Div_Range_R_*_R` | 48 | 1.627 s | 33.903 ms | 68.279 ms |
| Cora | CPU | 4 | `P_h*_Div_Range_DMinus1_*_J_tbl,P_h*_Div_Range_DMinus1_*_J_qry,P_h*_Div_Range_DMinus1_*_R` | 48 | 1.616 s | 33.670 ms | 74.480 ms |
| Cora | CPU | 5 | `P_h*_Div_Range_A_*_J_tbl,P_h*_Div_Range_A_*_J_qry,P_h*_Div_Range_A_*_R` | 48 | 1.162 s | 24.202 ms | 59.146 ms |
| Cora | CPU | 6 | `P_h*_Div_Range_Q_*_J_tbl,P_h*_Div_Range_Q_*_J_qry,P_h*_Div_Range_Q_*_R` | 48 | 1.067 s | 22.239 ms | 65.477 ms |
| Cora | CPU | 7 | `P_h*_J_src_node,P_h*_J_src_edge,P_h*_R_src_node,P_h*_R_src` | 8 | 0.617 s | 77.127 ms | 84.698 ms |
| Cora | CPU | 8 | `P_h*_AggBind_src_J_node,P_h*_AggBind_src_J_edge,P_h*_AggBind_src_R_node,P_h*_AggBind_src_R_edge` | 8 | 0.605 s | 75.579 ms | 80.600 ms |
| Cora | CPU | 9 | `P_h*_J_dst_node,P_h*_J_dst_edge,P_h*_R_dst_node,P_h*_R_dst` | 8 | 0.604 s | 75.440 ms | 76.579 ms |
| Cora | CPU | 10 | `P_h*_AggBind_dst_J_node,P_h*_AggBind_dst_J_edge,P_h*_AggBind_dst_R_node,P_h*_AggBind_dst_R_edge` | 8 | 0.601 s | 75.134 ms | 77.121 ms |
| Cora | GPU | 1 | `P_J_tbl_feat` | 1 | 3.268 s | 3268.101 ms | 3268.101 ms |
| Cora | GPU | 2 | `P_h*_Div_Range_R_*_J_tbl,P_h*_Div_Range_R_*_J_qry,P_h*_Div_Range_R_*_R` | 48 | 1.661 s | 34.595 ms | 69.443 ms |
| Cora | GPU | 3 | `P_h*_Div_Range_DMinus1_*_J_tbl,P_h*_Div_Range_DMinus1_*_J_qry,P_h*_Div_Range_DMinus1_*_R` | 48 | 1.656 s | 34.502 ms | 68.253 ms |
| Cora | GPU | 4 | `P_h*_Div_Range_Delta_*_J_tbl,P_h*_Div_Range_Delta_*_J_qry,P_h*_Div_Range_Delta_*_R` | 48 | 1.621 s | 33.780 ms | 68.092 ms |
| Cora | GPU | 5 | `P_h*_Div_Range_A_*_J_tbl,P_h*_Div_Range_A_*_J_qry,P_h*_Div_Range_A_*_R` | 48 | 1.156 s | 24.092 ms | 62.078 ms |
| Cora | GPU | 6 | `P_h*_Div_Range_Q_*_J_tbl,P_h*_Div_Range_Q_*_J_qry,P_h*_Div_Range_Q_*_R` | 48 | 1.059 s | 22.055 ms | 67.872 ms |
| Cora | GPU | 7 | `P_h*_J_t_node,P_h*_J_t_edge,P_h*_R_t_node,P_h*_R_t` | 8 | 0.611 s | 76.330 ms | 80.002 ms |
| Cora | GPU | 8 | `P_h*_J_src_node,P_h*_J_src_edge,P_h*_R_src_node,P_h*_R_src` | 8 | 0.608 s | 76.018 ms | 80.417 ms |
| Cora | GPU | 9 | `P_h*_J_dst_node,P_h*_J_dst_edge,P_h*_R_dst_node,P_h*_R_dst` | 8 | 0.605 s | 75.614 ms | 76.492 ms |
| Cora | GPU | 10 | `P_h*_AggBind_src_J_node,P_h*_AggBind_src_J_edge,P_h*_AggBind_src_R_node,P_h*_AggBind_src_R_edge` | 8 | 0.604 s | 75.496 ms | 77.261 ms |
| Citeseer | CPU | 1 | `P_J_tbl_feat` | 1 | 11.033 s | 11033.292 ms | 11033.292 ms |
| Citeseer | CPU | 2 | `P_h*_Div_Range_Q_*_J_tbl,P_h*_Div_Range_Q_*_J_qry,P_h*_Div_Range_Q_*_R` | 48 | 2.727 s | 56.809 ms | 69.246 ms |
| Citeseer | CPU | 3 | `P_h*_Div_Range_R_*_J_tbl,P_h*_Div_Range_R_*_J_qry,P_h*_Div_Range_R_*_R` | 48 | 2.684 s | 55.908 ms | 63.659 ms |
| Citeseer | CPU | 4 | `P_h*_Div_Range_Delta_*_J_tbl,P_h*_Div_Range_Delta_*_J_qry,P_h*_Div_Range_Delta_*_R` | 48 | 2.631 s | 54.806 ms | 61.586 ms |
| Citeseer | CPU | 5 | `P_h*_Div_Range_DMinus1_*_J_tbl,P_h*_Div_Range_DMinus1_*_J_qry,P_h*_Div_Range_DMinus1_*_R` | 48 | 2.613 s | 54.436 ms | 67.039 ms |
| Citeseer | CPU | 6 | `P_h*_Div_Range_A_*_J_tbl,P_h*_Div_Range_A_*_J_qry,P_h*_Div_Range_A_*_R` | 48 | 2.612 s | 54.417 ms | 66.250 ms |
| Citeseer | CPU | 7 | `P_m_feat` | 1 | 1.524 s | 1523.648 ms | 1523.648 ms |
| Citeseer | CPU | 8 | `P_h*_J_t_node,P_h*_J_t_edge,P_h*_R_t_node,P_h*_R_t` | 8 | 0.599 s | 74.884 ms | 76.873 ms |
| Citeseer | CPU | 9 | `P_h*_J_dst_node,P_h*_J_dst_edge,P_h*_R_dst_node,P_h*_R_dst` | 8 | 0.596 s | 74.505 ms | 77.153 ms |
| Citeseer | CPU | 10 | `P_h*_AggBind_src_J_node,P_h*_AggBind_src_J_edge,P_h*_AggBind_src_R_node,P_h*_AggBind_src_R_edge` | 8 | 0.589 s | 73.626 ms | 74.920 ms |
| Citeseer | GPU | 1 | `P_J_tbl_feat` | 1 | 11.091 s | 11090.631 ms | 11090.631 ms |
| Citeseer | GPU | 2 | `P_h*_Div_Range_Q_*_J_tbl,P_h*_Div_Range_Q_*_J_qry,P_h*_Div_Range_Q_*_R` | 48 | 2.703 s | 56.321 ms | 77.769 ms |
| Citeseer | GPU | 3 | `P_h*_Div_Range_R_*_J_tbl,P_h*_Div_Range_R_*_J_qry,P_h*_Div_Range_R_*_R` | 48 | 2.681 s | 55.848 ms | 67.939 ms |
| Citeseer | GPU | 4 | `P_h*_Div_Range_Delta_*_J_tbl,P_h*_Div_Range_Delta_*_J_qry,P_h*_Div_Range_Delta_*_R` | 48 | 2.634 s | 54.876 ms | 63.715 ms |
| Citeseer | GPU | 5 | `P_h*_Div_Range_A_*_J_tbl,P_h*_Div_Range_A_*_J_qry,P_h*_Div_Range_A_*_R` | 48 | 2.625 s | 54.683 ms | 67.084 ms |
| Citeseer | GPU | 6 | `P_h*_Div_Range_DMinus1_*_J_tbl,P_h*_Div_Range_DMinus1_*_J_qry,P_h*_Div_Range_DMinus1_*_R` | 48 | 2.599 s | 54.141 ms | 68.531 ms |
| Citeseer | GPU | 7 | `P_m_feat` | 1 | 1.550 s | 1550.434 ms | 1550.434 ms |
| Citeseer | GPU | 8 | `P_h*_J_src_node,P_h*_J_src_edge,P_h*_R_src_node,P_h*_R_src` | 8 | 0.595 s | 74.337 ms | 74.727 ms |
| Citeseer | GPU | 9 | `P_h*_J_t_node,P_h*_J_t_edge,P_h*_R_t_node,P_h*_R_t` | 8 | 0.594 s | 74.297 ms | 76.567 ms |
| Citeseer | GPU | 10 | `P_h*_AggBind_src_J_node,P_h*_AggBind_src_J_edge,P_h*_AggBind_src_R_node,P_h*_AggBind_src_R_edge` | 8 | 0.591 s | 73.819 ms | 74.380 ms |
| Pubmed | CPU | 1 | `P_J_tbl_feat` | 1 | 11.165 s | 11164.927 ms | 11164.927 ms |
| Pubmed | CPU | 2 | `P_h*_Div_Range_DMinus1_*_J_tbl,P_h*_Div_Range_DMinus1_*_J_qry,P_h*_Div_Range_DMinus1_*_R` | 48 | 5.725 s | 119.266 ms | 324.997 ms |
| Pubmed | CPU | 3 | `P_h*_Div_Range_R_*_J_tbl,P_h*_Div_Range_R_*_J_qry,P_h*_Div_Range_R_*_R` | 48 | 5.682 s | 118.374 ms | 342.321 ms |
| Pubmed | CPU | 4 | `P_h*_Div_Range_Delta_*_J_tbl,P_h*_Div_Range_Delta_*_J_qry,P_h*_Div_Range_Delta_*_R` | 48 | 5.539 s | 115.401 ms | 301.350 ms |
| Pubmed | CPU | 5 | `P_h*_J_dst_node,P_h*_J_dst_edge,P_h*_R_dst_node,P_h*_R_dst` | 8 | 3.556 s | 444.558 ms | 509.251 ms |
| Pubmed | CPU | 6 | `P_h*_J_t_node,P_h*_J_t_edge,P_h*_R_t_node,P_h*_R_t` | 8 | 3.230 s | 403.691 ms | 489.391 ms |
| Pubmed | CPU | 7 | `P_h*_J_src_node,P_h*_J_src_edge,P_h*_R_src_node,P_h*_R_src` | 8 | 3.226 s | 403.231 ms | 464.008 ms |
| Pubmed | CPU | 8 | `P_h*_AggBind_dst_J_node,P_h*_AggBind_dst_J_edge,P_h*_AggBind_dst_R_node,P_h*_AggBind_dst_R_edge` | 8 | 3.124 s | 390.527 ms | 413.633 ms |
| Pubmed | CPU | 9 | `P_h*_AggBind_src_J_node,P_h*_AggBind_src_J_edge,P_h*_AggBind_src_R_node,P_h*_AggBind_src_R_edge` | 8 | 3.109 s | 388.649 ms | 413.754 ms |
| Pubmed | CPU | 10 | `P_h*_Div_Range_A_*_J_tbl,P_h*_Div_Range_A_*_J_qry,P_h*_Div_Range_A_*_R` | 48 | 2.108 s | 43.924 ms | 246.224 ms |
| Pubmed | GPU | 1 | `P_J_tbl_feat` | 1 | 9.410 s | 9410.107 ms | 9410.107 ms |
| Pubmed | GPU | 2 | `P_h*_Div_Range_Delta_*_J_tbl,P_h*_Div_Range_Delta_*_J_qry,P_h*_Div_Range_Delta_*_R` | 48 | 5.369 s | 111.849 ms | 282.455 ms |
| Pubmed | GPU | 3 | `P_h*_Div_Range_DMinus1_*_J_tbl,P_h*_Div_Range_DMinus1_*_J_qry,P_h*_Div_Range_DMinus1_*_R` | 48 | 5.329 s | 111.022 ms | 283.899 ms |
| Pubmed | GPU | 4 | `P_h*_Div_Range_R_*_J_tbl,P_h*_Div_Range_R_*_J_qry,P_h*_Div_Range_R_*_R` | 48 | 5.321 s | 110.845 ms | 288.523 ms |
| Pubmed | GPU | 5 | `P_h*_J_src_node,P_h*_J_src_edge,P_h*_R_src_node,P_h*_R_src` | 8 | 3.042 s | 380.292 ms | 499.034 ms |
| Pubmed | GPU | 6 | `P_h*_AggBind_src_J_node,P_h*_AggBind_src_J_edge,P_h*_AggBind_src_R_node,P_h*_AggBind_src_R_edge` | 8 | 3.002 s | 375.257 ms | 415.841 ms |
| Pubmed | GPU | 7 | `P_h*_J_t_node,P_h*_J_t_edge,P_h*_R_t_node,P_h*_R_t` | 8 | 2.989 s | 373.567 ms | 466.514 ms |
| Pubmed | GPU | 8 | `P_h*_AggBind_dst_J_node,P_h*_AggBind_dst_J_edge,P_h*_AggBind_dst_R_node,P_h*_AggBind_dst_R_edge` | 8 | 2.960 s | 369.991 ms | 404.564 ms |
| Pubmed | GPU | 9 | `P_h*_J_dst_node,P_h*_J_dst_edge,P_h*_R_dst_node,P_h*_R_dst` | 8 | 2.957 s | 369.597 ms | 455.566 ms |
| Pubmed | GPU | 10 | `P_h*_Div_Range_A_*_J_tbl,P_h*_Div_Range_A_*_J_qry,P_h*_Div_Range_A_*_R` | 48 | 2.019 s | 42.053 ms | 243.496 ms |
| PPI | CPU | 1 | `P_out0_Div_Range_DMinus1_*_R` | 5 | 29.824 s | 5964.798 ms | 6015.991 ms |
| PPI | CPU | 2 | `P_avg_Div_Range_DMinus1_*_R` | 5 | 29.665 s | 5933.046 ms | 6025.312 ms |
| PPI | CPU | 3 | `P_avg_Div_Range_A_*_R` | 5 | 29.646 s | 5929.133 ms | 6034.948 ms |
| PPI | CPU | 4 | `P_avg_Div_Range_Q_*_R` | 5 | 29.570 s | 5913.972 ms | 6033.370 ms |
| PPI | CPU | 5 | `P_out0_Div_Range_Delta_*_R` | 5 | 29.568 s | 5913.601 ms | 6019.011 ms |
| PPI | CPU | 6 | `P_avg_Div_Range_Delta_*_R` | 5 | 29.511 s | 5902.240 ms | 6014.967 ms |
| PPI | CPU | 7 | `P_out0_Div_Range_A_*_R` | 5 | 29.476 s | 5895.227 ms | 6004.834 ms |
| PPI | CPU | 8 | `P_avg_Div_Range_R_*_R` | 5 | 29.447 s | 5889.364 ms | 6006.747 ms |
| PPI | CPU | 9 | `P_out0_Div_Range_R_*_R` | 5 | 29.311 s | 5862.183 ms | 5995.508 ms |
| PPI | CPU | 10 | `P_out0_Div_Range_Q_*_R` | 5 | 29.178 s | 5835.532 ms | 6012.150 ms |
| PPI | GPU | 1 | `P_out0_Div_Range_A_*_R` | 5 | 30.532 s | 6106.472 ms | 6177.687 ms |
| PPI | GPU | 2 | `P_out0_Div_Range_Q_*_R` | 5 | 29.959 s | 5991.863 ms | 6392.600 ms |
| PPI | GPU | 3 | `P_out0_Div_Range_DMinus1_*_R` | 5 | 29.921 s | 5984.169 ms | 6115.462 ms |
| PPI | GPU | 4 | `P_out0_Div_Range_R_*_R` | 5 | 29.803 s | 5960.557 ms | 6033.109 ms |
| PPI | GPU | 5 | `P_out0_Div_Range_Delta_*_R` | 5 | 29.759 s | 5951.818 ms | 6031.031 ms |
| PPI | GPU | 6 | `P_avg_Div_Range_R_*_R` | 5 | 29.679 s | 5935.889 ms | 6038.796 ms |
| PPI | GPU | 7 | `P_avg_Div_Range_Delta_*_R` | 5 | 29.662 s | 5932.439 ms | 6001.459 ms |
| PPI | GPU | 8 | `P_avg_Div_Range_A_*_R` | 5 | 29.464 s | 5892.746 ms | 6004.892 ms |
| PPI | GPU | 9 | `P_avg_Div_Range_Q_*_R` | 5 | 29.387 s | 5877.369 ms | 6040.117 ms |
| PPI | GPU | 10 | `P_avg_Div_Range_DMinus1_*_R` | 5 | 29.375 s | 5875.096 ms | 6009.973 ms |
| OGBN-Arxiv | CPU | 1 | `P_avg_Div_Range_Q_*_J_tbl` | 4 | 28.943 s | 7235.804 ms | 7270.624 ms |
| OGBN-Arxiv | CPU | 2 | `P_out0_Div_Range_DMinus1_*_J_tbl` | 4 | 28.804 s | 7201.076 ms | 7278.797 ms |
| OGBN-Arxiv | CPU | 3 | `P_out0_Div_Range_Delta_*_J_tbl` | 4 | 28.732 s | 7183.109 ms | 7297.722 ms |
| OGBN-Arxiv | CPU | 4 | `P_h*_Div_Range_Q_*_J_tbl` | 4 | 28.629 s | 7157.249 ms | 7255.477 ms |
| OGBN-Arxiv | CPU | 5 | `P_avg_Div_Range_DMinus1_*_J_tbl` | 4 | 28.624 s | 7155.976 ms | 7233.563 ms |
| OGBN-Arxiv | CPU | 6 | `P_avg_Div_Range_R_*_J_tbl` | 4 | 28.549 s | 7137.165 ms | 7284.601 ms |
| OGBN-Arxiv | CPU | 7 | `P_h*_Div_Range_R_*_J_tbl` | 4 | 28.528 s | 7132.072 ms | 7266.533 ms |
| OGBN-Arxiv | CPU | 8 | `P_avg_Div_Range_Delta_*_J_tbl` | 4 | 28.492 s | 7123.077 ms | 7278.704 ms |
| OGBN-Arxiv | CPU | 9 | `P_avg_Div_Range_A_*_J_tbl` | 4 | 28.480 s | 7119.932 ms | 7280.411 ms |
| OGBN-Arxiv | CPU | 10 | `P_out0_Div_Range_A_*_J_tbl` | 4 | 28.292 s | 7073.083 ms | 7288.128 ms |
| OGBN-Arxiv | GPU | 1 | `P_h*_Div_Range_Q_*_J_tbl` | 4 | 29.173 s | 7293.313 ms | 7461.436 ms |
| OGBN-Arxiv | GPU | 2 | `P_h*_Div_Range_DMinus1_*_J_tbl` | 4 | 28.857 s | 7214.165 ms | 7305.174 ms |
| OGBN-Arxiv | GPU | 3 | `P_h*_Div_Range_R_*_J_tbl` | 4 | 28.825 s | 7206.348 ms | 7283.896 ms |
| OGBN-Arxiv | GPU | 4 | `P_out0_Div_Range_Q_*_J_tbl` | 4 | 28.814 s | 7203.382 ms | 7280.078 ms |
| OGBN-Arxiv | GPU | 5 | `P_avg_Div_Range_Q_*_J_tbl` | 4 | 28.582 s | 7145.609 ms | 7278.539 ms |
| OGBN-Arxiv | GPU | 6 | `P_avg_Div_Range_R_*_J_tbl` | 4 | 28.503 s | 7125.697 ms | 7269.546 ms |
| OGBN-Arxiv | GPU | 7 | `P_h*_Div_Range_Delta_*_J_tbl` | 4 | 28.497 s | 7124.197 ms | 7273.976 ms |
| OGBN-Arxiv | GPU | 8 | `P_avg_Div_Range_A_*_J_tbl` | 4 | 28.437 s | 7109.167 ms | 7259.411 ms |
| OGBN-Arxiv | GPU | 9 | `P_h*_Div_Range_A_*_J_tbl` | 4 | 28.431 s | 7107.864 ms | 7294.080 ms |
| OGBN-Arxiv | GPU | 10 | `P_out0_Div_Range_A_*_J_tbl` | 4 | 28.361 s | 7090.241 ms | 7262.218 ms |

### 3.8 Trace linear-derived / MSM-input 记录

日志中的 `commit_batch:...:linear_derived=X:msm_inputs=Y` 用于记录该 batch 中 linear-derived 对象数量与实际 MSM 输入数。本表对所有这类行做直接计数与求和。

| 数据集 | 后端 | 记录数 | Σ linear_derived | Σ msm_inputs | 最大 linear_derived/行 | 最大 msm_inputs/行 |
|---|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 74 | 154 | 362 | 3 | 20 |
| Cora | GPU | 74 | 154 | 362 | 3 | 20 |
| Citeseer | CPU | 74 | 154 | 362 | 3 | 20 |
| Citeseer | GPU | 74 | 154 | 362 | 3 | 20 |
| Pubmed | CPU | 74 | 154 | 362 | 3 | 20 |
| Pubmed | GPU | 74 | 154 | 362 | 3 | 20 |
| PPI | CPU | 33 | 35 | 15 | 2 | 5 |
| PPI | GPU | 33 | 35 | 15 | 2 | 5 |
| OGBN-Arxiv | CPU | 33 | 35 | 6 | 2 | 2 |
| OGBN-Arxiv | GPU | 33 | 35 | 6 | 2 | 2 |

### 3.9 Taylor-shift commitment 计时

日志包含独立的 `trace:taylor_shift_commit:<object>:ms=...` 数值行。本节对该类记录进行完整计数，并与 `warm:trace:profile:derived_point_ms` 保持分开。

| 数据集 | 后端 | 记录数 | Σ taylor-shift | 最大单项 | 最大对象 | Σ explicit derived_point | Derived-point aggregate | Aggregate - taylor - explicit derived |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Cora | CPU | 47 | 3.232 s | 0.077 s | `P_h0_R_src_RouteShifted` | 0.462 s | 3.737 s | 42.965 ms |
| Cora | GPU | 47 | 3.258 s | 0.078 s | `P_h0_R_src_RouteShifted` | 0.456 s | 3.754 s | 39.452 ms |
| Citeseer | CPU | 47 | 3.246 s | 0.077 s | `P_out0_BiasBasis_Chunk_0_Acc_shifted` | 1.387 s | 4.683 s | 50.213 ms |
| Citeseer | GPU | 47 | 3.233 s | 0.078 s | `P_h0_R_src_RouteShifted` | 1.420 s | 4.702 s | 49.499 ms |
| Pubmed | CPU | 47 | 16.290 s | 0.466 s | `P_h2_R_src_RouteShifted` | 1.358 s | 17.675 s | 26.378 ms |
| Pubmed | GPU | 47 | 15.895 s | 0.440 s | `P_h0_R_src_RouteShifted` | 1.385 s | 17.306 s | 26.419 ms |
| PPI | CPU | 12 | 78.520 s | 7.249 s | `P_h0_AggBind_src_R_edge_shifted` | 2.347 s | 80.892 s | 25.164 ms |
| PPI | GPU | 12 | 77.917 s | 7.222 s | `P_out0_BiasBasis_Chunk_0_Acc_shifted` | 2.344 s | 80.286 s | 25.037 ms |
| OGBN-Arxiv | CPU | 12 | 77.655 s | 7.234 s | `P_out0_R_dst_RouteShifted` | 5.754 s | 83.421 s | 11.679 ms |
| OGBN-Arxiv | GPU | 12 | 78.960 s | 7.447 s | `P_out0_AggBind_dst_R_edge_shifted` | 5.616 s | 84.588 s | 11.793 ms |

`Aggregate - taylor - explicit derived` 仅是三个日志字段的直接算术差，用于说明 `derived_point_ms` 还包含未以这两类逐对象行单独展开的少量计时；不对其内部含义作进一步推断。

#### 3.9.1 Taylor-shift Top-5 单项

| 数据集 | 后端 | Rank | 对象 | ms |
|---|---:|---:|---|---:|
| Cora | CPU | 1 | `P_h0_R_src_RouteShifted` | 76.811 |
| Cora | CPU | 2 | `P_h7_AggBind_src_R_edge_shifted` | 73.338 |
| Cora | CPU | 3 | `P_out0_BiasBasis_Chunk_0_Acc_shifted` | 73.201 |
| Cora | CPU | 4 | `P_out0_R_t_RouteShifted` | 73.020 |
| Cora | CPU | 5 | `P_h1_R_dst_RouteShifted` | 72.459 |
| Cora | GPU | 1 | `P_h0_R_src_RouteShifted` | 78.253 |
| Cora | GPU | 2 | `P_out0_R_t_RouteShifted` | 73.243 |
| Cora | GPU | 3 | `P_h2_AggBind_src_R_edge_shifted` | 73.172 |
| Cora | GPU | 4 | `P_h4_R_t_RouteShifted` | 72.889 |
| Cora | GPU | 5 | `P_out0_AggBind_src_R_edge_shifted` | 72.646 |
| Citeseer | CPU | 1 | `P_out0_BiasBasis_Chunk_0_Acc_shifted` | 77.127 |
| Citeseer | CPU | 2 | `P_h4_R_t_RouteShifted` | 73.203 |
| Citeseer | CPU | 3 | `P_h0_R_src_RouteShifted` | 72.920 |
| Citeseer | CPU | 4 | `P_h0_AggBind_src_R_edge_shifted` | 72.628 |
| Citeseer | CPU | 5 | `P_h6_R_dst_RouteShifted` | 72.453 |
| Citeseer | GPU | 1 | `P_h0_R_src_RouteShifted` | 77.547 |
| Citeseer | GPU | 2 | `P_h1_R_t_RouteShifted` | 72.891 |
| Citeseer | GPU | 3 | `P_out0_BiasBasis_Chunk_0_Acc_shifted` | 72.678 |
| Citeseer | GPU | 4 | `P_h1_R_dst_RouteShifted` | 72.671 |
| Citeseer | GPU | 5 | `P_h4_R_t_RouteShifted` | 72.371 |
| Pubmed | CPU | 1 | `P_h2_R_src_RouteShifted` | 466.129 |
| Pubmed | CPU | 2 | `P_h2_R_dst_RouteShifted` | 452.003 |
| Pubmed | CPU | 3 | `P_h2_R_t_RouteShifted` | 450.430 |
| Pubmed | CPU | 4 | `P_h0_R_src_RouteShifted` | 377.096 |
| Pubmed | CPU | 5 | `P_h1_AggBind_src_R_edge_shifted` | 363.944 |
| Pubmed | GPU | 1 | `P_h0_R_src_RouteShifted` | 440.168 |
| Pubmed | GPU | 2 | `P_h0_R_t_RouteShifted` | 360.976 |
| Pubmed | GPU | 3 | `P_h0_R_dst_RouteShifted` | 358.057 |
| Pubmed | GPU | 4 | `P_h5_R_dst_RouteShifted` | 351.436 |
| Pubmed | GPU | 5 | `P_h3_AggBind_src_R_edge_shifted` | 348.302 |
| PPI | CPU | 1 | `P_h0_AggBind_src_R_edge_shifted` | 7249.076 |
| PPI | CPU | 2 | `P_h0_R_src_RouteShifted` | 7240.855 |
| PPI | CPU | 3 | `P_out0_AggBind_dst_R_edge_shifted` | 7236.642 |
| PPI | CPU | 4 | `P_out0_R_src_RouteShifted` | 7230.593 |
| PPI | CPU | 5 | `P_h0_R_t_RouteShifted` | 7220.143 |
| PPI | GPU | 1 | `P_out0_BiasBasis_Chunk_0_Acc_shifted` | 7222.457 |
| PPI | GPU | 2 | `P_out0_R_dst_RouteShifted` | 7219.491 |
| PPI | GPU | 3 | `P_h0_AggBind_src_R_edge_shifted` | 7216.739 |
| PPI | GPU | 4 | `P_h0_R_dst_RouteShifted` | 7205.620 |
| PPI | GPU | 5 | `P_out0_AggBind_dst_R_edge_shifted` | 7160.079 |
| OGBN-Arxiv | CPU | 1 | `P_out0_R_dst_RouteShifted` | 7233.666 |
| OGBN-Arxiv | CPU | 2 | `P_h0_R_src_RouteShifted` | 7225.950 |
| OGBN-Arxiv | CPU | 3 | `P_out0_AggBind_src_R_edge_shifted` | 7216.255 |
| OGBN-Arxiv | CPU | 4 | `P_h0_R_t_RouteShifted` | 7132.031 |
| OGBN-Arxiv | CPU | 5 | `P_h0_R_dst_RouteShifted` | 7097.840 |
| OGBN-Arxiv | GPU | 1 | `P_out0_AggBind_dst_R_edge_shifted` | 7447.285 |
| OGBN-Arxiv | GPU | 2 | `P_out0_BiasBasis_Chunk_0_Acc_shifted` | 7235.627 |
| OGBN-Arxiv | GPU | 3 | `P_h0_R_src_RouteShifted` | 7221.550 |
| OGBN-Arxiv | GPU | 4 | `P_out0_R_t_RouteShifted` | 7220.962 |
| OGBN-Arxiv | GPU | 5 | `P_out0_AggBind_src_R_edge_shifted` | 7220.164 |

### 3.10 Trace 辅助逐对象计时行的直接统计


下表统计 `proof_commit`、`model_reblind_derived`、`linear_value_validation`、`derived_point` 四类逐对象日志。`Σ` 仅表示对应日志行数值直接相加，不替代父阶段总时间。

| 数据集 | 后端 | 类别 | 记录数 | Σ ms | 最大单项 ms | 最大单项对象 |
|---|---:|---|---:|---:|---:|---|
| Cora | CPU | `proof_commit` | 161 | 30.449 | 0.371 | `P_h4_ScoreMask_src` |
| Cora | CPU | `model_reblind_derived` | 90 | 12.402 | 0.345 | `V_o0_b_proof` |
| Cora | CPU | `linear_value_validation` | 154 | 826.042 | 302.968 | `P_Query_feat` |
| Cora | CPU | `derived_point` | 154 | 462.017 | 167.936 | `P_J_qry_feat` |
| Cora | GPU | `proof_commit` | 161 | 27.951 | 0.358 | `P_h6_H_agg_pre_star_ExternalMask` |
| Cora | GPU | `model_reblind_derived` | 90 | 11.398 | 0.193 | `V_h5_W_col0_proof` |
| Cora | GPU | `linear_value_validation` | 154 | 816.888 | 302.314 | `P_Query_feat` |
| Cora | GPU | `derived_point` | 154 | 455.629 | 167.791 | `P_Query_feat` |
| Citeseer | CPU | `proof_commit` | 160 | 34.948 | 0.398 | `P_h7_ModelBind_proj_R` |
| Citeseer | CPU | `model_reblind_derived` | 89 | 15.160 | 0.355 | `V_o0_b_proof` |
| Citeseer | CPU | `linear_value_validation` | 154 | 2512.083 | 1168.465 | `P_J_qry_feat` |
| Citeseer | CPU | `derived_point` | 154 | 1386.872 | 651.928 | `P_J_qry_feat` |
| Citeseer | GPU | `proof_commit` | 160 | 34.236 | 0.394 | `P_h4_ScoreMask_src` |
| Citeseer | GPU | `model_reblind_derived` | 89 | 15.160 | 0.344 | `V_o0_b_proof` |
| Citeseer | GPU | `linear_value_validation` | 154 | 2621.226 | 1208.627 | `P_J_qry_feat` |
| Citeseer | GPU | `derived_point` | 154 | 1420.071 | 666.996 | `P_J_qry_feat` |
| Pubmed | CPU | `proof_commit` | 157 | 18.881 | 0.191 | `P_h1_H_agg_pre_star_ExternalMask` |
| Pubmed | CPU | `model_reblind_derived` | 86 | 7.330 | 0.167 | `V_o0_b_proof` |
| Pubmed | CPU | `linear_value_validation` | 154 | 3797.405 | 1175.910 | `P_Query_feat` |
| Pubmed | CPU | `derived_point` | 154 | 1358.476 | 640.971 | `P_J_qry_feat` |
| Pubmed | GPU | `proof_commit` | 157 | 18.856 | 0.191 | `P_h0_ScoreMask_src` |
| Pubmed | GPU | `model_reblind_derived` | 86 | 7.411 | 0.167 | `V_o0_b_proof` |
| Pubmed | GPU | `linear_value_validation` | 154 | 3860.172 | 1198.281 | `P_J_qry_feat` |
| Pubmed | GPU | `derived_point` | 154 | 1384.525 | 653.465 | `P_Query_feat` |
| PPI | CPU | `proof_commit` | 149 | 13.921 | 0.200 | `P_out0_ModelBind_proj_R` |
| PPI | CPU | `model_reblind_derived` | 134 | 11.191 | 0.166 | `V_o0_b_proof` |
| PPI | CPU | `linear_value_validation` | 35 | 8997.230 | 1135.503 | `P_h0_Query_ELU` |
| PPI | CPU | `derived_point` | 35 | 2346.667 | 649.282 | `P_h0_Query_ELU` |
| PPI | GPU | `proof_commit` | 149 | 13.770 | 0.200 | `P_out0_ModelBind_proj_R` |
| PPI | GPU | `model_reblind_derived` | 134 | 11.221 | 0.167 | `V_o0_b_proof` |
| PPI | GPU | `linear_value_validation` | 35 | 9227.869 | 1149.671 | `P_h0_Query_ELU` |
| PPI | GPU | `derived_point` | 35 | 2343.590 | 661.587 | `P_h0_Query_ELU` |
| OGBN-Arxiv | CPU | `proof_commit` | 68 | 7.139 | 0.204 | `P_h0_H_agg_pre_star_ExternalMask` |
| OGBN-Arxiv | CPU | `model_reblind_derived` | 53 | 4.490 | 0.165 | `V_o0_b_proof` |
| OGBN-Arxiv | CPU | `linear_value_validation` | 35 | 15487.863 | 2723.394 | `P_Query_feat` |
| OGBN-Arxiv | CPU | `derived_point` | 35 | 5753.922 | 1310.231 | `P_Query_feat` |
| OGBN-Arxiv | GPU | `proof_commit` | 68 | 7.178 | 0.203 | `P_H_cat_star_ExternalMask` |
| OGBN-Arxiv | GPU | `model_reblind_derived` | 53 | 4.567 | 0.166 | `V_o0_b_proof` |
| OGBN-Arxiv | GPU | `linear_value_validation` | 35 | 14985.840 | 2401.235 | `P_Query_feat` |
| OGBN-Arxiv | GPU | `derived_point` | 35 | 5615.720 | 1247.145 | `P_Query_feat` |

### 3.11 `linear_value_validation` 与 `derived_point` 的 Top-5 单项

| 数据集 | 后端 | 类别 | Rank | 对象 | ms |
|---|---:|---|---:|---|---:|
| Cora | CPU | `linear_value_validation` | 1 | `P_Query_feat` | 302.968 |
| Cora | CPU | `linear_value_validation` | 2 | `P_J_qry_feat` | 294.260 |
| Cora | CPU | `linear_value_validation` | 3 | `P_out0_Query_dst` | 5.296 |
| Cora | CPU | `linear_value_validation` | 4 | `P_h6_Query_dst` | 5.209 |
| Cora | CPU | `linear_value_validation` | 5 | `P_h6_Table_exp` | 3.361 |
| Cora | CPU | `derived_point` | 1 | `P_J_qry_feat` | 167.936 |
| Cora | CPU | `derived_point` | 2 | `P_Query_feat` | 167.101 |
| Cora | CPU | `derived_point` | 3 | `P_h6_Query_dst` | 3.389 |
| Cora | CPU | `derived_point` | 4 | `P_h6_Table_dst` | 3.383 |
| Cora | CPU | `derived_point` | 5 | `P_out0_Table_dst` | 3.163 |
| Cora | GPU | `linear_value_validation` | 1 | `P_Query_feat` | 302.314 |
| Cora | GPU | `linear_value_validation` | 2 | `P_J_qry_feat` | 290.729 |
| Cora | GPU | `linear_value_validation` | 3 | `P_out0_Query_dst` | 5.208 |
| Cora | GPU | `linear_value_validation` | 4 | `P_h3_Query_dst` | 5.207 |
| Cora | GPU | `linear_value_validation` | 5 | `P_h1_Table_exp` | 3.348 |
| Cora | GPU | `derived_point` | 1 | `P_Query_feat` | 167.791 |
| Cora | GPU | `derived_point` | 2 | `P_J_qry_feat` | 166.679 |
| Cora | GPU | `derived_point` | 3 | `P_h3_Table_dst` | 3.519 |
| Cora | GPU | `derived_point` | 4 | `P_h3_Query_dst` | 3.491 |
| Cora | GPU | `derived_point` | 5 | `P_out0_Table_dst` | 3.121 |
| Citeseer | CPU | `linear_value_validation` | 1 | `P_J_qry_feat` | 1168.465 |
| Citeseer | CPU | `linear_value_validation` | 2 | `P_Query_feat` | 1141.255 |
| Citeseer | CPU | `linear_value_validation` | 3 | `P_h1_Query_dst` | 5.009 |
| Citeseer | CPU | `linear_value_validation` | 4 | `P_h2_Query_dst` | 4.996 |
| Citeseer | CPU | `linear_value_validation` | 5 | `P_out0_Table_L` | 3.380 |
| Citeseer | CPU | `derived_point` | 1 | `P_J_qry_feat` | 651.928 |
| Citeseer | CPU | `derived_point` | 2 | `P_Query_feat` | 637.469 |
| Citeseer | CPU | `derived_point` | 3 | `P_h1_Query_dst` | 2.934 |
| Citeseer | CPU | `derived_point` | 4 | `P_h1_Table_dst` | 2.927 |
| Citeseer | CPU | `derived_point` | 5 | `P_h2_Query_dst` | 2.920 |
| Citeseer | GPU | `linear_value_validation` | 1 | `P_J_qry_feat` | 1208.627 |
| Citeseer | GPU | `linear_value_validation` | 2 | `P_Query_feat` | 1202.146 |
| Citeseer | GPU | `linear_value_validation` | 3 | `P_h0_Query_dst` | 5.001 |
| Citeseer | GPU | `linear_value_validation` | 4 | `P_h2_Query_dst` | 5.000 |
| Citeseer | GPU | `linear_value_validation` | 5 | `P_h6_Query_dst` | 4.983 |
| Citeseer | GPU | `derived_point` | 1 | `P_J_qry_feat` | 666.996 |
| Citeseer | GPU | `derived_point` | 2 | `P_Query_feat` | 649.319 |
| Citeseer | GPU | `derived_point` | 3 | `P_h2_Table_dst` | 2.949 |
| Citeseer | GPU | `derived_point` | 4 | `P_h6_Table_dst` | 2.945 |
| Citeseer | GPU | `derived_point` | 5 | `P_h2_Query_dst` | 2.933 |
| Pubmed | CPU | `linear_value_validation` | 1 | `P_Query_feat` | 1175.910 |
| Pubmed | CPU | `linear_value_validation` | 2 | `P_J_qry_feat` | 1161.943 |
| Pubmed | CPU | `linear_value_validation` | 3 | `P_h7_Query_dst` | 24.434 |
| Pubmed | CPU | `linear_value_validation` | 4 | `P_out0_Query_dst` | 24.266 |
| Pubmed | CPU | `linear_value_validation` | 5 | `P_h4_Query_dst` | 24.251 |
| Pubmed | CPU | `derived_point` | 1 | `P_J_qry_feat` | 640.971 |
| Pubmed | CPU | `derived_point` | 2 | `P_Query_feat` | 638.701 |
| Pubmed | CPU | `derived_point` | 3 | `P_h1_Table_dst` | 1.424 |
| Pubmed | CPU | `derived_point` | 4 | `P_h7_Table_dst` | 1.422 |
| Pubmed | CPU | `derived_point` | 5 | `P_h3_Table_dst` | 1.422 |
| Pubmed | GPU | `linear_value_validation` | 1 | `P_J_qry_feat` | 1198.281 |
| Pubmed | GPU | `linear_value_validation` | 2 | `P_Query_feat` | 1196.817 |
| Pubmed | GPU | `linear_value_validation` | 3 | `P_h4_Query_dst` | 24.100 |
| Pubmed | GPU | `linear_value_validation` | 4 | `P_h2_Query_dst` | 23.913 |
| Pubmed | GPU | `linear_value_validation` | 5 | `P_h0_Query_dst` | 23.887 |
| Pubmed | GPU | `derived_point` | 1 | `P_Query_feat` | 653.465 |
| Pubmed | GPU | `derived_point` | 2 | `P_J_qry_feat` | 651.998 |
| Pubmed | GPU | `derived_point` | 3 | `P_h7_Query_dst` | 1.428 |
| Pubmed | GPU | `derived_point` | 4 | `P_h4_Table_dst` | 1.426 |
| Pubmed | GPU | `derived_point` | 5 | `P_h5_Table_dst` | 1.426 |
| PPI | CPU | `linear_value_validation` | 1 | `P_h0_Query_ELU` | 1135.503 |
| PPI | CPU | `linear_value_validation` | 2 | `P_out0_w` | 649.743 |
| PPI | CPU | `linear_value_validation` | 3 | `P_h0_w_psq` | 625.634 |
| PPI | CPU | `linear_value_validation` | 4 | `P_out0_Query_dst` | 587.769 |
| PPI | CPU | `linear_value_validation` | 5 | `P_h0_Query_dst` | 581.236 |
| PPI | CPU | `derived_point` | 1 | `P_h0_Query_ELU` | 649.282 |
| PPI | CPU | `derived_point` | 2 | `P_out0_w` | 353.316 |
| PPI | CPU | `derived_point` | 3 | `P_h0_w_psq` | 353.277 |
| PPI | CPU | `derived_point` | 4 | `P_J_qry_feat` | 168.836 |
| PPI | CPU | `derived_point` | 5 | `P_Query_feat` | 168.627 |
| PPI | GPU | `linear_value_validation` | 1 | `P_h0_Query_ELU` | 1149.671 |
| PPI | GPU | `linear_value_validation` | 2 | `P_h0_Query_dst` | 687.111 |
| PPI | GPU | `linear_value_validation` | 3 | `P_h0_w_psq` | 653.635 |
| PPI | GPU | `linear_value_validation` | 4 | `P_out0_w` | 640.220 |
| PPI | GPU | `linear_value_validation` | 5 | `P_out0_Query_dst` | 589.544 |
| PPI | GPU | `derived_point` | 1 | `P_h0_Query_ELU` | 661.587 |
| PPI | GPU | `derived_point` | 2 | `P_h0_w_psq` | 353.336 |
| PPI | GPU | `derived_point` | 3 | `P_out0_w` | 344.068 |
| PPI | GPU | `derived_point` | 4 | `P_J_qry_feat` | 167.945 |
| PPI | GPU | `derived_point` | 5 | `P_Query_feat` | 165.045 |
| OGBN-Arxiv | CPU | `linear_value_validation` | 1 | `P_Query_feat` | 2723.394 |
| OGBN-Arxiv | CPU | `linear_value_validation` | 2 | `P_J_qry_feat` | 2702.945 |
| OGBN-Arxiv | CPU | `linear_value_validation` | 3 | `P_h0_Query_ELU` | 1133.807 |
| OGBN-Arxiv | CPU | `linear_value_validation` | 4 | `P_h0_Query_dst` | 750.732 |
| OGBN-Arxiv | CPU | `linear_value_validation` | 5 | `P_out0_Query_dst` | 744.831 |
| OGBN-Arxiv | CPU | `derived_point` | 1 | `P_Query_feat` | 1310.231 |
| OGBN-Arxiv | CPU | `derived_point` | 2 | `P_J_qry_feat` | 1304.291 |
| OGBN-Arxiv | CPU | `derived_point` | 3 | `P_h0_Query_ELU` | 650.842 |
| OGBN-Arxiv | CPU | `derived_point` | 4 | `P_out0_w` | 383.775 |
| OGBN-Arxiv | CPU | `derived_point` | 5 | `P_h0_w_psq` | 370.267 |
| OGBN-Arxiv | GPU | `linear_value_validation` | 1 | `P_Query_feat` | 2401.235 |
| OGBN-Arxiv | GPU | `linear_value_validation` | 2 | `P_J_qry_feat` | 2379.430 |
| OGBN-Arxiv | GPU | `linear_value_validation` | 3 | `P_h0_Query_ELU` | 1141.411 |
| OGBN-Arxiv | GPU | `linear_value_validation` | 4 | `P_h0_Query_dst` | 878.741 |
| OGBN-Arxiv | GPU | `linear_value_validation` | 5 | `P_out0_Query_dst` | 743.648 |
| OGBN-Arxiv | GPU | `derived_point` | 1 | `P_Query_feat` | 1247.145 |
| OGBN-Arxiv | GPU | `derived_point` | 2 | `P_J_qry_feat` | 1243.405 |
| OGBN-Arxiv | GPU | `derived_point` | 3 | `P_h0_Query_ELU` | 648.025 |
| OGBN-Arxiv | GPU | `derived_point` | 4 | `P_out0_w` | 386.185 |
| OGBN-Arxiv | GPU | `derived_point` | 5 | `P_h0_w_psq` | 378.958 |

---

## 4. Quotient 阶段

### 4.1 有直接聚合计时的 `t_edge` / `t_FH` / `t_N`

| 数据集 | 后端 | `t_edge` pre/eval/coef | `t_FH` pre/eval/coef | `t_N` pre/eval/coef | Quotient commit batch |
|---|---:|---:|---:|---:|---:|
| Cora | CPU | 13.214 / 6.178 / 0.025 s | 5.847 / 3.129 / 1.148 s | 4.002 / 4.293 / 0.008 s | 10.498 s |
| Cora | GPU | 13.114 / 6.002 / 0.026 s | 5.739 / 3.149 / 1.170 s | 3.894 / 5.474 / 0.006 s | 10.508 s |
| Citeseer | CPU | 10.969 / 6.733 / 0.033 s | 63.488 / 18.632 / 6.209 s | 3.984 / 4.246 / 0.005 s | 43.998 s |
| Citeseer | GPU | 10.399 / 6.669 / 0.023 s | 61.807 / 18.755 / 5.030 s | 3.795 / 4.346 / 0.005 s | 42.845 s |
| Pubmed | CPU | 79.110 / 66.373 / 0.153 s | 64.149 / 49.198 / 5.411 s | 20.750 / 8.375 / 0.023 s | 44.096 s |
| Pubmed | GPU | 80.177 / 69.315 / 0.129 s | 64.284 / 49.444 / 6.145 s | 21.249 / 8.674 / 0.023 s | 48.909 s |
| PPI | CPU | 1346.443 / 597.841 / 2.628 s | 5.163 / 19.613 / 1.042 s | 44.642 / 7.038 / 0.037 s | 30.971 s |
| PPI | GPU | 1484.190 / 598.838 / 2.612 s | 5.104 / 19.590 / 1.130 s | 47.023 / 7.558 / 0.044 s | 31.697 s |
| OGBN-Arxiv | CPU | 1166.060 / 509.347 / 2.351 s | 133.362 / 600.211 / 11.366 s | 44.167 / 7.583 / 0.130 s | 104.766 s |
| OGBN-Arxiv | GPU | 1281.845 / 516.582 / 2.367 s | 142.677 / 598.268 / 10.912 s | 44.272 / 8.116 / 0.132 s | 101.544 s |

日志同时存在 `t_in`、`t_d_h`、`t_cat`、`t_C` 的 `start/done` 标记，但本批次日志没有为这四个域输出与 `t_edge/t_FH/t_N` 同口径的独立聚合 `ms` 字段，因此本文件不自行估算它们的耗时。

### 4.2 `t_edge` coset prewarm 规模、语义复用与 worker

| 数据集 | 后端 | Base count | Derived count | Semantic objects | Unique | Duplicates | Reuse groups | Base workers | Derived workers |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 2496 | 133 | 2496 | 1349 | 1147 | 105 | 20 | 20 |
| Cora | GPU | 2496 | 133 | 2496 | 1349 | 1147 | 105 | 20 | 20 |
| Citeseer | CPU | 2496 | 133 | 2496 | 1024 | 1472 | 44 | 20 | 20 |
| Citeseer | GPU | 2496 | 133 | 2496 | 1024 | 1472 | 44 | 20 | 20 |
| Pubmed | CPU | 2496 | 133 | 2496 | 1024 | 1472 | 44 | 20 | 20 |
| Pubmed | GPU | 2496 | 133 | 2496 | 1024 | 1472 | 44 | 20 | 20 |
| PPI | CPU | 648 | 28 | 189 | 106 | 83 | 8 | 4 | 4 |
| PPI | GPU | 648 | 28 | 189 | 106 | 83 | 8 | 4 | 4 |
| OGBN-Arxiv | CPU | 543 | 28 | 159 | 92 | 67 | 6 | 4 | 4 |
| OGBN-Arxiv | GPU | 543 | 28 | 159 | 92 | 67 | 6 | 4 | 4 |

### 4.3 Coset tiered storage：完整记录（包括 `files=0`）

本表保留 `t_edge` 和 `t_FH` 的全部 tiered storage 字段，包括 `files=0` 的记录，从而同时保留内存驻留量、内存预算、evaluation block 和 lazy blinder 统计。

| 数据集 | 后端 | 域 | Memory bytes | Memory budget | Files | File-backed bytes | Lazy blinder views | Avoided materialization bytes | Evaluation block |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | `t_edge` | 6,673,137,664 | 8,589,934,592 | 0 | 0 | 1359 | 5,700,059,136 | 131,072 |
| Cora | CPU | `t_FH` | 3,221,225,472 | 8,589,934,592 | 0 | 0 | 0 | 0 | 16,777,216 |
| Cora | GPU | `t_edge` | 6,673,137,664 | 8,589,934,592 | 0 | 0 | 1363 | 5,716,836,352 | 131,072 |
| Cora | GPU | `t_FH` | 3,221,225,472 | 8,589,934,592 | 0 | 0 | 0 | 0 | 16,777,216 |
| Citeseer | CPU | `t_edge` | 5,309,988,864 | 8,589,934,592 | 0 | 0 | 1627 | 6,824,132,608 | 131,072 |
| Citeseer | CPU | `t_FH` | 8,589,934,592 | 8,589,934,592 | 2 | 4,294,967,296 | 0 | 0 | 32,768 |
| Citeseer | GPU | `t_edge` | 5,309,988,864 | 8,589,934,592 | 0 | 0 | 1625 | 6,815,744,000 | 131,072 |
| Citeseer | GPU | `t_FH` | 8,589,934,592 | 8,589,934,592 | 2 | 4,294,967,296 | 0 | 0 | 32,768 |
| Pubmed | CPU | `t_edge` | 8,589,934,592 | 8,589,934,592 | 1010 | 33,889,976,320 | 1626 | 54,559,506,432 | 32,768 |
| Pubmed | CPU | `t_FH` | 8,589,934,592 | 8,589,934,592 | 2 | 4,294,967,296 | 0 | 0 | 32,768 |
| Pubmed | GPU | `t_edge` | 8,589,934,592 | 8,589,934,592 | 1010 | 33,889,976,320 | 1627 | 54,593,060,864 | 32,768 |
| Pubmed | GPU | `t_FH` | 8,589,934,592 | 8,589,934,592 | 2 | 4,294,967,296 | 0 | 0 | 32,768 |
| PPI | CPU | `t_edge` | 8,589,934,592 | 8,589,934,592 | 610 | 654,982,512,640 | 102 | 109,521,666,048 | 32,768 |
| PPI | CPU | `t_FH` | 3,221,225,472 | 8,589,934,592 | 0 | 0 | 0 | 0 | 16,777,216 |
| PPI | GPU | `t_edge` | 8,589,934,592 | 8,589,934,592 | 610 | 654,982,512,640 | 102 | 109,521,666,048 | 32,768 |
| PPI | GPU | `t_FH` | 3,221,225,472 | 8,589,934,592 | 0 | 0 | 0 | 0 | 16,777,216 |
| OGBN-Arxiv | CPU | `t_edge` | 8,589,934,592 | 8,589,934,592 | 521 | 559,419,490,304 | 87 | 93,415,538,688 | 32,768 |
| OGBN-Arxiv | CPU | `t_FH` | 8,589,934,592 | 8,589,934,592 | 4 | 17,179,869,184 | 0 | 0 | 32,768 |
| OGBN-Arxiv | GPU | `t_edge` | 8,589,934,592 | 8,589,934,592 | 521 | 559,419,490,304 | 87 | 93,415,538,688 | 32,768 |
| OGBN-Arxiv | GPU | `t_FH` | 8,589,934,592 | 8,589,934,592 | 4 | 17,179,869,184 | 0 | 0 | 32,768 |

这些 `bytes` 是日志中的累计/逻辑统计字段，不表示同等数量数据同时常驻内存，也不等同于实际物理磁盘读写量。

### 4.4 `t_FH` spilled-lazy 对象

十组日志均出现 `prove:quotient:t_FH:spilled_lazy:*` 路径。每组日志中记录到的对象如下：

- Cora CPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- Cora GPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- Citeseer CPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- Citeseer GPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- Pubmed CPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- Pubmed GPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- PPI CPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- PPI GPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- OGBN-Arxiv CPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`
- OGBN-Arxiv GPU: `P_J_qry_feat`, `P_Query_feat`, `P_R_feat`, `P_m_feat`

---

## 5. Domain Opening：七个求值域的完整聚合记录

各域 opening 存在并行关系，因此各域子项不能相加得到 `Domain opening total`。每个域日志同时给出 `gather_ms` 与 `total_ms`。

| 数据集 | 后端 | Total | FH gather/total | edge gather/total | in gather/total | cat gather/total | N gather/total | d_h gather/total | C gather/total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 3.418 s | 2.247/2.247 s | 3.031/3.032 s | 0.006/0.006 s | 0.002/0.002 s | 0.125/0.125 s | 0.002/0.002 s | 0.005/0.005 s |
| Cora | GPU | 2.417 s | 2.043/2.043 s | 0.613/0.614 s | 0.007/0.007 s | 0.002/0.002 s | 0.064/0.065 s | 0.004/0.004 s | 0.002/0.002 s |
| Citeseer | CPU | 11.231 s | 10.532/10.532 s | 4.539/4.540 s | 0.032/0.032 s | 0.003/0.003 s | 0.141/0.142 s | 0.004/0.004 s | 0.002/0.002 s |
| Citeseer | GPU | 12.185 s | 10.676/10.676 s | 1.568/1.569 s | 0.008/0.008 s | 0.003/0.003 s | 0.062/0.062 s | 0.004/0.004 s | 0.002/0.002 s |
| Pubmed | CPU | 12.249 s | 9.403/9.403 s | 11.207/11.208 s | 0.005/0.005 s | 0.002/0.002 s | 1.191/1.191 s | 0.006/0.006 s | 0.005/0.005 s |
| Pubmed | GPU | 9.719 s | 7.781/7.781 s | 3.336/3.337 s | 0.004/0.004 s | 0.003/0.003 s | 1.229/1.229 s | 0.004/0.004 s | 0.003/0.003 s |
| PPI | CPU | 203.597 s | 2.574/2.574 s | 202.810/202.816 s | 0.001/0.001 s | 0.001/0.001 s | 0.793/0.794 s | 0.001/0.001 s | 0.001/0.001 s |
| PPI | GPU | 195.518 s | 2.397/2.397 s | 194.752/194.753 s | 0.001/0.001 s | 0.001/0.001 s | 0.639/0.640 s | 0.001/0.001 s | 0.001/0.001 s |
| OGBN-Arxiv | CPU | 184.814 s | 13.616/13.616 s | 181.066/181.066 s | 0.002/0.002 s | 0.002/0.002 s | 1.644/1.644 s | 0.001/0.001 s | 0.001/0.001 s |
| OGBN-Arxiv | GPU | 221.318 s | 35.088/35.088 s | 197.730/197.743 s | 0.001/0.001 s | 0.001/0.001 s | 18.160/18.160 s | 0.001/0.001 s | 0.001/0.002 s |

### 5.1 Domain opening total 的 GPU - CPU 差值

| 数据集 | CPU | GPU | GPU - CPU |
|---|---:|---:|---:|
| Cora | 3.418 s | 2.417 s | -1.001 s |
| Citeseer | 11.231 s | 12.185 s | +0.954 s |
| Pubmed | 12.249 s | 9.719 s | -2.530 s |
| PPI | 203.597 s | 195.518 s | -8.079 s |
| OGBN-Arxiv | 184.814 s | 221.318 s | +36.504 s |

---

## 6. External Opening：日志层级、Domain Witness 与 KZG Witness MSM

`prove:external_opening:start` 与 `prove:external_opening:done` 构成 console log 中可直接观察到的 External Opening 父范围。该范围内同时出现 `kzg:opening_witness_msm` 数值事件和 `prove:domain_witness:group/...:done` 事件。日志没有 `prove:external_opening:done:ms=...`，因此本文件不自行构造 External Opening 墙钟总时间。

### 6.1 实际日志顺序与包含关系

本批次十组日志均呈现以下事件顺序：

```text
prove:external_opening:start
    ├── kzg:opening_witness_msm ...          # 首条 domain_witness group 汇总之前
    ├── prove:domain_witness:group ...       # 7 个域 × 2 次，共 14 条 group 汇总
    ├── prove:domain_witness:done:ms=...
    ├── kzg:opening_witness_msm ...          # domain_witness:done 之后
    └── prove:external_opening:done
```

日志没有独立的 `prove:domain_witness:start` 标记，因此本文件不把首批 14 条 KZG witness MSM 进一步解释为某个未显式标记的子阶段，只按它们相对于首条 `domain_witness:group` 和 `domain_witness:done` 的出现顺序归类。十组日志在“首条 group 汇总至 `domain_witness:done`”之间均没有额外的独立 `kzg:opening_witness_msm` 数值行；该区间只看到 14 条 domain-witness group 汇总及 `domain_witness:done`。

`prove:domain_witness:done:ms` 是日志直接给出的聚合总时间。每组运行对七个求值域均记录两条 `Open_dom` group 调用；日志同时给出 `precompute_ms`、`inverse_ms`、`assembly_ms`、`witness_msm_ms`、`blinder_ms`、`workers` 和 `outer_workers`。

### 6.2 Domain Witness total

| 数据集 | CPU | GPU | GPU - CPU |
|---|---:|---:|---:|
| Cora | 30.258 s | 29.754 s | -0.505 s |
| Citeseer | 111.988 s | 112.014 s | +0.026 s |
| Pubmed | 124.627 s | 123.698 s | -0.929 s |
| PPI | 416.740 s | 402.960 s | -13.780 s |
| OGBN-Arxiv | 547.239 s | 539.764 s | -7.475 s |

### 6.3 七个域的两次 group 调用及其内部拆分

`Call 1/2` 仅按该域两条 group 日志在文件中的出现顺序编号，不推断它们对应的挑战点或其他协议语义。两次 group 调用可能在 `outer_workers=2` 下重叠执行；因此不能把下表各行简单相加得到 `domain_witness total`。

| 数据集 | 后端 | 域 | Call | Group total | Precompute | Inverse | Assembly | Witness MSM | Blinder | Workers | Outer workers |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | `FH` | 1 | 23.761 s | 1.127 s | 0.548 s | 0.032 s | 22.054 s | 0.222 ms | 16 | 2 |
| Cora | CPU | `FH` | 2 | 27.821 s | 1.423 s | 0.408 s | 0.052 s | 25.839 s | 0.212 ms | 16 | 2 |
| Cora | CPU | `edge` | 1 | 2.356 s | 2.031 s | 0.003 s | 0.003 s | 0.294 s | 0.281 ms | 16 | 2 |
| Cora | CPU | `edge` | 2 | 2.436 s | 2.108 s | 0.003 s | 0.001 s | 0.283 s | 0.279 ms | 16 | 2 |
| Cora | CPU | `N` | 1 | 0.103 s | 0.061 s | 0.000 s | 0.000 s | 0.036 s | 0.212 ms | 16 | 2 |
| Cora | CPU | `N` | 2 | 0.105 s | 0.061 s | 0.000 s | 0.000 s | 0.038 s | 0.212 ms | 16 | 2 |
| Cora | CPU | `in` | 1 | 0.024 s | 0.001 s | 0.000 s | 0.000 s | 0.022 s | 0.212 ms | 10 | 2 |
| Cora | CPU | `in` | 2 | 0.024 s | 0.001 s | 0.000 s | 0.000 s | 0.022 s | 0.174 ms | 10 | 2 |
| Cora | CPU | `cat` | 1 | 0.006 s | 0.000 s | 0.000 s | 0.000 s | 0.006 s | 0.175 ms | 10 | 2 |
| Cora | CPU | `cat` | 2 | 0.006 s | 0.000 s | 0.000 s | 0.000 s | 0.005 s | 0.167 ms | 10 | 2 |
| Cora | CPU | `d_h` | 1 | 0.004 s | 0.001 s | 0.000 s | 0.000 s | 0.002 s | 0.173 ms | 10 | 2 |
| Cora | CPU | `d_h` | 2 | 0.004 s | 0.001 s | 0.000 s | 0.000 s | 0.002 s | 0.172 ms | 10 | 2 |
| Cora | CPU | `C` | 1 | 0.003 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.169 ms | 10 | 2 |
| Cora | CPU | `C` | 2 | 0.003 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.169 ms | 10 | 2 |
| Cora | GPU | `FH` | 1 | 27.431 s | 0.813 s | 0.413 s | 0.061 s | 26.058 s | 0.211 ms | 16 | 2 |
| Cora | GPU | `FH` | 2 | 23.550 s | 0.750 s | 0.324 s | 0.030 s | 22.446 s | 0.225 ms | 16 | 2 |
| Cora | GPU | `edge` | 1 | 2.383 s | 2.044 s | 0.003 s | 0.001 s | 0.309 s | 0.332 ms | 16 | 2 |
| Cora | GPU | `edge` | 2 | 2.321 s | 1.998 s | 0.005 s | 0.001 s | 0.292 s | 0.551 ms | 16 | 2 |
| Cora | GPU | `N` | 1 | 0.104 s | 0.061 s | 0.000 s | 0.000 s | 0.036 s | 0.223 ms | 16 | 2 |
| Cora | GPU | `N` | 2 | 0.113 s | 0.062 s | 0.000 s | 0.000 s | 0.045 s | 0.492 ms | 16 | 2 |
| Cora | GPU | `in` | 1 | 2.723 s | 0.003 s | 0.000 s | 0.000 s | 2.719 s | 0.485 ms | 10 | 2 |
| Cora | GPU | `in` | 2 | 0.211 s | 0.003 s | 0.000 s | 0.000 s | 0.207 s | 0.379 ms | 10 | 2 |
| Cora | GPU | `cat` | 1 | 0.013 s | 0.000 s | 0.000 s | 0.000 s | 0.012 s | 0.363 ms | 10 | 2 |
| Cora | GPU | `cat` | 2 | 0.013 s | 0.000 s | 0.000 s | 0.000 s | 0.012 s | 0.353 ms | 10 | 2 |
| Cora | GPU | `d_h` | 1 | 0.007 s | 0.001 s | 0.000 s | 0.000 s | 0.004 s | 0.351 ms | 10 | 2 |
| Cora | GPU | `d_h` | 2 | 0.007 s | 0.001 s | 0.000 s | 0.000 s | 0.004 s | 0.351 ms | 10 | 2 |
| Cora | GPU | `C` | 1 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.005 s | 0.346 ms | 10 | 2 |
| Cora | GPU | `C` | 2 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.005 s | 0.354 ms | 10 | 2 |
| Citeseer | CPU | `FH` | 1 | 95.738 s | 3.690 s | 4.018 s | 0.224 s | 87.566 s | 0.210 ms | 16 | 2 |
| Citeseer | CPU | `FH` | 2 | 111.550 s | 3.692 s | 3.086 s | 0.205 s | 104.311 s | 0.199 ms | 16 | 2 |
| Citeseer | CPU | `edge` | 1 | 13.257 s | 2.743 s | 0.005 s | 0.001 s | 10.481 s | 0.573 ms | 16 | 2 |
| Citeseer | CPU | `edge` | 2 | 2.952 s | 2.670 s | 0.003 s | 0.001 s | 0.238 s | 0.265 ms | 16 | 2 |
| Citeseer | CPU | `N` | 1 | 0.103 s | 0.062 s | 0.000 s | 0.000 s | 0.035 s | 0.208 ms | 16 | 2 |
| Citeseer | CPU | `N` | 2 | 0.199 s | 0.061 s | 0.001 s | 0.000 s | 0.132 s | 0.211 ms | 16 | 2 |
| Citeseer | CPU | `in` | 1 | 0.093 s | 0.003 s | 0.001 s | 0.000 s | 0.089 s | 0.215 ms | 10 | 2 |
| Citeseer | CPU | `in` | 2 | 0.042 s | 0.003 s | 0.000 s | 0.000 s | 0.038 s | 0.208 ms | 10 | 2 |
| Citeseer | CPU | `cat` | 1 | 0.012 s | 0.000 s | 0.000 s | 0.000 s | 0.012 s | 0.170 ms | 10 | 2 |
| Citeseer | CPU | `cat` | 2 | 0.006 s | 0.000 s | 0.000 s | 0.000 s | 0.006 s | 0.177 ms | 10 | 2 |
| Citeseer | CPU | `d_h` | 1 | 0.006 s | 0.001 s | 0.000 s | 0.000 s | 0.004 s | 0.171 ms | 10 | 2 |
| Citeseer | CPU | `d_h` | 2 | 0.005 s | 0.001 s | 0.000 s | 0.000 s | 0.003 s | 0.175 ms | 10 | 2 |
| Citeseer | CPU | `C` | 1 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.001 s | 0.175 ms | 10 | 2 |
| Citeseer | CPU | `C` | 2 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.001 s | 0.172 ms | 10 | 2 |
| Citeseer | GPU | `FH` | 1 | 95.106 s | 3.670 s | 4.882 s | 0.223 s | 86.242 s | 0.406 ms | 16 | 2 |
| Citeseer | GPU | `FH` | 2 | 112.013 s | 3.727 s | 2.966 s | 0.194 s | 104.734 s | 0.203 ms | 16 | 2 |
| Citeseer | GPU | `edge` | 1 | 13.511 s | 2.601 s | 0.005 s | 0.001 s | 10.863 s | 0.576 ms | 16 | 2 |
| Citeseer | GPU | `edge` | 2 | 2.762 s | 2.500 s | 0.003 s | 0.001 s | 0.217 s | 0.307 ms | 16 | 2 |
| Citeseer | GPU | `N` | 1 | 0.105 s | 0.062 s | 0.000 s | 0.000 s | 0.037 s | 0.211 ms | 16 | 2 |
| Citeseer | GPU | `N` | 2 | 0.105 s | 0.063 s | 0.000 s | 0.000 s | 0.036 s | 0.257 ms | 16 | 2 |
| Citeseer | GPU | `in` | 1 | 0.042 s | 0.003 s | 0.000 s | 0.000 s | 0.038 s | 0.195 ms | 10 | 2 |
| Citeseer | GPU | `in` | 2 | 0.042 s | 0.003 s | 0.000 s | 0.000 s | 0.038 s | 0.218 ms | 10 | 2 |
| Citeseer | GPU | `cat` | 1 | 0.006 s | 0.000 s | 0.000 s | 0.000 s | 0.006 s | 0.176 ms | 10 | 2 |
| Citeseer | GPU | `cat` | 2 | 0.006 s | 0.000 s | 0.000 s | 0.000 s | 0.006 s | 0.178 ms | 10 | 2 |
| Citeseer | GPU | `d_h` | 1 | 0.004 s | 0.001 s | 0.000 s | 0.000 s | 0.002 s | 0.169 ms | 10 | 2 |
| Citeseer | GPU | `d_h` | 2 | 0.004 s | 0.001 s | 0.000 s | 0.000 s | 0.002 s | 0.172 ms | 10 | 2 |
| Citeseer | GPU | `C` | 1 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.001 s | 0.171 ms | 10 | 2 |
| Citeseer | GPU | `C` | 2 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.001 s | 0.174 ms | 10 | 2 |
| Pubmed | CPU | `FH` | 1 | 102.464 s | 6.108 s | 5.139 s | 0.394 s | 90.504 s | 0.770 ms | 16 | 2 |
| Pubmed | CPU | `FH` | 2 | 117.627 s | 5.825 s | 3.133 s | 0.207 s | 108.065 s | 0.211 ms | 16 | 2 |
| Pubmed | CPU | `edge` | 1 | 6.662 s | 4.724 s | 0.068 s | 0.004 s | 1.840 s | 0.389 ms | 16 | 2 |
| Pubmed | CPU | `edge` | 2 | 6.999 s | 4.732 s | 0.024 s | 0.002 s | 2.215 s | 0.212 ms | 16 | 2 |
| Pubmed | CPU | `N` | 1 | 12.429 s | 0.501 s | 0.005 s | 0.001 s | 11.917 s | 0.395 ms | 16 | 2 |
| Pubmed | CPU | `N` | 2 | 1.071 s | 0.516 s | 0.005 s | 0.001 s | 0.541 s | 0.388 ms | 16 | 2 |
| Pubmed | CPU | `in` | 1 | 0.014 s | 0.001 s | 0.000 s | 0.000 s | 0.012 s | 0.379 ms | 10 | 2 |
| Pubmed | CPU | `in` | 2 | 0.014 s | 0.001 s | 0.000 s | 0.000 s | 0.013 s | 0.378 ms | 10 | 2 |
| Pubmed | CPU | `cat` | 1 | 0.012 s | 0.000 s | 0.000 s | 0.000 s | 0.012 s | 0.359 ms | 10 | 2 |
| Pubmed | CPU | `cat` | 2 | 0.012 s | 0.000 s | 0.000 s | 0.000 s | 0.012 s | 0.360 ms | 10 | 2 |
| Pubmed | CPU | `d_h` | 1 | 0.007 s | 0.001 s | 0.000 s | 0.000 s | 0.004 s | 0.355 ms | 10 | 2 |
| Pubmed | CPU | `d_h` | 2 | 0.007 s | 0.001 s | 0.000 s | 0.000 s | 0.004 s | 0.350 ms | 10 | 2 |
| Pubmed | CPU | `C` | 1 | 0.003 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.523 ms | 10 | 2 |
| Pubmed | CPU | `C` | 2 | 0.003 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.519 ms | 10 | 2 |
| Pubmed | GPU | `FH` | 1 | 103.765 s | 5.866 s | 5.700 s | 0.252 s | 91.318 s | 0.214 ms | 16 | 2 |
| Pubmed | GPU | `FH` | 2 | 116.599 s | 5.572 s | 4.398 s | 0.211 s | 106.056 s | 0.212 ms | 16 | 2 |
| Pubmed | GPU | `edge` | 1 | 6.769 s | 4.815 s | 0.040 s | 0.005 s | 1.884 s | 0.202 ms | 16 | 2 |
| Pubmed | GPU | `edge` | 2 | 7.098 s | 4.820 s | 0.024 s | 0.002 s | 2.212 s | 0.210 ms | 16 | 2 |
| Pubmed | GPU | `N` | 1 | 11.180 s | 0.969 s | 0.005 s | 0.001 s | 10.199 s | 0.387 ms | 16 | 2 |
| Pubmed | GPU | `N` | 2 | 0.849 s | 0.502 s | 0.003 s | 0.001 s | 0.335 s | 0.199 ms | 16 | 2 |
| Pubmed | GPU | `in` | 1 | 0.010 s | 0.000 s | 0.000 s | 0.000 s | 0.009 s | 0.214 ms | 10 | 2 |
| Pubmed | GPU | `in` | 2 | 0.010 s | 0.001 s | 0.000 s | 0.000 s | 0.009 s | 0.212 ms | 10 | 2 |
| Pubmed | GPU | `cat` | 1 | 0.012 s | 0.000 s | 0.000 s | 0.000 s | 0.011 s | 0.359 ms | 10 | 2 |
| Pubmed | GPU | `cat` | 2 | 0.012 s | 0.000 s | 0.000 s | 0.000 s | 0.011 s | 0.360 ms | 10 | 2 |
| Pubmed | GPU | `d_h` | 1 | 0.004 s | 0.001 s | 0.000 s | 0.000 s | 0.002 s | 0.172 ms | 10 | 2 |
| Pubmed | GPU | `d_h` | 2 | 0.004 s | 0.001 s | 0.000 s | 0.000 s | 0.002 s | 0.171 ms | 10 | 2 |
| Pubmed | GPU | `C` | 1 | 0.003 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.524 ms | 10 | 2 |
| Pubmed | GPU | `C` | 2 | 0.003 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.528 ms | 10 | 2 |
| PPI | CPU | `FH` | 1 | 29.397 s | 4.000 s | 0.673 s | 0.033 s | 24.691 s | 0.218 ms | 16 | 2 |
| PPI | CPU | `FH` | 2 | 27.189 s | 3.545 s | 0.735 s | 0.053 s | 22.803 s | 0.415 ms | 16 | 2 |
| PPI | CPU | `edge` | 1 | 380.994 s | 333.126 s | 1.328 s | 0.063 s | 46.470 s | 0.410 ms | 16 | 2 |
| PPI | CPU | `edge` | 2 | 385.556 s | 333.087 s | 1.126 s | 0.128 s | 51.009 s | 0.209 ms | 16 | 2 |
| PPI | CPU | `N` | 1 | 6.345 s | 5.899 s | 0.048 s | 0.002 s | 0.385 s | 0.377 ms | 16 | 2 |
| PPI | CPU | `N` | 2 | 1.850 s | 1.342 s | 0.005 s | 0.003 s | 0.493 s | 0.194 ms | 16 | 2 |
| PPI | CPU | `in` | 1 | 0.016 s | 0.000 s | 0.001 s | 0.000 s | 0.015 s | 0.383 ms | 10 | 2 |
| PPI | CPU | `in` | 2 | 0.014 s | 0.000 s | 0.000 s | 0.000 s | 0.014 s | 0.350 ms | 10 | 2 |
| PPI | CPU | `cat` | 1 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.354 ms | 10 | 2 |
| PPI | CPU | `cat` | 2 | 0.004 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.388 ms | 10 | 2 |
| PPI | CPU | `d_h` | 1 | 0.005 s | 0.000 s | 0.001 s | 0.000 s | 0.004 s | 0.361 ms | 10 | 2 |
| PPI | CPU | `d_h` | 2 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.349 ms | 10 | 2 |
| PPI | CPU | `C` | 1 | 1.244 s | 0.001 s | 0.000 s | 0.000 s | 1.241 s | 0.838 ms | 10 | 2 |
| PPI | CPU | `C` | 2 | 0.038 s | 0.000 s | 0.000 s | 0.000 s | 0.036 s | 0.551 ms | 10 | 2 |
| PPI | GPU | `FH` | 1 | 25.565 s | 2.395 s | 0.417 s | 0.065 s | 22.601 s | 0.407 ms | 16 | 2 |
| PPI | GPU | `FH` | 2 | 27.516 s | 1.953 s | 0.547 s | 0.030 s | 24.986 s | 0.251 ms | 16 | 2 |
| PPI | GPU | `edge` | 1 | 368.438 s | 324.533 s | 1.063 s | 0.064 s | 42.772 s | 0.399 ms | 16 | 2 |
| PPI | GPU | `edge` | 2 | 374.741 s | 324.510 s | 0.809 s | 0.115 s | 49.118 s | 0.203 ms | 16 | 2 |
| PPI | GPU | `N` | 1 | 5.523 s | 1.436 s | 0.009 s | 0.002 s | 4.055 s | 0.407 ms | 16 | 2 |
| PPI | GPU | `N` | 2 | 1.482 s | 1.184 s | 0.005 s | 0.002 s | 0.281 s | 0.193 ms | 16 | 2 |
| PPI | GPU | `in` | 1 | 0.014 s | 0.000 s | 0.000 s | 0.000 s | 0.014 s | 0.352 ms | 10 | 2 |
| PPI | GPU | `in` | 2 | 0.014 s | 0.000 s | 0.000 s | 0.000 s | 0.014 s | 0.349 ms | 10 | 2 |
| PPI | GPU | `cat` | 1 | 0.004 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.351 ms | 10 | 2 |
| PPI | GPU | `cat` | 2 | 0.004 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.344 ms | 10 | 2 |
| PPI | GPU | `d_h` | 1 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.356 ms | 10 | 2 |
| PPI | GPU | `d_h` | 2 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.349 ms | 10 | 2 |
| PPI | GPU | `C` | 1 | 1.744 s | 0.000 s | 0.000 s | 0.000 s | 1.743 s | 0.539 ms | 10 | 2 |
| PPI | GPU | `C` | 2 | 0.023 s | 0.000 s | 0.000 s | 0.000 s | 0.022 s | 0.548 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `FH` | 1 | 193.757 s | 19.839 s | 10.639 s | 0.563 s | 161.756 s | 0.508 ms | 16 | 2 |
| OGBN-Arxiv | CPU | `FH` | 2 | 210.410 s | 14.571 s | 9.028 s | 0.560 s | 185.621 s | 0.219 ms | 16 | 2 |
| OGBN-Arxiv | CPU | `edge` | 1 | 336.819 s | 285.691 s | 0.636 s | 0.031 s | 50.454 s | 0.219 ms | 16 | 2 |
| OGBN-Arxiv | CPU | `edge` | 2 | 329.669 s | 285.667 s | 1.240 s | 0.106 s | 42.502 s | 0.404 ms | 16 | 2 |
| OGBN-Arxiv | CPU | `N` | 1 | 19.450 s | 1.167 s | 0.039 s | 0.004 s | 18.234 s | 0.218 ms | 16 | 2 |
| OGBN-Arxiv | CPU | `N` | 2 | 1.950 s | 0.533 s | 0.021 s | 0.002 s | 1.391 s | 0.227 ms | 16 | 2 |
| OGBN-Arxiv | CPU | `in` | 1 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.005 s | 0.213 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `in` | 2 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.220 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `cat` | 1 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.169 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `cat` | 2 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.181 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `d_h` | 1 | 0.003 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.173 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `d_h` | 2 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.168 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `C` | 1 | 0.010 s | 0.001 s | 0.000 s | 0.000 s | 0.009 s | 0.257 ms | 10 | 2 |
| OGBN-Arxiv | CPU | `C` | 2 | 0.009 s | 0.000 s | 0.000 s | 0.000 s | 0.008 s | 0.249 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `FH` | 1 | 195.140 s | 17.702 s | 8.395 s | 0.551 s | 168.149 s | 0.417 ms | 16 | 2 |
| OGBN-Arxiv | GPU | `FH` | 2 | 209.914 s | 15.468 s | 9.867 s | 0.516 s | 183.595 s | 0.219 ms | 16 | 2 |
| OGBN-Arxiv | GPU | `edge` | 1 | 324.759 s | 278.633 s | 1.114 s | 0.059 s | 44.902 s | 0.405 ms | 16 | 2 |
| OGBN-Arxiv | GPU | `edge` | 2 | 329.849 s | 278.675 s | 1.122 s | 0.124 s | 49.765 s | 0.220 ms | 16 | 2 |
| OGBN-Arxiv | GPU | `N` | 1 | 15.050 s | 2.490 s | 0.050 s | 0.002 s | 12.503 s | 0.220 ms | 16 | 2 |
| OGBN-Arxiv | GPU | `N` | 2 | 2.134 s | 0.608 s | 0.021 s | 0.002 s | 1.500 s | 0.218 ms | 16 | 2 |
| OGBN-Arxiv | GPU | `in` | 1 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.005 s | 0.227 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `in` | 2 | 0.005 s | 0.000 s | 0.000 s | 0.000 s | 0.004 s | 0.171 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `cat` | 1 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.167 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `cat` | 2 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.171 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `d_h` | 1 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.169 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `d_h` | 2 | 0.002 s | 0.000 s | 0.000 s | 0.000 s | 0.002 s | 0.171 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `C` | 1 | 0.009 s | 0.000 s | 0.000 s | 0.000 s | 0.008 s | 0.250 ms | 10 | 2 |
| OGBN-Arxiv | GPU | `C` | 2 | 0.009 s | 0.000 s | 0.000 s | 0.000 s | 0.008 s | 0.254 ms | 10 | 2 |

### 6.4 KZG opening witness MSM 的日志顺序区间统计

下表只对 `prove:external_opening:start` 与 `done` 之间的 `kzg:opening_witness_msm` 数值事件按日志出现位置做直接计数和 `ms` 求和。这里的“前段/后段”是日志顺序描述，不赋予额外协议语义。

| 数据集 | 后端 | 首条 domain_witness group 前 Calls | 前段 Σ ms | group→done 区间 KZG Calls | `domain_witness:done` 后 Calls | 后段 Σ ms | External-opening KZG Calls | KZG Σ ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | CPU | 14 | 13696.916 | 0 | 92 | 27613.003 | 106 | 41309.920 |
| Cora | GPU | 14 | 14172.831 | 0 | 92 | 27247.334 | 106 | 41420.165 |
| Citeseer | CPU | 14 | 54652.765 | 0 | 92 | 28203.137 | 106 | 82855.902 |
| Citeseer | GPU | 14 | 50418.259 | 0 | 92 | 28073.333 | 106 | 78491.593 |
| Pubmed | CPU | 14 | 54660.453 | 0 | 92 | 200906.970 | 106 | 255567.423 |
| Pubmed | GPU | 14 | 50805.489 | 0 | 92 | 204074.873 | 106 | 254880.363 |
| PPI | CPU | 14 | 45178.773 | 0 | 22 | 121241.963 | 36 | 166420.737 |
| PPI | GPU | 14 | 42590.361 | 0 | 22 | 116644.567 | 36 | 159234.928 |
| OGBN-Arxiv | CPU | 14 | 131823.065 | 0 | 22 | 118981.657 | 36 | 250804.721 |
| OGBN-Arxiv | GPU | 14 | 135627.308 | 0 | 22 | 126466.137 | 36 | 262093.445 |

### 6.5 KZG opening witness MSM：全部记录的分组统计

原始日志中存在大量逐 opening 的 `kzg:opening_witness_msm` 行。本节不再只保留 `input ≥ 1,000,000`，而是按 `(domain, input, nonzero, effective, sparse, tail)` 对每组运行的全部记录进行分组，保留调用次数以及单条 `ms` 的直接统计。 解析同时覆盖带 `[progress]` 前缀和不带该前缀的同格式行；本批次 Citeseer CPU/GPU 日志各存在 1 条不带 `[progress]` 的 `edge` witness MSM 数值行，也已计入。

`Σ ms` 是该分组所有逐调用 `ms` 的直接相加值，不等同于 `external_opening` 墙钟总时间；日志没有提供一个完整的 `prove:external_opening:done:ms=...` 聚合字段，并行/重叠关系也不能从这些逐条记录直接消除。

| 数据集 | 后端 | Domain | Input | Nonzero | Effective | Sparse | Tail | Calls | Σ ms | Avg ms | Min ms | Max ms |
|---|---:|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|
| Cora | CPU | `C` | 16 | 16 | 16 | `false` | `false` | 3 | 2.264 | 0.755 | 0.545 | 1.165 |
| Cora | CPU | `FH` | 4,194,304 | 4,194,304 | 4,194,304 | `false` | `false` | 2 | 13409.657 | 6704.829 | 3501.958 | 9907.699 |
| Cora | CPU | `N` | 4,096 | 4,096 | 4,096 | `false` | `false` | 46 | 3380.068 | 73.480 | 12.410 | 141.935 |
| Cora | CPU | `cat` | 128 | 128 | 128 | `false` | `false` | 2 | 8.587 | 4.293 | 4.279 | 4.307 |
| Cora | CPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 1.090 | 0.545 | 0.545 | 0.545 |
| Cora | CPU | `edge` | 32,768 | 32,768 | 32,768 | `false` | `false` | 49 | 24493.709 | 499.872 | 88.909 | 864.114 |
| Cora | CPU | `in` | 2,048 | 2,048 | 2,048 | `false` | `false` | 2 | 14.544 | 7.272 | 7.182 | 7.362 |
| Cora | GPU | `C` | 16 | 16 | 16 | `false` | `false` | 3 | 2.861 | 0.954 | 0.564 | 1.158 |
| Cora | GPU | `FH` | 4,194,304 | 4,194,304 | 4,194,304 | `false` | `false` | 2 | 13765.614 | 6882.807 | 3510.160 | 10255.455 |
| Cora | GPU | `N` | 4,096 | 4,096 | 4,096 | `false` | `false` | 46 | 3358.301 | 73.007 | 12.458 | 142.091 |
| Cora | GPU | `cat` | 128 | 128 | 128 | `false` | `false` | 2 | 18.919 | 9.459 | 9.440 | 9.478 |
| Cora | GPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 2.301 | 1.151 | 1.142 | 1.159 |
| Cora | GPU | `edge` | 32,768 | 32,768 | 32,768 | `false` | `false` | 49 | 24113.207 | 492.106 | 69.313 | 864.802 |
| Cora | GPU | `in` | 2,048 | 2,048 | 2,048 | `false` | `false` | 2 | 158.961 | 79.481 | 15.329 | 143.632 |
| Citeseer | CPU | `C` | 8 | 8 | 8 | `false` | `false` | 3 | 1.490 | 0.497 | 0.304 | 0.647 |
| Citeseer | CPU | `FH` | 16,777,216 | 16,777,216 | 16,777,216 | `false` | `false` | 2 | 54194.141 | 27097.071 | 13775.090 | 40419.051 |
| Citeseer | CPU | `N` | 4,096 | 4,096 | 4,096 | `false` | `false` | 46 | 4646.187 | 101.004 | 12.908 | 155.702 |
| Citeseer | CPU | `cat` | 128 | 128 | 128 | `false` | `false` | 2 | 8.642 | 4.321 | 4.309 | 4.333 |
| Citeseer | CPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 2.906 | 1.453 | 0.542 | 2.363 |
| Citeseer | CPU | `edge` | 32,768 | 32,768 | 32,768 | `false` | `false` | 49 | 23922.513 | 488.215 | 90.966 | 861.033 |
| Citeseer | CPU | `in` | 4,096 | 4,096 | 4,096 | `false` | `false` | 2 | 80.023 | 40.011 | 14.298 | 65.725 |
| Citeseer | GPU | `C` | 8 | 8 | 8 | `false` | `false` | 3 | 0.916 | 0.305 | 0.300 | 0.315 |
| Citeseer | GPU | `FH` | 16,777,216 | 16,777,216 | 16,777,216 | `false` | `false` | 2 | 49961.847 | 24980.923 | 14236.123 | 35725.724 |
| Citeseer | GPU | `N` | 4,096 | 4,096 | 4,096 | `false` | `false` | 46 | 4561.035 | 99.153 | 12.968 | 142.048 |
| Citeseer | GPU | `cat` | 128 | 128 | 128 | `false` | `false` | 2 | 8.578 | 4.289 | 4.280 | 4.298 |
| Citeseer | GPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 1.110 | 0.555 | 0.550 | 0.560 |
| Citeseer | GPU | `edge` | 32,768 | 32,768 | 32,768 | `false` | `false` | 49 | 23929.226 | 488.352 | 74.304 | 861.223 |
| Citeseer | GPU | `in` | 4,096 | 4,096 | 4,096 | `false` | `false` | 2 | 28.881 | 14.440 | 14.377 | 14.504 |
| Pubmed | CPU | `C` | 8 | 8 | 8 | `false` | `false` | 4 | 1.875 | 0.469 | 0.299 | 0.643 |
| Pubmed | CPU | `FH` | 16,777,216 | 16,777,216 | 16,777,216 | `false` | `false` | 2 | 52887.313 | 26443.657 | 14566.610 | 38320.703 |
| Pubmed | CPU | `N` | 32,768 | 32,768 | 32,768 | `false` | `false` | 46 | 23409.787 | 508.908 | 230.454 | 883.506 |
| Pubmed | CPU | `cat` | 128 | 128 | 128 | `false` | `false` | 2 | 18.272 | 9.136 | 9.129 | 9.143 |
| Pubmed | CPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 2.308 | 1.154 | 1.151 | 1.156 |
| Pubmed | CPU | `edge` | 262,144 | 262,144 | 262,144 | `false` | `false` | 48 | 179239.319 | 3734.152 | 351.342 | 6201.941 |
| Pubmed | CPU | `in` | 512 | 512 | 512 | `false` | `false` | 2 | 8.550 | 4.275 | 4.253 | 4.296 |
| Pubmed | GPU | `C` | 8 | 8 | 8 | `false` | `false` | 4 | 2.549 | 0.637 | 0.630 | 0.646 |
| Pubmed | GPU | `FH` | 16,777,216 | 16,777,216 | 16,777,216 | `false` | `false` | 2 | 49139.124 | 24569.562 | 14460.991 | 34678.133 |
| Pubmed | GPU | `N` | 32,768 | 32,768 | 32,768 | `false` | `false` | 46 | 24085.941 | 523.607 | 128.307 | 1064.491 |
| Pubmed | GPU | `cat` | 128 | 128 | 128 | `false` | `false` | 2 | 18.302 | 9.151 | 9.134 | 9.168 |
| Pubmed | GPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 1.106 | 0.553 | 0.551 | 0.555 |
| Pubmed | GPU | `edge` | 262,144 | 262,144 | 262,144 | `false` | `false` | 48 | 181627.192 | 3783.900 | 357.023 | 5998.132 |
| Pubmed | GPU | `in` | 512 | 512 | 512 | `false` | `false` | 2 | 6.148 | 3.074 | 3.009 | 3.139 |
| PPI | CPU | `C` | 128 | 128 | 128 | `false` | `false` | 4 | 35.074 | 8.768 | 4.292 | 11.441 |
| PPI | CPU | `FH` | 4,194,304 | 4,194,304 | 4,194,304 | `false` | `false` | 2 | 15804.431 | 7902.215 | 5010.303 | 10794.128 |
| PPI | CPU | `N` | 65,536 | 65,536 | 65,536 | `false` | `false` | 11 | 1427.120 | 129.738 | 96.991 | 190.715 |
| PPI | CPU | `cat` | 16 | 16 | 16 | `false` | `false` | 2 | 2.468 | 1.234 | 1.152 | 1.316 |
| PPI | CPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 2.463 | 1.231 | 1.152 | 1.311 |
| PPI | CPU | `edge` | 8,388,608 | 8,388,608 | 8,388,608 | `false` | `false` | 13 | 149139.235 | 11472.249 | 8964.931 | 19432.846 |
| PPI | CPU | `in` | 64 | 64 | 64 | `false` | `false` | 2 | 9.946 | 4.973 | 4.480 | 5.466 |
| PPI | GPU | `C` | 128 | 128 | 128 | `false` | `false` | 4 | 32.824 | 8.206 | 7.294 | 9.130 |
| PPI | GPU | `FH` | 4,194,304 | 4,194,304 | 4,194,304 | `false` | `false` | 2 | 14670.096 | 7335.048 | 4249.196 | 10420.900 |
| PPI | GPU | `N` | 65,536 | 65,536 | 65,536 | `false` | `false` | 11 | 1712.878 | 155.716 | 96.793 | 480.914 |
| PPI | GPU | `cat` | 16 | 16 | 16 | `false` | `false` | 2 | 2.289 | 1.144 | 1.135 | 1.154 |
| PPI | GPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 2.298 | 1.149 | 1.147 | 1.151 |
| PPI | GPU | `edge` | 8,388,608 | 8,388,608 | 8,388,608 | `false` | `false` | 13 | 142805.551 | 10985.042 | 7507.474 | 19803.290 |
| PPI | GPU | `in` | 64 | 64 | 64 | `false` | `false` | 2 | 8.991 | 4.496 | 4.494 | 4.498 |
| OGBN-Arxiv | CPU | `C` | 64 | 64 | 64 | `false` | `false` | 4 | 24.811 | 6.203 | 2.133 | 12.485 |
| OGBN-Arxiv | CPU | `FH` | 33,554,432 | 33,554,432 | 33,554,432 | `false` | `false` | 2 | 102869.719 | 51434.859 | 29034.191 | 73835.528 |
| OGBN-Arxiv | CPU | `N` | 262,144 | 262,144 | 262,144 | `false` | `false` | 11 | 5632.290 | 512.026 | 336.267 | 1142.246 |
| OGBN-Arxiv | CPU | `cat` | 16 | 16 | 16 | `false` | `false` | 2 | 1.095 | 0.548 | 0.546 | 0.549 |
| OGBN-Arxiv | CPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 1.159 | 0.579 | 0.544 | 0.615 |
| OGBN-Arxiv | CPU | `edge` | 8,388,608 | 8,388,608 | 8,388,608 | `false` | `false` | 13 | 142271.483 | 10943.960 | 7520.749 | 19839.086 |
| OGBN-Arxiv | CPU | `in` | 256 | 256 | 256 | `false` | `false` | 2 | 4.165 | 2.082 | 1.629 | 2.536 |
| OGBN-Arxiv | GPU | `C` | 64 | 64 | 64 | `false` | `false` | 4 | 11.638 | 2.910 | 2.116 | 5.161 |
| OGBN-Arxiv | GPU | `FH` | 33,554,432 | 33,554,432 | 33,554,432 | `false` | `false` | 2 | 104938.983 | 52469.491 | 35210.627 | 69728.356 |
| OGBN-Arxiv | GPU | `N` | 262,144 | 262,144 | 262,144 | `false` | `false` | 11 | 5734.936 | 521.358 | 337.717 | 970.544 |
| OGBN-Arxiv | GPU | `cat` | 16 | 16 | 16 | `false` | `false` | 2 | 1.089 | 0.544 | 0.542 | 0.546 |
| OGBN-Arxiv | GPU | `d` | 16 | 16 | 16 | `false` | `false` | 2 | 1.079 | 0.539 | 0.536 | 0.542 |
| OGBN-Arxiv | GPU | `edge` | 8,388,608 | 8,388,608 | 8,388,608 | `false` | `false` | 13 | 151402.396 | 11646.338 | 8471.870 | 19961.799 |
| OGBN-Arxiv | GPU | `in` | 256 | 256 | 256 | `false` | `false` | 2 | 3.325 | 1.662 | 1.539 | 1.786 |

`kzg:opening_witness_msm` 行对该小域使用日志标签 `domain=d`；高层 Domain Opening / Domain Witness 使用 `d_h`。本文件按原始日志字段分别保留，不自行改名合并。

---

## 7. 主要阶段直接差值总览

本节只做直接算术整理，便于不重读日志即可看到 CPU/GPU 在若干已有聚合字段上的差值；不对差值原因作解释。

| 数据集 | Trace total Δ | Trace MSM Δ | t_edge prewarm Δ | t_edge eval Δ | t_FH prewarm Δ | t_FH eval Δ | t_N prewarm Δ | t_N eval Δ | Quotient commit Δ | Domain opening Δ | Domain witness Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | +0.034 s | -0.158 s | -0.101 s | -0.176 s | -0.108 s | +0.020 s | -0.109 s | +1.181 s | +0.010 s | -1.001 s | -0.505 s |
| Citeseer | +0.965 s | +0.218 s | -0.571 s | -0.064 s | -1.681 s | +0.123 s | -0.189 s | +0.100 s | -1.153 s | +0.954 s | +0.026 s |
| Pubmed | -5.204 s | -4.829 s | +1.067 s | +2.943 s | +0.135 s | +0.246 s | +0.499 s | +0.299 s | +4.813 s | -2.530 s | -0.929 s |
| PPI | +18.120 s | +6.912 s | +137.747 s | +0.997 s | -0.059 s | -0.023 s | +2.381 s | +0.519 s | +0.726 s | -8.079 s | -13.780 s |
| OGBN-Arxiv | +12.228 s | +0.653 s | +115.785 s | +7.235 s | +9.314 s | -1.943 s | +0.105 s | +0.532 s | -3.222 s | +36.504 s | -7.475 s |

---

## 8. 日志中保留但不在汇总表逐行复制的内容

为避免把本文件重新变成约 6.6 万行原始日志，以下内容不逐行复制，但原始日志仍作为最终真源保留：

- 单纯表示流程推进而没有独立数值字段的 `start` / `done` 行；
- Context 中 public/static polynomial、SRS 各域、权重列、attention 参数、bias 列等逐对象构造进度；
- `commit_batch` 的逐对象纯流程行不逐行复制；本文件已保留 start/materialized/MSM-profile 的调用计数、按 batch size 聚合、linear-derived/MSM-input 计数、profile 分布和每组 Top-10；
- `t_edge:coset_prewarm:derived:*` 的全部逐对象 start/done；本文件保留 prewarm 总计时、base/derived 数量、语义复用计数和 worker；
- Domain opening 的纯 start/done；本文件已保留七个域的 gather/total；
- Domain witness 的重复流程标记不逐行复制；本文件已保留 External Opening 中的实际日志层级、`domain_witness:done` 总时间以及七个域两次 group 的完整数值拆分；
- 逐条 external-opening KZG witness MSM 原始行；本文件已保留其相对 `domain_witness` group/done 的日志顺序区间计数，并对全部记录按完整 key 分组保存 count/sum/avg/min/max；
- export 的纯流程标记；进程墙钟与 `formal_total` 的差值已经保留，用于区分正式计时与整个进程时间。

本文件的目的仍然是“高信息密度的事实索引”，不是替代原始日志。任何需要核对单个 polynomial、单次 MSM、单个 commitment 对象或精确调用顺序的问题，仍应回到第 1 节索引的对应 full-run log。


---

## 9. 本批次 console log 中没有直接给出的聚合字段

以下项目在这十个 console full-run log 中没有可直接提取的统一数值字段，因此本文件没有自行推导或补造：

- `prove:external_opening:done` 只有完成标记，没有 `ms=...` 聚合总时间；
- `t_in`、`t_d_h`、`t_cat`、`t_C` quotient 只有阶段 start/done，没有与 `t_edge/t_FH/t_N` 相同的独立聚合计时；
- Context 内部 public/static/SRS 的绝大多数子阶段只有 start/done，没有统一毫秒聚合值；
- `warm:verify` 在 console log 中只有 start/done，Verifier 内部子阶段没有统一 breakdown；
- console log 没有 `optimization_flags`、`gpu_kernel_invocation_count`、GPU 利用率、显存峰值、CPU 利用率或进程峰值 RSS 等统一字段；如果这些字段存在于 `run_manifest.json` 或其他工件中，应由相应工件单独补充，而不能从当前十个 console log 猜测。
