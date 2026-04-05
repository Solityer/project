# GAT-ZKML 多层多头正式主线

## 项目简介

本项目当前目标是把真实 GAT 前向、多层多头 family 语义、统一图规范化、formal prove/verify 和四数据集基准结果收敛到同一条正式主线。

当前正式主线已经覆盖：

- 多层多头 GAT family：`L >= 2`、多 hidden layer、逐层 `hidden_profile / d_in_profile`、`K_out >= 1`
- hidden / output 语义分离：
  - hidden：`attention -> aggregate -> ELU -> concat`
  - output：`attention -> aggregate -> bias-add -> average(if K_out>1) -> logits`
- 统一图输入协议：
  `(T_H, I, src, dst, G_batch, node_ptr, edge_ptr, N_total, N, E)`
- 共享 schema 的 prover / verifier / serializer / manifest / benchmark 主线
- 四个正式数据集口径：
  - Cora：全图 transductive
  - Citeseer：全图 transductive
  - Pubmed：全图 transductive
  - PPI：全量 295 图的 inductive multi-graph batch

当前项目只保留一套正式主线：

- 单次运行结果：`run_manifest.json` 与 `benchmark.txt`
- 统一汇总脚本：`scripts/export_benchmark_table.py`
- 总表真源：
  - `runs/benchmarks/latest.json`
  - `runs/benchmarks/latest.csv`
  - `runs/benchmarks/summary.md`

## 项目结构

建议按下面顺序阅读：

1. 先看本 README：确认当前唯一正式主线、运行命令和最新结果。
2. 再看 `GAT-ZKML-多层多头.md`：查看最新语义约束与 formal 设计笔记。
3. 再看 `configs/`：确认四个正式数据集配置。
4. 再看 `src/model/`、`src/protocol/`、`src/data/`：理解真实 GAT 前向、formal 约束和统一图规范化。
5. 最后看 `scripts/`：训练、bundle 导出/导入、benchmark 汇总。

关键目录与文件：

- `src/model/`
  正式 GAT family 前向、bundle 加载、真实 output head 语义。
- `src/protocol/`
  prover / verifier / trace / quotient / shared schema。
- `src/data/`
  Cora / Citeseer / Pubmed / PPI 的统一数据加载与图规范化。
- `configs/`
  当前正式配置入口。
- `scripts/train_planetoid_gat.py`
  Cora / Citeseer / Pubmed 的正式训练入口。
- `scripts/train_ppi_gat.py`
  PPI 的正式训练入口。
- `scripts/export_torch_gat_bundle.py`
  将 torch checkpoint 导出为 family-schema bundle。
- `scripts/import_ppi_bundle.py`
  PPI 的唯一正式导入入口。
- `scripts/export_benchmark_table.py`
  统一汇总四数据集结果。
- `runs/benchmarks/`
  当前正式总表真源。

不属于正式主路径但仍保留的内容：

- `reference/`
  只作为参考实现或对照资料，不是当前 benchmark / formal / README 所依赖的主执行入口。
- `project` 同级目录中的 `GAT-for-PPI`
  只作为历史 PPI 训练参考来源；当前正式主线已经迁移到本项目内的 `scripts/train_ppi_gat.py` 与 `scripts/import_ppi_bundle.py`。

## 当前正式主线说明

### 1. bundle / manifest / formal / benchmark 的关系

- 训练脚本先产出真实模型参数 checkpoint。
- bundle 导出或导入脚本把参数转换成 family-schema bundle。
- `gatzk_run` 读取 bundle、数据和配置，执行真实 GAT 前向、build trace、prove、verify。
- 每次运行都会落盘：
  - `benchmark.txt`
  - `run_manifest.json`
- `scripts/export_benchmark_table.py` 统一读取各个 `run_manifest.json`，生成最终总表：
  - `runs/benchmarks/latest.json`
  - `runs/benchmarks/latest.csv`
  - `runs/benchmarks/summary.md`

### 2. 当前正式脚本

- Planetoid 训练：`scripts/train_planetoid_gat.py`
- PPI 训练：`scripts/train_ppi_gat.py`
- torch checkpoint 导出 family bundle：`scripts/export_torch_gat_bundle.py`
- PPI 正式导入：`scripts/import_ppi_bundle.py`
- benchmark 总表导出：`scripts/export_benchmark_table.py`

除以上路径外，不应再使用旧的人工抄表、旧 summary、旧 fallback 或旧实验入口来生成正式结果。

## 从零开始到最新结果的完整运行步骤

### 1. 构建

```bash
cmake -S . -B build
cmake --build build -j
```

### 2. 数据准备

Planetoid 数据与 PPI 原始数据都放在 `data/` 下。

PPI cache 会在运行时通过 `scripts/prepare_ppi.py` 自动生成；也可以手动执行：

```bash
./.venv/bin/python scripts/prepare_ppi.py --data-root data/ppi --output-root data/cache/ppi
```

### 3. Cora / Citeseer / Pubmed：训练、导出、formal prove/verify

训练：

```bash
./.venv/bin/python scripts/train_planetoid_gat.py --config configs/cora_full.cfg --output-dir runs/cora_train
./.venv/bin/python scripts/train_planetoid_gat.py --config configs/citeseer_full.cfg --output-dir runs/citeseer_train
./.venv/bin/python scripts/train_planetoid_gat.py --config configs/pubmed_train.cfg --output-dir runs/pubmed_train
```

导出 bundle：

```bash
./.venv/bin/python scripts/export_torch_gat_bundle.py --checkpoint runs/cora_train/best_model.pt --output-dir artifacts/checkpoints/cora_gat
./.venv/bin/python scripts/export_torch_gat_bundle.py --checkpoint runs/citeseer_train/best_model.pt --output-dir artifacts/checkpoints/citeseer_gat
./.venv/bin/python scripts/export_torch_gat_bundle.py --checkpoint runs/pubmed_train/best_model.pt --output-dir artifacts/checkpoints/pubmed_gat
```

正式 prove/verify：

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm
```

### 4. PPI：训练、导入、formal prove/verify

当前唯一正式主线已经不再依赖外部 TensorFlow 训练脚本；直接使用本项目内的真实 PPI 训练入口：

```bash
./.venv/bin/python scripts/train_ppi_gat.py --data-root data/ppi --output-dir runs/ppi_train --epochs 40 --batch-size 2 --hidden-profile "[1x8]" --k-out 1
```

将真实 torch checkpoint 安装到当前 formal 主线：

```bash
./.venv/bin/python scripts/import_ppi_bundle.py --torch-checkpoint runs/ppi_train/best_model.pt --output-dir artifacts/checkpoints/ppi_gat
```

运行全量 295 图的正式 prove/verify：

```bash
./build/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm
```

### 5. 导出四数据集总表

```bash
./.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/warm/run_manifest.json \
  --run citeseer=runs/citeseer_full/warm/run_manifest.json \
  --run pubmed=runs/pubmed_full/warm/run_manifest.json \
  --run ppi=runs/ppi_full_formal/warm/run_manifest.json \
  --output-dir runs/benchmarks
```

## 当前最新最快实验结果

主表口径：`warm`

| 数据集 | 正式口径 | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | ---: | ---: | ---: |
| Cora | 全图 transductive | 4687.953 | 1247.859 | 37283 |
| Citeseer | 全图 transductive | 10007.613 | 3818.313 | 37291 |
| Pubmed | 全图 transductive | 26191.926 | 4306.864 | 37288 |
| PPI | 全量 295 图 inductive multi-graph batch | 145050.530 | 62225.460 | 9048 |

以上结果必须与以下文件保持一致：

- `runs/benchmarks/latest.json`
- `runs/benchmarks/latest.csv`
- `runs/benchmarks/summary.md`

## 当前主要性能瓶颈

当前 warm 主路径的主要瓶颈如下：

- Cora / Citeseer / Pubmed：
  主要热点仍然是 `trace_generation_ms`，其中包含 hidden/output witness 构造、量化对象物化和部分 residual。
- Pubmed：
  `trace_generation_ms` 仍是最大热点，其次是 `commit_dynamic_ms` 与 `domain_opening_ms` 的剩余成本。
- PPI：
  当前全量正式路径的最大热点已经转移到 `domain_opening_ms` 与 `verify_misc_ms`。

已经完成的优化：

- 在 `src/protocol/trace.cpp` 对 warm 路径缓存 forward 结果与量化 witness，减少重复 field convert、重复量化和重复中间对象构造。
- 在 `src/protocol/prover.cpp` 对 proof evaluation backend 做共享缓存，warm 路径不再重复构造 packed backend。
- 统一 route2 组合保持为当前最优正式组合：
  `msm_fft_packed_kernel_layout_pairing`

下一轮优先继续处理：

- Pubmed 的 `verify_misc_ms`
- Pubmed 的 `domain_opening_ms`
- PPI 全量路径中的 `verify_misc_ms` 与 `domain_opening_ms`

## 当前限制与未完成项

- 当前真实 checkpoint-backed family formal 已经覆盖多层、多头、`K_out>1` 语义，但本地已有正式 benchmark 的真实 checkpoint 仍以 `L=2`、`K_out=1` 为主。
- PPI 已经闭合到真实 checkpoint-backed formal prove/verify，但当前正式模型结构仍采用 `L=2`、`hidden_profile=[1x8]`、`K_out=1` 这一条可证明主线。
- 若后续需要把更深 family 或 `K_out>1` 的真实 checkpoint 纳入正式 benchmark，继续沿用当前 family-schema bundle 与 formal 主线即可，不要再引入第二套导出或验证流程。
