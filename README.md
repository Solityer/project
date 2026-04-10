# 项目简介

本仓库保存当前 GAT ZKML formal proving 与 benchmark 的正式主线实现。它面向已经固化好的真实 checkpoint bundle，负责把全量数据集的 GAT 推理、trace 构造、证明生成、验证和结果导出收敛到一条可复现、可回归、可对比的主线。

当前 project 的正式主线只负责以下事项：

- 读取五个正式配置对应的 checkpoint bundle 与图数据。
- 生成 formal trace、proof、run manifest 与 benchmark 表。
- 对 CPU 正式主线和 CUDA hotspot 计算后端做同口径对比。
- 维护 runs/benchmarks/summary.md 与 runs/benchmarks/cuda_formal/summary.md 这两类正式结果表。

## 当前最新最快实验结果

当前唯一有效的正式 benchmark 口径是 full dataset + warm。官方 CPU 主表写入 runs/benchmarks/summary.md；正式 GPU 结果在完成 cuda_formal 导出后写入 runs/benchmarks/cuda_formal/summary.md。

正式数据集范围固定为：Cora、Citeseer、Pubmed、PPI、ogbn-arxiv。

阅读结果时请遵守以下约束：

- 只把 warm 主表当作正式结果。
- proof_size_bytes、hidden_profile、d_in_profile 必须按 formal 合约保持一致。
- GPU 路线只允许作为 compute backend 扩展，不改变 transcript、proof schema 和验证语义。

## 训练与 ZKML 的职责边界

PPI 以及其他模型参数的上游真实参数来源是训练仓库和训练产物，而不是本仓库内再做一条训练主线。对于 PPI，GAT-for-PPI 仅承担上游真实参数来源的角色；本仓库只接收已经导出的 bundle 或 checkpoint，然后通过 scripts/import_ppi_bundle.py 安装到 artifacts/checkpoints/ppi_gat。

formal / benchmark 运行主线与上游训练严格分离。训练时间、训练日志、训练超参数调优都不进入 formal benchmark 主表，也不计入当前仓库的正式 proving/verify 结果。

## 从零开始到最新结果的完整运行步骤

### 1. 准备 Python 解释器

如果仓库内还没有 .venv/bin/python，可以先创建一个最小虚拟环境：

```bash
python3 -m venv .venv
```

如果需要运行 PPI bundle 导入脚本，请保证虚拟环境里有 numpy。

### 2. 构建 CPU 正式主线

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 3. 构建 CUDA 正式计算后端

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j
```

### 4. 确认五个正式配置都有真实 checkpoint bundle

正式配置如下：

- configs/cora_full.cfg
- configs/citeseer_full.cfg
- configs/pubmed_full.cfg
- configs/ppi_batch_formal.cfg
- configs/ogbn_arxiv_full.cfg

它们都必须指向真实 checkpoint bundle。PPI 不在本仓库内重新训练；需要先从上游真实参数来源导入 bundle，例如：

```bash
.venv/bin/python scripts/import_ppi_bundle.py --bundle-dir /path/to/ppi_bundle --output-dir artifacts/checkpoints/ppi_gat
```

### 5. 运行官方 CPU full-dataset warm 主线

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm
./build/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm
```

然后导出官方 CPU 主表：

```bash
.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/warm \
  --run citeseer=runs/citeseer_full/warm \
  --run pubmed=runs/pubmed_full/warm \
  --run ppi=runs/ppi_full_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/warm \
  --output-dir runs/benchmarks
```

### 6. 运行正式 GPU full-dataset warm 主线

GPU 只作为 compute backend 扩展，命令必须显式指定 cuda_hotspots 与 cuda_formal：

```bash
./build_cuda/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
```

然后导出正式 GPU 主表：

```bash
.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/cuda_formal/warm \
  --run citeseer=runs/citeseer_full/cuda_formal/warm \
  --run pubmed=runs/pubmed_full/cuda_formal/warm \
  --run ppi=runs/ppi_full_formal/cuda_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/cuda_formal/warm \
  --output-dir runs/benchmarks/cuda_formal
```

### 7. 运行回归

```bash
cmake --build build --target gatzk_tests -j
./build/gatzk_tests
```

如果需要核验 CUDA 构建链，也可以运行：

```bash
cmake --build build_cuda --target gatzk_tests -j
./build_cuda/gatzk_tests
```

## 结果判定规则

- 只有 full dataset + warm + verified 的结果可以进入正式结论。
- CPU 主表和 GPU 主表必须分开导出，GPU 不能覆盖官方 CPU 表。
- 任何 smoke、局部样本、局部子图、warm_probe、cold、single 都不能当作阶段完成结论。
- 所有保留的优化都必须回到 Cora、Citeseer、Pubmed、PPI、ogbn-arxiv 五个正式 warm 结果上重新验证。
