# 项目简介

本仓库只保留当前 GAT ZKML formal proving 与 benchmark 的正式主线。README 是当前正式结果的唯一总入口。

当前唯一有效的正式 benchmark 口径是 full dataset + warm + verified。CPU 正式主表写入 runs/benchmarks；GPU formal 主表写入 runs/benchmarks/cuda_formal。GPU 路线只扩展 compute backend，不改变 transcript、proof schema 和验证语义。

当前 project 的正式主线只负责：

- 读取五个正式配置对应的真实 checkpoint bundle 与图数据。
- 运行 full-dataset warm trace、prove、verify 与 manifest 导出。
- 维护 CPU 与 CUDA formal 两张正式 benchmark 主表。
- 保持 proof_size_bytes、hidden_profile、d_in_profile 的 formal 合约一致性。

## 正式 warm 结果

正式数据集固定为 Cora、Citeseer、Pubmed、PPI、ogbn-arxiv。

### CPU 正式主表

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 279.067 | 1606.430 | 285.622 | 90755 |
| Citeseer | 475.126 | 1735.940 | 288.684 | 90763 |
| Pubmed | 6450.343 | 9870.719 | 312.142 | 90760 |
| PPI | 38662.963 | 69387.301 | 4148.822 | 21360 |
| ogbn-arxiv | 206909.775 | 498992.408 | 176094.810 | 21361 |

### GPU 正式主表

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 293.667 | 1626.915 | 287.523 | 90755 |
| Citeseer | 487.755 | 1795.860 | 289.004 | 90763 |
| Pubmed | 6228.826 | 9609.582 | 308.432 | 90760 |
| PPI | 38456.013 | 69110.563 | 4085.410 | 21360 |
| ogbn-arxiv | 207067.573 | 496885.861 | 175288.511 | 21361 |

### 正式结果真源

- CPU 主表：runs/benchmarks/latest.json、runs/benchmarks/latest.csv、runs/benchmarks/summary.md
- GPU 主表：runs/benchmarks/cuda_formal/latest.json、runs/benchmarks/cuda_formal/latest.csv、runs/benchmarks/cuda_formal/summary.md
- CPU per-run manifests：runs/cora_full/warm、runs/citeseer_full/warm、runs/pubmed_full/warm、runs/ppi_full_formal/warm、runs/ogbn_arxiv_full/warm
- GPU per-run manifests：runs/cora_full/cuda_formal/warm、runs/citeseer_full/cuda_formal/warm、runs/pubmed_full/cuda_formal/warm、runs/ppi_full_formal/cuda_formal/warm、runs/ogbn_arxiv_full/cuda_formal/warm
- 本轮正式回归证据：runs/tests/cpu_gatzk_tests.log、runs/tests/cpu_gatzk_tests.exit_code、runs/tests/cuda_gatzk_tests.log、runs/tests/cuda_gatzk_tests.exit_code

## 正式结论与测试状态

- 2026-04-10 的 CPU 全量回归使用 `./build/gatzk_tests`，退出码为 `0`。
- 2026-04-10 的 CUDA 全量回归使用 `./build_cuda/gatzk_tests`，退出码为 `0`。
- CPU 与 CUDA 的正式 benchmark、README 表格、`latest.json`、`latest.csv`、`summary.md` 与各自 `run_manifest.json` 现在保持一致。
- GPU formal 主线保持可用，但在五个 full-dataset warm 正式结果上，相对 CPU 的四项指标未形成广义性能收益。

## 训练与 ZKML 的职责边界

PPI 以及其他模型参数的上游真实参数来源是训练仓库与其导出的 bundle，而不是本仓库内部再维护一条训练主线。对于 PPI，GAT-for-PPI 仅承担上游真实参数来源的角色；本仓库只接收已经导出的 bundle 或 checkpoint，并将其安装到 artifacts/checkpoints/ppi_gat。

formal / benchmark 运行主线与上游训练严格分离。训练时间、训练日志、训练超参数调优都不进入 formal benchmark 主表，也不计入当前仓库的正式 proving 与 verify 结果。

## 正式复现命令

### 构建 CPU

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### 构建 CUDA formal 计算后端

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j
```

### 导入 PPI bundle

```bash
.venv/bin/python scripts/import_ppi_bundle.py --bundle-dir /path/to/ppi_bundle --output-dir artifacts/checkpoints/ppi_gat
```

### 运行 CPU full-dataset warm

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm
./build/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm
./build/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm
```

```bash
.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/warm \
  --run citeseer=runs/citeseer_full/warm \
  --run pubmed=runs/pubmed_full/warm \
  --run ppi=runs/ppi_full_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/warm \
  --output-dir runs/benchmarks
```

### 运行 GPU formal warm

```bash
./build_cuda/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --export-tag cuda_formal
```

```bash
.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/cuda_formal/warm \
  --run citeseer=runs/citeseer_full/cuda_formal/warm \
  --run pubmed=runs/pubmed_full/cuda_formal/warm \
  --run ppi=runs/ppi_full_formal/cuda_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/cuda_formal/warm \
  --output-dir runs/benchmarks/cuda_formal
```

### 最终回归

```bash
cmake --build build --target gatzk_tests -j
./build/gatzk_tests
cmake --build build_cuda --target gatzk_tests -j
./build_cuda/gatzk_tests
```
