# 项目简介

## GAT ZKML Formal Proving Pipeline

本仓库保留多层多头 GAT ZKML 的正式证明主线，目标是以单一 formal logic 支撑 full_cpu 与 full_gpu 两个完整执行后端。唯一协议语义真源是 `GAT-ZKML-多层多头.md`；本轮工作未修改该文件，也未改变 transcript、proof schema、serializer/parser 结构或 verifier equations。

唯一正式 benchmark 口径是 full-dataset warm verified。CPU 主表输出到 `runs/benchmarks`，GPU 主表输出到 `runs/benchmarks/cuda_formal`。

formal / benchmark 运行主线只保留正式 proving、verification、artifact 导出与 benchmark 汇总；上游训练仅作为参数来源，不构成本仓库中的第二条正式主线。

当前 project 的正式主线只负责：

- 读取五个正式配置对应的真实 checkpoint bundle 与图数据。
- 运行 full-dataset warm trace、prove、verify 与 manifest 导出。
- 维护 CPU 与 CUDA formal 两张正式 benchmark 主表。
- 保持 proof_size_bytes、hidden_profile、d_in_profile 的 formal 合约一致性。

## 当前正式架构

- 单一 formal semantics：所有证明含义、转录顺序和验证方程只在 `GAT-ZKML-多层多头.md` 中定义一次。
- full_cpu：`mcl/cpu`，采用多线程 FFT、打包 trace 布局和 CPU 侧 proving/verification 主路径。
- full_gpu：`mcl/cuda_hotspots`，在不改变 formal semantics 的前提下通过统一的 size-aware 路由策略接管 CUDA 可获益的 Fr 域热点，包括 barycentric 权重、shared weighted evaluation、shared tau-evaluation batch 与 lookup histogram。
- full_gpu 内部允许按 workload 大小在 GPU 与 CPU 之间选择具体实现，但这种分流只改变执行策略，不改变 transcript、proof shape、verifier equations 或 artifact contract。
- 两个后端共享同一 protocol、同一 artifact 契约、同一 proof_size_bytes / hidden_profile / d_in_profile 约束。

## 仓库结构

```text
src/           C++ 核心实现（algebra、crypto、data、model、protocol、util）
include/       公共头文件
tests/         正式回归测试
configs/       五个正式数据集配置
artifacts/     checkpoint bundles
data/          数据集与缓存
runs/          正式 benchmark 输出、per-run manifests、测试日志
scripts/       数据准备、bundle 导入、benchmark 导出脚本
cmake/         CMake 辅助模块
third_party/   mcl 依赖
```

## 构建依赖

- Linux
- CMake 3.18+
- 支持 C++17 的 GCC 或 Clang
- pthreads
- GPU 后端额外需要 CUDA 12+ 与可用 NVIDIA GPU

## CPU 构建命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## GPU 构建命令

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j
```

## 正式 full-dataset warm 复现命令

### CPU

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/warm \
  --run citeseer=runs/citeseer_full/warm \
  --run pubmed=runs/pubmed_full/warm \
  --run ppi=runs/ppi_full_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/warm \
  --output-dir runs/benchmarks
```

### GPU

```bash
./build_cuda/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/cuda_formal/warm \
  --run citeseer=runs/citeseer_full/cuda_formal/warm \
  --run pubmed=runs/pubmed_full/cuda_formal/warm \
  --run ppi=runs/ppi_full_formal/cuda_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/cuda_formal/warm \
  --output-dir runs/benchmarks/cuda_formal
```

## 正式 warm 结果

下表对应仓库当前保留的正式 warm artifacts：CPU 表保持当前已验收 official baseline；GPU 表是本轮针对 full_gpu 热点路径重新执行并导出的正式 warm 结果。

### CPU 正式结果表

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 282.317 | 1214.714 | 152.646 | 90755 |
| Citeseer | 480.219 | 1396.046 | 154.533 | 90763 |
| Pubmed | 3104.808 | 5849.602 | 175.702 | 90760 |
| PPI | 4046.398 | 7707.721 | 459.909 | 21360 |
| ogbn-arxiv | 19131.116 | 90263.318 | 38113.733 | 21361 |

### GPU 正式结果表

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 282.846 | 1206.953 | 152.430 | 90755 |
| Citeseer | 506.832 | 1444.127 | 154.263 | 90763 |
| Pubmed | 3127.943 | 5857.416 | 174.754 | 90760 |
| PPI | 3805.534 | 7388.570 | 491.354 | 21360 |
| ogbn-arxiv | 18817.280 | 88968.464 | 44262.121 | 21361 |

五个数据集在 full_cpu 与 full_gpu 下的 proof_size_bytes 保持一致，formal artifact 契约未改变。

## full_cpu 与 full_gpu 后端说明

- full_cpu 与 full_gpu 共享完全相同的 formal semantics。
- full_cpu 是默认完整执行路径，适用于所有支持的环境。
- full_gpu 是完整执行后端，不是辅助伪 GPU 路径；其职责是在不改变证明含义的前提下接管 CUDA 可获益的 Fr 域热点。
- full_gpu 现在通过单一 size-aware workload policy 统一决定 barycentric、shared weighted evaluation、shared tau-evaluation batch 与 lookup histogram 是否进入 CUDA，不再让不同文件各自维护互相矛盾的阈值。
- 小图或低并行度 workload 在 full_gpu 下仍可能回落到 CPU，这属于同一后端内部的实现策略，不构成第二条 formal logic。
- 仍以 CPU 为主的阶段主要是 context_build、trace_generation 与 pairing-backed verify_quotient；这些是当前端到端收益继续扩大的剩余瓶颈，而不是协议分叉点。

## Artifact 真源

- CPU 主表：`runs/benchmarks/latest.json`、`runs/benchmarks/latest.csv`、`runs/benchmarks/summary.md`
- GPU 主表：`runs/benchmarks/cuda_formal/latest.json`、`runs/benchmarks/cuda_formal/latest.csv`、`runs/benchmarks/cuda_formal/summary.md`
- CPU per-run manifest：`runs/<dataset>/warm/run_manifest.json`
- GPU per-run manifest：`runs/<dataset>/cuda_formal/warm/run_manifest.json`
- 正式测试日志：`runs/tests/cpu_gatzk_tests.log`、`runs/tests/cuda_gatzk_tests.log`

`summary.md` 由 `scripts/export_benchmark_table.py` 自动生成，现会明确展示 backend 列，并在文件顶部声明 backend 只是实现策略标签而不是第二协议语义。

## 训练与 ZKML 的职责边界

PPI 以及其他模型参数的上游真实参数来源是训练仓库与其导出的 bundle，而不是本仓库内部再维护一条训练主线。对于 PPI，上游训练仓库只作为上游真实参数来源；本仓库只接收已经导出的 bundle 或 checkpoint，并将其安装到 `artifacts/checkpoints/ppi_gat`。

formal / benchmark 运行主线与上游训练严格分离。训练时间、训练日志、训练超参数调优都不进入 formal benchmark 主表，也不计入当前仓库的正式 proving 与 verify 结果。

## 正式测试命令与当前状态

```bash
cmake --build build --target gatzk_tests -j
./build/gatzk_tests
cmake --build build_cuda --target gatzk_tests -j
./build_cuda/gatzk_tests
```

当前状态：CPU 全量回归通过，GPU 全量回归通过，README、latest.json、latest.csv、summary.md 与 per-run manifest 已同步。
本轮已重新执行 GPU 五组正式 warm benchmark，并重新导出 `runs/benchmarks/cuda_formal`、README GPU 表格与 `runs/tests/*.log`；CPU formal path 本轮未改动，因此 CPU 主表仍保持当前已验收 official baseline。若需要按当前环境重新刷新 CPU 主表，请重新执行上面的 CPU warm 命令与导出脚本。

## 可复现性说明

- 正式 benchmark 仅基于 `artifacts/checkpoints/` 中的真实 checkpoint bundles。
- PPI 上游训练仓库只作为上游真实参数来源；本仓库不保留独立训练主线，也不把训练时间或训练日志计入正式 proving / verify 结果。
- 运行前需准备对应数据集内容到 `data/`；辅助脚本保留在 `scripts/prepare_planetoid.py`、`scripts/prepare_ogbn_arxiv.py`、`scripts/prepare_ppi.py`。
- `GAT-ZKML-多层多头.md` 是唯一 formal note，本 README 只汇报实现级优化后的 retained state，不引入新的 protocol 语义。

## 保留限制

在当前硬件与实现边界下，full_gpu 已通过统一 size-aware 路由覆盖 barycentric 权重、shared weighted evaluation、shared tau-evaluation batch 与 lookup histogram 等 Fr 域热点，并且当前正式结果已在 PPI 与 ogbn-arxiv 上把 `commitment_time_ms` / `prove_time_ms` 压到 CPU 之下；但 large-graph verify 仍明显受 `verify_quotient_ms` 与 `verify_public_eval_ms` 的 CPU 主导阶段限制，因此 GPU 还没有在所有正式数据集上同时赢下三项 timing metric。
