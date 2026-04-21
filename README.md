# 项目简介

本仓库给出一个面向多层多头 `GAT` 的正式 `ZKML` 证明系统实现。项目当前仅保留一套已经收敛的 formal 语义，并用同一套证明契约同时支撑 `full_cpu` 与 `full_gpu` 两条完整执行路径。仓库目标不是继续探索新的协议变体，而是在固定 formal 语义下，提供可复核、可复现、可验证、可导出的正式实现。

从学术定位上看，本仓库承担三类工作：

- 将多层多头 `GAT` 的前向过程转写为有限域上的正式可验证对象。
- 在固定 `proof schema`、固定 `transcript` 顺序和固定验证方方程下生成并验证证明。
- 给出与正式实现一一对应的 artifact、benchmark 主表和回归测试。

## 研究定位与主要特点

### 研究问题

本项目关注的问题是：如何在不改变 `GAT` 多层多头结构表达能力的前提下，把图神经网络前向过程、查表约束、路由关系、状态机关系、商多项式关系与 `KZG` 打开关系整理成一条完整、可执行、可验证的正式主线。

### 主要特点

- 保持一套 formal 语义，不引入第二套协议对象。
- 中央处理器路径与图形处理器路径共享同一套证明对象、挑战值命名、打开关系和验证逻辑。
- 正式展示面只保留最小必要结果，内部复核面保留完整清单，二者职责明确分离。
- 证明尺寸、结果主表、README、测试与 per-run 清单保持一致，便于论文写作与审稿复核。

## formal 基准与硬约束

项目唯一 formal 基准为：

- `GAT-ZKML-多层多头.md`

该文件是唯一 formal 语义真源。当前正式主线中以下内容不能修改：

- 协议语义
- `proof schema`
- `transcript` 顺序
- `serializer / parser` 结构
- `verifier equations`
- `PublicMetadata`、`hidden_profile`、`d_in_profile`、`proof_size_bytes` 的正式契约

本 README 只说明当前正式实现，不重新定义协议。

## 当前正式架构

### one formal logic

- 中央处理器路径与图形处理器路径共享同一套 challenge、quotient、opening、verify 规则。
- `proof_block_order()`、`dynamic_commitment_labels()`、`domain_opening_labels()` 在全仓中只有一套正式命名与顺序。
- `canonical_public_metadata(...)` 是公开元数据的统一真源，证明方与验证方共同复用。

### full_cpu

- 默认完整后端。
- 使用 `mcl/cpu` 执行全部 proving / verifying 热路径。
- 与正式 benchmark 主表中的中央处理器结果一一对应。

### full_gpu

- 完整正式后端，不是试验性分支。
- 只在不改变 formal 语义的前提下接管 `CUDA` 可获益的 `Fr` 域热点。
- 路由策略采用 `size-aware` 判定：是否提升到 `CUDA` 由域规模、批量形状与运行时条件共同决定，不靠第二套协议或手工切换证明结构。
- 图形处理器路径与中央处理器路径的差异只属于执行层，不属于第二套协议。

## 方法概述

### 从模型到证明对象

项目首先读取正式配置、真实图数据和真实参数包，将多层多头 `GAT` 的前向对象整理成有限域上的正式对象。这个过程包括：

- 建立完整数据集上的标准化图输入。
- 对齐层数、隐藏层形状、输入维度链和输出头数量。
- 构造公开多项式、静态查表对象与静态承诺。
- 物化运行轨迹中的动态列、矩阵、查表对象、路由对象、`PSQ` 对象和状态机对象。

### 从证明对象到最终证明

在证明阶段，系统按固定顺序执行以下步骤：

- 对动态对象生成动态承诺。
- 按固定吸收顺序重放 challenge。
- 计算各工作域上的 quotient 值与 quotient 承诺。
- 依据 `domain_opening_labels(...)` 构造各工作域 opening bundle。
- 组装最终 proof，并由验证方按同一 formal 语义重放检查。

### 与当前代码实现的对应关系

当前仓库中的真实调用链为：

- `src/main.cpp`：正式命令行入口，串联整条运行主线。
- `build_context(...)`：构造静态上下文。
- `build_trace(...)`：物化运行轨迹。
- `prove(...)`：生成承诺、商多项式、opening bundle 与最终 proof。
- `verify(...)`：执行验证方重算与检查。
- `proof_size_bytes(...)`：统计 proof 传输尺寸。
- `export_run_artifacts(...)`：导出单次运行 artifact。
- `scripts/export_benchmark_table.py`：汇总正式 benchmark 主表。

## 仓库结构

```text
include/gatzk/   正式头文件
src/             正式实现
tests/           formal 回归测试
configs/         五个正式数据集配置
scripts/         数据准备、参数包导入、benchmark 导出脚本
artifacts/       checkpoint bundles
data/            数据集与缓存
runs/            benchmark 主表与 per-run 清单
docs/            补充技术说明
```

## 环境依赖

- Linux
- CMake 3.18+
- 支持 C++17 的 GCC 或 Clang
- `pthreads`
- `full_gpu` 额外需要 `CUDA 12+` 与可用图形处理器

## 构建方式

### CPU 构建命令

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

### GPU 构建命令

```bash
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j
```

## 正式复现命令

### 执行层开关说明

- `--fast-msm on`：开启批量 `MSM` 路线，主要作用于承诺与见证构造阶段。
- `--parallel-fft on`：开启域上并行计算路线，用于较大的域上代数热点。
- `--fft-backend-upgrade on`：启用打包域评估后端。
- `--fft-kernel-upgrade on`：在打包域评估后端之上启用更强的旋转点核心路线。
- `--trace-layout-upgrade on`：优化轨迹物化布局与相关求值路径。
- `--fast-verify-pairing on`：优化验证方 pairing 检查。
- `--compute-backend cuda_hotspots`：允许满足条件的热点落到图形处理器路径。
- `--export-tag cuda_formal`：把图形处理器结果导出到独立目录，不覆盖中央处理器主表。

### CPU 五个数据集

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
```

### GPU 五个数据集

```bash
./build_cuda/gatzk_run --config configs/cora_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ppi_batch_formal.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ogbn_arxiv_full.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
```

## 正式控制台输出

正式命令执行完成后，控制台只输出一行最小结果，不再打印过程性阶段日志、profile 字段或 route 细节。

CPU 示例：

```text
dataset=cora config=configs/cora_full.cfg backend=cpu status=VERIFY_OK commitment_time_ms=283.786 prove_time_ms=1207.818 verify_time_ms=153.379 proof_size_bytes=90755
```

GPU 示例：

```text
dataset=ppi config=configs/ppi_batch_formal.cfg backend=gpu status=VERIFY_OK commitment_time_ms=3572.921 prove_time_ms=7136.396 verify_time_ms=450.268 proof_size_bytes=21360
```

## 正式结果表

正式展示面只保留四项指标：

- `commitment_time_ms`
- `prove_time_ms`
- `verify_time_ms`
- `proof_size_bytes`

### CPU 正式结果

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 1207.818 | 153.379 | 90755 |
| Citeseer | 453.773 | 1418.744 | 153.837 | 90763 |
| Pubmed | 2962.933 | 5604.074 | 175.454 | 90760 |
| PPI | 3563.973 | 7111.447 | 424.710 | 21360 |
| ogbn-arxiv | 18071.562 | 66540.600 | 417.380 | 21361 |

### GPU 正式结果

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 287.779 | 1237.908 | 152.254 | 90755 |
| Citeseer | 472.522 | 1354.367 | 153.856 | 90763 |
| Pubmed | 2998.093 | 5671.482 | 172.095 | 90760 |
| PPI | 3572.921 | 7136.396 | 450.268 | 21360 |
| ogbn-arxiv | 17541.068 | 65736.261 | 431.560 | 21361 |

## benchmark 主表导出命令

```bash
.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/warm \
  --run citeseer=runs/citeseer_full/warm \
  --run pubmed=runs/pubmed_full/warm \
  --run ppi=runs/ppi_full_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/warm \
  --output-dir runs/benchmarks

.venv/bin/python scripts/export_benchmark_table.py \
  --run cora=runs/cora_full/cuda_formal/warm \
  --run citeseer=runs/citeseer_full/cuda_formal/warm \
  --run pubmed=runs/pubmed_full/cuda_formal/warm \
  --run ppi=runs/ppi_full_formal/cuda_formal/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_full/cuda_formal/warm \
  --output-dir runs/benchmarks/cuda_formal
```

## truth-source 与正式产物真源

### 正式展示产物

- CPU：`runs/benchmarks/latest.csv`、`runs/benchmarks/summary.md`
- GPU：`runs/benchmarks/cuda_formal/latest.csv`、`runs/benchmarks/cuda_formal/summary.md`

### 机器复核真源

- CPU：`runs/benchmarks/latest.json`、`runs/<dataset>/warm/run_manifest.json`
- GPU：`runs/benchmarks/cuda_formal/latest.json`、`runs/<dataset>/cuda_formal/warm/run_manifest.json`

### formal 回归真源

- `tests/test_main.cpp`
- `./build/gatzk_tests`
- `./build_cuda/gatzk_tests`

`latest.csv` 与 `summary.md` 只用于正式展示，因此只公开四项指标。更细的阶段时间只保留在 `run_manifest.json` 与 `latest.json` 中；`benchmark.txt` 只保留每次运行的最小正式摘要。

## 训练与 ZKML 的职责边界

- 上游真实参数来源：`PPI` 的训练结果只作为 checkpoint bundle 参数来源，不构成当前仓库的第二条正式主线。
- `formal / benchmark 运行主线` 只负责读取正式配置、真实数据、真实 checkpoint bundle，随后执行 trace、prove、verify 与 artifact 导出。
- 训练时间、训练日志、训练超参数调优都不进入 formal benchmark 主表。
- 当前 project 的正式主线只负责证明与校验，不负责在 benchmark 表面复述上游训练过程。

## 证明尺寸说明

### 统计口径

`proof_size_bytes` 的唯一真值来源是 `src/protocol/prover.cpp` 中的 `proof_size_bytes(const Proof&)`。该函数统计的是当前 formal proof 的传输口径，不是目录中文件大小，也不是 `C++` 对象内存占用。

当前代码会计入：

- `PublicMetadata` 中需要随 proof 发送的字符串
- `block_order`
- dynamic commitments
- quotient commitments
- domain opening 中各列在 opening 点上的值与 witness
- external evaluations
- witness scalars
- external witness

当前代码不会计入：

- `proof.challenges`
- dynamic / quotient / opening 的标签名
- domain opening 的点值

原因是这些对象可以由验证方根据固定 `schema` 与 `transcript` 重放得到，不需要重复传输。因此 `proof_size_bytes` 统计的是“真正需要随 proof 发送的内容”，不是实现细节对象的总占用。

### 为什么小图更大，大图更小

本仓库中 `proof_size_bytes` 主要由 proof 形状决定，而不是由节点数或边数直接线性决定。

代码与正式产物核查结论如下：

- `cora`、`citeseer`、`pubmed` 当前都使用 `hidden_profile=8x8`
- `PPI`、`ogbn-arxiv` 当前都使用 `hidden_profile=1x8`
- `proof_block_order.txt` 在两类 family 间相同，说明 proof block 结构没有分叉
- `commitment_order.txt` 的动态承诺数量差异显著：
  - `cora_full/warm/commitment_order.txt`：`734`
  - `pubmed_full/warm/commitment_order.txt`：`734`
  - `ppi_full_formal/warm/commitment_order.txt`：`160`
  - `ogbn_arxiv_full/warm/commitment_order.txt`：`160`

这意味着当前 `8x8` family 会引入更多的：

- dynamic commitments
- 对应 domain opening 值
- 与这些列绑定的 external evaluations / witness scalars

因此：

- `cora` / `citeseer` / `pubmed` 的 `proof_size_bytes` 约为 `90755`
- `PPI` / `ogbn-arxiv` 的 `proof_size_bytes` 约为 `21360` / `21361`

因此小图 family proof 更大、大图 family proof 更小，根因是 proof shape 不同，而不是图规模单独决定传输尺寸。这是合法的 formal proof 负载差异，不是漏统计、漏序列化，也不是大图走了第二套协议。

## 可复现性与审稿复核建议

为保证论文写作与审稿复核的一致性，建议按以下顺序使用仓库：

1. 先阅读 `GAT-ZKML-多层多头.md`，明确唯一 formal 语义真源。
2. 再阅读 `项目整体逻辑与方法流程.md`，建立对对象流和调用链的整体理解。
3. 使用 `configs/*.cfg` 与 `artifacts/checkpoints/*` 进行正式复现。
4. 通过 `runs/<dataset>/.../run_manifest.json` 复核单次运行细节。
5. 通过 `runs/benchmarks/` 与 `runs/benchmarks/cuda_formal/` 复核正式结果表。

## 当前状态说明

- formal note 未改动。
- `proof schema`、`transcript` 顺序、`verifier equations` 未改动。
- `full_cpu` 与 `full_gpu` 共享同一 formal 语义。
- README、tests、benchmark 主表与 per-run 清单以当前 accepted truth 为准。
- 正式展示面只公开四项指标。

## formal 测试

```bash
cmake --build build --target gatzk_tests -j
./build/gatzk_tests

cmake --build build_cuda --target gatzk_tests -j
./build_cuda/gatzk_tests
```
