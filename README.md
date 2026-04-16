# 项目简介

本仓库实现多层多头 `GAT` 的正式 `ZKML` proving 主线。当前仓库只保留一套已经收敛的 formal 语义，并用同一套语义支撑 `full_cpu` 与 `full_gpu` 两个完整执行后端。

仓库当前职责只有三类：

- 读取正式配置、真实图数据与真实 checkpoint bundle。
- 生成 trace、proof、verify 结果与正式产物。
- 维护 CPU / GPU 两张正式 benchmark 主表。

## formal 基准与硬约束

项目唯一 formal 基准为：

- `GAT-ZKML-多层多头.md`

该文件是唯一 formal 语义真源。当前正式主线中以下内容不能修改：

- 协议语义
- proof schema
- transcript 顺序
- serializer / parser 结构
- verifier equations
- `PublicMetadata`、`hidden_profile`、`d_in_profile`、`proof_size_bytes` 的正式契约

本 README 只说明当前正式实现，不重新定义协议。

## 当前正式架构

### one formal logic

- CPU 与 GPU 共享同一套 challenge、quotient、opening、verify 规则。
- `proof_block_order()`、`dynamic_commitment_labels()`、`domain_opening_labels()` 在全仓中只有一套正式命名与顺序。

### full_cpu

- 默认完整后端。
- 使用 `mcl/cpu` 执行全部 proving / verifying 热路径。

### full_gpu

- 完整正式后端，不是试验性分支。
- 只在不改变 formal 语义的前提下接管 `CUDA` 可获益的 `Fr` 域热点。
- 路由策略采用 `size-aware` 判定：是否提升到 `CUDA` 由实际域规模、批量形状与运行时内存条件共同决定，不靠第二套协议或手工切换证明结构。
- GPU 与 CPU 的差异只属于执行层，不属于第二套协议。

## 仓库结构

```text
include/gatzk/   正式头文件
src/             正式实现
tests/           formal 回归测试
configs/         五个正式数据集配置
scripts/         数据准备、bundle 导入、benchmark 导出脚本
artifacts/       checkpoint bundles
data/            数据集与缓存
runs/            benchmark 主表、per-run manifest、测试日志
docs/            补充技术说明
```

## 构建依赖

- Linux
- CMake 3.18+
- 支持 C++17 的 GCC 或 Clang
- pthreads
- `full_gpu` 额外需要 CUDA 12+ 与可用 NVIDIA GPU

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

## 正式复现命令

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

### benchmark 主表导出命令

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

## 正式结果表

正式展示面只保留四项指标：

- `commitment_time_ms`
- `prove_time_ms`
- `verify_time_ms`
- `proof_size_bytes`

### CPU 正式结果

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 262.605 | 1174.454 | 153.686 | 90755 |
| Citeseer | 445.371 | 1327.556 | 153.984 | 90763 |
| Pubmed | 2941.120 | 5564.076 | 175.585 | 90760 |
| PPI | 3510.734 | 7133.975 | 426.190 | 21360 |
| ogbn-arxiv | 17569.742 | 65329.058 | 398.058 | 21361 |

### GPU 正式结果

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 287.779 | 1237.908 | 152.254 | 90755 |
| Citeseer | 472.522 | 1354.367 | 153.856 | 90763 |
| Pubmed | 2998.093 | 5671.482 | 172.095 | 90760 |
| PPI | 3572.921 | 7136.396 | 450.268 | 21360 |
| ogbn-arxiv | 17541.068 | 65736.261 | 431.560 | 21361 |

## truth-source 与正式产物真源

### 正式展示真源

- CPU：`runs/benchmarks/latest.json`、`runs/benchmarks/latest.csv`、`runs/benchmarks/summary.md`
- GPU：`runs/benchmarks/cuda_formal/latest.json`、`runs/benchmarks/cuda_formal/latest.csv`、`runs/benchmarks/cuda_formal/summary.md`

### per-run 真源

- CPU：`runs/<dataset>/warm/run_manifest.json`
- GPU：`runs/<dataset>/cuda_formal/warm/run_manifest.json`

### formal 回归真源

- `tests/test_main.cpp`
- `runs/tests/cpu_gatzk_tests.log`
- `runs/tests/cuda_gatzk_tests.log`

`latest.csv` 与 `summary.md` 只用于正式展示，因此只公开四项指标。更细的阶段时间仍保留在 `run_manifest.json`、`latest.json` 与 `benchmark.txt`，供 formal 测试、产物校验和内部分析使用。

## 证明尺寸说明

### 统计口径

`proof_size_bytes` 的唯一真值来源是 `src/protocol/prover.cpp` 中的 `proof_size_bytes(const Proof&)`。该函数统计的是当前 formal proof 的传输口径，不是目录中文件大小，也不是 C++ 对象内存占用。

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

原因是这些对象可以由 verifier 根据固定 schema 与 transcript 重放得到，不需要重复传输。

### 为什么小图更大，大图更小

本仓库中 `proof_size_bytes` 主要由 proof 形状决定，而不是由节点数或边数直接线性决定。

代码与正式产物核查结论如下：

- `cora`、`citeseer`、`pubmed` 当前都使用 `hidden_profile=8x8`
- `PPI`、`ogbn-arxiv` 当前都使用 `hidden_profile=1x8`
- `proof_block_order.txt` 在两类 family 间相同，说明 proof block 结构没有分叉
- 但 `commitment_order.txt` 的动态承诺数量差异很大：
  - `cora_full/warm/commitment_order.txt`：`734`
  - `pubmed_full/warm/commitment_order.txt`：`734`
  - `ppi_full_formal/warm/commitment_order.txt`：`160`
  - `ogbn_arxiv_full/warm/commitment_order.txt`：`160`

这意味着当前 `8x8` family 会带入更多的：

- dynamic commitments
- 对应 domain opening 值
- 与这些列绑定的 external evaluations / witness scalars

因此：

- `cora` / `citeseer` / `pubmed` 的 `proof_size_bytes` 约为 `90755`
- `PPI` / `ogbn-arxiv` 的 `proof_size_bytes` 约为 `21360` / `21361`

这是由 family 形状更宽导致的合法 proof 负载差异，不是漏统计、漏序列化，也不是大图走了第二套协议。

## 当前状态说明

- formal note 未改动
- proof schema、transcript 顺序、verifier equations 未改动
- `full_cpu` 与 `full_gpu` 共享同一 formal 语义
- README、tests、benchmark 主表与 per-run manifest 以当前 accepted truth 为准
- 正式展示面只公开四项指标

## 可复现性说明

- 运行前需要准备 `data/` 中的正式数据集内容。
- 数据准备脚本保留在 `scripts/prepare_planetoid.py`、`scripts/prepare_ogbn_arxiv.py`、`scripts/prepare_ppi.py`。
- checkpoint bundle 只从 `artifacts/checkpoints/` 读取。
- PPI 的上游训练只作为参数来源，不构成本仓库中的第二主线。

## formal 测试

```bash
cmake --build build --target gatzk_tests -j
./build/gatzk_tests

cmake --build build_cuda --target gatzk_tests -j
./build_cuda/gatzk_tests
```
