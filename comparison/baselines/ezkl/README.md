# GAT-ZKML 的 ezkl 全量数据集对比实验

## 项目定位

本目录是 `GAT-ZKML` 主仓库的 `ezkl` 严格全量数据集对比实验基线。它只服务于一个目标：在相同 `GAT` 证明语句、相同全量数据集、相同图预处理、相同 checkpoint、相同量化配置与相同最终 logits 语义下，将主仓库证明系统替换为 `ezkl/Halo2` 路线进行对照。

所有源码、配置、复现脚本、结果表与审计产物均位于 `comparison/baselines/ezkl/`。主仓库其它目录只作为只读 formal 真源使用。

## formal 基准与约束

唯一 formal 真源为：

```text
../../../GAT-ZKML-多层多头.md
```

CPU 与 GPU 路线证明同一个 compressed CSR native `GAT` statement。GPU 路线只允许改变 Halo2/Icicle 执行后端和必要工程环境，不允许改变数据集、图预处理、checkpoint、`model_arch_id`、`model_param_id`、`quant_cfg_id`、`hidden_profile`、`d_in_profile`、`K_out`、`output_average_rule`、最终 logits 语义或公开输出策略。

禁止使用 output-binding workaround、digest、folding、identity、precomputed logits bypass、subset、shard、sample、partial dataset、toy dataset 或 CPU fallback 冒充 GPU。

## 当前路线

| 路线 | route_id | 后端 | 当前状态 |
| --- | --- | --- | --- |
| CPU 严格路线 | `ezkl_gat_native_compressed_csr_attention_full` | `CPU` | 五个全量数据集均为 `verified` |
| GPU 严格路线 | `ezkl_gat_native_compressed_csr_attention_full_gpu` | `Icicle CUDA` | 已接入严格预检，本机 `aarch64` 后端不可用，未产生 `verified` 结果 |

GPU 路线只有在 `backend=gpu`、`gpu_backend_required=true`、`gpu_backend_active=true`、`cpu_fallback_used=false` 且真实 verify 通过时，才允许进入 verified 统计。

## 构建命令

```bash
cd /home/pzh/project/comparison/baselines/ezkl

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo test native_gat --no-default-features --features ezkl,no-banner
RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner
RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner,gpu-accelerated

cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j

python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py scripts/check_gpu_backend.py scripts/package_gpu_experiment.py
```

## GPU 后端自检

```bash
python3 scripts/check_gpu_backend.py
python3 scripts/check_gpu_backend.py --json > results/gpu_backend_check.json
```

自检会记录主机架构、GPU 设备、CUDA 工具链、`ICICLE_BACKEND_INSTALL_DIR`、四个关键 Icicle CUDA 后端库、库架构、`ldd` 输出和最小 GPU backend 初始化。当前本机结论为：

```text
host_arch=aarch64
gpu_device_count=1
backend_arch_match=false
library_arches=["x86_64"]
usable=false
cpu_fallback_used=false
```

## 本机 GPU 后端构建尝试

本机为 `aarch64`，GPU 为 `NVIDIA GB10`，CUDA 工具链为 `13.0`。本轮重新执行了源码审计、远程源码获取、本地 CMake 构建入口和系统库搜索。

关键结论：

- `gpu-accelerated` 依赖 `zkonduit/halo2` 的 `ac/conditional-compilation-icicle2` 分支和 `ingonyama-zk/icicle` 的 `emir/gate_eval_2` 分支。
- 当前 Icicle 前端源码存在，但没有 `icicle/backend/cuda` 源码目录。
- Icicle 文档说明 CUDA backend 是闭源授权组件。
- `git clone https://github.com/ingonyama-zk/icicle-cuda-backend.git` 与 `git clone git@github.com:ingonyama-zk/icicle-cuda-backend.git` 均失败。
- `cmake -DCUDA_BACKEND=local` 失败于 `backend/cuda` 不存在。
- `cmake -DCUDA_BACKEND=main` 失败于无法拉取 `git@github.com:ingonyama-zk/icicle-cuda-backend.git`。
- `/usr`、`/usr/local`、`/opt`、`/home/pzh` 下未找到 `aarch64` 的 `libicicle_backend_cuda*.so`。

详细证据见：

```text
results/gpu_backend_source_audit.md
results/gpu_backend_build_attempt.md
```

## 远端 x86-64 CUDA 运行包

由于本机无法获得可加载的 `aarch64` Icicle CUDA backend，本目录提供可迁移到 `x86_64` CUDA 主机的运行包生成脚本：

```bash
python3 scripts/package_gpu_experiment.py
```

生成：

```text
gpu_experiment_bundle/
gpu_experiment_bundle.tar.gz
```

运行包包含 `Cargo.toml`、`Cargo.lock`、`src/`、`cpp/gat_ezkl_adapter/`、`configs/`、`scripts/`、`README.md` 和必要结果审计文档；不包含 `target/`、`build_gat_adapter/`、proof、key、witness、compiled circuit 或后端二进制。

远端执行和回填流程见：

```text
results/gpu_remote_runbook.md
```

## 正式复现命令

### CPU 五个数据集

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --backend cpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --backend cpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --backend cpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full --backend cpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --backend cpu --clean --run
```

### GPU 五个数据集

GPU 命令会先执行后端二进制、架构、设备与初始化预检。只有 `usable=true` 时才进入证明和验证；否则写入 `blocked`，不允许 CPU fallback。

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
```

汇总与验证：

```bash
python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

## CPU 四项指标

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 39418.157 | 20340.017 | 372.952 | 1280306 |
| Citeseer | 39775.686 | 20593.018 | 489.516 | 1347736 |
| Pubmed | 79890.208 | 43471.667 | 966.513 | 3973400 |
| PPI | 1289180 | 1921540 | 23240 | 461655303 |
| ogbn-arxiv | 1290660 | 1860150 | 24660 | 453849446 |

## GPU 四项指标

当前 GPU route 未 verified，不能填入 GPU proof 指标。五个全量数据集均在 GPU 后端预检阶段阻断。

| 数据集 | status | failure_step | block_reason | gpu_backend_active | cpu_fallback_used | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| Citeseer | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| Pubmed | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| PPI | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| ogbn-arxiv | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |

## 结果文件

CPU 结果：

```text
results/gat_ezkl_full_results.json
results/gat_ezkl_full_results.csv
results/main_vs_ezkl_full_comparison.json
results/main_vs_ezkl_full_comparison.csv
results/summary.md
```

GPU 状态：

```text
results/gat_ezkl_gpu_full_results.json
results/gat_ezkl_gpu_full_results.csv
results/main_vs_ezkl_gpu_full_comparison.json
results/main_vs_ezkl_gpu_full_comparison.csv
results/gpu_summary.md
results/gpu_route_status.md
results/gpu_backend_check.json
```

GPU manifest 路径：

```text
runs/cora_full/gpu/run_manifest.json
runs/citeseer_full/gpu/run_manifest.json
runs/pubmed_full/gpu/run_manifest.json
runs/ppi_full_formal/gpu/run_manifest.json
runs/ogbn_arxiv_full/gpu/run_manifest.json
```

当前验证命令输出：

```text
final_result_validation=OK cpu_verified=5/5 gpu_status=not_verified_0/5
```
