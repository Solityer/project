# GAT-ZKML 的 ezkl 全量数据集对比实验

## 项目定位

本目录是 `GAT-ZKML` 主仓库的 `ezkl` 严格全量数据集对比实验基线。它只服务于一个目标：在相同 `GAT` 证明语句、相同全量数据集、相同图预处理、相同 checkpoint、相同量化配置与相同最终 logits 语义下，将主仓库证明系统替换为 `ezkl/Halo2` 路线进行对照。

本目录不是 `ezkl` 官方通用说明。所有源码、配置、复现脚本、结果表与审计产物均位于 `comparison/baselines/ezkl/`。主仓库其它目录只作为只读 formal 真源使用。

## formal 基准与对比约束

唯一 formal 真源为：

- `../../../GAT-ZKML-多层多头.md`

两条路线证明同一 compressed CSR native `GAT` 关系。允许差异仅为证明系统后端执行方式、公开实例分列、确定性查找表、资源复用和 GPU 后端切换等工程适配；不允许改变模型、数据、图预处理、量化、输出平均规则或最终 logits 公开语义。

## 当前路线

| 路线 | route_id | 后端 | 当前状态 |
| --- | --- | --- | --- |
| CPU 严格路线 | `ezkl_gat_native_compressed_csr_attention_full` | `CPU` | 五个全量数据集均为 `verified` |
| GPU 严格路线 | `ezkl_gat_native_compressed_csr_attention_full_gpu` | `Icicle CUDA` | 已接入并实际检查，当前本机后端不可用，未产生 `verified` 结果 |

CPU 与 GPU 路线属于同一 statement equivalence group。GPU 路线只有在 `backend=gpu`、`gpu_backend_required=true`、`gpu_backend_active=true` 且 `cpu_fallback_used=false` 时，才允许进入 verified 统计。

## 严格性说明

当前 CPU accepted results 满足：

- 五个全量数据集均为 `verified`。
- `strict_public_input_equivalent=true`。
- `full_gat_forward_in_ezkl_proof=true`。
- `final_logits_derivation_complete=true`。
- 最终 logits 是 row-major node-by-class public instances。
- 最终 logits 由 proof 内完整 native `GAT` forward 推导并绑定。
- 未使用 output-binding workaround、digest、folding、identity、precomputed logits、subset、shard、sample。
- 未使用 toy、smoke 或 debug 数据集。

GPU 路线沿用同一严格口径。若 GPU 构建、设备、后端加载、prove 或 verify 任一步未完成，结果不得标记为 `verified`。

## 构建方式

### CPU 构建

```bash
cd /home/pzh/project/comparison/baselines/ezkl

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo test native_gat --no-default-features --features ezkl,no-banner
RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner

cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j

python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py scripts/check_gpu_backend.py
```

### GPU feature 构建

```bash
cd /home/pzh/project/comparison/baselines/ezkl

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner,gpu-accelerated
```

### Icicle CUDA 后端构建尝试

本机为 `aarch64`，GPU 为 `NVIDIA GB10`，CUDA 工具链为 `13.0`。本轮已按 Icicle CMake 入口尝试：

```bash
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle -B build_gpu_backend/icicle_cuda_local -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_local -DBUILD_TESTS=OFF
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle -B build_gpu_backend/icicle_cuda_fetch -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DCUDA_BACKEND=main -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_fetch -DBUILD_TESTS=OFF
```

结果：

- `-DCUDA_BACKEND=local` 失败，因为本地 Icicle 源码树没有 `icicle/backend/cuda` 源码目录。
- `-DCUDA_BACKEND=main` 失败，因为 `git@github.com:ingonyama-zk/icicle-cuda-backend.git` 无访问权限，CMake 无法取得 CUDA backend 源码。
- 系统路径和用户目录中未找到其它 `aarch64` 的 `libicicle_backend_cuda*.so`。
- 当前可见的四个关键后端库均为 `x86_64` ELF，不能作为本机 `aarch64` GPU 后端。

本地构建目录和后端二进制目录被 `.gitignore` 忽略，不进入提交。

## GPU 后端自检

正式自检脚本为：

```bash
python3 scripts/check_gpu_backend.py
python3 scripts/check_gpu_backend.py --json > results/gpu_backend_check.json
```

自检内容包括：

- 主机架构。
- GPU 设备与数量。
- CUDA 工具链信息。
- `ICICLE_BACKEND_INSTALL_DIR` 或自动发现的后端目录。
- 四个关键 `Icicle CUDA` 后端库。
- 每个 `.so` 的 `file` 与 `ldd` 输出。
- 后端库架构与主机架构是否匹配。
- 使用带 GPU feature 的 `ezkl gen-srs --logrows 1` 触发最小 GPU backend 初始化。

当前自检结论见：

- `results/gpu_backend_check.json`

当前结论为 `usable=false`：GPU 设备可见，但 `Icicle CUDA` 后端库为 `x86_64`，主机为 `aarch64`，最小初始化失败，未激活 CUDA backend。

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

当前命令会先执行 GPU 后端预检。只有发现可加载且架构匹配的 `Icicle CUDA` 后端时，才会进入 prove/verify；否则立刻写入 `blocked`，不允许 CPU 回退。

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
```

运行结束后汇总并验证：

```bash
python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

## 正式结果

### CPU 四项指标

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 39418.157 | 20340.017 | 372.952 | 1280306 |
| Citeseer | 39775.686 | 20593.018 | 489.516 | 1347736 |
| Pubmed | 79890.208 | 43471.667 | 966.513 | 3973400 |
| PPI | 1289180 | 1921540 | 23240 | 461655303 |
| ogbn-arxiv | 1290660 | 1860150 | 24660 | 453849446 |

### GPU 四项指标

当前 GPU route 未 verified，不能填入 GPU proof 指标。五个数据集均在 GPU 后端预检阶段被阻断，原因是本机 `aarch64` 无法使用当前 `x86_64` 的 `Icicle CUDA` 后端库。

| 数据集 | status | failure_step | block_reason | gpu_backend_active | cpu_fallback_used | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| Citeseer | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| Pubmed | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| PPI | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |
| ogbn-arxiv | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` |  |  |  |  |

## 对比结果真源

CPU 结果真源：

- `results/gat_ezkl_full_results.json`
- `results/gat_ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
- `results/summary.md`

GPU 状态和未验证结果真源：

- `results/gat_ezkl_gpu_full_results.json`
- `results/gat_ezkl_gpu_full_results.csv`
- `results/main_vs_ezkl_gpu_full_comparison.json`
- `results/main_vs_ezkl_gpu_full_comparison.csv`
- `results/gpu_summary.md`
- `results/gpu_route_status.md`
- `results/gpu_backend_check.json`

## 机器复核真源

CPU 每个数据集保留以下最小审计 artifact：

- `runs/cora_full/run_manifest.json`
- `runs/citeseer_full/run_manifest.json`
- `runs/pubmed_full/run_manifest.json`
- `runs/ppi_full_formal/run_manifest.json`
- `runs/ogbn_arxiv_full/run_manifest.json`

GPU route 的最小审计 artifact 使用独立目录：

- `runs/cora_full/gpu/run_manifest.json`
- `runs/citeseer_full/gpu/run_manifest.json`
- `runs/pubmed_full/gpu/run_manifest.json`
- `runs/ppi_full_formal/gpu/run_manifest.json`
- `runs/ogbn_arxiv_full/gpu/run_manifest.json`

GPU manifest 必须记录 `backend=gpu`、`gpu_backend_required`、`gpu_backend_active`、`cpu_fallback_used`、`gpu_backend_name`、`gpu_backend_library_path`、`gpu_backend_library_arch`、`gpu_device_name`、`gpu_device_count`、完整命令、证明验证状态与四项指标。未验证时不得填入成功状态。

## 仓库清理状态

最终仓库只保留源码、配置、脚本、README、正式结果表和最小审计 artifact。proof、key、SRS、witness、compiled circuit、二进制 formal table、构建目录、缓存目录和临时文件不进入最终仓库表面。
