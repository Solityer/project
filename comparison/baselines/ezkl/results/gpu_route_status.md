# GPU route 状态

## 结论

GPU strict route 已接入五个全量数据集，并保持与 CPU route 相同的 compressed CSR native `GAT` statement：

```text
ezkl_gat_native_compressed_csr_attention_full_gpu
```

当前本机未产生 GPU `verified` 结果。所有 GPU 行均保持 `blocked`，未进入真实 GPU prove/verify，也没有写入 GPU 四项成功指标。CPU strict accepted results 仍为五个全量数据集 `verified`。

这些 GPU 行均未标记为 verified。

## 当前机器与 GPU

| 项 | 值 |
| --- | --- |
| 主机架构 | `aarch64` |
| GPU | `NVIDIA GB10` |
| GPU 架构 | `Blackwell` |
| GPU 数量 | `1` |
| 驱动版本 | `580.142` |
| CUDA 工具链 | `13.0, V13.0.88` |

机器可读自检结果见：

```text
results/gpu_backend_check.json
```

## GPU feature 构建状态

GPU feature 构建命令为：

```bash
RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner,gpu-accelerated
```

该构建只证明 Rust GPU feature 可编译，不等价于 CUDA backend 可加载。

## 后端来源审计

已确认：

- `gpu-accelerated` 依赖 `halo2_proofs/gpu-accelerated` 和 `icicle-runtime`。
- `Cargo.lock` 固定 `ingonyama-zk/icicle` 的 `012e00694f4cf399fe7a42d9cfbfa6cd7a60f876`。
- Icicle 前端源码存在于 `.cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle`。
- 当前 `halo2` checkout 中的 `icicle/backend/cuda` 只有四个已构建 `.so`，没有 CUDA backend 源码。
- 当前四个 `.so` 均为 `x86_64` ELF。

详细审计见：

```text
results/gpu_backend_source_audit.md
```

## 本机构建尝试

本轮实际执行了以下尝试：

```bash
git clone --depth 1 --branch main https://github.com/ingonyama-zk/icicle-cuda-backend.git .vendor_local/icicle-cuda-backend-main
git clone --depth 1 --branch main git@github.com:ingonyama-zk/icicle-cuda-backend.git .vendor_local/icicle-cuda-backend-main-ssh
```

核心错误：

```text
fatal: could not read Username for 'https://github.com': No such device or address
ERROR: Repository not found.
fatal: Could not read from remote repository.
```

本地 CMake 入口：

```bash
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle -B build_gpu_backend/icicle_cuda_local -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_local -DBUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
```

核心错误：

```text
add_subdirectory given source "backend/cuda" which is not an existing directory.
```

远程 CMake 入口：

```bash
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle -B build_gpu_backend/icicle_cuda_fetch_main -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DCUDA_BACKEND=main -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_fetch_main -DBUILD_TESTS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
```

核心错误：

```text
Failed to clone repository: 'git@github.com:ingonyama-zk/icicle-cuda-backend.git'
```

详细构建记录见：

```text
results/gpu_backend_build_attempt.md
```

## 系统后端搜索

搜索命令：

```bash
find /usr /usr/local /opt /home/pzh -type f -name "libicicle_backend_cuda*.so" 2>/dev/null | head -200
```

只找到 Cargo checkout 中的 `x86_64` 后端库，未找到可加载的 `aarch64` Icicle CUDA 后端。

## GPU 后端自检结果

自检命令：

```bash
python3 scripts/check_gpu_backend.py --json > results/gpu_backend_check.json
```

当前结论：

```text
host_arch=aarch64
gpu_available=true
gpu_device_count=1
backend_arch_match=false
library_arches=["x86_64"]
cuda_backend_initialization_status=failed
usable=false
cpu_fallback_used=false
```

因此 GPU route 不允许进入 full dataset prove/verify，也不得标记为 `verified`。

## GPU 数据集状态

| 数据集 | GPU 状态 | 失败步骤 | block_reason | gpu_backend_active | cpu_fallback_used | 是否 verified |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| Citeseer | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| Pubmed | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| PPI | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| ogbn-arxiv | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |

## 远端运行包

本机无法构建或加载 `aarch64` 后端时，使用以下脚本生成 `x86_64` CUDA 主机运行包：

```bash
python3 scripts/package_gpu_experiment.py
```

生成：

```text
gpu_experiment_bundle/
gpu_experiment_bundle.tar.gz
```

远端运行与结果回填步骤见：

```text
results/gpu_remote_runbook.md
```

## CPU accepted results 保持情况

CPU strict route：

```text
ezkl_gat_native_compressed_csr_attention_full
```

CPU 五个全量数据集仍满足：

```text
status=verified
strict_public_input_equivalent=true
full_gat_forward_in_ezkl_proof=true
final_logits_derivation_complete=true
no_precomputed_logits_bypass=true
```

验证命令当前输出：

```text
final_result_validation=OK cpu_verified=5/5 gpu_status=not_verified_0/5
```
