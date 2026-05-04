# GPU route 状态

## 结论

GPU strict route 已接入并对五个全量数据集执行正式后端预检：

- `ezkl_gat_native_compressed_csr_attention_full_gpu`

当前 GPU route 未产生 `verified` 结果，所有 GPU 行均未标记为 verified。CPU strict accepted results 仍为五个全量数据集 `verified`。

## 当前机器与 GPU

| 项 | 值 |
| --- | --- |
| 主机架构 | `aarch64` |
| GPU | `NVIDIA GB10` |
| GPU 数量 | `1` |
| 驱动版本 | `580.142` |
| CUDA 工具链 | `13.0` |

机器可读自检结果见：

- `results/gpu_backend_check.json`

## GPU feature 构建状态

GPU feature 构建已通过：

```bash
RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner,gpu-accelerated
```

构建结果：

```text
Finished `release` profile [optimized] target(s) in 3m 02s
```

## 后端构建与接入尝试

### 路线 A：本机构建 aarch64 CUDA 后端

已检查本地 Icicle 源码：

- `.cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle`
- `.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda`

检查结果：

- Icicle 前端源码存在。
- `halo2` checkout 中的 `icicle/backend/cuda` 只有四个已构建 `.so`，没有 CUDA backend 源码。
- Icicle CMake 支持 `-DCUDA_BACKEND=local` 或 `-DCUDA_BACKEND=<branch|commit>`，但 CUDA backend 源码不在当前本地树中。

执行过的本地构建入口：

```bash
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle -B build_gpu_backend/icicle_cuda_local -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DCUDA_BACKEND=local -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_local -DBUILD_TESTS=OFF
```

失败证据：

```text
add_subdirectory given source "backend/cuda" which is not an existing directory.
```

执行过的远程 CUDA backend 拉取入口：

```bash
cmake -S .cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle -B build_gpu_backend/icicle_cuda_fetch -DCMAKE_BUILD_TYPE=Release -DCURVE=bn254 -DCUDA_BACKEND=main -DCMAKE_INSTALL_PREFIX=/home/pzh/project/comparison/baselines/ezkl/.icicle_backend_local/cuda_fetch -DBUILD_TESTS=OFF
```

失败证据：

```text
Failed to clone repository: 'git@github.com:ingonyama-zk/icicle-cuda-backend.git'
ERROR: Repository not found.
fatal: Could not read from remote repository.
```

因此，本机具备 CUDA 13.0 编译工具链，但缺少可访问的 CUDA backend 源码，无法从源码生成 `aarch64` 的 Icicle CUDA 后端库。

### 路线 B：接入系统已有 aarch64 后端

已搜索 `/opt`、`/usr/local`、`/usr`、`/home/pzh`、用户 Cargo cache 和本目录 Cargo cache 中的 `libicicle_backend_cuda*.so`。

找到的库全部来自 `halo2` checkout，且均为 `x86_64` ELF。未找到可加载的 `aarch64` Icicle CUDA 后端库。

当前被检测到的后端目录为：

```text
/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda
```

关键库的文件类型为：

```text
libicicle_backend_cuda_device.so: x86_64
libicicle_backend_cuda_curve_bn254.so: x86_64
libicicle_backend_cuda_field_bn254.so: x86_64
libicicle_backend_cuda_hash.so: x86_64
```

## GPU 后端自检结果

自检命令：

```bash
python3 scripts/check_gpu_backend.py --json > results/gpu_backend_check.json
```

自检结论：

- `host_arch=aarch64`
- `gpu_available=true`
- `gpu_device_count=1`
- `backend_arch_match=false`
- `library_arches=["x86_64"]`
- `usable=false`

最小初始化命令：

```bash
ezkl gen-srs --srs-path gpu_backend_check.srs --logrows 1
```

最小初始化结果：

```text
gpu_backend_device_unavailable: CUDA device 0 is not available
```

该命令在自清理工作目录中执行，不保留 proof、key、SRS 或其它临时产物。

## GPU 数据集状态

| 数据集 | GPU 状态 | 失败步骤 | block_reason | gpu_backend_active | cpu_fallback_used | 是否 verified |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| Citeseer | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| Pubmed | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| PPI | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |
| ogbn-arxiv | `blocked` | `gpu_backend_preflight` | `gpu_backend_unavailable` | `false` | `false` | 否 |

## 未标记为 verified 的原因

GPU route 在 proof 生成前的后端预检阶段被阻塞，未进入 strict KZG prove，也未进入 verify。因此没有 GPU proof、没有 GPU 四项正式指标，也没有任何 GPU 行可标记为 `verified`。

当前阻塞不是 `GAT` 语义、数据、量化、最终 logits 绑定或验证逻辑失败，而是本机 `aarch64` 主机缺少可加载的 `aarch64` Icicle CUDA 后端库。现有后端库为 `x86_64`，不能用于本机。

## CPU accepted results 保持情况

CPU strict route 仍为：

- `ezkl_gat_native_compressed_csr_attention_full`

CPU 五个全量数据集仍满足：

- `status=verified`
- `strict_public_input_equivalent=true`
- `full_gat_forward_in_ezkl_proof=true`
- `final_logits_derivation_complete=true`
- `no_precomputed_logits_bypass=true`

`validate_final_results.py` 当前输出：

```text
final_result_validation=OK cpu_verified=5/5 gpu_status=not_verified_0/5
```

## 后续复现条件

要得到 GPU verified 结果，需要提供与本机匹配的 `aarch64` Icicle CUDA 后端库，或在 `x86_64` CUDA 主机上使用当前 `x86_64` 后端库运行同一 GPU route。完成后必须重新运行五个全量数据集，不得复用 CPU 指标作为 GPU 指标。
