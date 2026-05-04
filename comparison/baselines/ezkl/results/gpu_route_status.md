# GPU route 状态

## 结论

GPU route 已接入并对五个全量数据集实际启动：

- `ezkl_gat_native_compressed_csr_attention_full_gpu`

当前 GPU route 未产生 `verified` 结果，所有 GPU 行均未标记为 verified。CPU strict accepted results 仍为五个全量数据集 `verified`。

## GPU 构建状态

GPU feature 构建已通过：

```bash
RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner,gpu-accelerated
```

构建结果：

```text
Finished `release` profile [optimized] target(s) in 3m 42s
```

## GPU 设备检测

设备检测命令：

```bash
nvidia-smi
```

检测结果包含：

```text
NVIDIA GB10
Driver Version: 580.142
CUDA Version: 13.0
```

runner 记录的机器可读设备信息见：

- `runs/cora_full/gpu/run_manifest.json`
- `runs/citeseer_full/gpu/run_manifest.json`
- `runs/pubmed_full/gpu/run_manifest.json`
- `runs/ppi_full_formal/gpu/run_manifest.json`
- `runs/ogbn_arxiv_full/gpu/run_manifest.json`

## GPU 后端环境

GPU route 使用的后端目录：

```text
/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda
```

运行环境包含：

- `EZKL_REQUIRE_GPU_BACKEND=1`
- `ICICLE_BACKEND_INSTALL_DIR=/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda`
- `LD_LIBRARY_PATH=/home/pzh/project/comparison/baselines/ezkl/target/release/deps/icicle/lib`

## 后端不可用证据

本机主机架构：

```text
aarch64
```

当前 `Icicle CUDA` 后端库的文件类型为：

```text
libicicle_backend_cuda_curve_bn254.so: ELF 64-bit LSB shared object, x86-64
libicicle_backend_cuda_device.so: ELF 64-bit LSB shared object, x86-64
libicicle_backend_cuda_field_bn254.so: ELF 64-bit LSB shared object, x86-64
libicicle_backend_cuda_hash.so: ELF 64-bit LSB shared object, x86-64
```

ezkl 运行时错误为：

```text
eIcicleError::INVALID_DEVICE
Device API not found for type: CUDA
fatal runtime error: Rust cannot catch foreign exceptions, aborting
```

因此当前阻塞不是 GAT 语义、数据、量化、最终 logits 绑定或验证逻辑失败，而是本机可用 GPU 与当前 `Icicle CUDA` 后端二进制架构不匹配。

## 数据集状态

| 数据集 | GPU 状态 | 失败步骤 | block_reason | 是否 verified |
| --- | --- | --- | --- | --- |
| Cora | `blocked` | `gen_settings_ms` | `gpu_backend_unavailable` | 否 |
| Citeseer | `blocked` | `gen_settings_ms` | `gpu_backend_unavailable` | 否 |
| Pubmed | `blocked` | `gen_settings_ms` | `gpu_backend_unavailable` | 否 |
| PPI | `blocked` | `gen_settings_ms` | `gpu_backend_unavailable` | 否 |
| ogbn-arxiv | `blocked` | `gen_settings_ms` | `gpu_backend_unavailable` | 否 |

## 未标记为 verified 的原因

GPU route 在 proof 生成前的后端初始化阶段被阻塞，未进入 strict KZG prove，也未进入 verify。因此没有 GPU proof、没有 GPU 四项正式指标，也没有任何 GPU 行可标记为 `verified`。

## CPU accepted results 保持情况

CPU strict route 仍为：

- `ezkl_gat_native_compressed_csr_attention_full`

CPU 五个全量数据集仍满足：

- `status=verified`
- `strict_public_input_equivalent=true`
- `full_gat_forward_in_ezkl_proof=true`
- `final_logits_derivation_complete=true`
- `no_precomputed_logits_bypass=true`

## 下一步依赖

要得到 GPU verified 结果，需要为当前 `aarch64` 主机提供可加载的 `Icicle CUDA` 后端库，或在与现有后端二进制匹配的 `x86-64` CUDA 主机上运行同一 GPU route。完成后必须重新运行五个全量数据集，不得复用 CPU 指标作为 GPU 指标。
