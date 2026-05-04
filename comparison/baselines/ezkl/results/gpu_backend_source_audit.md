# GPU 后端来源审计

## 审计结论

本轮在本机重新审计了 `ezkl_gat_native_compressed_csr_attention_full_gpu` 使用的 Rust、Halo2 与 Icicle 依赖来源。结论是：Rust GPU feature 可以构建，但当前可见的 Icicle CUDA 后端库仍只存在 `x86_64` 版本；本机 `aarch64` 需要的 CUDA backend 源码不在当前 checkout 中，且远程 backend 仓库需要授权，无法拉取。

## Rust 依赖来源

`Cargo.toml` 中 GPU feature 定义为：

```text
gpu-accelerated = ["halo2_proofs/gpu-accelerated", "dep:icicle-runtime"]
```

相关依赖为：

```text
halo2_gadgets = https://github.com/zkonduit/halo2 branch ac/conditional-compilation-icicle2
halo2_proofs = https://github.com/zkonduit/halo2 package halo2_proofs branch ac/conditional-compilation-icicle2
icicle-runtime = https://github.com/ingonyama-zk/icicle branch emir/gate_eval_2 package icicle-runtime
```

`Cargo.lock` 中的关键版本为：

```text
halo2_gadgets 0.2.0: git+https://github.com/zkonduit/halo2?branch=ac%2Fconditional-compilation-icicle2#01c88842679b4308e43ae5ed91c4183e861669bd
halo2_proofs 0.3.0: git+https://github.com/zkonduit/halo2?branch=ac%2Fconditional-compilation-icicle2#01c88842679b4308e43ae5ed91c4183e861669bd
icicle-bn254 3.7.0: git+https://github.com/ingonyama-zk/icicle?branch=emir%2Fgate_eval_2#012e00694f4cf399fe7a42d9cfbfa6cd7a60f876
icicle-core 3.7.0: git+https://github.com/ingonyama-zk/icicle?branch=emir%2Fgate_eval_2#012e00694f4cf399fe7a42d9cfbfa6cd7a60f876
icicle-hash 3.7.0: git+https://github.com/ingonyama-zk/icicle?branch=emir%2Fgate_eval_2#012e00694f4cf399fe7a42d9cfbfa6cd7a60f876
icicle-runtime 3.7.0: git+https://github.com/ingonyama-zk/icicle?branch=emir%2Fgate_eval_2#012e00694f4cf399fe7a42d9cfbfa6cd7a60f876
mopro-msm 0.1.0: git+https://github.com/zkonduit/metal-msm-gpu-acceleration.git#be5f647b1a6c1a6ea9024390744a2b4d87f5d002
```

`cargo tree --features ezkl,no-banner,gpu-accelerated` 显示，`halo2_proofs` 依赖 `icicle-bn254`、`icicle-core` 与 `icicle-runtime`，本仓库自身也通过 `dep:icicle-runtime` 触发 GPU 后端设备初始化。

## 本地源码位置

Icicle 前端源码：

```text
.cargo_local/git/checkouts/icicle-e4fed18c5a8319f4/012e006/icicle
```

Halo2 checkout 中附带的 CUDA 后端二进制：

```text
.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda
.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/01c8884/icicle/backend/cuda
```

当前 `ICICLE_BACKEND_INSTALL_DIR` 自动发现值：

```text
/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda
```

## CUDA backend 源码状态

Icicle 前端源码包含：

```text
icicle/CMakeLists.txt
icicle/cmake/backend_include.cmake
icicle/backend/cpu
```

但不包含：

```text
icicle/backend/cuda
```

Halo2 checkout 中的 `icicle/backend/cuda` 只包含以下 `.so`，没有 `CMakeLists.txt`、`Makefile`、`build.rs` 或 CUDA 源码：

```text
libicicle_backend_cuda_device.so
libicicle_backend_cuda_curve_bn254.so
libicicle_backend_cuda_field_bn254.so
libicicle_backend_cuda_hash.so
```

文件架构均为：

```text
ELF 64-bit LSB shared object, x86-64
```

## 构建系统发现

Icicle `icicle/CMakeLists.txt` 提供后端选项：

```text
CUDA_BACKEND = local 或 branch/commit
```

`icicle/cmake/backend_include.cmake` 的 CUDA 行为为：

```text
- CUDA_BACKEND=local: add_subdirectory(backend/cuda)
- CUDA_BACKEND=<branch|commit>: FetchContent 拉取 git@github.com:ingonyama-zk/icicle-cuda-backend.git
```

Icicle 文档 `install_cuda_backend.md` 明确说明 CUDA backend 是闭源授权组件，需要 license。当前本机没有可构建的 CUDA backend 源码。

## 远程源码获取尝试

已尝试：

```bash
git ls-remote https://github.com/ingonyama-zk/icicle-cuda-backend.git HEAD refs/heads/main refs/heads/master
git clone --depth 1 --branch main https://github.com/ingonyama-zk/icicle-cuda-backend.git .vendor_local/icicle-cuda-backend-main
git clone --depth 1 --branch main git@github.com:ingonyama-zk/icicle-cuda-backend.git .vendor_local/icicle-cuda-backend-main-ssh
```

核心错误：

```text
fatal: could not read Username for 'https://github.com': No such device or address
ERROR: Repository not found.
fatal: Could not read from remote repository.
```

因此，本轮没有获得可构建的 aarch64 Icicle CUDA backend source。
