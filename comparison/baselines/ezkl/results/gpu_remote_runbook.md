# GPU 远端运行手册

## 适用目标

本手册用于把 `ezkl_gat_native_compressed_csr_attention_full_gpu` strict route 迁移到 `x86_64` CUDA 主机运行。远端运行必须证明与 CPU route 同一个 compressed CSR native `GAT` statement，只允许改变执行后端，不允许改变数据、图预处理、checkpoint、量化配置、最终 logits 语义或公开输出策略。

## 远端主机要求

- Linux `x86_64`。
- NVIDIA GPU 与可用驱动。
- CUDA toolkit，可执行 `nvcc --version`。
- Rust/Cargo，可使用本目录 `rust-toolchain` 指定版本。
- CMake 3.18+ 与 C++17 编译器。
- Python 3。
- 可加载的 `x86_64` Icicle CUDA backend，目录内至少包含：

```text
libicicle_backend_cuda_device.so
libicicle_backend_cuda_curve_bn254.so
libicicle_backend_cuda_field_bn254.so
libicicle_backend_cuda_hash.so
```

## 生成运行包

在本机执行：

```bash
cd /home/pzh/project/comparison/baselines/ezkl
python3 scripts/package_gpu_experiment.py
```

生成：

```text
gpu_experiment_bundle/
gpu_experiment_bundle.tar.gz
```

运行包不包含 proof、key、witness、compiled circuit、构建目录或 Icicle 后端二进制。

## 远端放置方式

在远端准备同一主仓库布局，主仓库数据和 artifact 仍按 `/home/pzh/project` 布局提供：

```bash
mkdir -p /home/pzh/project/comparison/baselines
tar -xzf gpu_experiment_bundle.tar.gz -C /home/pzh/project/comparison/baselines
mv /home/pzh/project/comparison/baselines/gpu_experiment_bundle /home/pzh/project/comparison/baselines/ezkl
cd /home/pzh/project/comparison/baselines/ezkl
```

如果远端路径不同，需先把主仓库补齐到 `/home/pzh/project`，因为 runner 和 exporter 会只读访问主仓库 `data/`、`artifacts/` 与 `runs/benchmarks/latest.json`。

## 构建命令

```bash
cd /home/pzh/project/comparison/baselines/ezkl

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup \
CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local \
cargo test native_gat --no-default-features --features ezkl,no-banner

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup \
CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local \
cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup \
CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local \
cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner,gpu-accelerated

cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j

python3 -m py_compile \
  scripts/run_gat_ezkl_baseline.py \
  scripts/collect_gat_ezkl_results.py \
  scripts/validate_final_results.py \
  scripts/check_gpu_backend.py \
  scripts/package_gpu_experiment.py
```

## 后端环境变量

把 `ICICLE_BACKEND_INSTALL_DIR` 指向远端 `x86_64` Icicle CUDA backend 目录：

```bash
export ICICLE_BACKEND_INSTALL_DIR=/path/to/icicle/backend/cuda
export EZKL_REQUIRE_GPU_BACKEND=1
export LD_LIBRARY_PATH=/home/pzh/project/comparison/baselines/ezkl/target/release/deps/icicle/lib:${LD_LIBRARY_PATH:-}
```

如果 Icicle license 需要显式配置，按远端授权方式设置：

```bash
export ICICLE_LICENSE=/path/to/license
```

或：

```bash
export ICICLE_LICENSE=port@ip
```

## GPU 后端检查

```bash
python3 scripts/check_gpu_backend.py
python3 scripts/check_gpu_backend.py --json > results/gpu_backend_check.json
```

只有同时满足以下条件，才允许进入五个 full dataset GPU 运行：

```text
usable=true
backend_arch_match=true
host_arch=x86_64
gpu_device_count >= 1
cuda_backend_initialization_status=ok
cpu_fallback_used=false
```

## 五个全量数据集 GPU 运行

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --clean --run

python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

验证成功时应输出：

```text
final_result_validation=OK cpu_verified=5/5 gpu_status=verified_5/5
```

## 回填文件

远端运行完成后，把以下文件回填到本仓库同名路径：

```text
results/gpu_backend_check.json
results/gat_ezkl_gpu_full_results.json
results/gat_ezkl_gpu_full_results.csv
results/main_vs_ezkl_gpu_full_comparison.json
results/main_vs_ezkl_gpu_full_comparison.csv
results/gpu_summary.md
runs/cora_full/gpu/run_manifest.json
runs/citeseer_full/gpu/run_manifest.json
runs/pubmed_full/gpu/run_manifest.json
runs/ppi_full_formal/gpu/run_manifest.json
runs/ogbn_arxiv_full/gpu/run_manifest.json
```

每个 GPU manifest 必须包含：

```text
backend=gpu
gpu_backend_required=true
gpu_backend_active=true
cpu_fallback_used=false
route_id=ezkl_gat_native_compressed_csr_attention_full_gpu
status=verified
strict_public_input_equivalent=true
full_gat_forward_in_ezkl_proof=true
final_logits_derivation_complete=true
no_precomputed_logits_bypass=true
commitment_time_ms
prove_time_ms
verify_time_ms
proof_size_bytes
gpu_device_name
gpu_backend_library_arch=x86_64
verify_passed=true
```

回填后在本机执行：

```bash
cd /home/pzh/project/comparison/baselines/ezkl
python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

若任何 GPU 行没有真实 verify 通过，不能把该行写成 `verified`。
