# GAT-ZKML 的 ezkl 全量数据集对比实验

## 项目定位

本目录是 `GAT-ZKML` 主仓库的 `ezkl` 严格全量数据集对比实验基线。它只服务于一个目标：在相同 `GAT` 证明语句、相同数据集、相同图预处理、相同 checkpoint、相同量化配置与相同最终 logits 语义下，将主仓库证明系统替换为 `ezkl/Halo2` 路线进行对照。

本目录不是 `ezkl` 官方通用仓库说明，也不承载 `ezkl` 官方示例、教程或贡献说明。当前实现依赖本目录内的 native `GAT` 适配，用于证明与主仓库 formal 语义对齐的全量数据集推理语句；允许变化的对象仅限证明系统后端，以及为承载该语句所需的 `ezkl/Halo2` 工程适配。

所有源码、配置、复现脚本、结果表与审计产物均位于 `comparison/baselines/ezkl/`。主仓库其它目录只作为只读 formal 真源使用。

## formal 基准与对比约束

唯一 formal 真源为：

- `../../../GAT-ZKML-多层多头.md`

本目录不重新定义 `GAT` 语义。下列控制变量必须与主仓库保持一致：

- 数据集与全量样本集合
- 图预处理、自环规则、边排序规则与批处理规则
- checkpoint bundle
- `model_arch_id`
- `model_param_id`
- `quant_cfg_id`
- `hidden_profile`
- `d_in_profile`
- `K_out`
- `output_average_rule`
- 最终 logits 语义与公开输出策略

允许差异仅为证明系统替换为 `ezkl/Halo2`，以及 native compressed CSR、公开实例分列、确定性查找表、资源复用和 GPU 后端切换等不改变证明语句的工程适配。

## 当前路线

| 路线 | route_id | 后端 | 状态 |
| --- | --- | --- | --- |
| CPU 严格路线 | `ezkl_gat_native_compressed_csr_attention_full` | `CPU` | 五个全量数据集均为 `verified` |
| GPU 严格路线 | `ezkl_gat_native_compressed_csr_attention_full_gpu` | `GPU` / `Icicle CUDA` | 已接入并实际启动，当前因后端二进制架构不匹配为 `blocked`，未产生 `verified` 结果 |

两条路线使用同一个 compressed CSR native `GAT` 关系、同一套数据与模型元数据、同一套量化和最终 logits 语义。GPU 路线只改变执行后端和独立输出目录，不允许改变模型、数据、预处理、量化、输出平均规则或公开输出语义。

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

GPU 路线沿用同一严格性口径。若 GPU 构建、设备、后端加载、prove 或 verify 任一步未完成，结果不得标记为 `verified`。

## 构建方式

### CPU 构建

```bash
cd /home/pzh/project/comparison/baselines/ezkl

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo test native_gat --no-default-features --features ezkl,no-banner
RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner

cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j

python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py
```

### GPU 构建

GPU 路线使用实际存在的 `gpu-accelerated` feature，并通过 `ICICLE_BACKEND_INSTALL_DIR` 指向 `Icicle CUDA` 后端目录。

```bash
cd /home/pzh/project/comparison/baselines/ezkl

RUSTUP_HOME=/home/pzh/project/comparison/baselines/ezkl/.rustup CARGO_HOME=/home/pzh/project/comparison/baselines/ezkl/.cargo_local cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner,gpu-accelerated
```

当前本机检测到的 `Icicle CUDA` 后端目录为：

```text
/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda
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

```bash
ICICLE_BACKEND_INSTALL_DIR=/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --icicle-backend-dir /home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda --clean --run
ICICLE_BACKEND_INSTALL_DIR=/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --icicle-backend-dir /home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda --clean --run
ICICLE_BACKEND_INSTALL_DIR=/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --icicle-backend-dir /home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda --clean --run
ICICLE_BACKEND_INSTALL_DIR=/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --icicle-backend-dir /home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda --clean --run
ICICLE_BACKEND_INSTALL_DIR=/home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full_gpu --backend gpu --icicle-backend-dir /home/pzh/project/comparison/baselines/ezkl/.cargo_local/git/checkouts/halo2-9199b7623b70ff0b/1dd2090/icicle/backend/cuda --clean --run
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

GPU route 已对五个全量数据集实际启动，但未产生 `verified` 结果。当前本机为 `aarch64`，检测到的 `Icicle CUDA` 后端库为 `x86-64`，运行时在 `gen_settings_ms` 阶段报 `Device API not found for type: CUDA`。正式状态与证据记录在 `results/gpu_route_status.md`；不得将 CPU 指标复制为 GPU 指标。

| 数据集 | status | block_reason | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | `blocked` | `gpu_backend_unavailable` |  |  |  |  |
| Citeseer | `blocked` | `gpu_backend_unavailable` |  |  |  |  |
| Pubmed | `blocked` | `gpu_backend_unavailable` |  |  |  |  |
| PPI | `blocked` | `gpu_backend_unavailable` |  |  |  |  |
| ogbn-arxiv | `blocked` | `gpu_backend_unavailable` |  |  |  |  |

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

## 机器复核真源

CPU 每个数据集保留以下最小审计 artifact：

- `runs/cora_full/run_manifest.json`
- `runs/citeseer_full/run_manifest.json`
- `runs/pubmed_full/run_manifest.json`
- `runs/ppi_full_formal/run_manifest.json`
- `runs/ogbn_arxiv_full/run_manifest.json`

每个 CPU 目录同时保留：

- `model_info.json`
- `graph_attention_spec.json`
- `final_logits_binding.json`
- `formal_tables.json`
- `formal_table_hashes.json`
- `native_full_witness_preflight.json`
- `keygen_memory_profile.json`
- `prove_memory_profile.json`

GPU route 的最小审计 artifact 使用独立目录：

- `runs/cora_full/gpu/run_manifest.json`
- `runs/citeseer_full/gpu/run_manifest.json`
- `runs/pubmed_full/gpu/run_manifest.json`
- `runs/ppi_full_formal/gpu/run_manifest.json`
- `runs/ogbn_arxiv_full/gpu/run_manifest.json`

GPU manifest 必须记录 `backend=gpu`、`gpu_backend`、`gpu_env`、`gpu_device_info`、完整命令、证明验证状态与四项指标；未验证时不得填入成功状态。

当前 GPU 阻塞状态还保留每个 `gpu/` 目录下的小型 JSON 复核文件：

- `model_info.json`
- `graph_attention_spec.json`
- `final_logits_binding.json`
- `formal_tables.json`
- `formal_table_hashes.json`
- `native_full_witness_preflight.json`

二进制输入、feature payload、proof、key、witness、compiled circuit 与控制台输出已清理，不进入最终仓库表面。

## 控制变量审计

控制变量审计真源为：

- `results/control_variable_audit.md`

审计结论必须同时覆盖 CPU 与 GPU 路线：两条路线证明同一 `GAT` 语义，差异只属于证明后端执行方式和工程资源组织方式。

## 大图 public output 与资源说明

大图 public output 复核真源为：

- `results/large_public_output_policy_review.md`

`PPI` 与 `ogbn-arxiv` 保留完整最终 logits 公开实例。chunked public instance columns、shape-only setup 与 prove-time payload deduplication 只改变资源组织方式，不改变公开输出值、顺序或验证语义。

## 仓库清理状态

清理与代码洁净审计真源为：

- `results/cleanup_report.md`
- `results/code_cleanliness_audit.md`

最终仓库只保留源码、配置、脚本、README、正式结果表和最小审计 artifact。proof、key、SRS、witness、compiled circuit、二进制 formal table、构建目录、缓存目录和临时文件不进入最终仓库表面。
