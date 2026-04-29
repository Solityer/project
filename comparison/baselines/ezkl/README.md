# GAT-ZKML ezkl Full-Dataset Baseline

## 项目定位

本目录是 `GAT-ZKML` 主仓库的 `ezkl` strict full-dataset comparison baseline。它用于在相同 `GAT statement`、相同 full dataset、相同 checkpoint、相同图预处理、相同量化配置与相同 final logits 输出语义下，将主仓库证明系统替换为 `ezkl/Halo2` native backend 进行对比。

本目录不是主仓库 formal proof 系统本身，也不是 `ezkl` 官方 demo。当前结果依赖本目录内对 `ezkl` 源码增加的 native GAT proof route，因此不能表述为原版 `ezkl` 开箱即用的 ONNX baseline。它证明的是与主仓库 formal 语义对齐的 full GAT statement；对比变量是证明后端以及为承载该 statement 所需的 `ezkl/Halo2` 适配。

所有构建、运行、结果汇总与审计产物均独立保存在 `comparison/baselines/ezkl/` 下。主仓库代码、数据、checkpoint、配置与 formal note 只作为只读真源使用。

## formal 基准与对比约束

唯一 formal 语义真源仍是项目根目录下的：

- `../../../GAT-ZKML-多层多头.md`

本目录不重新定义 GAT 语义，只实现与该 formal statement 对齐的 `ezkl` native backend 适配。对比实验中下列对象不得改变：

- `dataset`
- `graph preprocessing`
- `checkpoint bundle`
- `model_arch_id`
- `model_param_id`
- `quant_cfg_id`
- `hidden_profile`
- `d_in_profile`
- `K_out`
- `output_average_rule`
- `final logits semantics`

## 当前正式路线

当前唯一正式路线为：

```text
ezkl_gat_native_compressed_csr_attention_full
```

该路线是 `ezkl native extension for the same GAT statement`。native `ezkl/Halo2` 适配包括：

- compressed CSR graph attention
- deterministic formal lookup tables
- quantized inverse lookup policy
- chunked public instance columns
- shape-only setup for large public outputs
- tagged lookup tables
- prove-time payload deduplication

这些适配属于 statement-preserving proof backend adaptations，只改变证明后端承载方式与资源组织方式，不改变模型语义、数据语义、量化语义或输出语义。

## 严格性说明

当前 accepted truth 为：

- `route_id=ezkl_gat_native_compressed_csr_attention_full`
- verified rows: `5/5`
- 数据集：`Cora`、`Citeseer`、`Pubmed`、`PPI`、`ogbn-arxiv`

五个数据集均满足：

| 数据集 | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | no_precomputed_logits_bypass |
| --- | --- | --- | --- | --- | --- |
| Cora | verified | true | true | true | true |
| Citeseer | verified | true | true | true | true |
| Pubmed | verified | true | true | true | true |
| PPI | verified | true | true | true | true |
| ogbn-arxiv | verified | true | true | true | true |

严格性口径如下：

- full GAT forward 在 `ezkl` proof 内证明。
- final logits 是 row-major node-by-class public instances。
- final logits 由 native output derivation 产生并绑定。
- 未采用 output-binding workaround、precomputed logits、identity route、digest route 或 folding route。
- 未采用 subset、shard、sample 切分。
- 未采用 toy、smoke、debug 数据集。

## 构建方式

```bash
cd /home/pzh/project/comparison/baselines/ezkl

cargo test native_gat --no-default-features --features ezkl,no-banner
cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner

cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j

python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py
```

如本机 `rustup` 未配置 default toolchain，请按本地 toolchain 配置导出 `RUSTC` / `RUSTDOC`，或使用本机已有的 `cargo` toolchain 执行同一组命令。toolchain 路径不是实验 statement 的组成部分。

## 正式复现命令

五个 full dataset 的正式运行命令如下：

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
```

运行结束后汇总并验证最终结果：

```bash
python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

## 正式结果

`ezkl` 四项正式指标如下：

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 39418.157 | 20340.017 | 372.952 | 1280306 |
| Citeseer | 39775.686 | 20593.018 | 489.516 | 1347736 |
| Pubmed | 79890.208 | 43471.667 | 966.513 | 3973400 |
| PPI | 1289180 | 1921540 | 23240 | 461655303 |
| ogbn-arxiv | 1290660 | 1860150 | 24660 | 453849446 |

主仓库与 `ezkl` 的正式对比表不在本 README 中手写维护，其正式来源为：

- `results/main_vs_ezkl_full_comparison.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/summary.md`

## 对比结果真源

最终展示与机器可读结果保存在：

- `results/gat_ezkl_full_results.json`
- `results/gat_ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
- `results/summary.md`

上述文件由 `scripts/collect_gat_ezkl_results.py` 从 run manifests、per-run binding artifact 与主仓库 benchmark 真源生成，并由 `scripts/validate_final_results.py` 校验。

## 机器复核真源

每个正式数据集目录均保留最小复核 artifact：

- `runs/cora_full/`
- `runs/citeseer_full/`
- `runs/pubmed_full/`
- `runs/ppi_full_formal/`
- `runs/ogbn_arxiv_full/`

每个目录下的核心复核文件为：

- `run_manifest.json`
- `model_info.json`
- `graph_attention_spec.json`
- `final_logits_binding.json`
- `formal_tables.json`
- `formal_table_hashes.json`
- `native_full_witness_preflight.json`

其中 `final_logits_binding.json` 记录 row-major node-by-class public instance 绑定；`run_manifest.json` 记录 route、status 与四项指标；`formal_tables.json` 与 `formal_table_hashes.json` 记录 deterministic lookup table policy。

## 控制变量审计

控制变量审计真源为：

- `results/control_variable_audit.md`

审计结论为：除证明系统替换为 `ezkl/Halo2` native route 以及为承载 full public logits 与 native GAT relation 所必需的 backend adaptations 外，数据、模型、checkpoint、图预处理、量化配置、输出平均规则与 final logits 语义均保持与主仓库对齐。

`ezkl` proof size 大于主仓库正式 proof size，是 `ezkl/Halo2` native route 与 full public logits 绑定共同作用的结果，不表示任务、数据集或输出语义发生改变。

## 大图 public output 与资源说明

大图 public output 复核真源为：

- `results/large_public_output_policy_review.md`

`PPI` 与 `ogbn-arxiv` 的 final logits 规模较大，当前路线使用 chunked public instance columns、shape-only setup 与 prove-time payload deduplication 等资源组织方式。这些方式不改变 public output 语义：final logits 仍作为公开实例参与验证，并由 proof 内 native output derivation 绑定。

## 清理与仓库状态

清理与代码洁净审计真源为：

- `results/cleanup_report.md`
- `results/code_cleanliness_audit.md`

最终仓库只保留源码、配置、复现脚本、最终结果与最小审计 artifact。proof、key、SRS、witness、compiled circuit、二进制 formal table 与构建产物不保留在仓库中，可由正式复现脚本重新生成。当前结果表、审计文件与 per-run manifests 构成本目录的 accepted truth。
