# GAT-ZKML 的 ezkl CPU 严格全量数据集对比实验

本目录是主仓库 GAT-ZKML 的 ezkl CPU strict baseline。主仓库提供 formal 语义真源；本目录只改变证明系统承载方式，以及 ezkl/Halo2 对稀疏图、公开实例列和查找表的必要适配。

## 正式路线

唯一保留路线：

```text
ezkl_gat_native_compressed_csr_attention_full
```

该路线证明 full GAT forward。final logits 在电路内完整推导并公开绑定，`strict_public_input_equivalent=true` 只在真实 prove 与 verify 完成后成立。

禁用项：

- 没有 output-binding workaround。
- 没有 digest。
- 没有 folding。
- 没有 identity。
- 没有 precomputed logits。
- 没有 subset、shard 或 sample。
- 没有 toy、smoke、partial 或 scaled graph。

## 全量数据集状态

| 数据集 | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | no_precomputed_logits_bypass |
| --- | --- | --- | --- | --- | --- |
| Cora | verified | true | true | true | true |
| Citeseer | verified | true | true | true | true |
| Pubmed | verified | true | true | true | true |
| PPI | verified | true | true | true | true |
| ogbn-arxiv | verified | true | true | true | true |

## CPU 四项指标

| 数据集 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | ---: |
| Cora | 39418.157 | 20340.017 | 372.952 | 1280306 |
| Citeseer | 39775.686 | 20593.018 | 489.516 | 1347736 |
| Pubmed | 79890.208 | 43471.667 | 966.513 | 3973400 |
| PPI | 1289180 | 1921540 | 23240 | 461655303 |
| ogbn-arxiv | 1290660 | 1860150 | 24660 | 453849446 |

## 结果文件

- `results/gat_ezkl_full_results.json`
- `results/gat_ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
- `results/summary.md`
- `results/control_variable_audit.md`
- `results/large_public_output_policy_review.md`
- `results/native_full_experiment_report.md`
- `results/cleanup_report.md`
- `results/code_cleanliness_audit.md`

## 每个数据集的审计 artifact

每个 full dataset 目录均保留以下审计 artifact：

- `runs/<dataset>/run_manifest.json`
- `runs/<dataset>/model_info.json`
- `runs/<dataset>/graph_attention_spec.json`
- `runs/<dataset>/final_logits_binding.json`
- `runs/<dataset>/formal_tables.json`
- `runs/<dataset>/formal_table_hashes.json`
- `runs/<dataset>/native_full_witness_preflight.json`

适用的 `<dataset>`：

- `cora_full`
- `citeseer_full`
- `pubmed_full`
- `ppi_full_formal`
- `ogbn_arxiv_full`

## 构建与静态检查

```bash
cargo test native_gat --no-default-features --features ezkl,no-banner
cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner
cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j
python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py
```

## 运行与汇总

```bash
python3 scripts/run_gat_ezkl_baseline.py --dataset cora_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset citeseer_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset pubmed_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ppi_full_formal --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/run_gat_ezkl_baseline.py --dataset ogbn_arxiv_full --mode full --route ezkl_gat_native_compressed_csr_attention_full --clean --run
python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

期望验证输出：

```text
final_result_validation=OK cpu_verified=5/5
```
