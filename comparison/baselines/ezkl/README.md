# GAT-ZKML 的 ezkl CPU 严格全量数据集对比实验

本目录是主仓库 GAT-ZKML 的 ezkl CPU-only strict baseline。主仓库提供正式语义真源；本目录只改变证明系统承载方式，以及 ezkl/Halo2 对稀疏图、公开实例列和查找表的必要适配。

## 正式路线

唯一最终路线：`ezkl_gat_native_compressed_csr_attention_full`。该路线证明 full GAT forward。最终 logits 在电路内完整推导并公开绑定，`strict_public_input_equivalent=true` 只在真实 prove 与 verify 完成后成立。

不存在 GPU baseline；本目录最终结果全部为 `backend=cpu`。

## 全量数据集状态

| 数据集 | status | final benchmark | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | no_precomputed_logits_bypass | manifest_strict_check |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Cora | verified | true | true | true | true | true | passed |
| Citeseer | verified | true | true | true | true | true | passed |
| Pubmed | verified | true | true | true | true | true | passed |
| PPI | verified | true | true | true | true | true | passed |
| ogbn-arxiv | verified | true | true | true | true | true | passed |

## CPU 四项指标

| 数据集 | status | final benchmark | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Cora | verified | true | 39610.144 | 19551.087 | 399.098 | 1280317 |
| Citeseer | verified | true | 39191.259 | 20785.288 | 492.913 | 1347633 |
| Pubmed | verified | true | 80007.432 | 42373.141 | 989.291 | 3973375 |
| PPI | verified | true | 1280185.848 | 1956645.759 | 24571.380 | 461655281 |
| ogbn-arxiv | verified | true | 1287653.770 | 1947900.981 | 25975.184 | 453849491 |

## 结果可信性检查

最终表格由 `scripts/collect_gat_ezkl_results.py` 从 run manifest 自动生成。collector 和 validator 会拒绝以下情况：

- `status=verified` 但 `verify_passed` 或 `proof_generated` 不为 `true`。
- `status=verified` 但 `failed_command`、`failure_step` 或 `block_reason` 非空。
- `stdout_tail`、`stderr_tail` 或阶段日志包含 `manual continuation`。
- `gen_settings_ms`、`compile_ms`、`setup_ms`、`gen_witness_ms`、`prove_time_ms` 或 `verify_time_ms` 缺失。
- `proof_size_bytes` 与真实 `proof.json` 文件大小不一致。
- 缺少 `proof.json`、`verify.json`、`run_manifest.json`、模型/图/绑定/形式化表/预检查/内存 profile 文件。
- route 不是 CPU strict route，或出现 forbidden fallback / precomputed logits bypass。

## 结果文件

- `results/gat_ezkl_full_results.json`
- `results/gat_ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
- `results/summary.md`
- `results/strict_audit_report.md`
- `results/control_variable_audit.md`
- `results/native_full_experiment_report.md`
- `results/ppi_ogbn_arxiv_audit.md`
- `results/cleanup_report.md`

## 构建、运行与验证

```bash
cargo test native_gat --no-default-features --features ezkl,no-banner
cargo build --release --bin ezkl --no-default-features --features ezkl,no-banner
cmake -S cpp/gat_ezkl_adapter -B build_gat_adapter -DCMAKE_BUILD_TYPE=Release
cmake --build build_gat_adapter -j
python3 -m py_compile scripts/run_gat_ezkl_baseline.py scripts/collect_gat_ezkl_results.py scripts/validate_final_results.py
python3 scripts/collect_gat_ezkl_results.py
python3 scripts/validate_final_results.py
```

五个数据集的 clean rerun 命令见 `scripts/run_gat_ezkl_baseline.py --help` 和本目录正式审计报告。
