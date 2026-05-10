# 清理报告

## 保留的内容

源码、配置与脚本：

- `Cargo.toml`
- `Cargo.lock`
- `src/`
- `cpp/gat_ezkl_adapter/`
- `configs/full_datasets.json`
- `configs/ezkl_routes.json`
- `scripts/run_gat_ezkl_baseline.py`
- `scripts/collect_gat_ezkl_results.py`
- `scripts/validate_final_results.py`

CPU 结果：

- `results/gat_ezkl_full_results.json`
- `results/gat_ezkl_full_results.csv`
- `results/main_vs_ezkl_full_comparison.json`
- `results/main_vs_ezkl_full_comparison.csv`
- `results/summary.md`

审计文档：

- `results/control_variable_audit.md`
- `results/large_public_output_policy_review.md`
- `results/native_full_experiment_report.md`
- `results/native_constraint_debug_report.md`
- `results/code_cleanliness_audit.md`
- `results/cleanup_report.md`

## 每次运行的最小结果

每个正式数据集目录保留：

- `run_manifest.json`
- `model_info.json`
- `graph_attention_spec.json`
- `final_logits_binding.json`
- `formal_tables.json`
- `formal_table_hashes.json`
- `native_full_witness_preflight.json`
- `keygen_memory_profile.json`
- `prove_memory_profile.json`

未保留 proof、key、SRS、witness、compiled circuit 或二进制 payload。

## 验证状态

- `python3 scripts/validate_final_results.py`：`final_result_validation=OK cpu_verified=5/5`
- 五个 full dataset 均为 `verified`