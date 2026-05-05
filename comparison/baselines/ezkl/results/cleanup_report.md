# 清理报告

## 元数据

- 更新时间：2026-05-05 Asia/Shanghai
- 范围：`/home/pzh/project/comparison/baselines/ezkl`
- 主仓库其它目录：未修改

## 最终保留内容

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

CPU strict 结果：

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

## per-run 最小 artifact

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

## 已删除内容

本轮已删除此前的加速路线、后端检查脚本、远端执行包脚本、结果文件、状态文件、per-run manifests、运行包目录与压缩包。最终目录为 CPU-only strict baseline。

同时清理或确认忽略以下构建与临时对象：

- `target/`
- `build_gat_adapter/`
- `CMakeFiles/`
- `CMakeCache.txt`
- `cmake_install.cmake`
- `scripts/__pycache__/`
- `tmp/`
- `*.tmp`
- `*.bak`
- `*.old`
- `*.orig`
- `core`
- `core.*`
- `perf.data`
- `nohup.out`
- `*.log`

## 验证状态

- `python3 scripts/validate_final_results.py`：`final_result_validation=OK cpu_verified=5/5`
- 五个 full dataset 均为 `verified`
- `git status --short` 在提交后为空

## 主仓库污染检查

本轮所有修改均限制在 `comparison/baselines/ezkl/` 内。未修改：

- `/home/pzh/project/src`
- `/home/pzh/project/include`
- `/home/pzh/project/configs`
- `/home/pzh/project/data`
- `/home/pzh/project/artifacts`
- `/home/pzh/project/runs`
- `/home/pzh/project/scripts`
- `/home/pzh/project/README.md`
- `/home/pzh/project/GAT-ZKML-多层多头.md`
- `/home/pzh/project/comparison/baselines/GAT-ZKML-ezkl.md`
