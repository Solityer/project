# 清理报告

## 元数据

- 更新时间：2026-05-04 Asia/Shanghai
- 范围：`/home/pzh/project/comparison/baselines/ezkl`
- 主仓库其它目录：未修改

## 本轮保留内容

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

GPU 状态与结果：

- `results/gat_ezkl_gpu_full_results.json`
- `results/gat_ezkl_gpu_full_results.csv`
- `results/main_vs_ezkl_gpu_full_comparison.json`
- `results/main_vs_ezkl_gpu_full_comparison.csv`
- `results/gpu_summary.md`
- `results/gpu_route_status.md`

审计文档：

- `results/control_variable_audit.md`
- `results/large_public_output_policy_review.md`
- `results/native_full_experiment_report.md`
- `results/native_constraint_debug_report.md`
- `results/code_cleanliness_audit.md`
- `results/cleanup_report.md`

## per-run 最小 artifact

CPU 每个正式数据集目录保留：

- `run_manifest.json`
- `model_info.json`
- `graph_attention_spec.json`
- `final_logits_binding.json`
- `formal_tables.json`
- `formal_table_hashes.json`
- `native_full_witness_preflight.json`
- `keygen_memory_profile.json`
- `prove_memory_profile.json`

GPU 每个正式数据集的 `gpu/` 目录保留：

- `run_manifest.json`
- `model_info.json`
- `graph_attention_spec.json`
- `final_logits_binding.json`
- `formal_tables.json`
- `formal_table_hashes.json`
- `native_full_witness_preflight.json`

GPU 未保留 proof、key、SRS、witness、compiled circuit 或二进制 payload；当前 GPU 只保留后端不可用 evidence。

## 已清理或忽略内容

构建产物与临时目录：

- `target/`
- `build_gat_adapter/`
- `CMakeFiles/`
- `CMakeCache.txt`
- `cmake_install.cmake`
- `scripts/__pycache__/`
- `tmp/`

可复现大文件：

- `proof.json`
- `pk.key`
- `vk.key`
- `pk.srs`
- `vk.srs`
- `witness.json`
- `model.compiled`
- `formal_tables.bin`
- `csr_segment_layout.bin`
- `node_features.f64.bin`
- generated ONNX/native marker files
- stdout/stderr/time dump

`.gitignore` 已覆盖 CPU 与 GPU 嵌套 run 目录下的大型 proof/key/witness/compiled/payload 文件。最小 JSON artifact 通过强制加入版本控制保留。

## GPU 清理结论

GPU route 已对五个全量数据集实际启动，但全部在 `gen_settings_ms` 阶段因 `gpu_backend_unavailable` 阻塞。阻塞证据已汇总到：

- `results/gpu_route_status.md`
- `runs/<dataset>/gpu/run_manifest.json`

GPU 未产生 verified proof，因此没有 GPU 四项正式指标。

## 验证状态

- `python3 scripts/validate_final_results.py`：`final_result_validation=OK cpu_verified=5/5 gpu_status=not_verified_0/5`
- CPU 结果仍为五个全量数据集 `verified`
- GPU 结果为五个全量数据集 `blocked`

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
