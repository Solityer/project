# 主实验 targeted tests

本目录是 zkGAT 主实验唯一的 test-only 工作区。生产构建默认不编译这里的
runtime、diagnostic 和 parity 入口；不要把测试输出、测试配置或测试 fixture
写入正式的 `runs/`、`artifacts/` 或其他生产目录。

## 独立测试构建

使用独立的构建目录显式打开测试选项：

```bash
cmake -S . -B build_tests \
  -DCMAKE_BUILD_TYPE=Release \
  -DGATZK_ENABLE_CUDA_BACKEND=OFF \
  -DGATZK_BUILD_TESTS=ON
cmake --build build_tests --target gatzk_layered_schema_test -j2
```

其他 C++ runtime targets 也只在 `GATZK_BUILD_TESTS=ON` 时生成。只运行与当前修改
直接相关的短时 target；不要把这个构建目录当作正式 proof/verify 构建，也不要在
这里执行完整数据集、正式 SRS 或最终结果采集。

## Protocol tamper attestation

`gatzk_protocol_tamper_tests` 只消费已经存在的 `proof.bin`、point-only
verifier registry 和对应 `run_manifest.json`，不会生成 proof、加载 proving key
或启动正式数据集运行。registry 的 SHA-256 必须由调用者显式提供，不能从 registry
自身摘要推导信任根：

```bash
build_tests/tests/gatzk_protocol_tamper_tests \
  --proof runs/cora_secure/cpu_formal/warm/proof.bin \
  --registry artifacts/verifier_registry/cora_verifier_registry.bin \
  --trusted-registry-digest <reviewed-registry-sha256> \
  --run-manifest runs/cora_secure/cpu_formal/warm/run_manifest.json \
  --output tests/attestations/cora_cpu_tamper.json
```

该命令要求 run manifest、proof、registry、SRS、schema、public-input 和构建
provenance 字段彼此一致，并逐一执行 canonical 39-case mutation。任一 fixture
字段缺失、baseline verify 不通过或 case 不是实际 mutation 都 fail closed。输出
只能写入仓库 canonical `tests/` 目录；`runs/`、`artifacts/`、`results/` 和
`results/paper_secure/` 以及仓库外路径会被拒绝。生成的 attestation 仅是测试证据，
其自身摘要不是信任根。

attestation 是唯一的 20-field canonical JSON 对象：字段名必须与 producer 的
`binary_sha256`、`build_identity`、`build_receipt_sha256`、`config_sha256`、
`format`、`instance_schema_hash`、`proof_sha256`、`protocol_schema_id`、
`public_input_digest`、`public_output_digest`、`source_manifest_sha256`、
`source_tree_hash`、`srs_manifest_sha256`、`suite_id`、`tamper_test_case_map`、
`tamper_tests_ms`、`tamper_tests_passed`、`topology_digest`、
`trusted_registry_digest`、`verifier_registry_digest` 完全一致。提交给正式 consumer
前，用户必须在文件外部计算并审阅 SHA-256；不能使用 JSON 内的任意字段作为信任根。
`scripts/aggregate_secure_results.py` 要求每个 CPU/CUDA dataset 同时提供
`--*-attestation dataset=path` 与 `--*-attestation-digest dataset=sha256`，而
`scripts/export_benchmark_table.py` 要求对应的 `--attestation` 与
`--attestation-digest`。本地 verifier report 的 `false`/空 map/`0` 只是 pending，
不能替代 external attestation，也不能把测试输出复制到正式 results。

## Planetoid parity

`check_planetoid_parity.py` 和 `planetoid_parity_helpers.py` 都位于本目录。脚本
直接从项目根目录调用时使用同目录 helper，并通过项目的 `scripts/` 导入正式
checkpoint/模型实现：

```bash
parity_output=$(mktemp tests/.parity-cora.XXXXXX.json)
.venv/bin/python tests/check_planetoid_parity.py \
  --checkpoint artifacts/checkpoints/cora_gat/best_model.pt \
  --bundle-dir artifacts/checkpoints/cora_gat \
  --data-root data \
  --dataset cora \
  --output "$parity_output"
jq -e '.passed == true and .comparison_count > 0' "$parity_output"
unlink "$parity_output"
```

`citeseer` 和 `pubmed` 只需替换 `--dataset` 及对应 checkpoint/bundle 路径。parity
依赖真实 checkpoint 和数据，只作为有明确需求时的 targeted 检查，不是正式性能或
最终结果采集。

## PPI protocol diagnostic

PPI 的 test-only 配置位于
[`tests/configs/ppi_protocol_diag.cfg`](./configs/ppi_protocol_diag.cfg)。手动 partial
diagnostic 入口（不注册到 CTest）为：

```bash
build_tests/tests/gatzk_protocol_replay_diag \
  --config tests/configs/ppi_protocol_diag.cfg \
  --compute-backend cpu
```

该入口只接受约束允许的 partial multi-graph 上下文；配置中的诊断输出路径也位于
`tests/`，不会成为正式运行结果。
