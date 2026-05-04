# 代码洁净审计

## 审计时间

2026-05-04

## 当前正式路线

- CPU：`ezkl_gat_native_compressed_csr_attention_full`
- GPU：`ezkl_gat_native_compressed_csr_attention_full_gpu`

GPU 路线已接入 runner、配置、collector、validator、正式 GPU 后端自检脚本、远端运行包脚本与 README。当前 GPU 运行状态为 `blocked`，阻塞步骤为 `gpu_backend_preflight`，原因是 `Icicle CUDA` 后端二进制与本机 `aarch64` 架构不匹配；未写入任何 fake、stub、dummy 或 placeholder 成功路径。

## 检查命令

```bash
grep -R "TODO\|FIXME\|HACK\|XXX\|stub\|dummy\|fake\|placeholder\|temporary\|debug-only\|not implemented\|unimplemented!\|todo!\|panic!(\"TODO" -n src cpp scripts configs README.md || true
```

## 本轮新增代码结论

- `scripts/run_gat_ezkl_baseline.py` 未新增 TODO、stub、dummy、fake、placeholder 或未接通函数。
- `scripts/collect_gat_ezkl_results.py` 未新增 TODO、stub、dummy、fake、placeholder 或未接通函数。
- `scripts/validate_final_results.py` 未新增 TODO、stub、dummy、fake、placeholder 或未接通函数。
- `scripts/check_gpu_backend.py` 未出现禁用词命中。
- `scripts/package_gpu_experiment.py` 未出现禁用词命中。
- `configs/ezkl_routes.json` 未出现禁用词命中。
- `cpp/gat_ezkl_adapter/` 未新增禁用词命中。
- `README.md` 未出现禁用词命中。
- `src/native_gat.rs` 未新增禁用词命中。
- `src/execute.rs` 仍只存在 upstream 既有 TODO 命中，与本轮 GPU route 无关。

## 剩余命中分类

| 路径 | 分类 | 对 strict CPU/GPU route 的影响 |
| --- | --- | --- |
| `src/bin/py_stub_gen.rs` | upstream Python typing stub generator | 无 |
| `src/bindings/python.rs` | upstream `pyo3_stub_gen` 注解与绑定元数据 | 无 |
| `src/execute.rs` | upstream 注册验证键查询 TODO | 无 |
| `src/commands.rs` | upstream 命令行 placeholder 样式与 dummy SRS 帮助文本 | 无 |
| `src/eth.rs` | upstream VKA digest TODO | 无 |
| `src/graph/model.rs` | upstream generic dummy layout 估算机制 | 无 |
| `src/graph/mod.rs` | upstream generic graph loading placeholder/dummy 注释 | 无 |
| `src/graph/vars.rs` | upstream dummy vars helper | 无 |
| `src/tensor/var.rs` | upstream tensor placeholder/dummy helper | 无 |
| `src/circuit/ops/` | upstream generic op dummy layout 与未实现分支 | 无 |
| `src/circuit/modules/` | upstream planner TODO 与 Poseidon spec 未实现分支 | 无 |

## 禁用项确认

当前正式 CPU 结果与 GPU route 尝试均未引入：

- output-binding replacement
- digest-only output
- folding
- identity route
- precomputed logits bypass
- subset
- shard
- sample
- toy、smoke、debug 数据集
- fake verify
- dummy proof
- placeholder GPU 成功路径

GPU 未验证状态以 `results/gpu_route_status.md` 和 `runs/<dataset>/gpu/run_manifest.json` 记录。
