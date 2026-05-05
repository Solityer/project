# 代码洁净审计

## 审计时间

2026-05-05

## 当前正式路线

- `ezkl_gat_native_compressed_csr_attention_full`

## 检查命令

```bash
grep -R "TODO\|FIXME\|HACK\|XXX\|stub\|dummy\|fake\|placeholder\|temporary\|debug-only\|not implemented\|unimplemented!\|todo!\|panic!(\"TODO" -n src cpp scripts configs README.md || true
```

## 本实验新增和维护代码结论

- `scripts/run_gat_ezkl_baseline.py` 未新增 TODO、stub、dummy、fake、placeholder 或未接通函数。
- `scripts/collect_gat_ezkl_results.py` 未新增 TODO、stub、dummy、fake、placeholder 或未接通函数。
- `scripts/validate_final_results.py` 未新增 TODO、stub、dummy、fake、placeholder 或未接通函数。
- `configs/ezkl_routes.json` 未出现禁用词命中。
- `cpp/gat_ezkl_adapter/` 的 strict route 代码未新增禁用词命中。
- `README.md` 未出现禁用词命中。
- `src/native_gat.rs` 的 strict route 代码未新增禁用词命中。

## 剩余命中分类

以下命中来自 upstream ezkl 通用代码或通用测试/绑定辅助，不属于本实验 strict route 新增代码：

| 路径 | 分类 | 对 strict CPU route 的影响 |
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

当前正式结果未引入：

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
- placeholder 成功路径
