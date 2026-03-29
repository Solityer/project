# Full-Graph Mainline

正式配置：

- `configs/cora_full.cfg`

full-graph 语义要求：

1. `local_nodes == dataset.num_nodes`
2. `extract_local_subgraph()` 走 full-graph 快路径
3. `context.local.absolute_ids` 保持自然顺序
4. benchmark 日志中 `is_full_dataset=true`
5. `notes` 中包含 `dataset_scope=full`

协议层主对象：

- 公开对象：`P_I`, `P_src`, `P_dst`, `P_Q_new_edge`, `P_Q_end_edge`, `P_Q_edge_valid`, `P_Q_N`, `P_Q_proj_valid`, `P_Q_d_valid`
- 动态对象：`P_H`, `P_H_prime`, `P_E_src`, `P_E_dst`, `P_M`, `P_Sum`, `P_inv`, `P_H_agg`, `P_H_agg_star`, `P_Y` 及 lookup / route / zkMaP / PSQ 辅助对象
- quotient：`t_FH`, `t_edge`, `t_in`, `t_d`, `t_N`

当前 route 主线已经切到双累加器：

- src: `P_Table_src`, `P_m_src`, `P_R_src_node`, `P_Query_src`, `P_R_src`
- dst: `P_Table_dst`, `P_m_dst`, `P_R_dst_node`, `P_Query_dst`, `P_R_dst`
- route 终值通过 `S_src`, `S_dst` 显式进入 proof 并被 verifier 使用

真实来源脚本：

- `scripts/export_gat_checkpoint.py`
- `scripts/gat_reference.py`
- `scripts/checkpoint_reader_compat.py`

当前 `project` 内已经沉淀的真实来源闭环包括：

1. 从 `../GAT/pre_trained/cora/mod_cora.ckpt` 导出稳定中间格式；
2. 读取真实 Cora Planetoid 数据并复现原始 TensorFlow GAT 的全图 reference forward；
3. 在 `runs/cora_full/reference/` 下导出可用于 parity 的中间对象。

正式入口硬闸门：

1. `configs/cora_full.cfg` 不允许 synthetic 参数 fallback；
2. `build_context()` 会强制检查 `checkpoint_bundle/manifest.json`；
3. 若真实 bundle 不是“单隐藏头 + 仿射输出”结构，则直接失败退出；
4. 当前原始 GAT 真实 bundle 报告 `hidden_head_count=8` 且包含 `output_head`，因此正式 prove / verify 还不能被诚实标记为已闭口。

调试说明：

- 为了维持协议对象回归测试，测试代码仍保留 `allow_synthetic_model=true` 的 debug-only 配置。
- 该 debug 配置不再是正式入口，也不应出现在正式 README 命令链中。
