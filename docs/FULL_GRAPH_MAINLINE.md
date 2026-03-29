# Full-Graph Mainline

正式配置：

- `configs/cora_full.cfg`
- `configs/citeseer_full.cfg`
- `configs/pubmed_full.cfg`

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
