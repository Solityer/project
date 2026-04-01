# TEST_PLAN

## 主测试原则

- 主流程测试必须基于全量真实 Cora
- 不允许用 toy / mock / sampled graph 替代主验证
- 小规模 synthetic 只保留纯协议回归和纯函数单测

## 已有测试

### Full Cora

- `full_graph_config_parse`
  - 目标：验证正式 `cora_full.cfg`
- `real_checkpoint_bundle_loads_multihead_parameters`
  - 目标：验证真实 bundle 读取
- `reference_style_multihead_forward_shapes`
  - 目标：验证 hidden/output 多头结构 shape
- `reference_style_multihead_forward_matches_reference_artifacts`
  - 目标：验证 full Cora 多头前向 parity
- `full_cora_edges_are_dst_sorted_and_self_looped`
  - 目标：验证 full Cora 边按 `dst` 排序且每节点有 self-loop
- `full_cora_bias_matches_reference`
  - 目标：验证 bias 与 reference 一致

### 协议回归

- `prove_verify_round_trip`
- `tampered_witness_fails`
- `selector_padding_consistency`
- `transcript_order_consistency`
- `agg_witness_commitment_opening_consistency`

## 本轮后续待补

- hidden per-head formal object existence tests
- `H_cat / H_cat_star` object tests
- `H_C` domain tests
- output `Y'_star / Y_star / PSQ_out` object tests
- proof block order tests for `M_pub / Com_dyn / S_route / Eval_ext / Eval_dom / Com_quot / Open_dom / W_ext / Pi_bind`
