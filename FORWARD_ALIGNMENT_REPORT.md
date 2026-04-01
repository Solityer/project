# FORWARD_ALIGNMENT_REPORT

## 对齐对象

对齐真值：

- 官方语义：`GAT/models/gat.py` + `GAT/utils/layers.py`
- project reference artifacts：`runs/cora_full/reference`

## 当前结论

`project` 中的真实多头 reference-style 前向，已经与 `runs/cora_full/reference` 对齐。

## 已对齐的关键语义

- hidden 8 个 attention heads
- hidden head 聚合后 `ELU`
- 8 个 hidden heads 沿特征维拼接为 `H_cat`
- output 层是单个 attention head
- output 激活为恒等
- self-loop 已纳入 attention 归一化与聚合
- `bias` 与 reference artifact 一致
- `dst` 排序与显式 self-loop 工程规则已通过 full Cora 测试

## 已对齐的关键对象

- `H`
- hidden `H_prime / E_src / E_dst / S / Z / M / Delta / U / Sum / inv / alpha / H_agg`
- `hidden_concat`
- output `H_prime / E_src / E_dst / S / Z / M / Delta / U / Sum / inv / alpha / H_agg`
- `Y_lin`
- `Y`

## 目前仍未完成的 formal 对齐

- `H_cat` 尚未进入正式协议对象系统
- `H_cat_star` 尚未显式落地
- `H_C` 尚未落地
- output formal 对象 `Y'_star / Y'_star_edge / widehat_y_star / Y_star / Y_star_edge / PSQ_out` 尚未落地
- verifier / quotients 仍是单头口径

## 误差口径

- reference artifact 是 `float32` 落盘
- parity 测试按 `float32` artifact 口径比较
- 当前 full Cora parity 已通过
