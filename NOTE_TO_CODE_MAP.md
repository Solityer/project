# NOTE_TO_CODE_MAP

## 映射原则

- `GAT/` 提供官方前向语义真值
- `project/GAT-ZKML-单层多头.md` 提供协议对象、挑战顺序、工作域、证明顺序真值
- `project/src` 是正式实现落点

## 章节到代码映射

| 笔记章节 | 对象/语义 | project 落点 | 当前状态 |
| --- | --- | --- | --- |
| 0.6.2 / 0.6.3 | hidden 8 heads 的 `H' / E_src / E_dst / S / Z / M / U / alpha / H_agg` | `include/gatzk/model/gat.hpp`, `src/model/gat.cpp` | 已实现 |
| 0.6.4 / 2.3 | `H_cat / H_cat_star` | `src/model/gat.cpp`, `src/protocol/trace.cpp` | `H_cat` 明文已实现，formal 对象仍待实现 |
| 0.6.5 / 2.4 | output `Y' / E_src(out) / E_dst(out) / Y / Y_star / PSQ_out` | `src/model/gat.cpp`, `src/protocol/trace.cpp`, `src/protocol/quotients.cpp` | 明文 `Y' / Y` 已实现，formal output 对象大部分待实现 |
| 0.7.1 | 七个工作域 `FH / edge / in / d_h / cat / C / N` | `include/gatzk/protocol/proof.hpp`, `src/protocol/prover.cpp` | `FH / edge / in / d / N` 已有，`cat / C` 待实现 |
| 2.2.10 | hidden dst-route delayed finalize | `src/protocol/trace.cpp` | 待实现 |
| 2.3 | concat witness / bind | `src/protocol/trace.cpp`, `src/protocol/verifier.cpp` | 待实现 |
| 2.4 | output witness / bind / PSQ_out | `src/protocol/trace.cpp`, `src/protocol/verifier.cpp`, `src/protocol/quotients.cpp` | 待实现 |
| 3.3.11 | `t_cat` | `src/protocol/quotients.cpp` | 待实现 |
| 3.3.12 | `t_C` | `src/protocol/quotients.cpp` | 待实现 |
| 3.5 | `M_pub / Com_dyn / S_route / Eval_ext / Eval_dom / Com_quot / Open_dom / W_ext / Pi_bind` | `include/gatzk/protocol/proof.hpp`, `src/protocol/prover.cpp`, `src/protocol/verifier.cpp` | 待实现 |

## 当前单头硬编码点

- `src/protocol/trace.cpp`
  - 只有单套 `P_H_prime / P_E_src / P_E_dst / P_H_star / P_H_agg_star / P_Y_lin / P_Y`
- `src/protocol/quotients.cpp`
  - 只有 `t_FH / t_edge / t_in / t_d / t_N`
- `src/protocol/verifier.cpp`
  - 只按单头 bundle label 回放

## 本轮新增

- full Cora multi-head forward parity 测试
- context-aware dynamic commitment label scaffold
- attention self-loop aggregation 修正

## 仍待实现

- `H_cat_star`
- `H_C`
- `Y'_star / Y'_star_edge / widehat_y_star / Y_star / Y_star_edge`
- `PSQ_out`
- `t_cat / t_C`
- metadata `M_pub`
