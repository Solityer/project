# Note To Code Map

## Implemented now
- `0.3 图、局部子图与排序约定` -> `src/data/loader.cpp`, `tests/test_main.cpp`
  - Full Cora edges are stable-sorted by `dst`, and every node has an explicit self-loop.
- `0.5 模型参数` -> `scripts/export_gat_checkpoint.py`, `src/model/gat.cpp`
  - Real checkpoint bundle loads `8` hidden heads and `1` output attention head.
- `0.6.2/0.6.3 隐藏层变量` -> `src/model/gat.cpp`, `tests/test_main.cpp`
  - `H_prime`, `E_src`, `E_dst`, `S`, `Z`, `M`, `Delta`, `U`, `Sum`, `inv`, `alpha`, `H_agg` are checked against full-Cora reference artifacts.
- `0.6.4 拼接阶段变量` -> `src/model/gat.cpp`
  - Hidden-head ELU outputs are concatenated into `hidden_concat`.
- `0.6.5 输出层变量` -> `src/model/gat.cpp`, `tests/test_main.cpp`
  - Output head `H_prime/E_src/E_dst/.../Y_lin/Y` matches the local reference export.
- `0.7 工作域` -> `src/protocol/prover.cpp`, `include/gatzk/protocol/proof.hpp`
  - `H_cat` and `H_C` now exist as explicit work domains inside formal context metadata.
- `0.8.5 元数据字段表` -> `include/gatzk/protocol/proof.hpp`, `src/protocol/prover.cpp`, `src/protocol/verifier.cpp`
  - `M_pub` is explicit and verifier-checked.

## Added this round but still incomplete
- `3.5 最终证明对象` -> `include/gatzk/protocol/proof.hpp`, `src/protocol/prover.cpp`
  - Fixed proof block order is explicit in `Proof::block_order`.
- `4.2/4.3 公开对象重建与挑战重放` -> `src/protocol/verifier.cpp`, `src/protocol/challenges.cpp`
  - Metadata and proof-block order checks are live, but multi-head transcript replay is not finished.

## Still pending
- `2.3 拼接阶段`
  - Formal `H_cat` and `H_cat_star` witness columns.
- `2.4 输出层的完整见证生成`
  - Formal `Y'_star`, `Y'_star_edge`, `widehat_y_star`, `Y_star`, `Y_star_edge`, `PSQ_out`.
- `3.3.11/3.3.12`
  - `t_cat` and `t_C`.
- `4.5/4.6`
  - Seven-domain quotient verification and tensor binding checks under the multi-head layout.
