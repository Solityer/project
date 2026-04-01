# PROGRESS_LOG

## 2026-03-31

### 已完成

1. 读取并确认官方 GAT 前向入口：
   - `GAT/models/gat.py`
   - `GAT/utils/layers.py`
2. 读取并确认笔记中的目标结构：
   - hidden 8 heads
   - concat
   - output attention head
   - `H_cat / H_C / PSQ_out / fixed proof order`
3. 校正 `project` 的多头明文前向：
   - 修正 self-loop 参与 attention 聚合
   - 与 full Cora reference artifact 做 parity
4. 扩展测试：
   - `.npy` 读取
   - full Cora parity
   - full Cora 排序/self-loop/bias 规则
5. 给 formal 层加了 context-aware challenge label scaffold

### 本轮涉及文件

- `src/model/gat.cpp`
- `tests/test_main.cpp`
- `include/gatzk/protocol/proof.hpp`
- `include/gatzk/protocol/challenges.hpp`
- `src/protocol/challenges.cpp`
- `src/protocol/prover.cpp`

### 当前测试结果

- `gatzk_tests` 全通过
- full Cora multi-head forward parity 已通过

### 剩余主问题

- formal `trace / quotients / verifier` 仍是单头对象体系
- `H_cat / H_C / output-specific formal objects` 尚未进入正式 proof pipeline
