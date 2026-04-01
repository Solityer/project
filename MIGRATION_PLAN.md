# MIGRATION_PLAN

## 目标

把 `project/` 当前仍残留单头 formal 假设的实现，迁移为与 `GAT/` 官方单层 GAT 语义一致、并与 [GAT-ZKML-单层多头.md](/home/pzh/project/GAT-ZKML-单层多头.md) 对齐的版本：

- hidden: `K_hid = 8`
- output: `K_out = 1`
- hidden 每头聚合后 `ELU`
- 8 头 hidden 输出按特征维拼接成 `H_cat`
- output 层是 attention head，激活为恒等
- 七个工作域显式存在：`H_FH / H_edge / H_in / H_d_h / H_cat / H_C / H_N`

## 官方真值入口

- `GAT/models/gat.py`
- `GAT/utils/layers.py`
- `GAT/execute_cora.py`

## 当前 project 热区

- 模型/前向：
  - `include/gatzk/model/gat.hpp`
  - `src/model/gat.cpp`
- formal 轨迹/挑战/证明/验证：
  - `src/protocol/trace.cpp`
  - `src/protocol/challenges.cpp`
  - `src/protocol/quotients.cpp`
  - `src/protocol/prover.cpp`
  - `src/protocol/verifier.cpp`

## 已完成

- 真实 checkpoint bundle 已固化到 `project/artifacts/checkpoints/cora_gat`
- full Cora reference forward 已固化到 `project/runs/cora_full/reference`
- C++ 多头 reference-style 前向已与 full Cora reference artifact 对齐
- challenge label 接口已开始从单头静态列表转成 context-aware scaffold

## 核心差异

### 1. 前向层

- 已基本对齐：
  - hidden 8 头
  - hidden concat
  - output 1 头
  - self-loop / bias / softmax / normalization
- 仍待工程对象化：
  - `H_cat`
  - `H_cat_star`
  - `Y'`
  - `Y'_star`
  - `Y'_star_edge`
  - `widehat_y_star`
  - `Y`
  - `Y_star`
  - `Y_star_edge`
  - `PSQ_out`

### 2. formal 层

- 当前仍是单头硬编码：
  - `P_H_prime`
  - `P_E_src / P_E_dst`
  - `P_H_star`
  - `P_H_agg_star`
  - `P_Y_lin / P_Y`
  - `P_a_out / P_b_out / P_Acc_out`
- 当前缺失：
  - `H_cat` 域绑定
  - `H_C` 类别域绑定
  - output 专属 route / PSQ / quotient
  - delayed finalize for hidden dst-route and output dst-route
  - note-defined proof object order and metadata block

## 文件级计划

### 第一批

- `tests/test_main.cpp`
  - 补 full Cora 排序/self-loop/bias/reference parity 测试
- `MIGRATION_PLAN.md`
- `NOTE_TO_CODE_MAP.md`
- `FORWARD_ALIGNMENT_REPORT.md`
- `TEST_PLAN.md`
- `PROGRESS_LOG.md`

### 第二批

- `include/gatzk/protocol/proof.hpp`
  - 显式扩展工作域、metadata、proof block 骨架
- `src/protocol/trace.cpp`
  - 按 hidden heads / concat / output 三阶段重排对象生成

### 第三批

- `src/protocol/challenges.cpp`
  - 按笔记顺序重排 transcript
- `src/protocol/quotients.cpp`
  - 新增 `t_cat`、`t_C`
- `src/protocol/prover.cpp`
  - proof block 顺序、metadata、domain opening 编排
- `src/protocol/verifier.cpp`
  - 按 fixed-order / metadata / multi-domain 回放验证

## 风险

- formal 层当前仍依赖单头对象命名，重构跨度大
- `H_C` 与 `H_cat` 的引入会影响 quotient / opening / proof serialization 全链
- output 层不再是 affine 输出，旧 `Y_lin / a_out / b_out / Acc_out` 语义需要整体替换
