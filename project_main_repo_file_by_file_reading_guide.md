# `Solityer/project` 主仓库逐文件阅读与理解文档

> 目标：帮助你逐个文件阅读并理解 `project` 主仓库的代码方法、整体逻辑、文件内容与每个文件在正式 `GAT-ZKML` 主线中的作用。  
> 范围：聚焦主仓库正式实现，即多层多头 `GAT` 的 formal ZKML proving / verifying 主线。对比实验目录、第三方源码、原始数据大文件、运行缓存文件不逐行展开，只说明它们在主线中的位置。

---

## 0. 先建立总图：这个仓库到底在做什么

本仓库实现的是一个面向多层多头 `GAT` 的正式 `ZKML` 证明系统。它不是单纯跑一个 GNN，也不是继续探索多个协议版本，而是在一套固定 formal 语义下，把：

1. 图数据；
2. 多层多头 GAT 前向过程；
3. lookup 约束；
4. route / PSQ / state machine 关系；
5. quotient 多项式关系；
6. KZG commitment / opening；
7. proof assembly；
8. verifier replay；
9. benchmark artifact；

串成一条可复核、可复现、可验证、可导出的正式主线。

主执行链是：

```text
README / formal note / configs
  ↓
src/main.cpp
  ↓
util::load_config
  ↓
protocol::build_context
  ↓
protocol::build_trace
  ↓
protocol::prove
  ↓
protocol::verify
  ↓
protocol::proof_size_bytes
  ↓
protocol::export_run_artifacts
  ↓
runs/<dataset>/warm/run_manifest.json
  ↓
scripts/export_benchmark_table.py
  ↓
runs/benchmarks/latest.{json,csv}
  ↓
runs/benchmarks/summary.md
```

---

## 1. 阅读方法总原则

### 1.1 不要先从最大文件开始

`src/protocol/trace.cpp`、`src/protocol/prover.cpp`、`src/protocol/verifier.cpp` 这类文件很重，直接读会很容易迷失。正确顺序是：

```text
formal 文档
  ↓
构建入口
  ↓
头文件对象边界
  ↓
底层实现：util / algebra / crypto
  ↓
data / model
  ↓
protocol schema / challenge / lookup / PSQ / zkmap
  ↓
trace
  ↓
quotients
  ↓
prover
  ↓
verifier
  ↓
configs / scripts / tests / artifacts
```

### 1.2 始终区分三类对象

| 类型 | 代表文件 | 含义 |
|---|---|---|
| 语义真源 | `GAT-ZKML-多层多头.md` | 定义协议对象、顺序、方程，不应被实现文件重新定义 |
| 实现真源 | `include/` + `src/` | 把 formal 语义落成可执行对象 |
| 复核真源 | `runs/` + `tests/` + `scripts/export_benchmark_table.py` | 检查结果、artifact、README、benchmark 是否一致 |

### 1.3 读代码时必须反复问四个问题

1. 这个文件定义的是协议语义，还是执行层优化？
2. 它产生的是静态对象，还是一次运行中的动态 witness？
3. 它影响 proof shape / transcript / verifier equation 吗？
4. 它的结果最终会进入 proof、manifest、benchmark，还是只用于辅助计算？

---

## 2. 推荐阅读顺序总表

```text
1. README.md
2. GAT-ZKML-多层多头.md
3. 项目整体逻辑与方法流程.md
4. project_main_repo_full_logic_readme.md
5. docs/热点优化方法与处理逻辑.md
6. CMakeLists.txt
7. cmake/ConfigureMCL.cmake
8. include/gatzk/**/*.hpp / *.cuh
9. src/util/*.cpp
10. src/algebra/*.cpp / *.cu
11. src/crypto/*.cpp
12. src/data/loader.cpp
13. src/model/gat.cpp
14. src/protocol/schema.cpp
15. src/protocol/challenges.cpp
16. src/protocol/lookup.cpp
17. src/protocol/psq.cpp
18. src/protocol/zkmap.cpp
19. src/protocol/trace.cpp
20. src/protocol/quotients.cpp
21. src/protocol/prover.cpp
22. src/protocol/verifier.cpp
23. src/main.cpp
24. configs/*.cfg
25. scripts/*
26. tests/test_main.cpp
27. reference/*
28. runs/* 与 artifacts/*
```

---

# 3. 根目录文档与构建文件

## `README.md`

### 内容概览

这是项目当前正式说明文档。它说明：

- 项目定位：多层多头 `GAT` 的 formal ZKML 证明系统。
- 一套 formal 语义同时支撑 `full_cpu` 和 `full_gpu`。
- 硬约束：不能改 `proof schema`、`transcript` 顺序、serializer/parser、verifier equations、public metadata 契约。
- 当前调用链：`src/main.cpp → build_context → build_trace → prove → verify → proof_size_bytes → export_run_artifacts`。
- 五个正式数据集与 CPU/GPU 复现命令。
- 正式结果表只保留四项指标：
  - `commitment_time_ms`
  - `prove_time_ms`
  - `verify_time_ms`
  - `proof_size_bytes`
- `proof_size_bytes` 的统计口径。
- benchmark 与 artifact 的 truth source。

### 作用

`README.md` 是对外展示面，不重新定义协议。读它是为了知道当前仓库“正式承认”的结果、命令、口径和边界。

---

## `GAT-ZKML-多层多头.md`

### 内容概览

这是唯一 formal 语义真源。它定义：

- 多层多头 GAT family。
- hidden layer 与 output layer 的数学语义。
- 有限域、有符号解释、量化尺度。
- 图、batch、edge sort、self-loop、node/edge pointer 约定。
- 静态表：feature、LeakyReLU、ELU、exp、range。
- 模型参数：
  - hidden heads；
  - output heads；
  - 投影矩阵；
  - attention 向量；
  - output bias。
- 中间变量：
  - `H_prime`
  - `E_src`
  - `E_dst`
  - `S`
  - `Z`
  - `M`
  - `Delta`
  - `U`
  - `Sum`
  - `inv`
  - `alpha`
  - `H_agg_pre`
  - `H_agg`
  - `H_cat`
  - output logits。
- lookup、route、PSQ、state machine。
- quotient 关系。
- KZG opening。
- public metadata。
- transcript 吸收顺序。
- verifier equations。

### 作用

所有代码都应该服从这个文件。任何实现优化都不能改变这里定义的对象、名字、顺序和验证方程。

---

## `项目整体逻辑与方法流程.md`

### 内容概览

这是中文整体逻辑导读。它从方法角度解释：

- 项目证明什么。
- 一次正式运行从哪里开始、到哪里结束。
- 静态上下文与运行轨迹的区别。
- 为什么 CPU/GPU 是同一套 formal 语义下的两条执行路径。
- 运行流程十二步：
  1. 读取配置；
  2. 装入图数据和参数包；
  3. 构造静态上下文；
  4. 构造运行轨迹；
  5. 生成动态承诺；
  6. 重放挑战值；
  7. 计算 quotient；
  8. 构造 opening bundle；
  9. 组装 proof；
  10. verify；
  11. export artifacts；
  12. 汇总 benchmark。

### 作用

这是读源码前最重要的中文地图。它不定义新协议，但能帮助理解 `build_context`、`build_trace`、`prove`、`verify` 的职责边界。

---

## `project_main_repo_full_logic_readme.md`

### 内容概览

这是已有的主仓库完整代码逻辑阅读 README。它按正式执行链列出：

- 阅读目标；
- 仓库结构；
- 文档阅读顺序；
- 构建文件；
- 头文件阅读顺序；
- 实现文件阅读顺序；
- 配置；
- 脚本；
- 测试；
- reference；
- runs/artifacts；
- 推荐阅读顺序总表。

### 作用

它是本文件的上游基础。本文件在它的基础上进一步补充“每一个文件的内容和作用”。

---

## `CMakeLists.txt`

### 内容概览

这是主构建入口。它定义：

- CMake 最低版本。
- C++ 标准。
- 是否构建 tests。
- 是否启用 CUDA backend。
- 是否默认启用 route2 优化开关。
- crypto backend，目前只支持 `mcl`。
- `GATZK_SOURCES` 主库源码清单。
- `gatzkml` 静态库。
- `gatzk_run` 可执行文件。
- `gatzk_tests` 测试可执行文件。

### 作用

它是“哪些源码真正进入主库”的构建真源。读源码时，先以这里的 `GATZK_SOURCES` 为主线。

---

## `cmake/ConfigureMCL.cmake`

### 内容概览

该文件负责配置 `mcl` 后端：

- 设置 `GATZK_USE_MCL`。
- 优先尝试 vendored `third_party/mcl`。
- 如果 vendored 不存在，则尝试系统安装的 mcl。
- 明确拒绝非 `mcl` 后端。
- 配置 include path 和 link library。

### 作用

它决定 crypto/backend 能否构建。构建失败时，优先检查这里的 mcl 查找逻辑。

---

## `docs/热点优化方法与处理逻辑.md`

### 内容概览

这是性能优化与 retained/reverted 方法记录。它强调：

- 唯一 formal note 不可改。
- 只能做执行层优化，不能改 protocol。
- benchmark 只接受 `full-dataset + warm + verified + 官方 artifact 真源`。
- retained patch 必须保持：
  - CPU correctness；
  - GPU correctness；
  - proof contract；
  - `proof_size_bytes` 稳定。
- 当前热点集中在 PPI / ogbn-arxiv 的 trace generation 与 dynamic commitment。

### 作用

读 CUDA、route2、cache、layout 优化时必须对照它。它告诉你哪些优化是主线保留，哪些被回退。

---

# 4. 头文件逐文件说明

## 4.1 Algebra 头文件

### `include/gatzk/algebra/field.hpp`

#### 内容概览

定义 `FieldElement`，封装 mcl 的有限域元素操作，包括：

- 零元、单位元；
- signed/unsigned 构造；
- 加减乘除；
- 幂运算；
- 逆元；
- 序列化/字符串化；
- mcl native value 访问。

#### 作用

所有 trace、polynomial、KZG、quotient 的底层数值都依赖它。它是有限域语义进入 C++ 的核心入口。

---

### `include/gatzk/algebra/domain_evaluation.hpp`

#### 内容概览

定义 root-of-unity domain 与域上求值辅助对象，通常包含：

- domain size；
- generator / omega；
- inverse size；
- barycentric weight；
- rotation shift；
- zero polynomial evaluation；
- domain point cache key。

#### 作用

为 evaluation-form polynomial、domain opening、quotient evaluation 提供公共域工具。

---

### `include/gatzk/algebra/polynomial.hpp`

#### 内容概览

定义 `Polynomial` 对象及其表示方式：

- coefficient basis；
- evaluation basis；
- shared/spilled evaluation；
- from coefficients；
- from evaluations；
- evaluate at point；
- domain conversion；
- degree/size metadata。

#### 作用

承接 trace column、public polynomial、static table、quotient polynomial，是 KZG commit/open 的直接输入。

---

### `include/gatzk/algebra/eval_backend.hpp`

#### 内容概览

定义 packed evaluation backend 与批量求值接口，服务：

- 多项式批量求值；
- domain 上的 packed evaluation；
- rotated-point kernel；
- 大域上的热点求值；
- verifier public evaluation cache。

#### 作用

性能层核心接口。它不改变多项式语义，只改变域上求值的组织方式。

---

### `include/gatzk/algebra/vector_ops.hpp`

#### 内容概览

声明 field vector 操作：

- 向量加减乘；
- dot product；
- batch inverse；
- matrix flatten；
- vector transform；
- CPU/CUDA backend availability；
- backend route 名称。

#### 作用

为 trace materialization、projection、commitment input、quotient 等提供通用向量运算。

---

### `include/gatzk/algebra/cuda_field.cuh`

#### 内容概览

声明 CUDA field backend 需要的接口或设备侧结构：

- CUDA field element 表示；
- host/device 转换；
- kernel 使用的 field 运算入口；
- CUDA hotspot glue。

#### 作用

仅在 CUDA build 下参与。它属于执行层优化，不定义第二套 formal 语义。

---

## 4.2 Crypto 头文件

### `include/gatzk/crypto/curve.hpp`

#### 内容概览

定义椭圆曲线群对象：

- `G1Point`；
- `G2Point`；
- point add/sub/mul；
- generator；
- same-base batch multiplication；
- pairing check；
- backend name。

#### 作用

KZG commitment 和 witness 都以曲线点为载体。该文件是 crypto 层群运算接口。

---

### `include/gatzk/crypto/kzg.hpp`

#### 内容概览

定义 KZG 相关对象和接口：

- SRS / key pair；
- commitment；
- commit polynomial；
- commit tau evaluation；
- open；
- verify opening；
- batch/folded opening support。

#### 作用

证明系统把多项式承诺给 verifier 的核心密码学组件。

---

### `include/gatzk/crypto/transcript.hpp`

#### 内容概览

定义 Fiat-Shamir transcript：

- 初始化 domain separator；
- absorb scalar；
- absorb commitment；
- derive challenge；
- challenge label 管理。

#### 作用

证明方和验证方必须用同一 transcript 顺序得到同一挑战值。它与 `src/protocol/challenges.cpp` 紧密配合。

---

## 4.3 Data / Model 头文件

### `include/gatzk/data/dataset.hpp`

#### 内容概览

定义数据结构：

- `Edge`；
- `GraphDataset`；
- `LocalGraph`；
- `PublicInput`；
- features；
- labels；
- node_ptr；
- edge_ptr；
- graph_count；
- task/report metadata。

#### 作用

把不同来源的数据集统一成 formal 主线可消费的图结构。

---

### `include/gatzk/data/loader.hpp`

#### 内容概览

声明数据加载与规范化函数：

- `load_dataset`；
- `normalize_graph_input`。

#### 作用

把 Planetoid / PPI / ogbn-arxiv 缓存读入，并按 formal 约束补自环、排序、构造 batch 指针。

---

### `include/gatzk/model/gat.hpp`

#### 内容概览

定义模型相关结构：

- hidden layer shape；
- attention head parameters；
- output layer parameters；
- model parameters；
- dense tensor；
- checkpoint manifest 对象；
- forward trace；
- forward profile；
- 模型加载函数；
- synthetic model builder；
- GAT reference forward。

#### 作用

把 checkpoint bundle / synthetic config 转为正式 GAT 前向对象，并产出 trace 阶段要使用的中间量。

---

## 4.4 Util 头文件

### `include/gatzk/util/config.hpp`

#### 内容概览

定义 `AppConfig`：

- dataset；
- data/cache/export path；
- checkpoint bundle；
- task/report；
- batching/subgraph/self-loop/edge-sort/chunking rules；
- hidden profile；
- d_in_profile；
- K_out；
- range_bits；
- model/static/quant/degree ids；
- runtime flags。

#### 作用

配置对象是正式运行的入口约束。它决定后续数据、模型、domain、metadata、artifact 的形状。

---

### `include/gatzk/util/logging.hpp`

#### 内容概览

声明日志工具：

- logging level；
- debug/info 输出；
- 可能的 scoped output control。

#### 作用

辅助调试。正式控制台输出已经在 `src/main.cpp` 收口成一行，所以 logging 不应成为正式结果真源。

---

### `include/gatzk/util/route2.hpp`

#### 内容概览

定义 route2 执行层开关：

- fast MSM；
- parallel FFT；
- FFT backend upgrade；
- FFT kernel upgrade；
- trace layout upgrade；
- fast verifier pairing；
- CUDA trace hotspots；
- CUDA field hotspots；
- route label / notes。

#### 作用

控制执行后端与优化路径，但不应改变 proof schema、transcript、metadata 或 verifier equations。

---

## 4.5 Protocol 头文件

### `include/gatzk/protocol/proof.hpp`

#### 内容概览

定义 protocol 核心数据契约：

- `StaticTables`；
- `WorkDomains`；
- `PublicMetadata`；
- `ProtocolContext`；
- `TraceArtifacts`；
- `DomainOpeningBundle`；
- `Proof`；
- `RunMetrics`。

#### 作用

这是 artifact contract 的核心真源。证明对象、公开元数据、运行指标、trace 产物都在这里定义边界。

---

### `include/gatzk/protocol/schema.hpp`

#### 内容概览

声明 schema/label 函数：

- `proof_block_order()`；
- hidden head prefix；
- hidden concat label；
- output head label；
- output challenge / eval / witness label；
- single output compatibility labels。

#### 作用

标签与顺序是 proof contract 的一部分。该文件决定 proof/parser/verifier 共同使用的命名入口。

---

### `include/gatzk/protocol/challenges.hpp`

#### 内容概览

声明 challenge 与 metadata 相关函数：

- `canonical_public_metadata`；
- `absorb_public_metadata`；
- `dynamic_commitment_labels`；
- `quotient_commitment_labels`；
- `domain_opening_labels`；
- `replay_challenges`。

#### 作用

这是 transcript 与 public metadata 的统一入口。prove 和 verify 都必须复用它。

---

### `include/gatzk/protocol/lookup.hpp`

#### 内容概览

声明 lookup 构造工具：

- table/query encoding；
- multiplicity；
- accumulator；
- selector mask；
- feature/activation/range/exp lookup helpers。

#### 作用

把非线性、范围、特征读取等关系转为可验证 lookup 对象。

---

### `include/gatzk/protocol/psq.hpp`

#### 内容概览

声明 PSQ 相关构造函数：

- `w_psq`；
- `T_psq`；
- `T_psq_edge`；
- `PSQ`；
- challenge lambda 参与的组合逻辑。

#### 作用

约束图 attention 聚合中的加权压缩关系，是 route/aggregation proof 的关键组件之一。

---

### `include/gatzk/protocol/zkmap.hpp`

#### 内容概览

声明 map-style binding / route helper：

- key/value packing；
- route relation helper；
- binding accumulator；
- node-edge 映射对象。

#### 作用

把节点域与边域之间的广播、回收、绑定关系整理成可证明对象。

---

### `include/gatzk/protocol/quotients.hpp`

#### 内容概览

声明各工作域 quotient 计算接口：

- FH；
- edge；
- input dim；
- hidden dim；
- concat；
- class；
- node；
- public eval 与 opened eval 接口。

#### 作用

quotient 是“约束是否成立”的中心表达。该文件定义 prover/verifier 共享的 quotient 计算边界。

---

### `include/gatzk/protocol/trace.hpp`

#### 内容概览

声明：

- `build_trace`；
- trace artifact 构造入口；
- 可能的 trace helpers。

#### 作用

`build_trace` 把前向计算翻译成 dynamic witness、lookup、route、PSQ、state machine 对象。

---

### `include/gatzk/protocol/prover.hpp`

#### 内容概览

声明 proving 入口：

- `build_context`；
- `prove`；
- `proof_size_bytes`；
- `export_run_artifacts`。

#### 作用

prover 侧主接口。`src/main.cpp` 通过这里进入静态上下文、证明生成和产物导出。

---

### `include/gatzk/protocol/verifier.hpp`

#### 内容概览

声明：

- `verify`。

#### 作用

验证方入口。它不重新定义规则，而是复用 schema、metadata、challenge、quotient/opening 真源。

---

# 5. 实现文件逐文件说明

## 5.1 Util 层

### `src/util/config.cpp`

#### 内容概览

实现 `.cfg` 解析：

- 删除注释与空白；
- 解析 `key = value`；
- 支持字段白名单；
- 拒绝 legacy 字段；
- 解析 bool；
- 解析 `hidden_profile`；
- 解析 `d_in_profile`；
- 校验 `L = hidden_profile.size() + 1`；
- 校验 `d_in_profile[layer+1] = hidden_profile[layer].head_count * hidden_profile[layer].head_dim`；
- 派生 `config_dir` 与 `project_root`；
- 填补默认 `quant_cfg_id`、`static_table_id`、`degree_bound_id`。

#### 作用

防止错误配置进入正式 proving/verifying 主链。它是配置字段契约的实现真源。

---

### `src/util/route2.cpp`

#### 内容概览

实现 route2 全局选项：

- 保存当前 route2 options；
- 返回 feature label；
- 生成 notes；
- 处理默认开关；
- 让 main/prover/verifier/algebra 层共享执行层配置。

#### 作用

集中管理执行层优化开关。不能把它理解为 protocol switch。

---

### `src/util/logging.cpp`

#### 内容概览

实现日志辅助：

- 统一日志输出；
- 控制是否打印调试信息；
- 可能封装环境变量或开关判断。

#### 作用

辅助研发排查。正式 benchmark 输出不依赖它作为真源。

---

## 5.2 Algebra 层

### `src/algebra/vector_ops.cpp`

#### 内容概览

实现有限域向量操作：

- vector add/sub/mul；
- dot product；
- batch inverse；
- matrix flatten；
- row/column transform；
- CPU route；
- CUDA hotspots 调度入口；
- backend availability/name。

#### 作用

为 trace、projection、lookup packing、opening、quotient 的大量 field 运算提供基础工具。

---

### `src/algebra/polynomial.cpp`

#### 内容概览

实现多项式对象：

- coefficient polynomial 构造；
- evaluation polynomial 构造；
- domain 上插值/求值；
- spilled evaluation polynomial 的加载；
- point evaluation；
- basis conversion；
- zero-padding / shape control。

#### 作用

所有可承诺对象最终都会变成 `Polynomial`。它连接 trace columns 和 KZG。

---

### `src/algebra/eval_backend.cpp`

#### 内容概览

实现域上批量求值后端：

- packed evaluation backend；
- domain weight cache；
- barycentric evaluation；
- rotated point evaluation；
- batch evaluate；
- parallel/optimized path。

#### 作用

减少 verifier/prover 中大量重复域上求值开销，是性能热点核心。

---

### `src/algebra/cuda_backend.cu`

#### 内容概览

实现 CUDA field hotspots：

- CUDA build enabled 检测；
- CUDA runtime available 检测；
- GPU kernel；
- host/device 数据搬运；
- size-aware offload；
- 与 CPU 结果一致性保持。

#### 作用

执行层优化文件。它不改变 proof shape，只把合适的 field hotspots 放到 GPU。

---

## 5.3 Crypto 层

### `src/crypto/curve.cpp`

#### 内容概览

实现曲线点与 pairing：

- G1/G2 点构造；
- point add/sub/mul；
- scalar multiplication；
- same-base batch multiplication；
- fast MSM route；
- pairing check；
- serialization / debug string；
- backend name。

#### 作用

承诺、witness、pairing verification 的底层实现。

---

### `src/crypto/kzg.cpp`

#### 内容概览

实现 KZG：

- setup / key pair；
- polynomial commitment；
- tau evaluation commitment；
- opening witness；
- folded/batch opening；
- verifier opening check；
- quotient witness。

#### 作用

证明系统把多项式值和承诺绑定的核心密码学实现。

---

### `src/crypto/transcript.cpp`

#### 内容概览

实现 Fiat-Shamir transcript：

- absorb scalar；
- absorb commitment；
- 生成 challenge；
- domain separator；
- deterministic hash-to-field。

#### 作用

保证 prover/verifier 用相同对象、相同名字、相同顺序生成相同 challenge。

---

## 5.4 Data / Model 层

### `src/data/loader.cpp`

#### 内容概览

实现数据加载与规范化：

- 读取缓存数据集；
- 读取 `meta.cfg`；
- 读取 `features.txt` / `features.npy`；
- row normalize；
- 量化 feature 到 field；
- 读取 labels；
- 读取 node_ptr / edge_ptr；
- 读取 edges；
- Planetoid cache 准备；
- PPI cache 准备；
- ogbn-arxiv cache 准备；
- selected graph ids；
- local graph 构造；
- self-loop 补齐；
- edge deduplicate；
- edge sort：`graph_id → dst → stable_index`；
- public input 填充。

#### 作用

把外部数据整理成 formal 主线一致消费的 `GraphDataset` 和 `LocalGraph`。

---

### `src/model/gat.cpp`

#### 内容概览

实现模型参数读取与 GAT 前向：

- checkpoint manifest 解析；
- dense tensor dump 读取；
- hidden layer info 解析；
- hidden head spec 解析；
- output head spec 解析；
- seq kernel reshape；
- attention kernel reshape；
- bias reshape；
- synthetic model 构造；
- multi-layer/multi-head 参数构造；
- GAT attention head forward：
  - projection；
  - source/destination attention；
  - LeakyReLU；
  - stabilized softmax；
  - aggregation；
  - output bias；
  - ELU / identity activation；
- hidden layer concat；
- output head average；
- forward profile timing。

#### 作用

把模型参数和图输入转为 trace 阶段需要的前向中间量。它是“模型语义落地”的核心实现。

---

## 5.5 Protocol 基础层

### `src/protocol/schema.cpp`

#### 内容概览

实现 proof schema 标签：

- `proof_block_order()` 固定为：
  - `M_pub`
  - `Com_dyn`
  - `S_route`
  - `Eval_ext`
  - `Eval_dom`
  - `Com_quot`
  - `Open_dom`
  - `W_ext`
  - `Pi_bind`
- hidden head prefix：`P_h<i>`；
- hidden concat label；
- output head label；
- single-output compatible label；
- challenge/eval/witness label helper。

#### 作用

统一 label 命名与 proof block 顺序。任何改动都会影响 proof contract。

---

### `src/protocol/challenges.cpp`

#### 内容概览

实现 public metadata、commitment labels 和 transcript replay：

- `canonical_public_metadata`；
- `absorb_public_metadata`；
- hidden profile 字符串；
- d_in_profile 字符串；
- dynamic commitment label 生成；
- quotient commitment label 生成；
- domain opening label 生成；
- replay challenges；
- static/public/dynamic/quotient commitments 吸收顺序；
- hidden/output head challenge 命名；
- transcript cache。

#### 作用

这是 prover/verifier 共享挑战值和公开元数据的真源。证明方和验证方不能各自拼 metadata 或 challenge 顺序。

---

### `src/protocol/lookup.cpp`

#### 内容概览

实现 lookup argument：

- table packing；
- query packing；
- multiplicity；
- accumulator；
- feature lookup；
- LeakyReLU lookup；
- ELU lookup；
- exp lookup；
- range lookup；
- selector mask；
- padding；
- state-machine style accumulator。

#### 作用

把 feature 读取、非线性函数、范围检查等不方便直接代数化的关系转成可验证 lookup 关系。

---

### `src/protocol/psq.cpp`

#### 内容概览

实现 PSQ 关系：

- 构造 weighted compressed source；
- 构造 `w_psq`；
- 构造 `T_psq`；
- 构造 edge broadcast；
- 生成 `PSQ` column；
- 使用 `lambda_psq` challenge 绑定加权聚合关系。

#### 作用

PSQ 是 attention aggregation proof 的重要中间约束，把边上的加权源特征和节点聚合结果绑定起来。

---

### `src/protocol/zkmap.cpp`

#### 内容概览

实现 map/binding 关系：

- key/value 编码；
- source/destination route helper；
- node-to-edge broadcast；
- edge-to-node recovery；
- binding accumulator；
- route objects。

#### 作用

把图中的节点域与边域数据流动显式证明化，避免“代码里广播了但证明里没绑定”的漏洞。

---

## 5.6 Protocol Trace 层

### `src/protocol/trace.cpp`

#### 内容概览

这是仓库最重的文件之一，负责 `build_trace`。主要内容包括：

- trace helper；
- dynamic commitment spec；
- feature lookup trace；
- hidden head trace；
- hidden projection trace；
- hidden attention trace；
- hidden softmax chain；
- hidden route objects；
- hidden PSQ objects；
- ELU lookup objects；
- concat trace；
- output head trace；
- output attention trace；
- output aggregation/average trace；
- matrix/column materialization；
- field conversion；
- lazy/spilled large object 处理；
- dynamic polynomial 构造；
- dynamic commitment order 设置；
- external evaluations；
- witness scalars；
- metrics 拆分；
- route2 / CUDA hotspots 分支。

#### 作用

把 `ProtocolContext` 中的静态骨架和模型前向结果，翻译成最终 proof 需要承诺和打开的动态对象。它不定义新协议，只物化 witness 和动态列。

#### 建议分块阅读

```text
1. helper / timing / cache
2. dynamic commitment specs
3. feature lookup trace
4. hidden head trace
5. softmax / range / exp / ELU lookup
6. PSQ / route / state machine
7. concat trace
8. output head trace
9. dynamic polynomial materialization
10. external eval / witness scalars
11. TraceArtifacts assembly
12. route2 / CUDA / spill
```

---

## 5.7 Protocol Quotient 层

### `src/protocol/quotients.cpp`

#### 内容概览

实现各工作域 quotient：

- `t_FH`：feature / flatten / lookup 相关约束；
- `t_edge`：edge-domain route、softmax、PSQ、lookup 相关约束；
- `t_in`：输入维投影约束；
- `t_d_h`：隐藏头宽度压缩约束；
- `t_cat`：hidden concat 约束；
- `t_C`：output class width 约束；
- `t_N`：node-domain relations，例如 max/sum/inv/logits 等；
- legacy/single-output compat；
- real multihead 路线；
- quotient evaluation cache；
- verifier 重算支持。

#### 作用

quotient 是证明系统的“约束是否成立”表达。prover 计算 quotient 并承诺，verifier 重算并检查。

---

## 5.8 Protocol Prover 层

### `src/protocol/prover.cpp`

#### 内容概览

该文件非常重要，包含四大类职责。

#### 1. `build_context`

构造静态上下文：

```text
config
  ↓
dataset
  ↓
local graph
  ↓
model parameters
  ↓
work domains
  ↓
static tables
  ↓
public polynomials
  ↓
static commitments
  ↓
KZG keypair
  ↓
ProtocolContext
```

具体会做：

- load dataset；
- normalize graph；
- load checkpoint/model；
- check shape；
- create work domains；
- build static tables；
- build public polynomials；
- commit public/static objects；
- setup KZG；
- cache static context。

#### 2. `prove`

生成最终 proof：

```text
TraceArtifacts
  ↓
check dynamic commitment order
  ↓
replay challenges
  ↓
build quotient polynomials
  ↓
commit quotients
  ↓
replay challenges after quotients
  ↓
build domain openings
  ↓
collect external evaluations
  ↓
collect witness scalars
  ↓
assemble Proof
```

#### 3. `proof_size_bytes`

统计 proof 传输尺寸，通常计入：

- public metadata strings；
- block order；
- dynamic commitments；
- quotient commitments；
- domain opening values；
- opening witnesses；
- external evaluations；
- witness scalars；
- external witness。

不计入可由 verifier 重放得到的 label、challenge、opening point 标签等。

#### 4. `export_run_artifacts`

导出单次运行产物：

- `metadata.txt`；
- `challenges.txt`；
- `proof_block_order.txt`；
- `commitment_order.txt`；
- `benchmark.txt`；
- `run_manifest.json`；
- 其他内部复核文件。

#### 作用

prover 主链文件。它把静态上下文、动态轨迹、商多项式、opening 和 artifact 串成正式 proof 结果。

---

## 5.9 Protocol Verifier 层

### `src/protocol/verifier.cpp`

#### 内容概览

实现验证方：

```text
Proof
  ↓
check public metadata
  ↓
check proof block order
  ↓
replay transcript
  ↓
check dynamic/quotient/opening label order
  ↓
verify domain openings
  ↓
recompute quotient equations
  ↓
check external folds/witness scalars
  ↓
return accepted
```

内部还包括：

- public evaluation backend registry；
- verifier evaluation memoization；
- lazy large FH public polynomial evaluation；
- opened value indexing；
- KZG opening verification；
- quotient recomputation；
- external fold/witness scalar checks；
- metrics 记录。

#### 作用

验证方不实现第二套规则。它复用 `canonical_public_metadata`、`proof_block_order`、`domain_opening_labels`、quotient 函数来检查 proof。

---

## 5.10 主入口

### `src/main.cpp`

#### 内容概览

命令行入口，处理：

- 参数解析：
  - `--config`
  - `--benchmark-mode`
  - `--compute-backend`
  - `--fast-msm`
  - `--parallel-fft`
  - `--fft-backend-upgrade`
  - `--fft-kernel-upgrade`
  - `--trace-layout-upgrade`
  - `--fast-verify-pairing`
  - `--export-tag`
- 只接受 `--benchmark-mode warm`；
- 设置 route2 options；
- 检查 CUDA build/runtime；
- 加载 config；
- warm-up；
- measured run；
- 依次调用：
  - `build_context`
  - `build_trace`
  - `prove`
  - `verify`
  - `proof_size_bytes`
  - `export_run_artifacts`
- 统计并输出正式四项指标；
- 控制台只输出一行。

#### 作用

正式编排层。它不定义协议，不改 schema，不改 verifier equation，只负责把一次正式运行串起来。

---

# 6. 正式配置逐文件说明

## `configs/cora_full.cfg`

### 内容概览

Cora full dataset 正式配置，通常包含：

- `dataset = cora`
- data/cache/export 路径；
- checkpoint bundle；
- task/report 规则；
- whole graph / full dataset；
- self-loop 与 edge sort 规则；
- `hidden_profile = 8x8`；
- `d_in_profile` 对齐 Cora feature dim；
- `K_out`；
- `num_classes`；
- static/model/quant ids。

### 作用

定义 Cora 正式运行的公开配置。它会进入 `PublicMetadata` 和 benchmark artifact。

---

## `configs/citeseer_full.cfg`

### 内容概览

Citeseer full dataset 正式配置，结构与 Cora 类似，但：

- dataset 名称不同；
- feature dim 不同；
- class 数不同；
- checkpoint bundle 不同；
- export dir 不同。

### 作用

定义 Citeseer 正式运行。

---

## `configs/pubmed_full.cfg`

### 内容概览

Pubmed full dataset 正式配置，包含：

- Pubmed 数据源；
- Pubmed checkpoint；
- full graph rule；
- hidden/output profile；
- class 数；
- benchmark export dir。

### 作用

定义 Pubmed 正式运行。

---

## `configs/ppi_batch_formal.cfg`

### 内容概览

PPI 多图 batch formal 配置，重点字段通常包括：

- `dataset = ppi`
- `batching_rule` 指向 multi-graph batch；
- `batch_graphs`；
- `hidden_profile = 1x8`；
- `d_in_profile`；
- PPI checkpoint bundle；
- multi-graph node/edge pointer 规则。

### 作用

定义 PPI formal batch 运行。读它时要重点理解 graph_count、node_ptr、edge_ptr 与 batch semantics。

---

## `configs/ogbn_arxiv_full.cfg`

### 内容概览

ogbn-arxiv full graph 配置，重点字段包括：

- `dataset = ogbn_arxiv`
- whole graph single；
- 大图缓存路径；
- checkpoint bundle；
- `hidden_profile = 1x8`；
- input feature dim；
- class 数；
- export dir。

### 作用

定义 ogbn-arxiv 大图正式运行。它是大图性能优化和 CUDA hotspot 的重要目标。

---

# 7. 脚本逐文件说明

## `scripts/fetch_mcl.sh`

### 内容概览

准备或下载 mcl 依赖，可能拉取 third-party source 并放入 `third_party/mcl`。

### 作用

服务构建依赖，与 `cmake/ConfigureMCL.cmake` 配套。

---

## `scripts/planetoid_utils.py`

### 内容概览

Planetoid 数据集公共工具，通常封装：

- Cora/Citeseer/Pubmed 数据读取；
- raw object 解析；
- edge/index 处理；
- feature/label 转换；
- cache 写入辅助。

### 作用

Planetoid 系列脚本共用工具库。

---

## `scripts/prepare_planetoid.py`

### 内容概览

准备 Cora/Citeseer/Pubmed cache：

- 读取原始 Planetoid 数据；
- 生成 `meta.cfg`；
- 导出 features；
- 导出 labels；
- 导出 edges；
- 写入 cache root。

### 作用

把外部数据转成 C++ loader 读取的正式缓存格式。

---

## `scripts/train_planetoid_gat.py`

### 内容概览

训练 Planetoid GAT：

- 加载 Planetoid 数据；
- 定义/训练 GAT；
- 保存模型权重；
- 生成训练结果。

### 作用

为 Cora/Citeseer/Pubmed 生成上游模型参数来源。训练不是 formal proving 主线的一部分。

---

## `scripts/check_planetoid_parity.py`

### 内容概览

检查 Planetoid 训练/导出结果与 C++ 读取逻辑或参考前向的一致性。

### 作用

用于避免 Python/PyTorch 侧和 C++ formal 侧参数解释不一致。

---

## `scripts/export_gat_checkpoint.py`

### 内容概览

导出 GAT checkpoint bundle：

- 读取训练权重；
- 按 manifest 记录 tensor 名称和形状；
- 导出 dense tensor dump；
- 写出 hidden/output layer 结构信息。

### 作用

把训练结果变成 `src/model/gat.cpp` 可读取的正式参数包。

---

## `scripts/checkpoint_reader_compat.py`

### 内容概览

验证 checkpoint reader compatibility：

- 检查 manifest 字段；
- 检查 tensor shape；
- 检查 hidden profile；
- 检查 d_in_profile；
- 检查 output head。

### 作用

防止 checkpoint bundle 与 C++ reader 预期不一致。

---

## `scripts/prepare_ppi.py`

### 内容概览

准备 PPI 数据缓存：

- 读取 PPI 原始数据；
- 处理多图 batch；
- 导出 node_ptr；
- 导出 edge_ptr；
- 导出 features/labels/edges；
- 写入 cache。

### 作用

把 PPI 多图数据转成 formal loader 可读格式。

---

## `scripts/import_ppi_bundle.py`

### 内容概览

导入 PPI checkpoint bundle：

- 整理外部训练结果；
- 转成仓库规定的 bundle layout；
- 对齐 manifest。

### 作用

让 PPI 的真实参数包进入正式 benchmark 主线。

---

## `scripts/ogbn_arxiv_utils.py`

### 内容概览

ogbn-arxiv 工具库：

- 数据下载/读取 helper；
- graph edge 处理；
- feature/label 处理；
- split/metadata helper。

### 作用

为 ogbn-arxiv 数据准备和训练脚本复用。

---

## `scripts/prepare_ogbn_arxiv.py`

### 内容概览

准备 ogbn-arxiv cache：

- 读取 OGBN 原始数据；
- 导出 features；
- 导出 labels；
- 导出 edges；
- 生成 `meta.cfg`；
- 写入 cache。

### 作用

让 C++ loader 可以读取 ogbn-arxiv full graph。

---

## `scripts/train_ogbn_arxiv_gat.py`

### 内容概览

训练 ogbn-arxiv GAT：

- 加载 ogbn-arxiv；
- 定义 GAT；
- 训练/评估；
- 保存 PyTorch checkpoint。

### 作用

为 ogbn-arxiv formal run 提供上游模型参数来源。

---

## `scripts/export_torch_gat_bundle.py`

### 内容概览

从 PyTorch checkpoint 导出通用 GAT bundle：

- 提取 tensor；
- 写 manifest；
- 写 tensor dump；
- 记录 hidden profile / d_in_profile / K_out / C。

### 作用

把 PyTorch GAT 参数转成 C++ formal reader 使用的 checkpoint bundle。

---

## `scripts/export_benchmark_table.py`

### 内容概览

汇总 benchmark：

```text
runs/<dataset>/warm/run_manifest.json
  ↓
validate warm/full/VERIFY_OK
  ↓
latest.json
  ↓
latest.csv
  ↓
summary.md
```

只保留正式展示字段：

- commitment time；
- prove time；
- verify time；
- proof size。

### 作用

这是正式结果表导出的脚本真源。README 表和 tests 会围绕它的输出做一致性检查。

---

# 8. 测试文件

## `tests/test_main.cpp`

### 内容概览

formal 回归测试总入口，通常包括：

- 文件读写 helper；
- repo root helper；
- JSON/CSV 解析 helper；
- config fixture；
- proof fixture；
- schema tests；
- metadata tests；
- challenge replay tests；
- dynamic commitment order tests；
- domain opening labels tests；
- prove/verify success tests；
- tampering proof failure tests；
- metadata mismatch failure；
- block order mismatch failure；
- opening mismatch failure；
- artifact truth source tests；
- README/latest.json/latest.csv/summary.md 一致性；
- proof_size_bytes 一致性；
- cleanup regression：
  - conflict marker；
  - placeholder；
  - old trial/smoke artifact；
  - 实验字段污染正式表。

### 作用

它是仓库 formal contract 的回归保护层。任何改动都应该用它检查是否破坏正式主线。

---

# 9. Reference 文件

## `reference/gat_ogbn_arxiv/pyg_gat_example.py`

### 内容概览

PyG GAT 参考样例，服务 ogbn-arxiv：

- 展示 PyG GAT 模型结构；
- 数据加载示例；
- 前向/训练参考。

### 作用

帮助理解上游 PyG 模型与 C++ formal GAT 的对应关系。它不是 proof 语义真源。

---

# 10. 运行产物与数据文件说明

这些目录不适合逐个大文件读源码式展开，但要理解它们在闭环中的作用。

## `artifacts/checkpoints/`

### 内容概览

存放 checkpoint bundles：

- manifest；
- tensor dump；
- hidden layer metadata；
- output head metadata；
- model shape；
- parameter ids。

### 作用

是正式参数来源。`src/model/gat.cpp` 会读取它们，并与 `configs/*.cfg` 对齐。

---

## `data/`

### 内容概览

存放原始数据或缓存数据。缓存格式通常包括：

- `meta.cfg`；
- `features.txt` 或 `features.npy`；
- `labels.txt`；
- `edges.txt`；
- `node_ptr.txt`；
- 可能的 split 或额外 metadata。

### 作用

是 `src/data/loader.cpp` 的输入。formal run 使用的是规范化后的 cache，而不是随意读取原始训练对象。

---

## `runs/benchmarks/latest.json`

### 内容概览

CPU benchmark 的机器复核主表，包含比展示面更完整的结构化字段。

### 作用

机器复核真源之一。测试和正式 summary 会对齐它。

---

## `runs/benchmarks/latest.csv`

### 内容概览

CPU benchmark 的展示表，只保留正式四项指标。

### 作用

正式展示产物之一。

---

## `runs/benchmarks/summary.md`

### 内容概览

CPU benchmark 的 Markdown 摘要表。

### 作用

面向论文/README 的正式展示文本。

---

## `runs/benchmarks/cuda_formal/latest.json`

### 内容概览

GPU benchmark 的机器复核主表。

### 作用

GPU 机器复核真源。

---

## `runs/benchmarks/cuda_formal/latest.csv`

### 内容概览

GPU benchmark 展示 CSV。

### 作用

GPU 正式展示产物。

---

## `runs/benchmarks/cuda_formal/summary.md`

### 内容概览

GPU benchmark Markdown 摘要表。

### 作用

GPU 正式展示文本。

---

## `runs/<dataset>/warm/run_manifest.json`

### 内容概览

单次 CPU formal run 的完整内部复核清单，包含：

- dataset；
- backend；
- config；
- node/edge count；
- full dataset flag；
- route2 flags；
- timing breakdown；
- proof size；
- artifact paths；
- verify status；
- notes。

### 作用

单次运行机器复核真源。`export_benchmark_table.py` 从它汇总正式主表。

---

## `runs/<dataset>/warm/benchmark.txt`

### 内容概览

单次运行最小正式摘要。

### 作用

面向人读的一行/少量行结果，不是完整内部复核清单。

---

## `runs/<dataset>/warm/metadata.txt`

### 内容概览

导出的 `PublicMetadata`。

### 作用

用于复核 proof 中公开元数据是否与 config/model/data 一致。

---

## `runs/<dataset>/warm/challenges.txt`

### 内容概览

导出的 challenge 集合。

### 作用

用于复核 transcript replay 是否稳定。

---

## `runs/<dataset>/warm/proof_block_order.txt`

### 内容概览

导出的 proof block 顺序。

### 作用

用于复核 proof serializer/parser/verifier 共享顺序没有漂移。

---

## `runs/<dataset>/warm/commitment_order.txt`

### 内容概览

导出的 dynamic commitment 顺序。

### 作用

用于复核 `trace.commitment_order` 是否等于 `dynamic_commitment_labels(context)`。

---

## `runs/<dataset>/cuda_formal/warm/*`

### 内容概览

GPU formal run 的同类产物：

- `run_manifest.json`
- `benchmark.txt`
- `metadata.txt`
- `challenges.txt`
- `proof_block_order.txt`
- `commitment_order.txt`

### 作用

验证 GPU 路径与 CPU 路径共享同一 proof contract，只是执行层不同。

---

# 11. 模块之间的调用关系

## 11.1 从入口到 context

```text
src/main.cpp
  └── util::load_config
  └── protocol::build_context
        ├── data::load_dataset
        ├── data::normalize_graph_input
        ├── model::load_model_parameters / build_family_model_parameters
        ├── build static tables
        ├── build public polynomials
        ├── build static commitments
        └── setup KZG
```

## 11.2 从 context 到 trace

```text
ProtocolContext
  └── protocol::build_trace
        ├── model forward
        ├── feature lookup trace
        ├── hidden head traces
        ├── output head traces
        ├── route objects
        ├── PSQ objects
        ├── lookup accumulators
        ├── dynamic polynomials
        └── TraceArtifacts
```

## 11.3 从 trace 到 proof

```text
TraceArtifacts
  └── protocol::prove
        ├── check commitment order
        ├── replay challenges
        ├── build quotients
        ├── commit quotients
        ├── build domain openings
        ├── collect external evals
        ├── collect witness scalars
        └── Proof
```

## 11.4 从 proof 到 verify

```text
Proof
  └── protocol::verify
        ├── check PublicMetadata
        ├── check proof_block_order
        ├── replay_challenges
        ├── verify KZG openings
        ├── recompute quotient equations
        ├── check external fold/scalars
        └── bool accepted
```

## 11.5 从 run 到 benchmark

```text
Proof + RunMetrics
  └── export_run_artifacts
        ├── run_manifest.json
        ├── benchmark.txt
        ├── metadata.txt
        ├── challenges.txt
        ├── proof_block_order.txt
        └── commitment_order.txt

run_manifest.json
  └── scripts/export_benchmark_table.py
        ├── latest.json
        ├── latest.csv
        └── summary.md
```

---

# 12. 每类文件阅读时的检查清单

## 12.1 读 formal 文档时

- 协议对象是什么？
- 哪些对象是 public？
- 哪些对象是 witness？
- transcript 吸收顺序是什么？
- verifier equation 是什么？

## 12.2 读 config 时

- `L` 与 `hidden_profile` 是否一致？
- `d_in_profile` 是否与前一层 concat width 一致？
- `checkpoint_bundle` 是否与 model ids 一致？
- `batching_rule` 与 dataset 是否匹配？

## 12.3 读 trace 时

- 这个 column 的 label 是哪里来的？
- domain 是哪个？
- 是否进入 dynamic commitment？
- 是否被 quotient 使用？
- 是否被 domain opening 打开？
- 是否进入 external evaluation/witness scalar？

## 12.4 读 prover 时

- dynamic commitment 顺序是否检查？
- challenge replay 是否在正确阶段？
- quotient label 是否固定？
- opening label 是否固定？
- proof block order 是否来自 schema？
- public metadata 是否来自 canonical 函数？

## 12.5 读 verifier 时

- 是否复用同一 metadata？
- 是否复用同一 block order？
- 是否复用同一 challenge replay？
- 是否检查 KZG opening？
- 是否重算 quotient？
- 是否能拒绝 tampered proof？

## 12.6 读 benchmark/artifacts 时

- 是否是 warm？
- 是否是 full dataset？
- verify 是否 OK？
- CPU/GPU 是否分目录？
- 展示字段是否只保留四项？
- proof size 是否来自 `proof_size_bytes`？

---

# 13. 最容易误解的点

## 13.1 CPU/GPU 不是两套协议

CPU 与 GPU 的区别只在执行层：

- 哪些 field hotspots 放到 CUDA；
- 哪些大域计算并行；
- 是否启用 packed evaluation；
- 是否启用 fast MSM / fast pairing。

只要 label、challenge、quotient、opening、verifier equation 不变，就仍是一套 formal 语义。

## 13.2 proof size 不是图大小

`proof_size_bytes` 主要由 proof shape 决定，而 proof shape 又由：

- hidden_profile；
- output heads；
- dynamic commitments 数量；
- opening values 数量；
- external eval/witness scalars 数量；

决定。因此小图 proof 可能比大图更大。

## 13.3 trace 不是普通日志

trace 是可证明 witness 对象的物化，不是 debug trace。它会影响 dynamic commitments、challenges、quotients 和 openings。

## 13.4 README 是展示面，不是语义真源

README 说明当前正式结果与复现命令，但 formal 语义真源仍是 `GAT-ZKML-多层多头.md`。

---

# 14. 建议实际阅读路线

## 第一天：建立整体图

读：

1. `README.md`
2. `GAT-ZKML-多层多头.md`
3. `项目整体逻辑与方法流程.md`
4. `project_main_repo_full_logic_readme.md`

目标：知道仓库证明什么、主调用链是什么。

## 第二天：建立对象边界

读：

1. `include/gatzk/protocol/proof.hpp`
2. `include/gatzk/util/config.hpp`
3. `include/gatzk/model/gat.hpp`
4. `include/gatzk/data/dataset.hpp`
5. `include/gatzk/protocol/challenges.hpp`
6. `include/gatzk/protocol/schema.hpp`

目标：知道核心对象字段和接口。

## 第三天：读底层实现

读：

1. `src/util/config.cpp`
2. `src/data/loader.cpp`
3. `src/model/gat.cpp`
4. `src/algebra/*.cpp`
5. `src/crypto/*.cpp`

目标：知道数据、模型、field、polynomial、KZG 怎样落地。

## 第四天：读协议骨架

读：

1. `src/protocol/schema.cpp`
2. `src/protocol/challenges.cpp`
3. `src/protocol/lookup.cpp`
4. `src/protocol/psq.cpp`
5. `src/protocol/zkmap.cpp`

目标：知道命名、challenge、lookup、route、PSQ 如何组成证明对象。

## 第五天：读主证明链

读：

1. `src/protocol/trace.cpp`
2. `src/protocol/quotients.cpp`
3. `src/protocol/prover.cpp`
4. `src/protocol/verifier.cpp`
5. `src/main.cpp`

目标：完整串起 trace → prove → verify。

## 第六天：读配置、脚本、测试、产物

读：

1. `configs/*.cfg`
2. `scripts/export_benchmark_table.py`
3. `tests/test_main.cpp`
4. `runs/benchmarks/*`
5. 每个 dataset 的 `run_manifest.json`

目标：理解复现、测试与 benchmark truth source。

---

# 15. 快速定位表

| 想理解的问题 | 优先读 |
|---|---|
| 项目证明什么 | `GAT-ZKML-多层多头.md`、`README.md` |
| 入口怎么跑 | `src/main.cpp` |
| 配置怎么读 | `include/gatzk/util/config.hpp`、`src/util/config.cpp` |
| 数据怎么变成图 | `include/gatzk/data/dataset.hpp`、`src/data/loader.cpp` |
| checkpoint 怎么读 | `include/gatzk/model/gat.hpp`、`src/model/gat.cpp` |
| GAT forward 怎么算 | `src/model/gat.cpp` |
| proof 对象长什么样 | `include/gatzk/protocol/proof.hpp` |
| labels 在哪定义 | `include/gatzk/protocol/schema.hpp`、`src/protocol/schema.cpp` |
| challenges 在哪生成 | `src/protocol/challenges.cpp` |
| dynamic commitments 顺序 | `dynamic_commitment_labels` in `src/protocol/challenges.cpp` |
| domain opening labels | `domain_opening_labels` in `src/protocol/challenges.cpp` |
| trace 怎么构造 | `src/protocol/trace.cpp` |
| lookup 怎么构造 | `src/protocol/lookup.cpp` |
| PSQ 是什么 | `src/protocol/psq.cpp` |
| route/binding 是什么 | `src/protocol/zkmap.cpp` |
| quotient 怎么算 | `src/protocol/quotients.cpp` |
| proof 怎么组装 | `src/protocol/prover.cpp` |
| proof size 怎么算 | `proof_size_bytes` in `src/protocol/prover.cpp` |
| verifier 检查什么 | `src/protocol/verifier.cpp` |
| benchmark 怎么导出 | `scripts/export_benchmark_table.py` |
| 回归测试覆盖什么 | `tests/test_main.cpp` |
| GPU 是否第二协议 | `docs/热点优化方法与处理逻辑.md`、`src/algebra/cuda_backend.cu` |

---

# 16. 读完后应能回答的问题

1. 为什么 `GAT-ZKML-多层多头.md` 是唯一 formal 语义真源？
2. `PublicMetadata` 哪些字段进入 proof？
3. `hidden_profile` 为什么影响 `proof_size_bytes`？
4. `build_context` 和 `build_trace` 的边界是什么？
5. dynamic commitment order 为什么不能随便改？
6. challenge replay 为什么必须被 prover/verifier 共享？
7. lookup、route、PSQ、quotient、opening 分别解决什么？
8. verifier 为什么不需要重新执行完整前向？
9. CPU/GPU 为什么不是两套协议？
10. `latest.csv` 为什么只保留四项指标？
11. `run_manifest.json` 与 `summary.md` 的职责为什么不同？
12. 如何从一个 config 追踪到最终 proof 和 benchmark 表？

---

## 17. 最后的一句话理解

这个仓库的核心不是“跑 GAT”，而是把多层多头 `GAT` 前向过程中的所有关键值、名字、顺序和关系，变成有限域中的正式可验证对象，再用固定 transcript、固定 quotient、固定 KZG opening 和固定 verifier 逻辑把它们绑定成一份可复核的 proof。
