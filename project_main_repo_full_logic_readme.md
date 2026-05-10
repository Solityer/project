# project 主仓库完整代码逻辑阅读 README

本文用于指导从零开始完整阅读并理解 `project` 主仓库的代码逻辑。本文聚焦主仓库正式实现本身，即 `GAT-ZKML` 多层多头 GAT formal proving 主线；`comparison/baselines/ezkl` 是对比实验目录，已有单独 README，不纳入本文的主实现阅读顺序。

本文的核心目标是：按照主仓库真实执行链，把 formal 语义、配置、数据、模型、代数、密码、协议、证明、验证、artifact、benchmark、测试和脚本全部串起来，避免只看局部文件而误解整体协议。

---

## 1. 主仓库阅读目标

读完本文指定文件后，应能回答以下问题：

1. 项目到底在证明什么。
2. formal note 中的多层多头 GAT 语义如何落到 C++ 实现。
3. `gatzk_run` 从命令行到最终 proof 的完整调用链是什么。
4. `build_context`、`build_trace`、`prove`、`verify` 各自负责什么。
5. `proof schema`、`transcript` 顺序、dynamic commitment 顺序、domain opening label 顺序分别在哪里定义。
6. 静态上下文、运行轨迹、动态承诺、quotient、opening、proof、manifest 之间如何流动。
7. CPU 和 GPU 路径为什么是同一套 formal 语义，而不是两套协议。
8. 五个正式数据集配置如何进入运行。
9. benchmark 主表如何从 per-run artifact 汇总出来。
10. 测试如何保证 README、manifest、latest.json、latest.csv、summary.md 等 truth source 一致。

---

## 2. 仓库结构总览

主仓库可以按职责分为以下层次：

```text
project/
├── README.md                         项目当前正式说明和结果表
├── GAT-ZKML-多层多头.md               唯一 formal 语义真源
├── 项目整体逻辑与方法流程.md           中文整体流程说明
├── CMakeLists.txt                    主构建入口
├── cmake/                            第三方/后端配置
├── include/gatzk/                    正式头文件
├── src/                              正式 C++ 实现
├── configs/                          五个正式数据集配置
├── scripts/                          数据准备、训练、checkpoint、benchmark 汇总脚本
├── tests/                            formal 回归测试
├── docs/                             补充技术说明
├── reference/                        参考实现或参考样例
├── artifacts/                        checkpoint bundles 等输入产物
├── data/                             数据集和缓存
└── runs/                             per-run artifact 与 benchmark truth source
```

核心执行链：

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
  ↓
README result table / tests/test_main.cpp
```

---

## 3. 第一阶段：先读语义和项目总说明

第一阶段不要看代码细节，只建立“项目为什么这样写”的大框架。

阅读顺序：

```text
README.md
GAT-ZKML-多层多头.md
项目整体逻辑与方法流程.md
docs/热点优化方法与处理逻辑.md
```

### 3.1 `README.md`

阅读重点：

- 项目定位：多层多头 GAT 的正式 ZKML 证明系统。
- 当前正式主线：一套 formal 语义，同时支撑 `full_cpu` 与 `full_gpu`。
- 硬约束：不能改 proof schema、transcript 顺序、serializer/parser 结构、verifier equations、public metadata 契约。
- 当前调用链：`src/main.cpp → build_context → build_trace → prove → verify → proof_size_bytes → export_run_artifacts`。
- 五个正式数据集：Cora、Citeseer、Pubmed、PPI、ogbn-arxiv。
- CPU/GPU 运行命令。
- proof size 统计口径。
- benchmark truth source。

### 3.2 `GAT-ZKML-多层多头.md`

这是唯一 formal 语义真源。重点读：

- 多层多头 GAT family 定义。
- hidden layer 和 output layer 的数学语义。
- 有限域、有符号解释、量化尺度。
- 图、batch、edge sort、self-loop 和 pointer 约定。
- 静态表：feature、LeakyReLU、ELU、exp、range。
- 模型参数、hidden head、output head。
- 中间变量：投影、attention、softmax、聚合、ELU、concat、output logits。
- lookup、route、PSQ、state machine、quotient、KZG opening。
- public metadata。
- transcript 吸收顺序。
- verifier equations。

### 3.3 `项目整体逻辑与方法流程.md`

该文件是中文逻辑导读。重点读：

- 五层结构：模型层、代数层、密码层、协议层、工具层。
- 一次正式运行的十二步。
- `build_context`、`build_trace`、`prove`、`verify` 的真实职责。
- 静态上下文和运行轨迹的区别。
- artifact 导出链。
- 为什么 CPU/GPU 只属于执行层差异。

### 3.4 `docs/热点优化方法与处理逻辑.md`

该文件用于理解性能优化路线。重点读：

- route2 开关的语义。
- fast MSM、parallel FFT、packed domain、rotated kernel、trace layout、fast pairing 的边界。
- CUDA hotspots 为什么只改变执行后端，不改变 formal proof。
- 大图优化与 formal 语义之间的关系。

---

## 4. 第二阶段：读构建系统和主入口

阅读顺序：

```text
CMakeLists.txt
cmake/ConfigureMCL.cmake
src/main.cpp
```

### 4.1 `CMakeLists.txt`

这是主仓库构建真源。必须核对：

- C++ 标准。
- 是否启用 tests。
- 是否启用 CUDA backend。
- mcl 后端配置。
- `GATZK_SOURCES` 中进入 `gatzkml` 静态库的所有 `.cpp`。
- `gatzk_run` 可执行文件。
- `gatzk_tests` 测试可执行文件。

主库源码清单以此为准：

```text
src/algebra/eval_backend.cpp
src/algebra/polynomial.cpp
src/algebra/vector_ops.cpp
src/crypto/curve.cpp
src/crypto/kzg.cpp
src/crypto/transcript.cpp
src/data/loader.cpp
src/model/gat.cpp
src/protocol/lookup.cpp
src/protocol/challenges.cpp
src/protocol/prover.cpp
src/protocol/psq.cpp
src/protocol/quotients.cpp
src/protocol/schema.cpp
src/protocol/trace.cpp
src/protocol/verifier.cpp
src/protocol/zkmap.cpp
src/util/config.cpp
src/util/logging.cpp
src/util/route2.cpp
src/algebra/cuda_backend.cu   # 仅 CUDA build 启用
src/main.cpp                  # gatzk_run 入口，单独链接主库
tests/test_main.cpp           # gatzk_tests 入口
```

### 4.2 `cmake/ConfigureMCL.cmake`

阅读重点：

- mcl 如何被发现、配置、链接。
- `GATZK_CRYPTO_BACKEND` 如何确定。
- 如果构建失败，应先从这里排查 mcl include/lib 路径。

### 4.3 `src/main.cpp`

这是正式命令行入口，不定义协议，只编排运行。重点读：

```text
parse command line
  ↓
load config
  ↓
set route2 options
  ↓
warm-up
  ↓
measured run
  ↓
build_context
  ↓
build_trace
  ↓
prove
  ↓
verify
  ↓
proof_size_bytes
  ↓
export_run_artifacts
```

重点检查：

- 只支持 `--benchmark-mode warm`。
- `--compute-backend cpu` 和 `--compute-backend cuda_hotspots` 如何分叉执行层。
- route2 开关如何写入 metrics。
- full dataset 检查在哪里做。
- 控制台为何只输出一行。
- warm-up 和 measured run 的目录区别。
- `export_tag` 如何用于 GPU 结果目录。

---

## 5. 第三阶段：读头文件，建立对象模型

先读头文件，再读实现文件。头文件是对象边界和模块职责的索引。

### 5.1 algebra 头文件

阅读顺序：

```text
include/gatzk/algebra/field.hpp
include/gatzk/algebra/domain_evaluation.hpp
include/gatzk/algebra/polynomial.hpp
include/gatzk/algebra/eval_backend.hpp
include/gatzk/algebra/vector_ops.hpp
include/gatzk/algebra/cuda_field.cuh
```

理解：

- `FieldElement` 如何封装 mcl field。
- Root of unity domain、barycentric、evaluation domain 如何表示。
- `Polynomial` 支持 coefficient/evaluation/shared/spilled 等表示。
- eval backend 如何服务 opening 和 quotient 热路径。
- vector ops 和 CUDA field hotspots 的边界。

### 5.2 crypto 头文件

阅读顺序：

```text
include/gatzk/crypto/curve.hpp
include/gatzk/crypto/kzg.hpp
include/gatzk/crypto/transcript.hpp
```

理解：

- G1/G2 点封装。
- KZG keypair、commit、open、verify。
- transcript 如何吸收 scalar/commitment 并产生 challenge。
- fast MSM 和 fast pairing 在接口层如何暴露。

### 5.3 data/model 头文件

阅读顺序：

```text
include/gatzk/data/dataset.hpp
include/gatzk/data/loader.hpp
include/gatzk/model/gat.hpp
```

理解：

- `Dataset`、`LocalGraph`、edge、feature、label、graph pointer。
- dataset loader 如何把 Planetoid/PPI/ogbn-arxiv 读入统一结构。
- `ModelParameters`、hidden layer、output layer、attention head。
- `HeadForwardTrace` 和 `ForwardProfile` 的作用。
- checkpoint bundle 与 synthetic model 的兼容边界。

### 5.4 util 头文件

阅读顺序：

```text
include/gatzk/util/config.hpp
include/gatzk/util/logging.hpp
include/gatzk/util/route2.hpp
```

理解：

- `AppConfig` 字段。
- hidden profile、d_in_profile、K_out、dataset、checkpoint、export_dir。
- route2 运行开关。
- logging 是正式输出还是开发辅助。

### 5.5 protocol 头文件

阅读顺序：

```text
include/gatzk/protocol/proof.hpp
include/gatzk/protocol/schema.hpp
include/gatzk/protocol/challenges.hpp
include/gatzk/protocol/lookup.hpp
include/gatzk/protocol/psq.hpp
include/gatzk/protocol/zkmap.hpp
include/gatzk/protocol/quotients.hpp
include/gatzk/protocol/trace.hpp
include/gatzk/protocol/prover.hpp
include/gatzk/protocol/verifier.hpp
```

理解：

- `Proof`、`PublicMetadata`、`RunMetrics`、`ProtocolContext`、`TraceArtifacts`。
- proof block order。
- dynamic commitment labels。
- quotient commitment labels。
- domain opening labels。
- lookup argument、PSQ、zkMaP、quotient 结构。
- `build_context`、`build_trace`、`prove`、`verify` 的接口边界。

---

## 6. 第四阶段：按底层到上层读实现文件

实现文件建议按依赖层阅读，不按字母顺序阅读。

---

## 6.1 util 层

阅读顺序：

```text
src/util/config.cpp
src/util/route2.cpp
src/util/logging.cpp
```

### `src/util/config.cpp`

重点：

- `.cfg` 如何按 `key = value` 解析。
- supported keys 白名单。
- legacy field 如何被拒绝。
- hidden_profile 如何解析。
- d_in_profile 如何解析。
- shape consistency 如何校验。
- project_root、data_root、cache_root、checkpoint_bundle、export_dir 如何派生。

阅读时要对照 `configs/*.cfg`。

### `src/util/route2.cpp`

重点：

- 全局 route2 options 如何保存。
- `route2_feature_label` 如何生成 benchmark label。
- `route2_feature_notes` 如何写入 manifest。
- route2 是执行层开关，不是 protocol switch。

### `src/util/logging.cpp`

重点：

- 日志输出边界。
- 正式运行为什么不输出大量过程日志。
- logging 是否影响 artifact 或 benchmark。

---

## 6.2 algebra 层

阅读顺序：

```text
src/algebra/vector_ops.cpp
src/algebra/polynomial.cpp
src/algebra/eval_backend.cpp
src/algebra/cuda_backend.cu
```

### `src/algebra/vector_ops.cpp`

重点：

- field vector 运算。
- batch invert。
- dot product / matrix flatten / vector transform。
- CPU 和 CUDA hotspots 的边界。
- route2 的 parallel/packed 路径是否在这里被使用。

### `src/algebra/polynomial.cpp`

重点：

- polynomial basis。
- coefficient/evaluation 表示。
- domain conversion。
- opening point evaluation。
- spilled evaluation polynomial 的读取。
- 大图上多项式对象如何避免过度复制。

### `src/algebra/eval_backend.cpp`

重点：

- domain evaluation backend。
- packed domain evaluation。
- rotated point kernel。
- parallel FFT/NTT or barycentric evaluation。
- opening 和 quotient 计算如何复用该后端。

### `src/algebra/cuda_backend.cu`

仅 CUDA build 阅读。重点：

- CUDA backend 是否可用检测。
- 哪些 field hotspots 被 offload。
- CUDA 路线如何保持与 CPU 结果一致。
- size-aware routing 的阈值和策略。

---

## 6.3 crypto 层

阅读顺序：

```text
src/crypto/curve.cpp
src/crypto/kzg.cpp
src/crypto/transcript.cpp
```

### `src/crypto/curve.cpp`

重点：

- G1/G2 加减乘。
- same-base batch multiplication。
- fast MSM route。
- pairing 检查。
- backend_name 如何进入 artifact。

### `src/crypto/kzg.cpp`

重点：

- KZG setup。
- commit polynomial。
- open polynomial at point。
- verify opening。
- batch opening 或 folded opening。
- quotient witness 如何生成。
- proof 中 commitment/witness 的传输对象如何构造。

### `src/crypto/transcript.cpp`

重点：

- Fiat-Shamir transcript。
- absorb scalar / commitment。
- challenge 生成。
- challenge label 的稳定性。
- 为什么 prove/verify 必须用同一吸收顺序。

---

## 6.4 data/model 层

阅读顺序：

```text
src/data/loader.cpp
src/model/gat.cpp
```

### `src/data/loader.cpp`

重点：

- 数据集加载入口。
- Planetoid、PPI、ogbn-arxiv 读取和缓存规则。
- full dataset 和 local graph 如何构造。
- edge sort 和 self-loop 规则。
- node_ptr / edge_ptr / graph_count。
- 数据 shape 如何和 config 对齐。

### `src/model/gat.cpp`

重点：

- checkpoint bundle manifest 读取。
- dense tensor 读取。
- hidden layer / output layer 参数解析。
- synthetic model fallback。
- multi-head/multi-layer shape validation。
- GAT reference forward：
  - projection。
  - source/target attention。
  - LeakyReLU。
  - segment softmax。
  - aggregation。
  - ELU。
  - concat。
  - output attention。
  - output bias。
  - output head average。
- `ForwardProfile` 如何填充 metrics。

---

## 6.5 protocol 基础层

阅读顺序：

```text
src/protocol/schema.cpp
src/protocol/challenges.cpp
src/protocol/lookup.cpp
src/protocol/psq.cpp
src/protocol/zkmap.cpp
```

### `src/protocol/schema.cpp`

重点：

- `proof_block_order()`。
- hidden head label。
- concat label。
- output head label。
- single output compatible label。
- label 命名为什么是 proof contract 的一部分。

### `src/protocol/challenges.cpp`

重点：

- `canonical_public_metadata`。
- public metadata 哪些字段会进入 transcript 和 proof size。
- `dynamic_commitment_labels`。
- `quotient_commitment_labels`。
- `domain_opening_labels`。
- `replay_challenges`。
- static/public/dynamic/quotient commitments 的吸收顺序。
- hidden heads 和 output heads challenge 命名。
- prove 和 verify 如何共享挑战真源。

### `src/protocol/lookup.cpp`

重点：

- lookup table/query/multiplicity/accumulator。
- feature lookup。
- activation lookup。
- range lookup。
- exp lookup。
- state machine accumulator。
- selector mask 和 padding。

### `src/protocol/psq.cpp`

重点：

- PSQ 对象的数学含义。
- `w_psq`、`T_psq`、`T_psq_edge`、`PSQ` 如何构造。
- PSQ 如何约束 aggregation。
- lambda_psq challenge 如何参与。

### `src/protocol/zkmap.cpp`

重点：

- zkMap 或 map-style binding 的用途。
- route/binding/helper relation。
- 它和 lookup/PSQ/quotient 的接口关系。

---

## 6.6 protocol trace 层

阅读顺序：

```text
src/protocol/trace.cpp
include/gatzk/protocol/trace.hpp
```

`trace.cpp` 是主仓库中最重的文件之一。建议分块读，不要一次从头到尾硬读。

建议分块顺序：

1. helper functions。
2. dynamic commitment spec。
3. matrix/column materialization。
4. feature lookup trace。
5. hidden head trace。
6. concat trace。
7. output head trace。
8. route/lookup/PSQ objects。
9. dynamic commitment materialization。
10. trace artifact object 组装。
11. metrics 拆分。
12. spilling/cache/large object 处理。
13. CUDA/route2 分支。

核心对象流：

```text
ProtocolContext
  ↓
model forward trace
  ↓
feature / hidden / output witness columns
  ↓
lookup table/query/multiplicity/accumulator
  ↓
route relation
  ↓
PSQ relation
  ↓
dynamic commitment specs
  ↓
TraceArtifacts
```

重点核对：

- trace 只物化 witness 和 dynamic objects，不定义新协议。
- dynamic commitment order 必须等于 `dynamic_commitment_labels(context)`。
- 大图路径中的 cache/spill 不得改变值。
- route2 layout upgrade 只能改变存储和计算方式。
- CUDA hotspots 只能改变执行路径，不能改变 labels、challenge、proof shape。

---

## 6.7 protocol quotient 层

阅读顺序：

```text
src/protocol/quotients.cpp
include/gatzk/protocol/quotients.hpp
```

重点：

- 各 domain 的 quotient 如何计算。
- `t_FH`、`t_edge`、`t_in`、`t_d_h`、`t_cat`、`t_C`、`t_N` 分别约束什么。
- legacy/single-output compat 和 real multihead 之间的边界。
- quotient evaluation 如何依赖 challenge。
- quotient commitment 如何进入 proof 和 transcript。
- 大图上 quotient hotpath 如何优化。

建议按 domain 分块做笔记：

| quotient | domain | 约束对象 | 依赖 trace | 依赖 challenge |
|---|---|---|---|---|
| t_FH | FH | feature / hidden flatten | P_H 等 | eta/beta/y |
| t_edge | edge | route / softmax / PSQ | edge columns | lookup/psq challenges |
| t_in | input dim | projection | H/W/H_prime | y_proj |
| t_d_h | hidden width | hidden head compression | H_prime/H_star | xi/y |
| t_cat | concat width | hidden concat | H_agg/H_cat | xi_cat |
| t_C | class width | output head | Y/Y_star | xi_out |
| t_N | node | node-domain relations | M/Sum/inv/logits | node challenges |

---

## 6.8 protocol proving 层

阅读顺序：

```text
src/protocol/prover.cpp
include/gatzk/protocol/prover.hpp
```

该文件主要包含四类职责。

### 6.8.1 `build_context`

重点：

```text
config
  ↓
dataset
  ↓
local graph
  ↓
model parameters
  ↓
domains
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

核对：

- 数据和模型 shape 如何对齐。
- domain size 如何选择。
- public polynomial 和 static commitment 如何构造。
- context cache 是否只缓存静态对象。
- checkpoint bundle 是否进入 public metadata。

### 6.8.2 `prove`

重点：

```text
TraceArtifacts
  ↓
check dynamic commitment order
  ↓
replay challenges before quotient
  ↓
build quotient polynomials
  ↓
commit quotients
  ↓
replay challenges after quotient
  ↓
build domain openings
  ↓
build external evaluations/witnesses
  ↓
assemble Proof
```

核对：

- proof block order 来自 `proof_block_order()`。
- dynamic commitment labels 来自 `dynamic_commitment_labels()`。
- quotient labels 来自 `quotient_commitment_labels()`。
- domain opening labels 来自 `domain_opening_labels()`。
- public metadata 来自 `canonical_public_metadata()`。

### 6.8.3 `proof_size_bytes`

重点：

- 统计的是传输口径。
- 计入 public metadata、block order、commitments、opening values/witnesses、external evaluations、witness scalars、external witness。
- 不计入可重放的 label/challenge/point。
- README、manifest、tests 使用同一函数作为真源。

### 6.8.4 `export_run_artifacts`

重点：

导出内容通常包括：

```text
metadata.txt
challenges.txt
proof_block_order.txt
commitment_order.txt
benchmark.txt
run_manifest.json
```

核对：

- 正式展示面和机器复核面分开。
- benchmark.txt 只保留最小正式摘要。
- run_manifest.json 保留完整内部字段。
- CPU/GPU export dir 是否分开。

---

## 6.9 protocol verification 层

阅读顺序：

```text
src/protocol/verifier.cpp
include/gatzk/protocol/verifier.hpp
```

重点：

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

必须确认：

- verifier 不做第二套规则。
- verifier 复用 `canonical_public_metadata`。
- verifier 复用 `proof_block_order`。
- verifier 复用 `domain_opening_labels`。
- verifier 的 quotient equation 和 prover 构造保持一致。
- metrics 中 verify_metadata、verify_transcript、verify_domain_opening、verify_quotient、verify_external_fold 如何统计。

---

## 7. 第五阶段：读正式配置

阅读顺序：

```text
configs/cora_full.cfg
configs/citeseer_full.cfg
configs/pubmed_full.cfg
configs/ppi_batch_formal.cfg
configs/ogbn_arxiv_full.cfg
```

每个配置都要记录：

- dataset。
- data_root / cache_root。
- checkpoint_bundle。
- export_dir。
- task_type。
- report_unit。
- batching_rule。
- subgraph_rule。
- self_loop_rule。
- edge_sort_rule。
- chunking_rule。
- hidden_dim。
- num_classes。
- range_bits。
- seed。
- L / layer_count。
- hidden_profile。
- d_in_profile。
- K_out。
- batch_graphs。
- allow_synthetic_model。
- model_arch_id / model_param_id / static_table_id / quant_cfg_id / degree_bound_id。

核对规则：

```text
hidden_profile[layer].head_count * hidden_profile[layer].head_dim
  == d_in_profile[layer + 1]
```

配置阅读要和 `src/util/config.cpp` 一起看，因为配置字段白名单、过时字段拒绝和 shape 校验都在那里实现。

---

## 8. 第六阶段：读脚本工具

阅读顺序：

```text
scripts/fetch_mcl.sh
scripts/planetoid_utils.py
scripts/prepare_planetoid.py
scripts/train_planetoid_gat.py
scripts/check_planetoid_parity.py
scripts/export_gat_checkpoint.py
scripts/checkpoint_reader_compat.py
scripts/prepare_ppi.py
scripts/import_ppi_bundle.py
scripts/ogbn_arxiv_utils.py
scripts/prepare_ogbn_arxiv.py
scripts/train_ogbn_arxiv_gat.py
scripts/export_torch_gat_bundle.py
scripts/export_benchmark_table.py
```

### 8.1 构建依赖脚本

`fetch_mcl.sh`

重点：

- mcl 依赖如何下载/准备。
- 和 `cmake/ConfigureMCL.cmake` 的关系。

### 8.2 Planetoid 数据与训练链

```text
planetoid_utils.py
prepare_planetoid.py
train_planetoid_gat.py
check_planetoid_parity.py
export_gat_checkpoint.py
checkpoint_reader_compat.py
```

重点：

- Cora/Citeseer/Pubmed 数据如何准备。
- PyTorch GAT 如何训练。
- parity 如何检查。
- checkpoint 如何导出为 C++ 读取的 bundle。
- checkpoint reader compatibility 如何验证。

### 8.3 PPI 链

```text
prepare_ppi.py
import_ppi_bundle.py
```

重点：

- PPI 数据如何准备。
- PPI checkpoint bundle 如何导入。
- 多图 batch metadata 如何进入 config。

### 8.4 ogbn-arxiv 链

```text
ogbn_arxiv_utils.py
prepare_ogbn_arxiv.py
train_ogbn_arxiv_gat.py
export_torch_gat_bundle.py
```

重点：

- ogbn-arxiv 数据如何准备。
- 大图 GAT 训练/导出如何处理。
- bundle shape 如何对应 `ogbn_arxiv_full.cfg`。

### 8.5 benchmark 汇总链

`export_benchmark_table.py`

重点：

```text
runs/<dataset>/warm/run_manifest.json
  ↓
validate warm/full/verify ok
  ↓
latest.json
  ↓
latest.csv
  ↓
summary.md
```

核对：

- 是否只接收 `warm`。
- 是否只接收完整数据集。
- 是否只接收 verify ok。
- CPU/GPU output dir 如何区分。
- 展示字段是否只保留四项正式指标。

---

## 9. 第七阶段：读测试

阅读顺序：

```text
tests/test_main.cpp
```

该文件很长，建议按测试职责分块读。

### 9.1 基础 helper

重点：

- repo_root。
- slurp_file/write_text。
- JSON/CSV 解析 helper。
- require 函数。

### 9.2 config fixture

重点：

- small_debug_config。
- full_graph_formal_config。
- full_citeseer_formal_config。
- ppi_full_formal_config。
- ogbn_arxiv_full_formal_config。
- synthetic family config。

### 9.3 proof fixture

重点：

```text
build_context
  ↓
build_trace
  ↓
prove
```

该 helper 用于快速构造可测试 proof。

### 9.4 schema / metadata / challenge tests

重点：

- proof block order。
- dynamic commitment order。
- public metadata。
- replay_challenges。
- domain opening labels。

### 9.5 proof / verify tests

重点：

- prove 成功。
- verify 成功。
- tampering proof 是否失败。
- metadata mismatch 是否失败。
- block order mismatch 是否失败。
- opening mismatch 是否失败。

### 9.6 artifact truth source tests

重点：

- run_manifest.json。
- latest.json。
- latest.csv。
- summary.md。
- README table。
- CPU/GPU official results 是否一致。
- proof_size_bytes 是否一致。

### 9.7 cleanliness / cleanup regression

重点：

- 是否存在已删除实验目录。
- 是否存在 placeholder marker。
- 是否存在 conflict marker。
- 是否存在旧 trial/smoke artifact。
- 正式结果表是否没有混入实验性字段。

---

## 10. 第八阶段：读 reference

阅读顺序：

```text
reference/gat_ogbn_arxiv/pyg_gat_example.py
```

用途：

- 作为 ogbn-arxiv GAT 参考样例。
- 辅助理解上游 PyG 模型和本仓库 C++/formal GAT 语义的对应关系。
- 不作为正式 proof 语义真源。

---

## 11. 第九阶段：读正式运行产物

`runs/` 和 `artifacts/` 不都是代码，但它们是理解代码逻辑闭环的必要真源。

### 11.1 CPU benchmark truth source

阅读顺序：

```text
runs/benchmarks/latest.json
runs/benchmarks/latest.csv
runs/benchmarks/summary.md
```

### 11.2 GPU benchmark truth source

阅读顺序：

```text
runs/benchmarks/cuda_formal/latest.json
runs/benchmarks/cuda_formal/latest.csv
runs/benchmarks/cuda_formal/summary.md
```

### 11.3 每个 dataset 的 per-run artifact

CPU 目录：

```text
runs/cora_full/warm/
runs/citeseer_full/warm/
runs/pubmed_full/warm/
runs/ppi_full_formal/warm/
runs/ogbn_arxiv_full/warm/
```

GPU 目录：

```text
runs/cora_full/cuda_formal/warm/
runs/citeseer_full/cuda_formal/warm/
runs/pubmed_full/cuda_formal/warm/
runs/ppi_full_formal/cuda_formal/warm/
runs/ogbn_arxiv_full/cuda_formal/warm/
```

每个目录按以下顺序读：

```text
run_manifest.json
benchmark.txt
metadata.txt
challenges.txt
proof_block_order.txt
commitment_order.txt
```

如果目录中存在更多内部文件，也按“先 manifest，再正式摘要，再 proof/block/order/challenge，再其他 detail”顺序读。

### 11.4 checkpoint bundles

目录通常位于：

```text
artifacts/checkpoints/
```

阅读重点：

- checkpoint manifest。
- tensor shape。
- hidden profile。
- d_in_profile。
- K_out。
- model_param_id。
- 是否和 config 对齐。

---

## 12. 全仓推荐阅读顺序总表

下面是完整建议顺序。按这个顺序读，可以最大程度避免漏掉逻辑链。

### 12.1 文档和构建

```text
README.md
GAT-ZKML-多层多头.md
项目整体逻辑与方法流程.md
docs/热点优化方法与处理逻辑.md
CMakeLists.txt
cmake/ConfigureMCL.cmake
```

### 12.2 头文件

```text
include/gatzk/algebra/field.hpp
include/gatzk/algebra/domain_evaluation.hpp
include/gatzk/algebra/polynomial.hpp
include/gatzk/algebra/eval_backend.hpp
include/gatzk/algebra/vector_ops.hpp
include/gatzk/algebra/cuda_field.cuh

include/gatzk/crypto/curve.hpp
include/gatzk/crypto/kzg.hpp
include/gatzk/crypto/transcript.hpp

include/gatzk/data/dataset.hpp
include/gatzk/data/loader.hpp

include/gatzk/model/gat.hpp

include/gatzk/util/config.hpp
include/gatzk/util/logging.hpp
include/gatzk/util/route2.hpp

include/gatzk/protocol/proof.hpp
include/gatzk/protocol/schema.hpp
include/gatzk/protocol/challenges.hpp
include/gatzk/protocol/lookup.hpp
include/gatzk/protocol/psq.hpp
include/gatzk/protocol/zkmap.hpp
include/gatzk/protocol/quotients.hpp
include/gatzk/protocol/trace.hpp
include/gatzk/protocol/prover.hpp
include/gatzk/protocol/verifier.hpp
```

### 12.3 实现文件

```text
src/util/config.cpp
src/util/route2.cpp
src/util/logging.cpp

src/algebra/vector_ops.cpp
src/algebra/polynomial.cpp
src/algebra/eval_backend.cpp
src/algebra/cuda_backend.cu

src/crypto/curve.cpp
src/crypto/kzg.cpp
src/crypto/transcript.cpp

src/data/loader.cpp
src/model/gat.cpp

src/protocol/schema.cpp
src/protocol/challenges.cpp
src/protocol/lookup.cpp
src/protocol/psq.cpp
src/protocol/zkmap.cpp
src/protocol/trace.cpp
src/protocol/quotients.cpp
src/protocol/prover.cpp
src/protocol/verifier.cpp

src/main.cpp
```

### 12.4 配置

```text
configs/cora_full.cfg
configs/citeseer_full.cfg
configs/pubmed_full.cfg
configs/ppi_batch_formal.cfg
configs/ogbn_arxiv_full.cfg
```

### 12.5 脚本

```text
scripts/fetch_mcl.sh
scripts/planetoid_utils.py
scripts/prepare_planetoid.py
scripts/train_planetoid_gat.py
scripts/check_planetoid_parity.py
scripts/export_gat_checkpoint.py
scripts/checkpoint_reader_compat.py
scripts/prepare_ppi.py
scripts/import_ppi_bundle.py
scripts/ogbn_arxiv_utils.py
scripts/prepare_ogbn_arxiv.py
scripts/train_ogbn_arxiv_gat.py
scripts/export_torch_gat_bundle.py
scripts/export_benchmark_table.py
```

### 12.6 测试和参考

```text
tests/test_main.cpp
reference/gat_ogbn_arxiv/pyg_gat_example.py
```

### 12.7 产物和结果

```text
runs/benchmarks/latest.json
runs/benchmarks/latest.csv
runs/benchmarks/summary.md
runs/benchmarks/cuda_formal/latest.json
runs/benchmarks/cuda_formal/latest.csv
runs/benchmarks/cuda_formal/summary.md

runs/cora_full/warm/run_manifest.json
runs/citeseer_full/warm/run_manifest.json
runs/pubmed_full/warm/run_manifest.json
runs/ppi_full_formal/warm/run_manifest.json
runs/ogbn_arxiv_full/warm/run_manifest.json

runs/cora_full/cuda_formal/warm/run_manifest.json
runs/citeseer_full/cuda_formal/warm/run_manifest.json
runs/pubmed_full/cuda_formal/warm/run_manifest.json
runs/ppi_full_formal/cuda_formal/warm/run_manifest.json
runs/ogbn_arxiv_full/cuda_formal/warm/run_manifest.json
```

---

## 13. 按功能链路二轮精读

第一轮按文件读，第二轮按功能链路读。

### 13.1 配置链路

```text
configs/*.cfg
  ↓
include/gatzk/util/config.hpp
  ↓
src/util/config.cpp
  ↓
src/main.cpp
```

目标：理解一次 run 的所有公共输入。

### 13.2 数据链路

```text
scripts/prepare_*.py
  ↓
include/gatzk/data/*.hpp
  ↓
src/data/loader.cpp
  ↓
ProtocolContext.local
```

目标：理解 full dataset、graph、edge sort、self-loop、batch pointer。

### 13.3 模型链路

```text
scripts/train_*.py / export_*checkpoint*.py
  ↓
artifacts/checkpoints/*
  ↓
include/gatzk/model/gat.hpp
  ↓
src/model/gat.cpp
  ↓
build_context / build_trace
```

目标：理解参数、shape、reference forward。

### 13.4 代数链路

```text
include/gatzk/algebra/*.hpp
  ↓
src/algebra/*.cpp
  ↓
trace materialization
  ↓
quotient / opening
```

目标：理解 domain、polynomial、evaluation backend。

### 13.5 密码链路

```text
include/gatzk/crypto/*.hpp
  ↓
src/crypto/*.cpp
  ↓
prover.cpp
  ↓
verifier.cpp
```

目标：理解 KZG commit/open/verify 和 transcript。

### 13.6 schema / challenge 链路

```text
schema.cpp
  ↓
challenges.cpp
  ↓
trace.cpp
  ↓
prover.cpp
  ↓
verifier.cpp
```

目标：理解 label/order/challenge 是 proof contract。

### 13.7 witness / trace 链路

```text
build_context
  ↓
build_trace
  ↓
TraceArtifacts
  ↓
dynamic commitments
```

目标：理解模型前向如何变成可承诺列。

### 13.8 quotient / opening 链路

```text
TraceArtifacts
  ↓
quotients.cpp
  ↓
KZG commitments
  ↓
domain openings
  ↓
Proof
```

目标：理解约束如何变成 quotient 和 opening。

### 13.9 verify 链路

```text
Proof
  ↓
verifier.cpp
  ↓
replay_challenges
  ↓
check openings
  ↓
check quotients
  ↓
VERIFY_OK
```

目标：理解验证方不重跑完整前向，但重放 formal 检查。

### 13.10 artifact / benchmark 链路

```text
export_run_artifacts
  ↓
runs/<dataset>/warm/run_manifest.json
  ↓
scripts/export_benchmark_table.py
  ↓
runs/benchmarks/latest.*
  ↓
tests/test_main.cpp
```

目标：理解正式结果表如何和代码真源一致。

---

## 14. 主仓库核心对象关系图

```text
AppConfig
  ├─ dataset/checkpoint/export settings
  ├─ hidden_profile / d_in_profile / K_out
  └─ route constraints

Dataset + LocalGraph
  ├─ node features
  ├─ edges
  ├─ node_ptr / edge_ptr
  └─ full dataset metadata

ModelParameters
  ├─ hidden_layers
  ├─ output_layer
  ├─ checkpoint metadata
  └─ reference forward

ProtocolContext
  ├─ config
  ├─ dataset/local graph
  ├─ model
  ├─ domains
  ├─ static tables
  ├─ public polynomials
  ├─ static/public commitments
  └─ KZG keypair

TraceArtifacts
  ├─ forward trace
  ├─ dynamic commitments
  ├─ dynamic polynomials
  ├─ lookup objects
  ├─ route objects
  ├─ PSQ objects
  ├─ external evaluations
  └─ witness scalars

Proof
  ├─ PublicMetadata
  ├─ proof_block_order
  ├─ dynamic commitments
  ├─ quotient commitments
  ├─ domain openings
  ├─ external evaluations
  ├─ witness scalars
  └─ challenges

RunArtifacts
  ├─ metadata.txt
  ├─ challenges.txt
  ├─ proof_block_order.txt
  ├─ commitment_order.txt
  ├─ benchmark.txt
  └─ run_manifest.json
```

---

## 15. 不可改边界清单

阅读代码时，凡涉及以下内容，都要视为 formal contract：

- `GAT-ZKML-多层多头.md` 的协议语义。
- `proof_block_order()`。
- `dynamic_commitment_labels(context)`。
- `quotient_commitment_labels(context)`。
- `domain_opening_labels(context)`。
- `canonical_public_metadata(context)`。
- `replay_challenges(...)` 的吸收顺序。
- proof serializer/parser 结构。
- verifier equations。
- `proof_size_bytes(const Proof&)` 统计口径。
- per-run artifact 与 benchmark 主表字段对应关系。
- README、latest.json、latest.csv、summary.md、run_manifest.json 的一致性。

---

## 16. 常见误读点

### 16.1 误把 `src/main.cpp` 当作协议定义

`src/main.cpp` 只是运行编排层。协议定义分散在 formal note、schema、challenges、trace、quotients、prover、verifier 中。

### 16.2 误把 GPU 当第二套协议

GPU 路径只改变执行层热点，不改变 proof schema、challenge、label、quotient、opening、verifier equations。

### 16.3 误把 proof size 当作文件大小

`proof_size_bytes` 是传输口径，不是 run directory 大小，也不是 C++ 对象内存占用。

### 16.4 误以为节点/边越多 proof 一定越大

当前 proof size 主要取决于 proof shape，例如 hidden profile、dynamic commitment 数量、opening 数量，而不是单独由图规模线性决定。

### 16.5 误以为 verifier 会重跑完整 GAT forward

verifier 主要重放 public metadata、transcript、opening 和 quotient equations，不是重新跑一遍完整模型前向。

### 16.6 误把 scripts 中训练逻辑当 formal benchmark

训练脚本只产生参数来源。formal benchmark 主线从配置、数据、checkpoint bundle 进入 `gatzk_run`，只统计 prove/verify 等正式字段。

---

## 17. 阅读笔记模板

建议读每个文件时记录：

| 文件 | 层级 | 主要职责 | 输入 | 输出 | 关键函数/结构 | 是否 formal contract | 疑问 |
|---|---|---|---|---|---|---:|---|
| src/main.cpp | entry | 运行编排 | config/flags | artifacts/result | run_once | 否 |  |
| src/protocol/challenges.cpp | protocol | transcript/metadata | context/commitments | challenges | replay_challenges | 是 |  |
| src/protocol/prover.cpp | protocol | context/proof/artifact | config/trace | proof | build_context/prove | 是 |  |
| src/protocol/verifier.cpp | protocol | 验证 | proof/context | bool | verify | 是 |  |
| scripts/export_benchmark_table.py | tool | 汇总结果 | manifests | latest/summary | main | 否 |  |

---

## 18. 最终完成标准

当你能不看本文，独立复述下面这条链，并指出每一步在哪个文件实现，才算真正读完主仓库代码逻辑：

```text
config file
  ↓
util::load_config
  ↓
data::load_dataset / model::load_checkpoint_bundle
  ↓
protocol::build_context
  ↓
model::forward
  ↓
protocol::build_trace
  ↓
dynamic commitments
  ↓
protocol::replay_challenges
  ↓
protocol::build_quotients
  ↓
KZG quotient commitments
  ↓
domain openings
  ↓
Proof assembly
  ↓
protocol::verify
  ↓
proof_size_bytes
  ↓
export_run_artifacts
  ↓
export_benchmark_table.py
  ↓
latest.json / latest.csv / summary.md
  ↓
tests/test_main.cpp consistency checks
```

---

## 19. 推荐阅读节奏

### 第一天

- README。
- formal note。
- 项目整体流程。
- CMake。
- `src/main.cpp`。
- 全部头文件。

### 第二天

- util。
- algebra。
- crypto。
- data。
- model。
- configs。

### 第三天

- protocol schema。
- challenges。
- lookup。
- psq。
- zkmap。
- trace。

### 第四天

- quotients。
- prover。
- verifier。
- tests。
- scripts。
- runs/benchmarks。

### 第五天

- 做一次完整反向追溯：
  - 从 README 结果表追到 summary。
  - 从 summary 追到 latest.csv/latest.json。
  - 从 latest.json 追到 run_manifest。
  - 从 run_manifest 追到 `export_run_artifacts`。
  - 从 proof_size 追到 `proof_size_bytes`。
  - 从 VERIFY_OK 追到 `verify`。
  - 从 proof labels 追到 `schema/challenges`。
  - 从 trace commitments 追到 `trace.cpp`。
  - 从 forward 中间量追到 `model/gat.cpp`。

---

## 20. 附：最小必读清单

如果时间有限，最少不能跳过：

```text
README.md
GAT-ZKML-多层多头.md
项目整体逻辑与方法流程.md
CMakeLists.txt
src/main.cpp
include/gatzk/protocol/proof.hpp
include/gatzk/protocol/schema.hpp
include/gatzk/protocol/challenges.hpp
include/gatzk/protocol/trace.hpp
include/gatzk/protocol/prover.hpp
include/gatzk/protocol/verifier.hpp
src/util/config.cpp
src/data/loader.cpp
src/model/gat.cpp
src/protocol/schema.cpp
src/protocol/challenges.cpp
src/protocol/trace.cpp
src/protocol/quotients.cpp
src/protocol/prover.cpp
src/protocol/verifier.cpp
configs/*.cfg
scripts/export_benchmark_table.py
tests/test_main.cpp
runs/benchmarks/latest.json
runs/*/warm/run_manifest.json
```

但如果目标是“完整理解主仓库全部逻辑”，不能停在最小清单，应按本文完整顺序读完。
