# ezkl C++ 对比实验完整阅读 README

本文用于指导从零开始完整阅读并理解 `project/comparison/baselines/ezkl` 目录下的 ezkl 对比实验代码。该目录不是主协议实现本身，而是为了和主项目 `GAT-ZKML` 正式实现进行对比而保留的一套 baseline 实验目录。当前实验的核心特征是：使用 C++ adapter 生成 GAT full-dataset 对比输入、元数据、ONNX/graph 相关 artifact，再通过 ezkl/Rust 侧命令完成 setup、witness、prove、verify，最后由脚本汇总正式结果。

本 README 的目标不是替代源码阅读，而是给出一个不跳步、不遗漏主线文件的阅读顺序。建议严格按本文顺序阅读；如果直接从某个实现文件开始，很容易把 C++ adapter、ezkl CLI、Rust 承载层、结果脚本和 run artifact 之间的关系看散。

---

## 1. 阅读目标

完成本文阅读后，应能回答以下问题：

1. 这个 baseline 实验到底比较什么。
2. 为什么它被称为 `ezkl_gat_native_compressed_csr_attention_full` strict route。
3. C++ adapter 负责哪些事情，Rust/ezkl 负责哪些事情。
4. 五个 full dataset 的输入、模型、图结构和 checkpoint 是如何进入对比流程的。
5. GAT forward、segment softmax、量化、formal table、metadata、ONNX 导出和 witness 生成分别在哪些文件中实现。
6. 每个 run directory 中的 artifact 是怎么产生、怎么验证、怎么被汇总到最终结果表中的。
7. 哪些文件是实验主线必须读的，哪些属于 upstream ezkl 通用代码，但仍然需要理解其在 prove/verify 中的作用。
8. 如何从最终结果反向追溯到单次运行的配置、输入、模型、证明和验证记录。

---

## 2. 总体架构

该目录可以按职责分成六层：

```text
comparison/baselines/ezkl/
├── README.md                         实验说明和正式运行入口
├── Cargo.toml / Cargo.lock           ezkl/Rust 构建与依赖
├── src/                              ezkl/Rust CLI、graph、circuit、pfsys、tensor 等承载层
├── cpp/gat_ezkl_adapter/             本项目用于 baseline 的 C++ GAT adapter
├── configs/                          full dataset 和 strict route 配置
├── scripts/                          运行、汇总、校验脚本
├── results/                          正式结果、审计报告、清理报告、对比汇总
└── runs/                             每个数据集单次运行产物
```

最重要的执行链如下：

```text
configs/*.json
  ↓
cpp/gat_ezkl_adapter
  ↓
导出 graph_attention_spec / formal_tables / binding metadata / model info / ONNX or graph inputs
  ↓
scripts/run_gat_ezkl_baseline.py
  ↓
ezkl Rust CLI commands
  ↓
setup / witness / prove / verify
  ↓
runs/<dataset>/run_manifest.json
  ↓
scripts/collect_gat_ezkl_results.py
  ↓
results/gat_ezkl_full_results.*
  ↓
scripts/validate_final_results.py
  ↓
results/main_vs_ezkl_full_comparison.*
```

阅读时要始终记住：C++ adapter 是 baseline 实验的输入构造与模型语义侧重点；ezkl/Rust 侧是证明系统、CLI 和电路承载层；scripts 是把两者串起来的正式实验编排层；results 和 runs 是可复核输出层。

---

## 3. 第一阶段：先读实验声明和边界

这一阶段只做一件事：弄清楚这个目录保留的到底是哪一条实验路线，以及哪些路线已经被排除。

阅读顺序：

```text
comparison/baselines/ezkl/README.md
comparison/baselines/ezkl/results/cleanup_report.md
comparison/baselines/ezkl/results/code_cleanliness_audit.md
comparison/baselines/ezkl/results/control_variable_audit.md
comparison/baselines/ezkl/results/large_public_output_policy_review.md
comparison/baselines/ezkl/results/native_full_experiment_report.md
comparison/baselines/ezkl/results/native_constraint_debug_report.md
```

阅读重点：

### 3.1 `README.md`

先看该 baseline 的正式定位：

- 是否是 full dataset。
- 是否是 CPU strict route。
- 是否允许 digest、folding、identity、precomputed logits、subset、sample、shard 等替代路线。
- 五个数据集分别如何运行。
- 输出结果有哪些。
- 每个 run directory 中应包含哪些审计 artifact。

### 3.2 `cleanup_report.md`

这个文件用于确认最终保留下来的实验表面。重点看：

- 哪些目录是最终保留目录。
- 哪些文件属于正式对比实验主线。
- 哪些中间产物被删除。
- 每个 run directory 保留哪些最小 artifact。

### 3.3 `code_cleanliness_audit.md`

用于判断哪些 TODO、stub、placeholder 属于 upstream ezkl 通用代码，哪些会影响当前 strict route。阅读时不要看到 TODO 就立刻认为实验无效；要看审计是否把它排除在当前 route 外。

### 3.4 `control_variable_audit.md`

用于理解对比实验的控制变量。重点检查：

- 数据集是否一致。
- 模型参数是否一致。
- 输入规模是否一致。
- 是否使用 full graph / full dataset。
- 是否引入主项目没有使用的额外捷径。

### 3.5 `large_public_output_policy_review.md`

用于理解 public output / large output 的处理策略。重点看哪些输出被公开、哪些被绑定、哪些被压缩或记录为 metadata。

### 3.6 `native_full_experiment_report.md`

这是 full experiment 的总结性文件。重点看每个 dataset 的运行状态、最终结果字段、是否 verify 成功。

### 3.7 `native_constraint_debug_report.md`

这个文件用于理解 strict route 中 constraint/debug 相关问题如何被处理。读它可以帮助你后面理解 `native_gat.rs` 和 C++ adapter 的关系。

---

## 4. 第二阶段：读构建入口和依赖

阅读顺序：

```text
comparison/baselines/ezkl/Cargo.toml
comparison/baselines/ezkl/Cargo.lock
comparison/baselines/ezkl/cpp/gat_ezkl_adapter/CMakeLists.txt
```

阅读重点：

### 4.1 `Cargo.toml`

关注：

- package name。
- binary 名称。
- feature flags。
- 依赖的 proving、halo2、onnx、serde、clap、rayon、tensor 相关 crate。
- 是否存在 `native_gat` 或 graph GAT command 相关 feature。

### 4.2 `Cargo.lock`

不用逐行背依赖，但要知道：

- 它固定了 ezkl/Rust 侧依赖版本。
- 它保证对比实验可复现。
- 如果 prove/verify 行为变化，`Cargo.lock` 是必须检查的文件之一。

### 4.3 `cpp/gat_ezkl_adapter/CMakeLists.txt`

这是 C++ adapter 的构建真源。必须核对：

- C++ 标准。
- target 名称。
- include 路径。
- 编译进 target 的所有 `.cpp` 文件。
- 是否链接外部库。
- 是否有 test target 或单独 executable。

该文件中列出的 `.cpp` 文件必须全部阅读，不能只读 `main.cpp`。

---

## 5. 第三阶段：读 C++ adapter 头文件

先读头文件，是为了建立对象模型。不要一开始直接进入 `.cpp`，否则会被细节淹没。

阅读顺序：

```text
cpp/gat_ezkl_adapter/include/gat_ezkl/hash.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/config.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/dataset.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/graph.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/checkpoint.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/quant.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/bounds.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/formal_tables.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/segment_softmax.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/gat.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/metadata.hpp
cpp/gat_ezkl_adapter/include/gat_ezkl/onnx_export.hpp
```

### 5.1 `hash.hpp`

理解：

- hash 工具用于什么 artifact。
- hash 是用于绑定文件、表、metadata，还是用于命名和审计。
- hash 输入是否稳定排序。
- hash 输出格式是否固定。

### 5.2 `config.hpp`

理解：

- C++ adapter 接收哪些配置字段。
- full dataset 配置如何被解析。
- route 配置如何绑定。
- export directory、dataset name、checkpoint path、graph path 等路径如何表示。
- 哪些字段属于正式实验约束，哪些只是运行参数。

### 5.3 `dataset.hpp`

理解：

- dataset 的基本结构。
- 节点数、边数、特征维、类别数如何记录。
- label、mask、feature 是否读取。
- 是否支持多图 batch。
- 是否强制 full dataset。

### 5.4 `graph.hpp`

理解：

- 图结构如何表示。
- CSR 或 compressed sparse row 表达如何构造。
- src/dst、row pointer、col index 如何排序。
- self-loop 是否添加。
- edge order 是否稳定。
- 图输入如何和 GAT attention 对齐。

### 5.5 `checkpoint.hpp`

理解：

- checkpoint bundle 的结构。
- 隐藏层、输出层、head、weight、attention vector、bias 如何读取。
- 参数 shape 如何校验。
- checkpoint metadata 如何绑定到实验输出。

### 5.6 `quant.hpp`

理解：

- float 到整数或定点表示如何转换。
- scale、rounding、clamp、signed range 如何定义。
- C++ adapter 中的量化语义是否和主项目 formal note 对齐。
- 输出 logits、attention score、softmax weight 是否使用同一量化策略。

### 5.7 `bounds.hpp`

理解：

- 各类值的上下界如何计算。
- bounds 是用于 ezkl settings、input range、table range，还是 debug。
- 是否存在为某个数据集单独放宽的范围。
- bounds 是否进入 artifact。

### 5.8 `formal_tables.hpp`

理解：

- formal table 包含哪些表。
- LeakyReLU / ELU / exp / range 等表如何生成。
- table hash 如何生成。
- table 是否与主协议对齐。
- table 是否进入 `formal_tables.json` 和 `formal_table_hashes.json`。

### 5.9 `segment_softmax.hpp`

理解：

- GAT attention 的按目标节点 segment softmax 如何实现。
- max-subtract、exp、sum、inverse、alpha 如何计算。
- 空 segment、孤立节点、自环如何处理。
- 量化 softmax 和 formal table 如何衔接。

### 5.10 `gat.hpp`

理解：

- GAT forward 的核心数据结构。
- hidden layer head 如何表示。
- output head 如何表示。
- projection、attention score、activation、concat、output average 如何表示。
- C++ baseline 是否完全复现 full GAT，而不是只导出简化 logits。

### 5.11 `metadata.hpp`

理解：

- 输出 metadata 包含哪些字段。
- model info、graph attention spec、binding metadata 如何生成。
- dataset、checkpoint、route、table hash、shape、runtime config 如何绑定。

### 5.12 `onnx_export.hpp`

理解：

- ONNX 导出内容。
- 哪些输入被导入 ONNX。
- graph attention spec 和 ONNX 之间如何对应。
- ezkl 侧使用的是 ONNX 图、native graph route，还是两者组合。

---

## 6. 第四阶段：读 C++ adapter 实现

实现文件按依赖顺序读，而不是按字母顺序读。

阅读顺序：

```text
cpp/gat_ezkl_adapter/src/hash.cpp
cpp/gat_ezkl_adapter/src/config.cpp
cpp/gat_ezkl_adapter/src/dataset.cpp
cpp/gat_ezkl_adapter/src/graph.cpp
cpp/gat_ezkl_adapter/src/checkpoint.cpp
cpp/gat_ezkl_adapter/src/quant.cpp
cpp/gat_ezkl_adapter/src/bounds.cpp
cpp/gat_ezkl_adapter/src/formal_tables.cpp
cpp/gat_ezkl_adapter/src/segment_softmax.cpp
cpp/gat_ezkl_adapter/src/gat.cpp
cpp/gat_ezkl_adapter/src/metadata.cpp
cpp/gat_ezkl_adapter/src/onnx_export.cpp
cpp/gat_ezkl_adapter/src/main.cpp
```

### 6.1 `hash.cpp`

核对所有 hash 的确定性：

- 文件内容 hash。
- JSON hash。
- table hash。
- graph spec hash。
- 是否有非确定性字段，如当前时间、绝对路径、随机数。

### 6.2 `config.cpp`

跟踪配置读取流程：

```text
argv / config path
  ↓
load JSON config
  ↓
resolve dataset config
  ↓
resolve route config
  ↓
construct adapter config
```

检查点：

- 缺字段时是否报错。
- 默认值是否合理。
- dataset 名称是否和后续脚本一致。
- route 是否被固定为 strict baseline。

### 6.3 `dataset.cpp`

跟踪数据读取流程：

```text
data root
  ↓
features / labels / masks / graph edges
  ↓
dataset object
  ↓
shape validation
```

检查点：

- full dataset 是否被完整读取。
- feature shape 是否匹配 checkpoint 输入维度。
- edge list 是否完整。
- label/mask 是否只用于报告，不影响证明 shortcut。

### 6.4 `graph.cpp`

跟踪图结构转换：

```text
edge list
  ↓
self-loop rule
  ↓
sort rule
  ↓
CSR row pointer / col index
  ↓
graph attention layout
```

检查点：

- src/dst 是否和 formal note 一致。
- CSR 是按目标节点聚合，还是按源节点聚合。
- segment softmax 的 segment 是 incoming edges 还是 outgoing edges。
- edge sort 是否稳定。
- self-loop 是否和主项目保持一致。

### 6.5 `checkpoint.cpp`

跟踪参数读取：

```text
checkpoint bundle
  ↓
hidden layers
  ↓
attention heads
  ↓
output heads
  ↓
shape and metadata validation
```

检查点：

- hidden profile 是否对应 `8x8` 或 `1x8` 等配置。
- output head 数是否正确。
- bias 是否只在 output head 加入。
- 参数维度是否和 dataset feature/class 对齐。

### 6.6 `quant.cpp`

跟踪所有量化入口：

- feature quantization。
- weight quantization。
- attention score quantization。
- activation quantization。
- exp/softmax quantization。
- logits quantization。

检查点：

- rounding 方式是否固定。
- signed integer 到 field/int 的解释是否一致。
- 是否有 dataset-specific 特例。
- 是否使用 hidden shortcut。

### 6.7 `bounds.cpp`

理解 ezkl settings 或 range 相关输入如何生成。

检查点：

- bounds 是否来自真实前向统计。
- bounds 是否可能泄露 witness。
- bounds 是否只是为了生成安全范围。
- bounds 是否和 results 中的 policy review 对齐。

### 6.8 `formal_tables.cpp`

跟踪 table 生成：

```text
quant config
  ↓
range table
  ↓
activation table
  ↓
exp table
  ↓
hash and export
```

检查点：

- table domain 是否完整。
- table 是否确定性生成。
- table hash 是否写入 artifact。
- 表内容是否和 Rust/native route 读取一致。

### 6.9 `segment_softmax.cpp`

这是 GAT attention 的关键文件之一。逐函数理解：

```text
for each destination node:
    collect incoming edges
    compute score max
    compute shifted scores
    exp lookup / approximation
    sum
    inverse
    alpha
```

检查点：

- segment 边界来自 CSR。
- max-subtract 是否使用目标节点 incoming edges。
- exp 是真实计算、lookup 还是近似。
- alpha scale 是否固定。
- degree 为 0 的节点如何处理。
- 数值稳定性如何保证。

### 6.10 `gat.cpp`

这是 C++ adapter 的核心。建议按 GAT forward 语义读：

```text
input features
  ↓
hidden layer projection
  ↓
source/target attention score
  ↓
edge score
  ↓
LeakyReLU
  ↓
segment softmax
  ↓
weighted aggregation
  ↓
ELU
  ↓
multi-head concat
  ↓
next hidden layer or output layer
  ↓
output attention
  ↓
output bias
  ↓
multi-output-head average
  ↓
final logits
```

检查点：

- 是否支持多层多头。
- 每个 head 的参数是否独立。
- concat 顺序是否固定。
- 输出层是否仍然是 attention head，而不是普通 linear。
- output bias 加在什么位置。
- 多输出头是否取平均。
- final logits 是否写入 binding artifact。

### 6.11 `metadata.cpp`

理解 artifact 的生成和绑定：

- `model_info.json`
- `graph_attention_spec.json`
- `final_logits_binding.json`
- `formal_tables.json`
- `formal_table_hashes.json`
- `native_full_witness_preflight.json`

检查点：

- metadata 是否包含 dataset、shape、route、checkpoint。
- hash 是否绑定输入和输出。
- final logits 是否可追溯。
- 是否有足够信息复核单次运行。

### 6.12 `onnx_export.cpp`

理解导出给 ezkl 的模型/图输入。检查：

- 导出的 graph 是否包含 full attention。
- 是否导出真实 logits，而不是预计算 output。
- 输入 tensor 名称、shape、dtype 是否稳定。
- settings 生成是否和 route 对齐。

### 6.13 `main.cpp`

最后读入口。把前面所有对象串起来：

```text
parse args
  ↓
load config
  ↓
load dataset
  ↓
build graph layout
  ↓
load checkpoint
  ↓
generate formal tables
  ↓
run C++ GAT forward
  ↓
generate metadata and bindings
  ↓
export ONNX / graph spec / witness preflight
```

检查点：

- 有没有绕过真实 GAT forward。
- 有没有只导出 digest。
- 有没有只处理 sample/subset。
- 出错是否 fail-fast。
- 输出目录结构是否和 scripts 期待一致。

---

## 7. 第五阶段：读配置文件

阅读顺序：

```text
comparison/baselines/ezkl/configs/full_datasets.json
comparison/baselines/ezkl/configs/ezkl_routes.json
```

### 7.1 `full_datasets.json`

需要记录每个 dataset 的：

- dataset key。
- raw data path。
- processed/cache path。
- checkpoint path。
- node count。
- edge count。
- feature dimension。
- class count。
- task type。
- graph batching rule。
- self-loop rule。
- full dataset flag。

### 7.2 `ezkl_routes.json`

需要确认：

- route name。
- 是否 strict。
- 是否 native compressed CSR attention full。
- 是否禁用 digest/folding/identity/subset/sample/precomputed logits。
- settings path 或 command 参数。
- proving/verification pipeline 是否固定。

---

## 8. 第六阶段：读脚本编排

阅读顺序：

```text
comparison/baselines/ezkl/scripts/run_gat_ezkl_baseline.py
comparison/baselines/ezkl/scripts/collect_gat_ezkl_results.py
comparison/baselines/ezkl/scripts/validate_final_results.py
```

### 8.1 `run_gat_ezkl_baseline.py`

这是正式运行脚本。按函数读：

```text
parse CLI args
  ↓
load dataset config
  ↓
select route
  ↓
invoke C++ adapter
  ↓
invoke ezkl commands
  ↓
capture timings
  ↓
write run_manifest.json
```

重点核对：

- 是否逐 dataset 运行。
- 是否每次都使用 full dataset。
- 是否调用 C++ adapter executable。
- 是否调用 graph-gat 相关 ezkl command。
- setup、witness、prove、verify 的顺序是否固定。
- 错误是否停止运行。
- timing 字段如何统计。
- proof size 如何统计。

### 8.2 `collect_gat_ezkl_results.py`

这是结果汇总脚本。理解：

```text
runs/<dataset>/run_manifest.json
  ↓
extract formal metrics
  ↓
write results/gat_ezkl_full_results.json
  ↓
write results/gat_ezkl_full_results.csv
```

检查点：

- 只汇总 verify 成功的 run。
- 字段名称是否和 README 一致。
- 数据集顺序是否固定。
- 是否混入旧结果。
- 是否读取了正确 run directory。

### 8.3 `validate_final_results.py`

这是最终校验脚本。理解：

- 如何检查五个 dataset 是否齐全。
- 如何检查 route 是否一致。
- 如何检查 artifact 是否存在。
- 如何检查 result 表和 run manifest 是否一致。
- 如何检查 main vs ezkl comparison 是否完整。

---

## 9. 第七阶段：读 ezkl/Rust 承载层

虽然你的对比实验核心是 C++ adapter，但 strict baseline 最终仍然依赖 ezkl/Rust 侧完成电路、witness、prove 和 verify。因此 Rust 文件不能完全跳过。

优先阅读顺序如下。

### 9.1 CLI 和命令层

```text
comparison/baselines/ezkl/src/bin/ezkl.rs
comparison/baselines/ezkl/src/commands.rs
comparison/baselines/ezkl/src/native_gat.rs
comparison/baselines/ezkl/src/lib.rs
comparison/baselines/ezkl/src/execute.rs
```

重点：

- CLI 如何注册 command。
- graph-gat 相关 command 如何进入 `native_gat.rs`。
- `graph-gat-gen-settings`、`graph-gat-compile`、`graph-gat-setup`、`graph-gat-gen-witness`、`graph-gat-prove`、`graph-gat-verify` 分别做什么。
- C++ adapter 导出的 artifact 如何被 Rust 侧读取。

### 9.2 proving system 层

```text
comparison/baselines/ezkl/src/pfsys/mod.rs
comparison/baselines/ezkl/src/pfsys/srs.rs
comparison/baselines/ezkl/src/pfsys/errors.rs
comparison/baselines/ezkl/src/srs_sha.rs
```

重点：

- SRS 如何读取或生成。
- setup/prove/verify 使用哪些 proving system API。
- SRS hash 是否进入 manifest。
- proof 和 vk/pk 的路径如何处理。

### 9.3 graph 层

```text
comparison/baselines/ezkl/src/graph/mod.rs
comparison/baselines/ezkl/src/graph/model.rs
comparison/baselines/ezkl/src/graph/input.rs
comparison/baselines/ezkl/src/graph/node.rs
comparison/baselines/ezkl/src/graph/vars.rs
comparison/baselines/ezkl/src/graph/modules.rs
comparison/baselines/ezkl/src/graph/utilities.rs
comparison/baselines/ezkl/src/graph/errors.rs
```

重点：

- graph node 如何表示。
- input tensor 如何被解释。
- model graph 如何编译。
- vars 如何绑定 witness。
- modules 如何组合。
- errors 如何暴露给脚本。

### 9.4 circuit 总层

```text
comparison/baselines/ezkl/src/circuit/mod.rs
comparison/baselines/ezkl/src/circuit/table.rs
comparison/baselines/ezkl/src/circuit/utils.rs
comparison/baselines/ezkl/src/circuit/tests.rs
```

重点：

- circuit 总入口。
- lookup table 如何进入 circuit。
- utility 是否影响当前 GAT route。
- tests 覆盖哪些行为。

### 9.5 circuit ops 层

```text
comparison/baselines/ezkl/src/circuit/ops/mod.rs
comparison/baselines/ezkl/src/circuit/ops/base.rs
comparison/baselines/ezkl/src/circuit/ops/layouts.rs
comparison/baselines/ezkl/src/circuit/ops/lookup.rs
comparison/baselines/ezkl/src/circuit/ops/poly.rs
comparison/baselines/ezkl/src/circuit/ops/hybrid.rs
comparison/baselines/ezkl/src/circuit/ops/region.rs
comparison/baselines/ezkl/src/circuit/ops/errors.rs
comparison/baselines/ezkl/src/circuit/ops/chip.rs
```

重点：

- 基础 op 如何映射到 circuit。
- lookup op 如何使用 formal tables。
- polynomial / hybrid ops 是否参与当前 route。
- layout 和 region 如何影响约束数量。

### 9.6 einsum chip 子层

```text
comparison/baselines/ezkl/src/circuit/ops/chip/einsum/mod.rs
comparison/baselines/ezkl/src/circuit/ops/chip/einsum/analysis.rs
comparison/baselines/ezkl/src/circuit/ops/chip/einsum/circuit_params.rs
comparison/baselines/ezkl/src/circuit/ops/chip/einsum/layouts.rs
comparison/baselines/ezkl/src/circuit/ops/chip/einsum/reduction_planner.rs
```

重点：

- matrix multiply / einsum 如何规划。
- projection 和 output linear 相关操作是否落到 einsum。
- reduction planner 如何影响 constraints。
- layout 是否可能改变 benchmark 结果。

### 9.7 circuit modules 层

```text
comparison/baselines/ezkl/src/circuit/modules/mod.rs
comparison/baselines/ezkl/src/circuit/modules/planner.rs
comparison/baselines/ezkl/src/circuit/modules/polycommit.rs
comparison/baselines/ezkl/src/circuit/modules/poseidon.rs
comparison/baselines/ezkl/src/circuit/modules/poseidon/spec.rs
comparison/baselines/ezkl/src/circuit/modules/poseidon/poseidon_params.rs
comparison/baselines/ezkl/src/circuit/modules/errors.rs
```

重点：

- module planner 如何组织 circuit。
- polycommit 是否用于 proof pipeline。
- poseidon 是否用于 transcript、hash、commitment 或其他模块。
- errors 如何传播。

### 9.8 tensor 层

```text
comparison/baselines/ezkl/src/tensor/mod.rs
comparison/baselines/ezkl/src/tensor/val.rs
comparison/baselines/ezkl/src/tensor/var.rs
comparison/baselines/ezkl/src/tensor/ops.rs
comparison/baselines/ezkl/src/tensor/errors.rs
comparison/baselines/ezkl/src/tensor/metal/tensor_ops.metal
```

重点：

- tensor value 和 tensor variable 的区别。
- tensor ops 如何映射到 graph/circuit。
- 当前 CPU strict route 是否使用 Metal；如果不用，也要知道该文件是 upstream 通用加速路径。

### 9.9 其他辅助文件

```text
comparison/baselines/ezkl/src/fieldutils.rs
comparison/baselines/ezkl/src/eth.rs
comparison/baselines/ezkl/src/logger.rs
comparison/baselines/ezkl/src/bindings/mod.rs
comparison/baselines/ezkl/src/bindings/python.rs
comparison/baselines/ezkl/src/bin/py_stub_gen.rs
```

重点：

- field conversion。
- logging。
- Python binding 是否影响当前实验；通常不影响 strict route，但仍然属于目录内源码。
- `eth.rs` 多数和链上验证或 Solidity/verifier 相关，读时确认是否进入当前 baseline pipeline。

---

## 10. 第八阶段：读正式结果文件

阅读顺序：

```text
comparison/baselines/ezkl/results/gat_ezkl_full_results.json
comparison/baselines/ezkl/results/gat_ezkl_full_results.csv
comparison/baselines/ezkl/results/main_vs_ezkl_full_comparison.json
comparison/baselines/ezkl/results/main_vs_ezkl_full_comparison.csv
comparison/baselines/ezkl/results/summary.md
```

阅读重点：

### 10.1 `gat_ezkl_full_results.json`

看机器可读结果。重点字段：

- dataset。
- route。
- status。
- setup time。
- witness time。
- prove time。
- verify time。
- proof size。
- memory profile。
- artifact path。

### 10.2 `gat_ezkl_full_results.csv`

看论文或表格展示用字段。检查：

- 是否和 JSON 一致。
- 是否五个 dataset 都齐全。
- 数值单位是否一致。

### 10.3 `main_vs_ezkl_full_comparison.json`

看主项目实现和 ezkl baseline 的横向对比。检查：

- 主项目字段和 ezkl 字段是否同口径。
- proof size 是否同一统计口径。
- 时间是否同一运行模式。
- 是否有 CPU/GPU 混比。

### 10.4 `main_vs_ezkl_full_comparison.csv`

看正式展示表。检查字段顺序、单位、dataset 顺序。

### 10.5 `summary.md`

最后读总结。它应该能对应前面的 JSON/CSV，而不是引入新的结论来源。

---

## 11. 第九阶段：逐个 dataset 阅读 run artifact

五个目录建议按正式顺序读：

```text
comparison/baselines/ezkl/runs/cora_full/
comparison/baselines/ezkl/runs/citeseer_full/
comparison/baselines/ezkl/runs/pubmed_full/
comparison/baselines/ezkl/runs/ppi_full_formal/
comparison/baselines/ezkl/runs/ogbn_arxiv_full/
```

每个目录内部按如下顺序读：

```text
run_manifest.json
model_info.json
graph_attention_spec.json
final_logits_binding.json
formal_tables.json
formal_table_hashes.json
native_full_witness_preflight.json
keygen_memory_profile.json
prove_memory_profile.json
```

### 11.1 `run_manifest.json`

这是单次运行总索引。必须先读它。重点字段：

- dataset。
- route。
- command。
- input paths。
- output paths。
- status。
- timings。
- proof size。
- verify result。
- artifact hashes。
- memory profile paths。

### 11.2 `model_info.json`

确认模型结构：

- layer count。
- hidden profile。
- output head count。
- input dimension。
- class count。
- checkpoint id。
- parameter shapes。

### 11.3 `graph_attention_spec.json`

确认图和 attention 结构：

- node count。
- edge count。
- CSR layout。
- self-loop rule。
- edge sort rule。
- segment layout。
- attention route。
- input/output tensor names。

### 11.4 `final_logits_binding.json`

确认 final logits 的绑定方式：

- logits shape。
- hash。
- public/private policy。
- binding metadata。
- 是否和 large public output policy 一致。

### 11.5 `formal_tables.json`

确认 formal table 实体：

- range table。
- activation table。
- exp table。
- scale。
- table length。
- table value range。

### 11.6 `formal_table_hashes.json`

确认 table hash 是否和 metadata、manifest 一致。

### 11.7 `native_full_witness_preflight.json`

确认 witness 生成前检查：

- 输入 shape。
- graph spec。
- model info。
- table hash。
- route。
- full dataset。
- 是否通过 preflight。

### 11.8 `keygen_memory_profile.json`

理解 setup/keygen 阶段内存消耗。

### 11.9 `prove_memory_profile.json`

理解 prove 阶段内存消耗。

---

## 12. 反向追溯方法

读完整体后，需要从结果反向追溯一遍，确认理解没有断层。

以 `cora_full` 为例：

```text
results/gat_ezkl_full_results.json
  ↓
runs/cora_full/run_manifest.json
  ↓
runs/cora_full/model_info.json
  ↓
runs/cora_full/graph_attention_spec.json
  ↓
runs/cora_full/formal_tables.json
  ↓
runs/cora_full/final_logits_binding.json
  ↓
scripts/run_gat_ezkl_baseline.py
  ↓
cpp/gat_ezkl_adapter/src/main.cpp
  ↓
cpp/gat_ezkl_adapter/src/gat.cpp
  ↓
cpp/gat_ezkl_adapter/src/segment_softmax.cpp
  ↓
src/native_gat.rs
  ↓
src/bin/ezkl.rs / src/commands.rs
```

如果任意一步无法解释某个字段的来源，就回到对应实现文件重读。

---

## 13. 主线代码核对清单

阅读完后，用下面清单检查是否真的覆盖完整。

### 13.1 实验边界

- [ ] 明确当前 baseline 是 full dataset strict route。
- [ ] 明确没有使用 subset/sample/shard。
- [ ] 明确没有使用 precomputed logits 替代证明。
- [ ] 明确没有使用 digest/folding/identity route 替代 full attention。
- [ ] 明确五个 dataset 都进入正式汇总。

### 13.2 C++ adapter

- [ ] 读完所有 `.hpp`。
- [ ] 读完所有 `.cpp`。
- [ ] 能解释配置如何解析。
- [ ] 能解释 dataset 如何加载。
- [ ] 能解释 CSR 如何构造。
- [ ] 能解释 checkpoint 如何读取。
- [ ] 能解释 quant 和 bounds。
- [ ] 能解释 formal tables。
- [ ] 能解释 segment softmax。
- [ ] 能解释 GAT forward。
- [ ] 能解释 metadata 和 binding。
- [ ] 能解释 ONNX/graph export。
- [ ] 能解释 adapter main 的完整执行链。

### 13.3 Rust/ezkl

- [ ] 读完 CLI command 注册。
- [ ] 读完 `native_gat.rs`。
- [ ] 读完 proving system 入口。
- [ ] 读完 graph 层核心文件。
- [ ] 读完 circuit 层核心文件。
- [ ] 读完 tensor 层核心文件。
- [ ] 明确哪些 Rust 文件属于当前 route，哪些只是 upstream 通用能力。

### 13.4 scripts

- [ ] 能手动复述 `run_gat_ezkl_baseline.py` 的步骤。
- [ ] 能解释 `collect_gat_ezkl_results.py` 如何汇总。
- [ ] 能解释 `validate_final_results.py` 如何校验。
- [ ] 能说明脚本如何防止旧结果混入。

### 13.5 results/runs

- [ ] 能从 summary 追溯到 CSV。
- [ ] 能从 CSV 追溯到 JSON。
- [ ] 能从 JSON 追溯到 run manifest。
- [ ] 能从 run manifest 追溯到 C++ adapter 输出。
- [ ] 能解释每个 dataset artifact 的用途。

---

## 14. 推荐的二轮精读顺序

第一轮按文件顺序读，第二轮应按功能链路读。

### 14.1 配置链路

```text
configs/full_datasets.json
configs/ezkl_routes.json
scripts/run_gat_ezkl_baseline.py
cpp/gat_ezkl_adapter/src/config.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/config.hpp
```

### 14.2 数据和图链路

```text
cpp/gat_ezkl_adapter/src/dataset.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/dataset.hpp
cpp/gat_ezkl_adapter/src/graph.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/graph.hpp
runs/<dataset>/graph_attention_spec.json
```

### 14.3 模型和参数链路

```text
cpp/gat_ezkl_adapter/src/checkpoint.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/checkpoint.hpp
cpp/gat_ezkl_adapter/src/gat.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/gat.hpp
runs/<dataset>/model_info.json
```

### 14.4 数值语义链路

```text
cpp/gat_ezkl_adapter/src/quant.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/quant.hpp
cpp/gat_ezkl_adapter/src/bounds.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/bounds.hpp
cpp/gat_ezkl_adapter/src/formal_tables.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/formal_tables.hpp
runs/<dataset>/formal_tables.json
runs/<dataset>/formal_table_hashes.json
```

### 14.5 Attention 链路

```text
cpp/gat_ezkl_adapter/src/segment_softmax.cpp
cpp/gat_ezkl_adapter/include/gat_ezkl/segment_softmax.hpp
cpp/gat_ezkl_adapter/src/gat.cpp
src/native_gat.rs
runs/<dataset>/native_full_witness_preflight.json
```

### 14.6 Prove/Verify 链路

```text
src/bin/ezkl.rs
src/commands.rs
src/native_gat.rs
src/pfsys/mod.rs
src/pfsys/srs.rs
scripts/run_gat_ezkl_baseline.py
runs/<dataset>/run_manifest.json
```

### 14.7 结果链路

```text
scripts/collect_gat_ezkl_results.py
scripts/validate_final_results.py
results/gat_ezkl_full_results.json
results/gat_ezkl_full_results.csv
results/main_vs_ezkl_full_comparison.json
results/main_vs_ezkl_full_comparison.csv
results/summary.md
```

---

## 15. 常见误读点

### 15.1 误把 C++ adapter 当成证明系统

C++ adapter 主要负责 GAT baseline 输入构造、前向语义、metadata、formal tables 和导出。真正的 proving/verification 由 ezkl/Rust 侧完成。

### 15.2 误把 upstream ezkl 示例当成本实验主线

`src/` 中存在大量通用 ezkl 能力。当前实验应以 `native_gat.rs`、CLI command、scripts 和 C++ adapter 为主线。

### 15.3 误把结果 CSV 当成唯一真源

CSV 是展示面。机器复核时应优先看：

```text
results/gat_ezkl_full_results.json
runs/<dataset>/run_manifest.json
runs/<dataset>/*.json
```

### 15.4 误把 proof size 当成文件夹大小

proof size 应看 manifest 或结果表中的正式统计字段，不应直接用 run directory 大小代替。

### 15.5 误以为大图必须 proof 更大

proof size 与电路 shape、public output policy、route、commitment/opening 结构有关，不一定和节点数或边数线性一致。需要结合 `large_public_output_policy_review.md` 和 manifest 字段判断。

---

## 16. 建议做的阅读笔记表

建议阅读时维护一张表：

| 文件 | 读完 | 作用 | 输入 | 输出 | 关键函数/结构 | 与哪一阶段相关 | 疑问 |
|---|---:|---|---|---|---|---|---|
| README.md |  | 实验说明 | 无 | 运行命令 | 无 | 总览 |  |
| config.cpp |  | 配置解析 | JSON/CLI | AdapterConfig | load_config | C++ adapter |  |
| gat.cpp |  | GAT forward | graph/features/checkpoint | logits/intermediates | run_gat_forward | C++ adapter |  |
| native_gat.rs |  | ezkl native route | exported artifacts | witness/proof commands | graph-gat-* | Rust/ezkl |  |
| run_manifest.json |  | 单次运行真源 | 全部运行输出 | metrics/status | JSON fields | results |  |

---

## 17. 最终阅读完成标准

当你能不看 README，独立画出下面这条链路，并能指出每一步对应的文件时，才算真正读完：

```text
full_datasets.json / ezkl_routes.json
  ↓
C++ adapter parses config
  ↓
loads dataset + graph + checkpoint
  ↓
builds CSR attention layout
  ↓
runs quantized/full GAT forward
  ↓
generates formal tables and table hashes
  ↓
exports graph_attention_spec + model_info + final_logits_binding
  ↓
run_gat_ezkl_baseline.py invokes ezkl commands
  ↓
Rust native_gat route generates settings / compiles / setups / witnesses / proves / verifies
  ↓
run_manifest records status, timings, hashes, memory, proof size
  ↓
collect script builds result table
  ↓
validate script checks final consistency
  ↓
summary and comparison files become final reporting surface
```

---

## 18. 附：最小必读文件清单

如果时间有限，最小不能跳过的文件是：

```text
README.md
results/cleanup_report.md
results/code_cleanliness_audit.md
results/control_variable_audit.md
Cargo.toml
cpp/gat_ezkl_adapter/CMakeLists.txt
configs/full_datasets.json
configs/ezkl_routes.json
cpp/gat_ezkl_adapter/include/gat_ezkl/*.hpp
cpp/gat_ezkl_adapter/src/*.cpp
scripts/run_gat_ezkl_baseline.py
scripts/collect_gat_ezkl_results.py
scripts/validate_final_results.py
src/bin/ezkl.rs
src/commands.rs
src/native_gat.rs
src/pfsys/mod.rs
src/pfsys/srs.rs
src/graph/*.rs
src/circuit/**/*.rs
src/tensor/*.rs
results/gat_ezkl_full_results.json
results/main_vs_ezkl_full_comparison.json
runs/*/run_manifest.json
runs/*/model_info.json
runs/*/graph_attention_spec.json
runs/*/final_logits_binding.json
runs/*/formal_tables.json
runs/*/formal_table_hashes.json
runs/*/native_full_witness_preflight.json
```

如果目标是论文复核或答辩，应读完整清单，而不是只读最小清单。

---

## 19. 附：推荐阅读节奏

建议按三天安排：

第一天：

- 读 README 和 results 审计报告。
- 读 CMake/Cargo/configs。
- 读 C++ headers。

第二天：

- 读 C++ `.cpp`。
- 手工画出 C++ adapter 的对象流。
- 对照一个 dataset 的 artifact。

第三天：

- 读 Rust/ezkl route。
- 读 scripts。
- 读 results 和全部 runs。
- 做一次从 result 到 source 的反向追溯。

如果只看一遍，容易漏掉 C++ adapter 和 Rust route 之间的接口；至少要做一次反向追溯。
