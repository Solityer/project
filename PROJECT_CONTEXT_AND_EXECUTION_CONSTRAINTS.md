
# zkGAT 项目背景与执行约束

> 适用范围：`/home/pzh/project` 中除 `comparison/` 外的主实验。  
> 最高技术规范：`GAT-ZKML-多层多头.md`。  
> 本文件用于明确当前主实验的最终目标、开发边界、执行权限、测试要求和工作区治理规则。后续 Codex 任务应先完整阅读 `GAT-ZKML-多层多头.md`、本文件和 `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md` 三份只读约束文件，再直接基于主实验真实代码执行，不需要在每次 Prompt 中重复本文件内容。

---

## 1. 项目背景与判断原则

zkGAT 主实验研究 GAT 推理的零知识证明与验证。

当前工作的目标不是重新设计 zkGAT 协议，也不是重新做一套实验系统，而是在现有已经实现的主实验代码和正式证明流程基础上，把论文最终使用的 GAT 模型实例正式调整为真正的多层多头 GAT，并保证现有训练、模型加载、前向计算、trace、prove、verify、CPU/GPU 和结果导出流程能够正确处理新的模型结构。

### 1.1 只处理主实验

当前工作区：/home/pzh/project

`comparison/` 为完全独立的对比实验目录，本轮强制忽略。

不得：分析 `comparison/` 的内部实现；修改、删除、运行或清理 `comparison/`；使用 `comparison/` 的代码、协议组件、配置、结果或命名推断主实验应如何实现；把 `comparison/` 的任何机制写入主实验约束；让主实验依赖 `comparison/`。

主实验当前实际上实现了什么，只能根据主实验真实源码、真实调用链和实际数据流判断。`GAT-ZKML-多层多头.md` 用于约束正式模型语义、正式方法和当前实现必须满足的技术边界，但不得用文档描述代替对当前代码实现状态的检查，也不得因为文档中出现某个概念就推断主实验代码已经实现对应机制。

### 1.2 判断真实实现必须以代码执行链为准

README、源码注释、旧说明文档和历史描述不得作为判断当前主实验实际实现状态的事实依据；它们只能用于辅助定位文件或理解背景，不能用于证明某项能力已经实现。判断主实验是否真正支持当前模型结构时，必须沿真实代码调用和数据流分析，而不能只看 README、注释、文件名、配置 parser、结构体字段、历史描述或 `comparison/`。至少应沿实际主实验链路检查：

```text
配置读取
-> 模型结构建立
-> checkpoint / manifest 加载
-> GAT 前向
-> quantized / registered forward
-> trace 构造
-> prove
-> proof serialize / deserialize
-> verify
-> CPU backend
-> GPU backend
-> 结果导出
```

训练相关链路应实际检查：

```text
训练入口
-> 模型构造
-> 真实训练
-> checkpoint export
-> manifest
-> C++ loader
-> 正式前向与证明流程
```

只有真实代码路径能够完整消费新模型结构，才算真正支持多层多头。

## 2. 当前正式实验目标

这次不是增加一个额外的 architecture scaling 实验，而是把论文最终五个主实验本身统一改为真正的多层多头 GAT。

### 2.1 最终统一模型结构

五个正式数据集统一采用：L = 3；hidden_profile = [8x8,8x8]；K_out = 2

含义：两个隐藏 GAT 层；每个隐藏层 8 个 attention heads；每个 hidden head 维度为 8；每个隐藏层 multi-head concat 后宽度为 64；一个输出 GAT 层；输出层 2 个 attention heads。

`L = 3` 表示：2 hidden GAT layers + 1 output GAT layer

当前正式 GAT 前向语义保持现有主实验已经实现并采用的正式语义，不得为了这次模型扩展主动改变隐藏层 concat、激活、attention、aggregation、输出层 bias、输出 heads 聚合等已有计算规则。

### 2.2 五个正式数据集

| Dataset | `d_in_profile` | `num_classes` |
|---|---:|---:|
| Cora | `[1433,64]` | 7 |
| Citeseer | `[3703,64]` | 6 |
| Pubmed | `[500,64]` | 3 |
| PPI | `[50,64]` | 121 |
| OGBN-Arxiv | `[128,64]` | 40 |

第二隐藏层输入必须真实接收第一隐藏层 `8 heads x 8 dims = 64` 的 concat 输出，不能只在配置层写成 64，而真实 forward、checkpoint 或 proof 路径仍按旧结构运行。本轮只调整正式模型架构。五个数据集当前已经正确实现的数据加载、图结构、节点/边语义、预处理及数据集特有边界处理均保持现状；不得为了适配多层多头主动修改正式数据语义。

### 2.3 最终实验矩阵

最终论文正式主实验：5 datasets x 2 backends = 10 formal runs

backend：

- CPU；
- GPU：沿用当前主实验已经实现的正式 CUDA backend / hotspot 路径，具体 backend 名称、CUDA 覆盖范围和 CPU fallback 以当前真实代码为准。

对于同一数据集，CPU/GPU 必须使用同一个正式 checkpoint、同一个正式 config、同一个该数据集对应的 canonical SRS 和相同公开输入，只允许当前正式 backend 所必需的执行设备和执行参数不同。本轮多层多头改造不要求把当前仍在 CPU 执行的协议阶段迁移到 GPU，也不得为了所谓“完整 GPU”重新设计、扩大或替换现有 backend。不同数据集使用各自独立的 checkpoint、config 和 dataset-specific SRS。

### 2.4 不做 architecture scaling

本轮不新增不同 `L`、不同 hidden heads、不同 `K_out` 或 depth/heads/K_out/combined 的正式 scaling curve。之前未正式执行、未正式采用且已经不再服务于当前主实验的 architecture-scaling 草案，在确认没有当前正式依赖后可以清理。这里禁止的是模型 architecture scaling，不等于删除现有 CPU/GPU 工程优化或论文原本需要的工程性能能力。

---

## 3. 本轮 Codex 需要实际完成的工作

Codex 的任务不是重新设计方案，而是全面检查当前主实验真实代码，并把与当前正式多层多头模型相关的工程问题一次性处理到位。

### 3.1 先判断现有实现到底哪里需要改

必须沿真实代码路径确认：

- 当前模型结构是否真正支持两个隐藏层；
- 第二隐藏层是否真实消费第一层 multi-head concat 输出；
- 每个隐藏层是否真正按实际 head 数完整执行；
- 输出层是否真正支持 `K_out = 2`；
- checkpoint shape 和参数组织是否真实支持当前结构；
- manifest 是否能完整描述并检查当前结构；
- C++ loader 是否能正确加载；
- C++ FP32 forward 是否正确；
- quantized / registered forward 是否正确；
- trace 是否按实际 layer/head 组织；
- prove / verify 是否已经能够直接处理当前结构；
- CPU/GPU 是否都支持相同正式模型结构；
- SRS 是否与新的实际 proof shape 完整兼容，以及当前正式 envelope、capacity 和真实 binding 条件是否需要相应调整；
- 正式结果导出是否仍然正确。

如果现有某一部分本来已经通用且正确，保持现状，不为了产生修改量而重构。本轮保持当前主实验“一次完整 GAT 推理对应一次完整正式 prove/verify”的组织方式。不得为了适配多层或多头，主动把一个完整正式推理拆分为按 layer、按 head 或其他方式组织的多个彼此独立的正式 proof。

### 3.2 现有协议默认保持不变

当前已经实现并能够正确工作的主实验正式证明流程，是本轮工作的基线。

本轮目标是：让新的多层多头模型正确进入现有正式 zkGAT 流程，而不是借多层多头改造重新设计 zkGAT 协议。

因此：

- 已经正确工作的 trace / prove / verify 等正式协议代码默认不改；
- 已经正确工作的公开/私有语义默认不改；
- 已经正确工作的 proof 组织、serialization 和验证逻辑默认不改；
- 已经正确工作的 CPU/GPU 正式优化路径默认保留；
- 不得因为代码美观、重构、所谓通用化、方便扩展、方便训练、方便 SRS 或性能优化而主动改写协议核心；
- 不得根据本文件、历史记录或 `comparison/` 凭空新增主实验原本不存在的 protocol relation、subprotocol 或安全机制。

只有真实主实验代码或测试证明当前正式多层多头实例存在明确问题，并且不能通过配置、checkpoint、训练、维度、SRS 适配或执行入口等非协议层解决时，才允许做最小必要协议修改。协议相关修改必须保持 `GAT-ZKML-多层多头.md` 已确定的方法和安全目标，并完成与修改直接对应的 correctness / regression 验证。当前主实验已有正式安全参数和安全强度保持不变，包括当前 120-bit security target。不得为了降低多层多头实验成本主动降低安全参数、跳过正式验证或启用非正式快速路径。

### 3.3 五个正式配置直接改成最终状态

继续使用：

```text
configs/cora_secure.cfg
configs/citeseer_secure.cfg
configs/pubmed_secure.cfg
configs/ppi_secure.cfg
configs/ogbn_arxiv_secure.cfg
```

不得另建 `*_multilayer.cfg`、`*_new.cfg`、`*_v2.cfg`、`*_final.cfg` 等第二套正式配置。

修改正式架构时，当前真实代码中所有由模型结构派生、用于结构识别或用于 artifact/config 一致性绑定的 metadata / ID 也必须同步到新的正式架构。不得保留仍表示旧 `L=2`、旧 hidden profile、旧 `K_out=1` 或其他旧模型 shape 的结构派生值。具体字段和值必须依据现有生成、加载和校验逻辑确定，本文件不重新定义第二套命名规则。

### 3.4 训练和 checkpoint

五个数据集都必须具有能够真实训练当前最终架构的正式训练路径。Codex 负责把训练入口、模型构造、参数传递、checkpoint/export/manifest 链准备并验证到可正式执行状态；五个数据集的正式长时间训练由项目负责人按照 §4.3 执行，最终正式 checkpoint 必须来自本轮真实训练。

checkpoint/bundle import、旧 checkpoint 转换、tensor reshape、bundle export 或其他 artifact 转换流程都不等同于训练。Codex 必须逐数据集检查真实训练入口；如果某个数据集当前只有导入/转换路径，或者训练路径仍对旧架构存在硬编码，必须把这一问题作为当前多层多头适配的一部分处理。

正式训练继续使用：seed = 11固定 seed 只用于 checkpoint 生成过程可复现，不新增多 seed accuracy 实验。

正式训练不得静默采用 smoke/minimal 默认参数。若某个 trainer 的默认 `epochs`、`patience` 或其他训练参数明显只服务于短时验证，不得直接把这些默认值作为论文正式 checkpoint 的训练设置。若当前项目没有已经确定的正式训练 schedule，Codex 必须明确报告项目负责人确认，不得自行把测试默认值升级为正式实验参数。

training-only 参数不得为了方便直接塞入当前 C++ 正式 config parser 不接受的 canonical `*_secure.cfg`。训练参数应通过当前真实 trainer 已支持的 CLI 或项目当前已有的合法训练入口传递；只有主实验真实正式配置链本身要求某字段时才进入 canonical secure config。

禁止复制旧 head、复制旧隐藏层、reshape 旧 checkpoint、拼接旧权重、随机补层、synthetic weights 或其他没有经过真实训练的方式生成正式模型。

训练只用于生成正式 proof 实验需要的 checkpoint。本项目不新增 Accuracy、F1、AUC、多 seed 模型效果、训练收敛曲线或训练性能研究。

checkpoint / manifest / loader 必须与正式 config 的完整模型 shape 严格一致。结构不一致应拒绝，不得通过放松检查兼容错误 artifact。

canonical checkpoint 路径：

```text
artifacts/checkpoints/cora_gat/
artifacts/checkpoints/citeseer_gat/
artifacts/checkpoints/pubmed_gat/
artifacts/checkpoints/ppi_gat/
artifacts/checkpoints/ogbn_arxiv_gat/
```

### 3.5 `range_bits`

先保持当前已有正式值：

| Dataset | `range_bits` |
|---|---:|
| Cora | 12 |
| Citeseer | 12 |
| Pubmed | 12 |
| PPI | 14 |
| OGBN-Arxiv | 21 |

新 checkpoint 产生后执行真实整数范围审计。如果不足，不得自动修改；必须报告 dataset、layer/head、对应中间关系或中间量、实际最大范围、当前允许范围和最低需要 bits，由项目负责人决定是否调整。已知 range audit 未通过时，不得把该数据集后续完整运行结果当作正式论文结果。

### 3.6 SRS

SRS 必须根据当前主实验真实的 SRS 生成、加载、manifest/binding 和实际 proof shape 判断是否与新的正式多层多头实例完整兼容，而不能只看某一个表面容量字段。不得因为改成多层多头而重新设计一套 SRS 协议。

需要做的是：

- 按当前主实验真实代码检查现有 dataset-specific canonical SRS 是否与新的正式 proof shape 完整兼容；
- 检查当前代码实际要求的 capacity 以及其他真实存在的绑定条件；
- 如果当前正式实现判定旧 SRS 不兼容，就按照现有正式 SRS 工具和规则准备重新生成；
- 不得仅因为 nominal capacity 看起来足够就直接认定旧 SRS 一定可以继续使用；
- 保持当前正式 SRS 生成、manifest、PK/VK 和 loader 语义；
- 不引入新的非正式 SRS 路径；
- 不使用已知 trapdoor 绕过正式 setup。

canonical SRS 路径：

```text
artifacts/srs/cora_formal_srs/
artifacts/srs/citeseer_formal_srs/
artifacts/srs/pubmed_formal_srs/
artifacts/srs/ppi_formal_srs/
artifacts/srs/ogbn_arxiv_formal_srs/
```

正式大型 SRS 由项目负责人执行。

### 3.7 correctness / parity

论文不研究模型 accuracy，但必须确认新的多层多头模型真的被当前主实验正确计算和证明。

根据当前真实代码验证：

```text
trained model
-> exported checkpoint
-> C++ forward
-> quantized / registered forward
-> secure trace / proof path
-> public output
-> verify
```

具体测试内容由当前主实验真实实现决定，不得从 `comparison/` 或历史说明增加主实验不存在的协议测试项。最终完整正式运行必须 `VERIFY_OK`。

## 4. 测试和执行权限

### 4.1 `tests/` 是唯一 test-only 工作区

所有 test-only 的 unit/regression/correctness/parity/security/negative test、helper、fixture、mock、测试配置、测试脚本、测试输入、测试日志、测试 proof、测试 cache、测试中间 artifact，以及仅用于开发验证的 benchmark/profiling 代码，都只能位于 `tests/`。

是否属于 test-only 内容，必须根据真实职责和调用关系判断，而不能根据文件名中是否含有 `test`、`check`、`validate`、`parity` 等字样机械判断。

如果某个 checker / validator 是正式 checkpoint、artifact、SRS 或正式执行 Gate 的生产工具，应保持其真实正式职责和合理目录，不得仅为了目录形式统一强行迁入 `tests/`；如果其真实职责仅为开发测试，则必须收敛到 `tests/`。

test-only 内容不得散落到 `src/`、`include/`、`tools/`、`scripts/`、`configs/`、`artifacts/`、`runs/`、`results/` 等正式目录。

正式运行本身需要的性能统计、backend 开关和正式工程优化能力不属于 test-only 代码，不应被迁移到 `tests/`。

### 4.2 Codex 必须自己完成短时验证

Codex 不能只修改代码然后把所有验证交给项目负责人。

能够在合理时间内完成的必要验证，由 Codex 自己执行，包括 CMake configure、CPU build、GPU build、必要 unit test、targeted regression、correctness/parity、与本轮实际修改直接相关的既有 security/negative regression，以及必要短时局部集成验证。只有新架构真实暴露新的 correctness/security defect 时，才针对该真实问题补最小必要测试，不得为了本轮多层多头适配扩展新的协议测试体系。

### 4.3 长时间任务由项目负责人执行

由项目负责人执行：

- 十分钟级及以上或明显高资源的测试；
- 接近完整数据集规模的验证；
- 五数据集正式训练；
- 正式大型 SRS；
- 完整数据集 proof / verify；
- 五数据集 CPU 完整实验；
- 五数据集 GPU 完整实验；
- 会触发上述重型计算的正式结果采集。

Codex 不得因为这些操作由项目负责人执行，就提前停止其他仍可完成的代码、配置、测试、构建和静态检查工作。

只有后续工作真实依赖项目负责人执行后产生的新 checkpoint、SRS 或长时间结果，并且其他可完成工作已经完成，才算真正受到阻塞。

### 4.4 Codex 必须准备长时间任务的完整命令

需要项目负责人执行的任务，Codex 必须根据修改后的真实项目准备完整命令、实际参数、输入/输出路径、执行顺序、成功判据以及需要返回继续检查的结果。

不得只给示例命令或使用省略号。

正式长时间训练、SRS、测试和 CPU/GPU 实验默认运行至自然完成或真实失败，不应人为加入固定 timeout，除非项目负责人明确要求。

### 4.5 其他执行边界

本轮默认只读：

```text
GAT-ZKML-多层多头.md
PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md
PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md
```

Codex 不得为了迁就当前代码自行修改这三份文件。

Codex 可以使用 `git status`、`git diff`、`git log`、`git ls-files` 等只读 Git 命令检查当前状态、差异和 tracked/untracked 情况。

不得自行执行 `git pull`、`git commit`、`git push`、merge、rebase、tag、创建/删除分支、`git reset --hard`，也不得执行其他会改变当前 Git 工作区、历史或远程状态的操作。

执行测试前必须确认测试入口不会隐式触发正式训练、大型 SRS、完整数据集 proof 或最终正式实验。

### 4.6 后续正式执行顺序

Codex 完成本轮工程准备后，后续正式长时间任务按以下依赖顺序执行：

```text
正式训练
-> checkpoint / manifest 检查
-> forward / parity / range 检查通过
-> 正式 SRS 生成与检查
-> 必要长时间验证
-> CPU/GPU 十组正式实验
-> VERIFY_OK
-> 正式结果汇总
```

任何一步发现真实问题，都必须先解决并重新通过对应检查，再进入下一步。不得在已知前置检查失败的情况下继续生成后续正式结果。

这一顺序只是正式执行和验收顺序，不改变当前主实验已经实现的协议流程。

---

## 5. 正式结果要求

最终论文正式结果只能来自当前：

```text
L = 3
hidden_profile = [8x8,8x8]
K_out = 2
```

的五个完整数据集。

禁止 toy dataset、sampled subgraph、缩减图、部分节点/边或测试 fixture 替代论文正式结果。

旧 `L=2`、旧 hidden profile、`K_out=1` 或其他历史结果不得与新的十组结果混合形成最终主表。

正式性能指标继续使用当前正式口径：

```text
commitment_time_ms
prove_time_ms
verify_time_ms
proof_size_bytes
VERIFY_OK
```

正式总时间的统计口径固定为：

```text
commitment_time_ms +
prove_time_ms +
verify_time_ms
```

沿用当前正式实现已有的总时间字段命名。不得仅为了满足本文件重新命名或新增一个语义等价的总时间字段，也不得为了新的模型修改已有 timer 边界或把原本计时内工作移到 timer 外制造性能收益。`proof_size_bytes` 必须是真实 serialized proof 的字节数。

论文正式结果统一收敛到：results/paper_secure/

至少包括：

```text
results/paper_secure/summary.md
results/paper_secure/latest.csv
results/paper_secure/latest.json
results/paper_secure/security_tests_report.md
```

`security_tests_report.md` 是论



文级正式安全测试汇总；raw test logs、临时测试 proof 和其他测试产物仍然只存在于 `tests/`。

正式结果必须能够追溯到当前 dataset、backend、config、checkpoint、SRS、run/proof、`VERIFY_OK` 和正式性能指标。不要求为了本轮重新设计结果 schema；当前导出机制已经具备时保持现状即可。

---

## 6. 工作区治理

主实验最终只保留当前唯一正式状态。

不得创建或长期保留 `old`、`new`、`backup`、`legacy`、`previous`、`preserved`、`pre_*`、`phase*`、`final/final2`、日期型副本、`multilayer_v2`、`formal_new` 等开发历史式文件或目录。

主实验自行控制的 schema、file format、security profile、protocol、envelope、artifact、manifest、registry、report、transcript/domain identity、magic/header 及其他正式内部 identity 均只保留当前唯一 canonical 定义，不得携带 `v1/v2/vN`、`_vN`、`-vN` 或其他体现项目历史版本演进的编号。清理必须依据真实引用关系沿 producer、serialization、hash/digest/transcript、consumer 和 validation 调用链完整传播，不能以删除正式语义、弱化校验或破坏协议绑定为代价。第三方标准、外部 ABI/API、依赖自身版本、数据集官方版本以及算法或密码学参数中的固有版本/数字不属于本规则的清理对象。

同一职责只保留一个 canonical 实现。不得通过复制文件、平行实现、大段注释旧代码、备份目录或历史 artifact 保留修改过程。

除非项目负责人明确要求，不得新增 `TASK_REPORT.md`、`IMPLEMENTATION_NOTES.md`、`MIGRATION_NOTES.md`、`CHANGES.md`、`TODO_MULTILAYER.md` 或其他本轮任务过程文档。

清理文件时必须根据主实验真实引用关系确认已经失效，不得使用宽泛通配符或推测性大范围删除 checkpoint、SRS、results、runs 或其他 artifact。

当新的正式 checkpoint / SRS 尚未生成时：

- canonical config 仍保持最终多层多头目标；
- 不得为了短时测试临时改回旧模型配置；
- 不得复制、改名或伪造旧 checkpoint / SRS 填充正式状态；
- 需要轻量测试 artifact 时使用 `tests/` 内合法 fixture，不污染正式 canonical artifact。

需要人工保留的历史文件移出 `/home/pzh/project`。

`comparison/` 始终不参与任何清理。

---

## 7. 当前任务完成标准

本节区分两个状态：

### 7.1 Codex 本轮工程准备完成

Codex 本轮的目标不是在没有新 checkpoint、正式 SRS 和完整数据集运行结果的情况下宣称最终实验已经完成，而是把项目准备到只剩项目负责人必须执行的长时间任务。

Codex 只有在以下条件满足后，才可以认为本轮“工程准备完成”并停止等待项目负责人执行长时间任务：

1. 已基于主实验真实代码执行链完成分析，而不是依赖 README、注释、历史描述或 `comparison/`。
2. 五个正式 config 已直接表示 `L=3`、`hidden_profile=[8x8,8x8]`、`K_out=2`，并具有正确 `d_in_profile`；所有由模型结构派生或用于结构绑定的 metadata / ID 已同步到新架构，没有残留旧模型 shape。
3. 已逐数据集确认存在真正能够训练当前多层多头模型的正式训练路径，固定 seed 为 `11`，不会把 import/export/旧 checkpoint 转换当成训练，也不会把 smoke/minimal 默认参数静默当成正式训练参数；能够导出匹配 checkpoint / manifest。
4. 已确认 checkpoint loader、模型构造和真实 forward 能够正确处理两个隐藏层和两个 output heads。
5. 已确认 trace、prove、verify、CPU 和当前正式 CUDA backend / hotspot 路径对新结构的真实支持情况；没有为了本轮改造扩张 GPU 覆盖范围或重构现有 backend。
6. 对已经正确工作的正式协议代码没有无意义重构。
7. 如果真实代码发现旧单层/单输出头等硬编码，只进行了当前多层多头所需的最小必要修复，并完成对应短时验证。
8. 已准备当前模型所需 range audit，且没有擅自修改现有 `range_bits`。
9. 已根据当前主实验真实的 SRS 生成、加载、manifest/binding 和实际 proof shape，检查当前 dataset-specific canonical SRS 与新的正式多层多头实例是否完整兼容；如果当前正式实现判定需要重新生成，已经按照现有正式 SRS 流程准备好对应生成入口。没有仅依据 nominal capacity 作结论，也没有重新设计 SRS 协议。
10. 所有 test-only 代码和 test-only 产物均收敛在 `tests/`；正式 checker/validator 是否迁移已依据真实职责和调用关系判断，没有按文件名机械处理。
11. Codex 已实际完成 CPU/GPU build 及所有合理时间内能够执行的必要短时验证。
12. 旧 architecture-scaling 草案、真实失效的历史副本和重复实现，在确认没有正式依赖后已经清理；主实验自行控制的 schema/format/security/profile/protocol/envelope/artifact/manifest/registry/report/transcript 等正式内部 identity 已统一为当前唯一无历史版本号的 canonical 定义，不存在 `v1/v2/vN`、legacy、compatibility、migration 或旧格式 fallback；现有正式 CPU/GPU 工程优化能力没有被误删或回退。
13. 没有新建第二套正式 config、checkpoint、SRS、protocol 或 backend 实现。
14. 没有从 `comparison/` 引入任何主实验不存在的协议关系、代码或测试要求。
15. 对仍需要项目负责人执行的正式训练、checkpoint 后检查、SRS 生成、长时间测试和最终 CPU/GPU 十组实验，已给出经过检查的完整执行命令和成功判据。
16. 在要求项目负责人开始长时间任务前，所有不依赖这些任务实际输出即可完成的开发工作已经完成。
17. 已明确区分“当前代码路径经静态分析/短时验证确认具备支持能力”和“依赖新 checkpoint、新 SRS、完整数据集运行后才能最终确认”的事项；不得在尚未完成长时间正式任务时把后者描述为已经最终验证通过；正式时间仍沿用当前实现既有字段命名和 timer 边界。

### 7.2 最终正式实验完成

在 Codex 完成本轮工程准备后，由项目负责人按本文件规定的顺序执行正式训练、训练后检查、正式 SRS、必要长时间验证以及五个数据集 CPU/GPU 共十组完整实验。

只有在新的正式 checkpoint、正式 SRS 和完整数据集实际运行均完成，并且十组当前统一多层多头正式实验全部 `VERIFY_OK`、结果能够正确汇总到当前 canonical 正式结果后，才可以认为“最终正式实验完成”。

是否开始正式训练、生成正式 SRS、执行长时间测试、运行最终十组实验以及是否冻结论文最终结果，由项目负责人决定。
