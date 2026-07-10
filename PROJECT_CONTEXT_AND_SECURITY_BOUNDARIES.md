# zkGAT 项目背景与强制约束

> 文档版本：v1.0  
> 生效日期：2026-07-10  
> 状态：已确认，只读  
> 维护权限：仅由项目负责人创建、修改和确认  
> 使用方式：任何设计、实现、测试、重构、性能优化或实验任务开始前，必须先完整阅读本文件与 `GAT-ZKML-多层多头.md`  
> 只读要求：本文件与 `GAT-ZKML-多层多头.md` 均只读，执行过程中不得修改、补写、删减或根据当前实现反向调整其内容

---

## 1. 文件定位

本文件用于说明项目目标、主实验范围、安全边界、CPU/GPU 边界、代码治理原则、性能优化规则、实验限制和最终完成门槛。

完整技术定义、数学公式、证明关系、约束系统、商多项式、KZG opening、Fiat–Shamir 顺序和验证方程，以 `GAT-ZKML-多层多头.md` 为唯一技术依据。

本文件只保存长期背景和强制边界。当前完成状态、剩余问题、测试状态、性能状态和可运行范围以 `README.md` 为准。

规范优先级：

1. `GAT-ZKML-多层多头.md`
2. 本文件
3. 当前代码、配置、测试和结果
4. `README.md`

当当前实现与只读规范不一致时，以只读规范为准，修正实现。

---

## 2. 项目目标

项目名称为 **zkGAT**。

目标是构建面向多层多头图注意力网络推理的零知识可验证系统，使验证者能够确认一次 GAT 节点分类推理严格按照公开声明的模型结构、图输入、量化规则和证明协议执行，同时不获得私有模型参数和私有中间计算状态。

项目面向高水平安全与系统研究投稿，所有实现和实验必须满足：

- 协议语义真实；
- 零知识隐藏由真实密码学机制实现；
- 证明者与验证者边界清晰；
- CPU 与 GPU 路径语义一致；
- 性能数据真实、完整、可追溯；
- 代码结构可审计、可理解、可维护；
- 论文结论能够由正式实现和正式结果直接支持。

---

## 3. 正式任务范围

正式配置固定为：

- `protocol_name = zkgat`
- `proof_task = node_classification_full_gat`
- `privacy_policy = model_and_trace_hidden`
- `input_privacy = public_graph_public_features`
- `output_mode = public_logits`
- `opening_mode = domain_grouped_batch_opening`
- `binding_mode = zk_subproof_for_crpc`
- `security_profile_id = zkgat_secure_profile_v1`
- `srs_type = trusted_srs` 或 `mpc_srs`

正式任务只证明多层多头 GAT 节点分类推理，不包含训练、梯度计算和参数更新证明。

上述配置必须被公共元数据、Fiat–Shamir transcript、证明对象、验证逻辑、运行配置和正式结果共同绑定。任一关键字段不一致都必须拒绝验证。

---

## 4. 正式数据集与处理范围

正式主实验使用五个全量数据集：

- Cora
- Citeseer
- Pubmed
- PPI
- ogbn-arxiv

最终正式结果必须包含五个数据集的 CPU 与 GPU 两条路径，共十组结果。

要求：

- 不允许用 toy graph、人工小图或裁剪子图替代最终全量实验；
- PPI 必须正确处理多图批次、`node_ptr`、`edge_ptr` 和跨图隔离；
- 十组正式结果只能在协议、安全、结构和性能全部冻结后统一运行；
- 启动十组正式实验前，必须汇报协议冻结状态、Cora CPU/GPU 最终 profile、全部测试结果、剩余风险和正式运行计划，并获得项目负责人明确授权。

当前阶段只处理主实验内容。

`comparison/` 目录包含三个独立对比实验，本阶段完整忽略：

- 不检查；
- 不修改；
- 不删除；
- 不运行；
- 不纳入主实验清理、重构和性能统计；
- 不让主实验依赖其中的实现或结果。

---

## 5. 多层多头 GAT 固定语义

完整定义以 `GAT-ZKML-多层多头.md` 为准。

实现必须保证：

- 总层数、每层头数、每头维度和输出头数由公共元数据唯一确定；
- 前 \(L-1\) 层为隐藏注意力层；
- 隐藏层每个头完成注意力聚合后施加 ELU；
- 同层多个隐藏头沿特征维拼接；
- 下一隐藏层只接收上一层完整拼接输出；
- 最终输出层仍由注意力头组成；
- 每个输出头完成聚合后加本头偏置；
- 输出层不施加 ELU；
- 多个输出头按算术平均得到最终 logits；
- 所有实现对象、承诺、评值、子证明、序列化字段和日志显式携带层索引与头索引；
- 图排序、自环、多图边界和稀疏边语义在 CPU 与 GPU 路径中保持一致。

---

## 6. 公开与私有边界

公开信息包括：

- 图结构和节点特征；
- 数据集与任务配置；
- 模型结构；
- 量化与静态表配置；
- 图排序、padding、domain 和维度信息；
- SRS manifest、`srs_id`、`srs_type` 和 degree bound；
- 协议版本与安全配置；
- 最终公开 logits；
- 证明大小和运行统计。

私有信息包括：

- 原始模型参数；
- 中间激活；
- 注意力分数和 Softmax 权重；
- lookup、route、PSQ 和 folding witness；
- 私有多项式裸评值；
- 盲化随机量；
- KZG setup 隐藏陷门。

验证者不得读取、持有或间接访问：

- witness；
- 私有模型参数或私有多项式；
- 盲化随机量；
- Prover Key；
- setup 隐藏陷门；
- 可恢复 witness 的辅助数据；
- 证明者专属缓存。

证明者上下文与验证者上下文必须在类型、接口和数据所有权上明确分离，不能只依赖注释或调用约定。

---

## 7. KZG、SRS 与零知识边界

### 7.1 KZG 与 SRS

KZG 隐藏陷门 \(\tau\) 只允许在参数生成阶段临时存在。

强制要求：

- \(\tau\) 由密码学安全随机过程产生；
- 不允许 `seed -> tau`；
- 不允许 `tau = seed + c`；
- 不允许在证明阶段直接评估 `P(tau)` 作为承诺主路径；
- \(\tau\) 不得保存、打印或序列化；
- 生成 Prover Key 和 Verifier Key 后必须销毁 \(\tau\)；
- 证明者只持有 Prover Key 或工作域 SRS；
- 验证者只持有 Verifier Key；
- 验证接口不得接收完整 key pair；
- SRS 的曲线、有限域、degree bound、domain、文件哈希和 manifest 哈希必须严格校验；
- 任一 SRS 不一致必须硬失败。

### 7.2 零知识隐藏

所有私有多项式必须按照技术笔记规定启用真实 domain-vanishing blinding。

强制要求：

- 使用密码学安全随机源；
- 正式随机性不得依赖实验 seed；
- commitment、quotient、opening、外点绑定和 transcript 统一消费盲化后的对象；
- proof object 不得直接携带模型参数、中间激活、注意力权重、lookup/route/PSQ witness、folding 向量或私有裸评值；
- 需要绑定裸语义值时，必须使用零知识绑定子证明；
- 相同公共输入和 witness 多次生成的证明应具有随机性；
- 数据集 seed、模型 seed 和密码学随机源必须分离。

`MaskedPrivateValue`、`PrivateWitness`、`private_plain_value_count = 0`、payload 类型标签、schema audit 和字段重命名，只能作为结构审计的一部分，不能替代真实盲化和零知识绑定。

---

## 8. 公共元数据、transcript 与证明完整性

所有影响证明语义或验证结果的信息都必须进入公共元数据，并按技术笔记规定顺序吸收进 Fiat–Shamir transcript。

至少绑定：

- 协议名称、版本和安全配置；
- proof task 与各项隐私、opening、binding 模式；
- SRS 标识、类型和 manifest 哈希；
- 数据集；
- 模型结构和模型参数标识或承诺；
- 量化配置和静态表；
- 图排序、自环、batching 和 padding；
- domain、维度、层数和头数；
- 最终公开输出。

禁止：

- 漏吸收、重复吸收或重排；
- challenge 在对应消息提交前生成；
- Prover 与 Verifier 使用不同编码；
- 跨数据集、模型、配置、SRS 或协议版本重放证明。

证明必须完整绑定：

- 图结构与层间衔接；
- 每层每头线性变换、注意力、Softmax 和聚合；
- 隐藏层 ELU 与多头拼接；
- 输出头偏置与多头平均；
- 量化、范围、lookup、route 和 PSQ；
- 矩阵关系、GKR、sumcheck 或等价子证明；
- 商多项式；
- KZG commitment 与 opening；
- 外点评值绑定；
- 最终公开 logits。

不得存在未验证字段、未绑定自由字段、未验证关键子证明、安全检查快捷路径或 debug/benchmark/GPU 模式下的验证绕过。

---

## 9. CPU 与 GPU 路径

项目以 GPU 性能实现为主要方向，同时保留完整 CPU 路径。

两条路径必须：

- 使用相同公共元数据；
- 实现相同 GAT 语义和证明关系；
- 使用相同 transcript 与量化规则；
- 生成可由同一正式验证逻辑接受的证明；
- 对相同输入产生一致公开输出；
- 通过相同正确性、安全性和篡改测试。

CPU 路径主要用于：

- 正确性参考；
- 正式 CPU 基线；
- 单元测试和安全测试；
- GPU 结果交叉验证；
- 明确标注的控制逻辑和小规模阶段。

GPU 路径应优先覆盖真实 profile 中的主要热点，包括：

- 大规模有限域运算；
- G1 MSM；
- FFT、NTT、IFFT；
- quotient 构造和承诺；
- opening witness 与 domain opening；
- lookup/histogram；
- 大规模多项式和矩阵运算；
- host/device 数据流与并行调度。

设备边界要求：

- GPU 模式不得静默退化为 CPU 模式；
- 仍由 CPU 执行的阶段必须明确记录；
- 部分 CUDA kernel 不能描述为完整 GPU 证明；
- 不得用 CPU 结果填充 GPU 字段；
- 不得只统计 kernel 时间而忽略主机准备、传输、同步和回退时间；
- GPU 不可用或关键 GPU 路径失败时必须硬失败。

---

## 10. 代码结构治理

结构治理覆盖全部主实验实现，不限于 `prover.cpp`。

重点包括：

- `prover.cpp`
- `verifier.cpp`
- `trace.cpp`
- `quotients.cpp`
- KZG、SRS、commitment、opening
- transcript 与 proof schema
- lookup、route、PSQ
- matrix proof、GKR、sumcheck
- GPU 后端
- runner、profiling、serialization 与 tests

治理目标：

- 单一职责；
- 协议层与执行后端分离；
- Prover 与 Verifier 边界分离；
- CPU 与 GPU 后端分离；
- 数据准备、密码学计算、缓存、序列化和统计不堆积在同一超长函数；
- 核心协议阶段可以独立测试；
- 公共接口最小化；
- 避免重复实现、循环依赖和隐式全局状态；
- 关键安全边界由类型和接口保证。

治理规则：

- 先分析当前结构和依赖，再渐进式重构；
- 不套用与现有项目差异过大的全新目录模板；
- 不机械按行数拆文件；
- 每次重构形成范围清晰、可审查的修改包；
- 纯结构重构不得改变协议、transcript 和验证结果；
- 性能关键路径不得因无必要抽象产生明显退化；
- 每次重构后必须构建、测试并进行必要的 Cora 验证；
- 当当前结构阻碍协议审计、独立测试或安全边界落实时，先完成最小必要重构；
- 不得在协议尚未确认时进行大规模纯美化式迁移；
- 不得继续向已明显失控的超大文件堆积新逻辑。

清理前必须检查构建、链接、脚本、测试、配置、文档、数据准备和结果复现依赖。无法确认用途的内容不得直接删除，应先记录为待确认项。

---

## 11. 固定任务顺序

### 阶段 A：规范映射与审计

- 将技术规范中的模型对象、证明对象、约束、承诺、评值和验证检查映射到实际实现；
- 确认每一项安全声明都有真实密码学实现；
- 不因已有测试通过而跳过协议核查。

### 阶段 B：协议与安全修正

- 修正 GAT 前向语义；
- 修正 KZG/SRS；
- 分离 Prover/Verifier 上下文；
- 完成真实 blinding；
- 修正 transcript 与 proof schema；
- 补齐 quotient、opening 和 verifier checks；
- 完成负向测试与泄漏检查。

### 阶段 C：结构治理

- 拆分多职责模块和超长函数；
- 清理重复、失效和无用途内容；
- 建立清晰的协议层、后端层、数据层和测试边界。

### 阶段 D：协议冻结

只有以下条件全部满足后才能冻结协议：

- 技术规范逐项映射完成；
- 参考前向一致；
- CPU/GPU 证明语义一致；
- Prover/Verifier 边界明确；
- 真实 blinding 完成；
- transcript、proof schema 和 KZG/SRS 格式固定；
- 正确性、安全性和篡改测试全部通过；
- Cora CPU/GPU 均真实验证通过。

### 阶段 E：性能充分收敛

协议冻结后，在不改变证明方案和安全边界的前提下，依照 profile 优化主要热点，直至满足本文件规定的性能收敛标准。

### 阶段 F：正式实验

只有协议、安全、结构和性能全部收敛，并获得项目负责人明确授权后，才运行五个全量数据集 CPU/GPU 十组正式实验。

---

## 12. 开发阶段的数据集限制

在项目负责人明确解除本限制前：

- 只允许使用 Cora 进行完整端到端证明与性能测试；
- 可以运行 Cora CPU 和 Cora GPU；
- 可以运行必要的单元测试、安全测试、篡改测试和 microbenchmark；
- 可以使用最小化单元测试数据验证 PPI 多图边界、`node_ptr`、`edge_ptr`、跨图隔离和相关索引逻辑；
- 不得运行 PPI 全量证明；
- 不得将最小化测试数据作为实验结果；
- 不运行完整五数据集正式实验；
- 不生成或覆盖十组正式论文结果；
- 不把 Cora 中间结果表述为最终结果。

只有在项目负责人明确授权后，才统一运行五个数据集的 CPU/GPU 正式结果。

---

## 13. 正确性与安全测试

每轮修改必须运行与修改范围对应的测试。

至少覆盖：

- 参考前向、CPU 前向、GPU 前向及结果一致性；
- hidden concat、output-head bias 和 multi-head average；
- 图排序、自环和 PPI 多图边界；
- commitment、opening、proof generation 和 formal verification；
- 公共输入、图结构、元数据、SRS、commitment、quotient、opening、transcript、lookup、route、PSQ、matrix、GKR、sumcheck、输出和 proof payload 的篡改失败；
- proof 不含私有明文；
- 多次证明具有随机性；
- Verifier 不读取 witness；
- Verifier 不持有 Prover Key；
- 不同 SRS、模型、数据集和配置不能重放。

静态检查必须禁止：

- runtime tau；
- `seed -> tau`；
- 证明路径直接评估 `P(tau)`；
- Verifier 访问 Prover Key；
- Verifier 读取 witness；
- 正式路径使用 mock 或 replay；
- proof-specific cache 进入 fresh proof；
- GPU 模式静默 CPU fallback；
- 未验证字段影响接受结果。

---

## 14. 性能优化边界与收敛标准

性能优化只能在协议等价和安全等价前提下进行。

禁止通过以下方式提速：

- 改变 public statement 或 witness relation；
- 减少必须证明的计算；
- 删除 verifier check；
- 跳过 quotient、opening 或必要 challenge；
- 重排 transcript；
- 把私有值改为公开；
- 降低盲化强度；
- 复用上一份证明的 quotient、opening witness 或验证结果；
- replay proof；
- 使用子图、toy graph 或 synthetic model 替代正式数据；
- 只统计局部 kernel；
- 省略传输、同步和 CPU fallback；
- 将 CPU 计算标记为 GPU；
- 修改证明方案换取性能。

允许的优化包括：

- 等价算法替换；
- 多线程；
- GPU kernel；
- MSM、FFT/NTT 和 pairing 优化；
- 内存布局与缓冲区复用；
- 减少中间分配；
- 公共静态预处理；
- 合法静态缓存；
- 流式 quotient-to-MSM；
- 计算与传输重叠；
- 序列化优化；
- 不改变 transcript 的并行执行。

每项优化必须有优化前基线、真实 profile、优化后结果、正确性和安全回归，并明确内存、proof size 和协议语义影响。无收益、倒退或引入风险时必须完整回滚。

只有满足以下条件，才能认为性能已经充分收敛：

- 已完成 Cora CPU/GPU 阶段级与函数级 profile；
- 证明、验证、CPU fallback、GPU kernel、host/device 传输和峰值内存均已分析；
- 所有主要热点均有保留优化、回滚记录或不可继续优化的明确结论；
- 每个不可继续优化的主要热点均说明协议边界、工程限制或实测原因；
- 不允许仅凭一次倒退实验宣布某个方向永久结束；
- 连续多轮优化未产生稳定、可复现且有实际价值的收益；
- GPU 利用率、CPU fallback 比例、数据传输开销和内存瓶颈均已确认；
- 最终由项目负责人确认是否进入正式实验阶段。

---

## 15. 缓存与计时规则

warm-static 模式只允许复用与具体 proof 和 witness 无关的静态内容，包括已校验 SRS、公共图元数据、domain roots、barycentric weights、公共 selector、公共 topology polynomial、公共静态表、拉格朗日 SRS，以及与 proof 无关的设备缓冲区和执行计划。

fresh proof 中不得复用：

- witness-dependent polynomial；
- 盲化随机量；
- quotient polynomial 或 commitment；
- opening evaluation 或 witness；
- transcript state；
- proof object；
- proof-specific MSM 或 pairing 结果；
- 前一份证明的验证结果。

正式计时至少区分：

- `cold_end_to_end`
- `warm_static_fresh_proof`
- `repeat_same_proof_verify`

论文主要在线性能使用 `warm_static_fresh_proof`。

`repeat_same_proof_verify` 只用于验证吞吐测试，不能作为新证明验证时间。

正式统计至少包括：

- commitment、proof、verification 和 end-to-end wall time；
- proof size；
- peak host/device memory；
- CPU thread 数；
- GPU 与 CUDA 信息；
- host/device transfer time；
- GPU kernel time；
- CPU fallback time；
- cache policy；
- benchmark mode；
- verification status。

---

## 16. README 与结果目录

每轮任务结束后必须更新 `README.md`，准确反映：

- 当前协议配置和支持范围；
- 多层多头 GAT 语义；
- 公开与私有边界；
- KZG/SRS 使用方式；
- CPU/GPU 实际边界；
- 当前主要模块；
- 构建、运行和 Cora 测试方法；
- 当前已完成与未完成内容；
- 当前安全测试和性能状态；
- 当前 GPU 覆盖范围；
- 当前是否允许正式实验；
- 当前结果是否可用于论文；
- 已知限制。

README 只描述当前有效状态，不累积逐轮修改过程。详细性能尝试、失败优化、回滚原因和历史对照应记录在独立性能记录或提交说明中。

README 不得把计划写成已完成，不得把字段标签写成真实零知识机制，不得把局部 GPU 加速写成完整 GPU 证明，不得把 Cora 中间测试写成十组正式实验，也不得把 proof-specific cache 命中结果写成 fresh proof。

开发阶段产生的新结果不得覆盖或混入正式结果目录。

正式结果目录只有在以下条件全部满足后才能重新生成或覆盖：

1. 协议、安全、结构和性能已冻结；
2. Cora CPU/GPU 最终 profile 和测试结果已汇报；
3. 项目负责人明确授权。

每条最终结果必须能够追溯到最终代码、构建环境、数据集、模型、量化配置、SRS、协议版本、backend、计时模式和缓存策略，并包含真实的 commitment、proof、verification 时间、proof size 与验证状态。

---

## 17. 最终完成条件

只有以下条件全部满足，才能认定主实验完成：

### 协议与安全

- 多层多头 GAT 语义完全符合技术笔记；
- KZG/SRS 正确，setup 隐藏陷门已销毁；
- Prover Key 与 Verifier Key 结构分离；
- 真实 domain-vanishing blinding 已实现；
- transcript、proof schema 和验证逻辑固定；
- quotient、opening 和全部子证明均被验证；
- public logits 被完整绑定；
- Verifier 不读取 witness、不持有 Prover Key；
- proof 不含私有明文；
- 多次证明具有盲化随机性；
- SRS mismatch、schema/leakage、reference consistency、tamper/negative tests 和静态检查全部通过。

### 结构与性能

- 主实验模块职责清晰；
- 超大、多职责实现已合理拆分；
- CPU/GPU 和 Prover/Verifier 边界清晰；
- 无重复正式路径；
- 构建、测试和运行入口明确；
- README 与当前状态一致；
- Cora CPU/GPU profile 可信；
- proof-specific cache 不进入 fresh proof；
- 主要热点均已完成分析；
- GPU 主路径覆盖关键可并行热点；
- 保留的优化均有真实收益；
- 已达到性能充分收敛标准。

### 正式实验

- 已获得项目负责人明确授权；
- 五个全量数据集全部使用最终冻结实现；
- CPU/GPU 共十组运行全部成功；
- 全部使用 fresh proof 口径；
- 全部 `VERIFY_OK`；
- 全部安全闸门通过；
- proof size 统计真实；
- 结果、配置、环境和日志可追溯；
- 最终结果能够直接支持论文中的正确性、安全性和性能结论。

---

## 18. 每轮任务固定要求

### 开始前

1. 完整读取本文件；
2. 完整读取 `GAT-ZKML-多层多头.md`；
3. 确认任务属于主实验范围；
4. 排除 `comparison/`；
5. 检查当前代码、测试、结果和 README；
6. 明确本轮属于协议、安全、结构还是纯性能修改。

### 执行中

- 不修改两个只读文件；
- 不改变未授权的协议语义；
- 不把推测写成完成结论；
- 不用旧结果代替当前验证；
- 不以测试标志代替密码学机制；
- 未解除限制前只使用 Cora 进行完整端到端测试；
- 性能优化必须由 profile 驱动；
- 结构重构必须保持可审查和可回归；
- 未获得明确授权前不得启动十组正式实验。

### 结束前

1. 完成与修改范围对应的构建；
2. 运行相应单元测试、安全测试和负向测试；
3. 必要时运行 Cora CPU/GPU 验证；
4. 检查 CPU/GPU 实际执行边界；
5. 检查 witness 泄漏和 proof-specific cache；
6. 清理本轮无用途产物；
7. 更新 `README.md`；
8. 明确记录已完成、未完成、测试结果、性能变化和下一阻塞点；
9. 未达到全部最终完成条件时，不得宣称主实验完成；
10. 未获得项目负责人明确授权时，不得运行五个全量数据集 CPU/GPU 十组正式实验。
