# zkGAT 项目背景与当前执行约束

> 适用范围：`/home/pzh/project` 中除 `comparison/` 外的主实验工作区。 
> 最高技术规范：`GAT-ZKML-多层多头.md`。 
> 本文件定位：只描述当前项目长期有效的背景、技术边界、性能目标、实验权限和工作区治理要求。  
> 本文件不记录某一轮优化的具体热点、异常数值、日志位置、临时待办、测试命令、失败尝试或执行流水；这些内容应由当轮任务说明和当前实验材料承载。 
> 本文件仅在项目负责人明确要求时修改。

## 1. 项目定位与当前阶段

zkGAT 主实验围绕 `GAT-ZKML-多层多头.md` 定义的多层多头 GAT 推理、证明与验证展开，目标是在保持正式协议和安全性的前提下实现可验证、可复现且高性能的完整 zkGAT 系统。

主实验范围不扩展到：

- GAT 训练流程；
- 与技术规范无关的其他模型任务；
- 与当前证明、验证和正式性能无关的附加功能；
- `comparison/` 中的对比实验实现。

当前阶段为：

> **System-wide / Global Performance Optimization**

当前阶段默认以下内容已经确定并作为冻结边界：

- 正式推理语义；
- 证明关系；
- 安全边界；
- Prover / Verifier 职责；
- ProofSchema、transcript 和验证关系；
- 正式计时口径。

性能优化只能改变协议等价的计算方法和工程实现，不重新设计被证明的内容。

---

## 2. 技术规范与不可突破的协议、安全边界

`GAT-ZKML-多层多头.md` 是主实验最高技术规范，默认只读。实现与技术规范发生冲突时，应修正实现，不得为了迁就代码、降低开发难度或获得更好的性能数字而修改技术规范。

任何优化都不得改变或削弱技术规范定义的：

- 多层多头 GAT 推理语义；
- 图结构、多图边界和跨图隔离；
- 公开输入、私有模型及私有中间状态边界；
- `trusted_srs` 与正式 KZG 语义；
- commitment、quotient、opening 和 verification 关系；
- transcript、Fiat–Shamir challenge、ProofSchema 和 proof block；
- `SignedRoundDiv`、负指数 `ExpMap`、范围、符号、余数、tie、重缩放和无模回绕语义；
- proof-local 模型再盲化、coefficient-tail blinding、CRPC 和张量绑定；
- 完整公开 logits、`output_digest` 以及 Verifier 必须执行的检查。

严格禁止：

- 直接使用、泄露或依赖已知 `tau` 参与在线证明；
- 利用 trapdoor 绕过正式 KZG commitment 或 opening；
- 删除证明关系、验证检查或降低安全强度；
- 使用不安全、非正式或协议不等价的更快路径替代正式实现；
- replay proof；
- 跨 fresh proof 复用 proof-specific 动态结果；
- 将 Prover 应承担的工作转移给 Verifier 以制造 Prover 加速；
- 修改正式 timer 边界；
- 将原本属于正式 timer 的工作迁移到 startup、warmup、timer 外预处理或其他不计时阶段。

历史、非正式或不安全实现得到的更快性能数字只能作为计算潜力参考，不得作为恢复相应实现、削弱协议或改变安全边界的依据。

如果性能优化过程中发现真实的正确性或安全缺陷，应优先修复并完成必要回归；此类修复属于正确性维护，不作为性能成果，也不应借此扩展为与当前阶段无关的协议改写或一般性功能重构。

---

## 3. 正式性能目标与评价口径

### 3.1 唯一正式性能指标

当前性能阶段只关注三个正式指标：

- `commitment_time_ms`
- `prove_time_ms`
- `verify_time_ms`

定义：

`formal_total_time_ms = commitment_time_ms + prove_time_ms + verify_time_ms`

后文所称 `formal_total_time` 均指上述三项正式时间之和，不代表新的独立计时区间。

正式生产性能入口统一为：

`gatzk_run`

最终性能结论必须落到正式生产路径的三个指标及其总和，而不是其他辅助时间。

### 3.2 只优化正式 timer 内的真实工作

只有真实计入 `commitment`、`prove` 或 `verify` 的计算才属于当前性能优化范围，例如：

- MSM；
- commitment 构造；
- quotient 构造与 commitment；
- opening 与 opening witness；
- polynomial evaluation；
- NTT / INTT；
- lookup；
- 正式 timer 内的 CUDA kernel；
- 正式 timer 内的数据转换、同步、内存操作和 host-device transfer。

如果以下内容位于三个正式 timer 之外，则不得作为当前性能优化方向：

- SRS 加载和解析；
- startup；
- GLV 启动加载；
- 数据集加载；
- 配置解析；
- timer 外 context 初始化或预计算；
- timer 外日志、导出、序列化；
- 编译和构建；
- 其他只降低 end-to-end wall time、但不降低三个正式指标的路径。

proof size、GPU utilization、peak RSS、显存、线程数等可以作为正确性、资源和性能诊断依据，但不是独立性能目标。

### 3.3 大幅优化必须相对于正式总时间成立

当前阶段要求的是对完整正式运行有实质影响的性能提升，而不是局部函数或 benchmark 自身的高百分比提升。

重大性能优化必须结合以下因素判断：

- 对 `formal_total_time` 能减少多少绝对时间；
- 相对于优化前正式总时间下降多少；
- 是否真正作用于当前主要瓶颈；
- 是否能在一个或多个真实数据集规模下产生稳定收益。

不得仅因为某个 microbenchmark、stage benchmark 或 CUDA kernel 提升 50%、80% 或数倍，就将其描述为重大优化。

在仍存在几十秒、几百秒或上千秒正式热点时，不应把主要优化精力长期投入到只影响亚秒、毫秒或极小正式占比的路径。

小幅但真实、正确、安全、稳定且不增加明显复杂度的正式收益可以保留，但不能据此认为系统级性能目标已经完成，也不能围绕同一小路径长期进行低收益微调。

### 3.4 正式指标之间允许有限权衡

不要求 `commitment`、`prove`、`verify` 三项机械地同时下降。

当 `commitment` 或 `prove` 获得大幅下降时，可以接受 `verify` 小幅、可解释、受控且不会形成新热点的上升，但必须满足：

- `formal_total_time` 获得实质性的绝对和相对下降；
- `commitment` 与 `prove` 之间不得互相以实质性退化换取收益；
- 不得通过转移工作量人为降低 Prover 指标；
- `verify` 不得出现数倍、数量级或形成新主要热点的回归。

---

## 4. 性能优化原则

### 4.1 绝对正式耗时优先

性能优化优先级以完整正式运行中的绝对耗时贡献为核心，而不是局部代码复杂度、局部占比或 benchmark 百分比。

优先处理：

1. 对 `formal_total_time` 绝对时间贡献最大的结构性问题；
2. 在绝对耗时显著的前提下，占某个正式指标较大比例的路径；
3. 能同时减少多个主要正式热点的系统级重构；
4. 具有显著绝对时间影响的 CPU/GPU 性能差距或异常；
5. 最后才考虑小规模、低绝对贡献的局部路径。

一个路径即使占某个很小正式指标的很高比例，也不能仅凭“占比高”获得高优化优先级。

### 4.2 系统级重构优先于连续微调

当前阶段允许并鼓励协议等价的较大范围工程重构，包括但不限于：

- large / batch MSM 的组织；
- 同 SRS、同域对象的批处理；
- 消除语义重复的 commitment；
- 在数学关系允许时线性导出 commitment；
- opening claim 聚合和 witness 构造重组；
- quotient 构造与 materialization；
- polynomial 生命周期和有效长度感知；
- 稀疏、常量、selector 等对象的专用执行路径；
- NTT / INTT 组织；
- chunk、streaming 和 buffer reuse；
- CPU worker、任务调度和 NUMA；
- GPU kernel 划分、数据驻留、传输与同步；
- size / shape-aware CPU/GPU routing；
- 对正式计时语义本来就允许跨运行复用的 proof-independent 静态内容进行合法复用，但其目的必须是减少三个正式 timer 内的真实重复工作；静态内容在 timer 外的生成、加载或预处理本身不得作为当前性能优化成果。

任何具体模块都不是永久固定的优化目标。某个方向只有在当前有效证据表明它仍对正式时间有显著贡献时才值得继续深入。

### 4.3 测试、benchmark 与 profiling 只服务于决策

测试、benchmark 和 profiling 是正确性验证与性能决策工具，不是独立工作目标。

长期执行原则为：

- 优先复用仍能代表当前实现的有效证据，不为了“刷新数据”重复采集同类结果；
- 局部修改只进行与风险相匹配的必要检查；
- 一个结构性优化尚未形成时，不应反复运行完整回归、长 benchmark 或大规模 profiling；
- 较重的测试或 profiling 只有在其结果会直接影响“继续、保留、回滚或重新选择方案”时才有必要；
- 已有证据足以回答问题时，不继续增加同类测试；
- 临时 benchmark、profiler 或测试工具只在现有工具无法回答关键问题时创建。

因此，性能开发应以**完整结构问题和完整优化候选**为基本单位，而不是以“修改一个函数就进行一轮完整测试和 profiling”为基本单位。

micro / stage / kernel benchmark 只能证明局部路径的性质；它们不能替代正式生产路径的整体性能证据。

---

## 5. CPU、GPU 与数据规模专项边界

### 5.1 CPU 与 GPU 独立评价

CPU 和 GPU 是两个独立的正式性能路径，应分别评价其：

- `commitment_time_ms`
- `prove_time_ms`
- `verify_time_ms`
- `formal_total_time`

CPU 上的优化收益不能自动视为 GPU 收益，CUDA build 成功也不能视为 GPU 加速证据。

凡声称某项计算实际运行在 GPU 或获得 GPU 加速，必须能够确认真实 CUDA kernel 使用情况。

GPU 的最终目标是：在保持协议、安全性和正式计时真实性的前提下，同一数据集的 GPU `formal_total_time` 明显低于 CPU `formal_total_time`。

### 5.2 允许显式 size / shape-aware CPU fallback

GPU 模式不要求所有算子强制运行在 CUDA 上。

如果可靠性能证据表明某种规模、shape 或执行模式在 CPU 上更快，可以在 GPU 总体模式中使用显式 CPU 路由，但必须满足：

- 路由规则明确；
- 有真实性能依据；
- 不进行静默 fallback；
- 实际执行设备可识别；
- fallback 的时间真实计入 GPU 正式指标；
- 不得把 CPU 工作描述为 GPU kernel 性能。

目标是提高完整 GPU 模式的正式性能，而不是追求“所有算子形式上都在 GPU”。

### 5.3 PPI 多图专项

PPI 可以采用协议等价的多图专项工程策略，包括：

- 图级并行与图间调度；
- 多图批处理；
- 跨图可安全复用且实际减少正式 timer 内重复计算的工作；
- 图边界感知的数据布局；
- 同类 polynomial / commitment 的跨图批处理；
- 多图内存生命周期；
- CPU/GPU 多图任务划分。

任何专项优化都不得跨图错误聚合、改变图边界、改变 proof relation 或改变验证方程。

### 5.4 OGBN-Arxiv 大图专项

OGBN-Arxiv 可以采用协议等价的超大单图专项工程策略，包括：

- 超大域算法；
- large MSM 批处理；
- chunk / streaming；
- effective-length-aware 处理；
- 大对象和大域内存布局；
- NUMA；
- GPU residency / transfer；
- 超大域 verifier evaluation；
- size-aware routing；
- 避免不必要的完整域重复 materialization。

本节只规定长期允许的工程边界，不记录任何一轮 OGBN-Arxiv 的具体异常、数值或临时待办。

---

## 6. 正确性、安全性与资源约束

所有准备保留的性能修改必须满足：

- 正确；
- 安全；
- 协议等价；
- 与修改范围匹配的正常回归通过；
- 与受影响安全边界对应的 security / negative test 通过；
- 修改涉及共享数学逻辑、公共协议路径或双后端实现时，CPU/GPU 数学语义和公开输出一致；
- ProofSchema、transcript 和 verifier relation 不被破坏；
- private value 不泄漏；
- 用于正式性能结论的完整运行必须 `VERIFY_OK`。

`VERIFY_FAIL` 只能用于错误诊断，不得进入正式性能结果。

fresh proof 不得跨运行复用：

- witness-dependent polynomial；
- blinding randomness；
- quotient；
- opening；
- transcript；
- proof-specific MSM / pairing；
- verifier result；
- 其他 proof-specific 动态结果。

并行优化必须避免 outer/inner workers、NTT/MSM workers、异步任务和 CUDA concurrency 等组合造成：

- oversubscription；
- 线程数量乘法增长；
- NUMA 跨节点开销；
- 过量 context switch；
- memory-bandwidth saturation。

内存不是独立性能目标，但不得通过 swap、不可控的大域副本或巨大内存增长换取很小的正式时间收益。

---

## 7. 完整数据集实验权限与性能证据

正式数据集为：

- Cora
- Citeseer
- Pubmed
- PPI
- OGBN-Arxiv

最终需要五个数据集 CPU/GPU 共十组完整结果。

### 7.1 完整实验权限

最终十组完整实验必须由项目负责人亲自运行。

开发阶段：

- Citeseer full 仅由项目负责人运行；
- Pubmed full 仅由项目负责人运行；
- PPI full 仅由项目负责人运行；
- OGBN-Arxiv full 仅由项目负责人运行；
- 不允许由其他开发执行方自行运行五数据集 CPU/GPU 完整矩阵。

项目负责人在性能优化过程中运行的完整数据集结果，除非明确指定为最终冻结实验，否则默认属于当前性能分析和诊断证据，不自动成为论文最终结果。

### 7.2 Cora full 的唯一开发例外

当形成**预计能够显著降低 `formal_total_time` 的重大结构性性能候选**，并且已有较低成本证据充分支持其生产级价值时，可以进行必要的 Cora full 确认。

该例外仅用于重大候选的生产级验证，不用于：

- 常规 profiling；
- 刷新基线；
- 每个小候选的逐项比较；
- 仅由 micro/stage/kernel benchmark 高百分比触发的验证。

仅运行实际受候选影响的 backend；没有新的重大正式时间变化时，不重复运行。

开发期 Cora full 不能代替最终十组实验，也不能直接计入最终冻结结果。

### 7.3 最终结果冻结要求

最终十组结果必须来自**同一个冻结候选版本**，不得混用不同中间版本的数据。

同一数据集 CPU/GPU 运行必须保持相同的：

- 数据集；
- 模型；
- 协议参数；
- 证明参数；
- 当前唯一正式 SRS。

只允许 backend 所必需的设备和执行参数不同。

任何进入最终性能表、论文或正式报告的完整运行都必须得到 `VERIFY_OK`。

### 7.4 证据等级

性能结论应按证据范围准确表述：

1. unit / security / equivalence：证明正确性、安全性或数学等价性；
2. micro / stage / kernel：证明局部路径性质；
3. representative / integrated measurement：判断正式路径中的绝对贡献和候选价值；
4. 允许范围内的 Cora full：重大候选的生产级确认；
5. 五数据集 CPU/GPU full：完整数据集最高级别性能证据。

不得把低等级证据描述成高等级整体结论。

---

## 8. 性能候选的保留与回滚

性能修改只有在同时满足以下条件时才能作为当前正式实现保留：

- 正确；
- 安全；
- 协议等价；
- 不产生不可接受的内存、并发或设备副作用；
- 有与证据等级相匹配的依据支持其对至少一个正式指标具有真实价值；
- 整体不违反正式指标之间的权衡规则。

以下候选应回滚或删除：

- 正确性失败；
- 安全性失败；
- 明确造成 `formal_total_time` 退化；
- 违反正式指标权衡边界；
- 已有充分证据表明没有正式时间价值；
- 产生新的明显正式性能异常；
- 实际无收益却增加代码复杂度；
- 只有局部 benchmark 好看，但无法影响正式路径。

full dataset 不是每个候选保留的前置条件。没有 full 确认时，可以根据充分的局部或集成证据保留候选，但只能按实际证据范围描述收益，不得声称已经确认完整数据集整体提升。

---

## 9. 本地主实验工作区唯一当前状态

除 `comparison/` 外，`/home/pzh/project` 的持久状态必须始终表示当前唯一有效的主实验版本。

开发过程中可以临时产生必要的 benchmark、profile、proof、日志和对比产物，但在相应工作完成后必须删除、正式化或归并为唯一 canonical 状态。

主实验工作区不得长期保存开发历史。

禁止长期保留：

- `old`
- `backup`
- `legacy`
- `previous`
- `pre_*`
- `preserved`
- `v1/v2/vN`
- `phase*`
- 日期型历史副本
- 修改前备份
- 被替代实现
- 失败候选
- 大段注释掉的旧实现
- 新旧两套重复实现
- 已失效接口、宏、debug path 和临时 workaround

如果某个带有历史或版本式名称的文件仍被当前正式实现引用，应先把正式依赖迁移到按当前职责命名的 canonical 路径，再删除旧名称；不能因为“仍被引用”而永久保留历史命名。

### 9.1 本地实际文件系统是清理依据

工作区治理以 `/home/pzh/project` 当前实际本地内容为依据，包括主实验相关的：

- 源码；
- 配置；
- 测试；
- 数据和 artifacts；
- SRS；
- `runs/`；
- 隐藏的主实验文件/目录；
- spill；
- 失效 lock；
- logs；
- temporary proof；
- benchmark/profile/debug 输出；
- 项目自身 cache；
- 临时 target；
- 失效 build 产物。

编辑器、操作系统或外部开发工具自身维护且与主实验无关的元数据不在清理范围内。

### 9.2 SRS 唯一性

每种当前正式用途只保留一个 canonical SRS。

不得为了保留历史而同时保存 `pre`、`v2`、`preserved` 等多个版本。

删除旧 SRS 前应确认当前源码、配置、测试和运行时不再依赖它。

### 9.3 临时测试、benchmark 与运行结果

一次性 benchmark、profiler、test helper、临时 CMake target 和调试输出在完成诊断后：

- 具有长期回归价值的，规范命名并纳入正式体系；
- 仅服务临时诊断的，删除源码、target、日志和输出。

`runs/` 不作为开发历史档案。

当前仍用于当前实现性能判断、正确性验证或正式结果的内容可以保留；一旦被新的有效状态完全替代且结论已经提取，应删除旧状态结果、旧 proof、旧 profile 和旧日志。

删除任何疑似历史内容前，应先检查当前源码、构建、配置、测试和运行时依赖，不能仅凭文件名盲删。

---

## 10. 文档治理

正式文档只描述当前有效状态，不承担开发历史记录功能。

### `GAT-ZKML-多层多头.md`

作为最高技术规范保留，默认只读。

### `PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md`

只描述当前长期有效的项目背景和约束，不记录：

- 某一轮热点；
- 某一轮异常；
- 临时性能数字；
- 当前日志路径；
- 当轮执行步骤；
- 失败优化历史；
- 临时待办。

### `README.md`

只描述当前有效的：

- 项目用途；
- 构建与运行方式；
- CPU/GPU 边界；
- 安全状态；
- 当前正式配置；
- 实验权限；
- 与当前实现对应的有效性能信息；
- 当前真实限制。

README 不记录逐轮开发历史。

### Final report

只有真正进入最终冻结结果阶段后，才应存在与同一冻结版本最终十组结果相对应的 final report。

历史阶段文档、旧 benchmark 汇总、旧回滚记录或已经失去当前用途的说明文件，不应仅为了“留档”继续存在于主实验工作区。

---

## 11. `comparison/` 完整隔离

`comparison/` 不属于当前主实验处理范围。

主实验相关工作必须：

- 不检查其内部实现；
- 不修改；
- 不删除；
- 不运行；
- 不清理；
- 不纳入主实验性能统计；
- 不让主实验依赖其中代码、配置或结果。

所有“主实验工作区清理”和“唯一当前状态”要求均不进入 `comparison/`。

---

## 12. 阶段完成与最终冻结权限

性能阶段是否已经完成，不以以下情况判断：

- 完成了若干局部 benchmark；
- 优化了若干函数；
- 某几个 CUDA kernel 大幅提速；
- 已进行了很多轮修改；
- 某个局部路径已经没有继续优化空间。

只有当主要正式耗时已经获得实质改善、主要结构性瓶颈均已有明确结论，并且不存在仍然现实可行且预计能显著降低 `formal_total_time` 的大型优化方向时，才可以认为性能阶段接近收敛。

是否：

- 结束当前性能优化阶段；
- 冻结正式实现；
- 运行最终五数据集 CPU/GPU 十组实验；
- 生成最终报告；

均由项目负责人决定。
