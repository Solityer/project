# PERFORMANCE_OPTIMIZATION_CONSTRAINTS

## 1. 文档定位

本文档**仅用于项目负责人后续明确启动的 zkGAT 主实验性能优化阶段**。当前机制级消融重构期间，本文件的性能审计、热点优化、正式重跑等专项任务尚未启用，不得由 Codex 自动执行。

当前正式模型采用多层多头 GAT，正式证明路径与该模型结构保持一致，性能优化阶段的目标是：

> **在不修改协议方案、证明语义、安全边界和正式模型语义的前提下，尽可能从执行层、GPU 路径和结构性实现上降低 commitment / prove / verify 的端到端正式耗时。**

本文档只约束主实验性能优化，不用于重新设计协议、模型或实验方案。

---

## 2. 约束优先级

项目中的约束优先级如下：

1. `GAT-ZKML-多层多头.md`
2. `PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md`
3. `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md`
4. README、普通代码注释、临时分析记录和当前 Codex prompt

如果低优先级内容与高优先级内容冲突，必须以高优先级内容为准。

### 2.1 最高权限文档只读约束

以下三份文件属于项目只读约束文件：

- `GAT-ZKML-多层多头.md`
- `PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md`
- `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md`

性能优化阶段不得：

- 修改；
- 重命名；
- 移动；
- 删除；
- 通过性能优化文档、README、代码注释或当前 prompt 覆盖其约束。

如果当前实现与上述三份文件发生冲突，应修改实现，而不是修改只读约束文档以适配代码。

### 2.2 `comparison/` 完全隔离

本性能优化阶段只针对主实验。

`comparison/` 目录及其全部子目录：

- 不分析；
- 不修改；
- 不删除；
- 不运行；
- 不作为本阶段测试对象；
- 不得根据其中实现反推或替代主实验优化方案；
- 不得将 comparison 的结果、代码路径、配置或性能数据混入主实验判断。

---


## 3. 当前阶段基本前提

当前阶段默认以下内容已经冻结，不再重新设计：

- 正式多层多头 GAT 架构；
- 正式 proof schema；
- prover / verifier 的证明语义；
- transcript、opening、KZG、quotient 等协议关系；
- 安全边界与安全参数；
- 正式训练架构、训练配置和模型语义；
- CPU / GPU 必须执行同一正式协议语义；
- 已经验证有效的既有性能优化。

性能优化必须建立在这些前提之上。

---

## 4. 最高优先级硬约束

### 4.1 不允许修改协议方案内容

不得为了提高性能修改协议本身。

包括但不限于：

- 不修改 `GAT-ZKML-多层多头.md` 中已经确定的协议方案；
- 不改变正式证明关系；
- 不改变正式对象定义；
- 不减少或删除证明对象；
- 不减少或弱化约束；
- 不改变 transcript 语义；
- 不改变 opening 语义；
- 不改变 KZG 语义；
- 不改变 quotient 语义；
- 不改变 verifier 的正式验证逻辑；
- 不新增、删除或改变正式 public input；
- 不新增、删除或改变 proof 字段；
- 不改变 SRS 的正式绑定语义；
- 不通过弱化安全性、正确性或完整性换取性能；
- 不改变多层多头 GAT 正式前向计算语义。

核心原则：

> **数学关系不变、证明内容不变、安全边界不变，只优化“如何更快地执行”。**

### 4.2 正式计时语义和统计边界冻结

`commitment_time_ms`、`prove_time_ms`、`verify_time_ms` 的正式计时语义、阶段归属和统计边界必须保持不变。当前正式源码使用互不重叠的 commitment 与 prove 时间分区，`prove_time_ms` 不包含 `commitment_time_ms`；具体以项目长期约束和真实计时实现共同核对，任何“优化”都不得以改变此口径为手段。

不得通过以下方式人为降低正式时间：

- 将原本属于某阶段的计算移动到该阶段计时区间之外；
- 在正式计时开始前预执行本应计入正式阶段的计算；
- 将本应计入当前阶段的工作延迟到计时结束后；
- 通过隐藏预计算、延迟计算、异步未同步结束等方式缩短表面计时；
- 修改 benchmark accounting 规则以获得更低数字；
- 将真实执行成本转移到未统计的初始化、导出、缓存准备或其他非正式区间；
- 通过跨独立正式运行复用本应重新计算的协议状态来降低论文计时。

允许协议语义内合法的 resident cache、跨阶段中间量复用和执行层缓存优化，但：

- 必须属于同一次正式运行内合法可复用的执行状态；
- 不能改变正式协议语义；
- 不能改变正式计时口径；
- 不能把本应属于正式阶段的真实成本隐藏到统计范围之外。

### 4.3 尽可能只做执行层优化

优先优化范围包括：

- GPU kernel；
- GPU 覆盖范围；
- CPU → GPU 热点迁移；
- GPU batching；
- kernel fusion；
- GPU stream 调度；
- CPU / GPU 并行调度；
- 数据布局；
- 内存布局；
- temporary buffer 生命周期；
- resident cache；
- SRS/base resident reuse；
- 减少重复 host ↔ device transfer；
- 减少重复 field conversion；
- 减少重复 materialization；
- 减少重复 FFT / NTT；
- 减少重复 MSM；
- 减少重复 opening 中间量构造；
- streaming；
- chunking；
- workload-aware scheduling；
- 大规模 MSM / NTT / quotient / opening 的执行实现。

---

### 4.4 正向优化与缺陷修复的交付门槛

主实验性能优化在下一次独立任务启动后，必须以**真实正确且可测量的正向收益**为目标。优化候选应先有源码与 profiling 依据，再有可验证的算法/执行设计和与之对应的短时回归。

优化后若出现构建、CUDA、prover、verifier、证明序列化或资源准入错误，必须定位新实现的根因并修复；不得把关闭新路径、回退旧实现、绕过断言或安全检查作为“问题已经修复”的交付结论。

若某候选始终无法同时通过正确性、安全性与有效收益验收，不得把它认定为有效优化，也不得为了交差声称已经解决；必须如实报告未完成原因和阻塞证据。此规则不意味着允许运行带已知缺陷的生产代码。

性能收益必须以固定计时定义下、相同模型/工作负载/安全参数/后端条件的配对测量为依据。开发阶段只允许短时的真实计算和局部/父级代表性测量，可据已有正式 profile 估计对 commitment / prove / verify 的潜在贡献，但不得据此宣称已实测完整正式数据集的加速。单个 kernel 加速或 profiler 内的缩短不能替代父级收益的合理证据；最终全量正式收益仍须由项目负责人后续统一实测确认。

---

## 5. 已有有效优化必须保留

所有已经验证有效的优化点都必须继续保留。

不得：

- 删除已有有效优化；
- 回退已有有效优化；
- 绕开已有有效优化路径；
- 为统一代码路径而取消已经证明有收益的专项优化；
- 因新优化导致旧优化失效而不处理。

包括现有已验证有效的优化开关与执行路径，例如：

- `fast_msm`
- `parallel_fft`
- `packed_fft_backend`
- `rotated_fft_kernel`
- `compact_trace_layout`
- `fast_verify_pairing`
- CUDA hotspot 路径
- 已有 multi-graph / large-graph 通用优化

新增优化不能破坏现有优化的正确性和实际收益。

---

## 6. 优化目标：只处理高收益热点

性能优化必须优先处理：

1. **父级大热点**
2. **结构性热点**
3. **端到端耗时占比高的热点**
4. **GPU 路径上的明显 CPU fallback**
5. **对 commitment / prove / verify 总时间有显著影响的问题**

不得以“某个函数可以更快”为理由就直接深入优化。

每个候选优化点必须先回答：

- 当前耗时占 commitment / prove / verify 总时间多少？
- 是否属于父级或结构性热点？
- 理论上最多能降低多少端到端时间？
- 是否值得承担新的复杂度、同步成本、内存成本和维护成本？

如果预期端到端收益很小，则不应继续深入。

---

## 7. 当前已知重点热点

当前真实 profiling 中已经暴露出的重点包括：

- quotient materialization；
- `t_edge`；
- `t_N`；
- FH 相关 quotient / opening；
- edge 相关 opening；
- domain opening witness；
- dynamic commitment MSM；
- 大规模 MSM；
- GPU 未覆盖或覆盖不足的 NTT / IFFT；
- GPU 未覆盖或覆盖不足的 quotient materialization；
- 大规模 opening fold / witness construction；
- CPU ↔ GPU 数据搬运和同步；
- resident / streamed CUDA 路径的批处理与复用。

以上是当前重点，但**不是唯一允许优化的范围**。

如果全仓库审计发现更大的新结构性热点，应按端到端收益重新排序。

---

## 8. 必须进行全仓库 GPU 正式路径审计

不能只盯当前某一个函数或某一个文件。

必须沿正式调用链检查：

- 哪些大阶段仍然落在 CPU；
- 哪些 GPU 路径只覆盖了一部分工作；
- 是否存在重复 host ↔ device 传输；
- 是否存在 GPU kernel launch 过碎；
- 是否存在同步点过多；
- 是否存在可融合的连续 kernel；
- 是否存在重复 allocation / free；
- 是否存在已经上传的数据反复上传；
- 是否存在 SRS/base 可 resident 但没有复用；
- 是否存在大规模中间量被重复构造；
- 是否存在重复 field conversion；
- 是否存在重复 polynomial materialization；
- 是否存在重复 FFT / NTT；
- 是否存在重复 MSM；
- 是否存在 opening 中间量重复生成；
- 是否存在 CPU/GPU pipeline 无法重叠；
- 是否存在 multi-stream 或批处理可以显著减少等待；
- 是否存在大图时 working set、显存和内存生命周期不合理；
- 是否存在多图时跨图调度和 batching 不合理。

---

## 9. 禁止沉迷局部小热点

以下行为不允许：

- 对单个小函数持续进行多轮优化；
- 对单个小 kernel 持续深挖；
- 为几十毫秒甚至更小的收益增加大量复杂代码；
- 一个热点对 commitment / prove 总时间影响微乎其微，却持续优化；
- 局部 benchmark 提升明显，但端到端理论收益极低，仍继续投入；
- 因“还能更快一点”而无限优化已经足够小的阶段。

停止条件：

> 当某个热点对端到端总时间已经不再产生显著影响，应立即停止深入，转向下一个更大的父级或结构性热点。

---

## 10. 优化成功必须以端到端收益为导向

局部函数、单 kernel 或单阶段变快，不等于优化成功。

必须围绕其所针对的正式父级指标判断：

- commitment 类优化是否能够显著降低 `commitment_time`；
- prove 类优化是否能够显著降低 `prove_time`；
- verify 类优化是否能够显著降低 `verify_time`；
- 如果优化跨越多个正式阶段，则检查相应多个父级指标；
- 其他正式指标不得出现无合理原因的明显回退；
- 是否把瓶颈转移成新的严重热点；
- 是否引入新的同步或数据搬运成本；
- 是否导致其他已优化阶段明显回退。

不要求每一个优化同时降低 commitment、prove、verify 三项。

性能开发期间不得为了核对父级收益运行完整正式 GAT proof。应使用可控的小型真实父级路径或上下游集成测量，结合既有正式 profile 的热点占比分析其潜在端到端收益。如果短时证据无法支撑父级受益，不能把该候选认定为已取得主实验正式加速；留待负责人后续全量结果确认，不允许以此为由扩张开发测试规模。

---

## 11. 优先减少重复工作

必须重点排查并优先处理以下重复工作：

- 重复 FFT / NTT；
- 重复 MSM；
- 重复 polynomial materialization；
- 重复 domain conversion；
- 重复 field conversion；
- 重复 host ↔ device transfer；
- 重复 SRS/base preparation；
- 重复 opening witness / fold 中间量构造；
- 重复 selector / evaluation helper 构造；
- 可以跨阶段复用但当前没有复用的中间结果。

---

## 12. 优先扩大 GPU 对父级大热点的覆盖

如果一个父级大热点仍然明确处于 CPU fallback，并且适合 GPU：

> 应优先评估将该父级阶段迁移或扩大到 GPU，而不是继续优化已经很快的 GPU 小函数。

重点评估：

- quotient materialization；
- large-domain NTT / IFFT；
- large opening witness；
- large opening folds；
- large-domain polynomial operations；
- large MSM；
- 多个相同结构对象的批量 GPU 执行。

---

## 13. GPU 优化必须同时考虑调度、同步和数据移动

任何 GPU 优化都不能只看 kernel 本身。

同时必须审查：

- kernel launch 数量；
- kernel 粒度；
- stream synchronization；
- device synchronization；
- host-side blocking；
- temporary allocation；
- pinned / pageable memory；
- H2D；
- D2H；
- unified memory；
- resident data 生命周期；
- SRS/base cache 生命周期；
- batch size；
- stream overlap；
- compute / transfer overlap。

不得把原来的 CPU 计算瓶颈简单替换成 GPU 同步或搬运瓶颈。

---

## 14. CPU 正式路径不得被破坏

虽然本阶段重点优化 GPU 正式路径，但：

- CPU 正式路径必须继续正确；
- CPU / GPU 正式语义必须一致；
- 共享代码修改不能使 CPU 路径失效；
- 不能为了 CUDA 性能直接删除 CPU 必需逻辑；
- 不能引入 GPU-only 的协议语义。

必要时应通过轻量 regression 验证共享逻辑没有破坏 CPU 路径。

---

## 15. 多图 workload 专项优化允许且鼓励

允许针对 PPI 所代表的**多图 workload**继续进行专项优化。

但优化必须基于多图结构本身，而不是数据集名称。

允许优化：

- `G_batch > 1` 的执行调度；
- `node_ptr / edge_ptr`；
- 图边界管理；
- 多图 batching；
- 多图 GPU kernel 合并；
- 多图 opening 批处理；
- 多图 quotient 批处理；
- 多图内存复用；
- 多图 temporary buffer reuse；
- 多图 stream scheduling；
- 跨图 workload balancing。

严禁：

```text
if dataset == "ppi"
```

或其他针对 PPI 名称的特殊路径。

---

## 16. 超大图 workload 专项优化允许且鼓励

允许针对 OGBN-Arxiv 所代表的**超大图 workload**继续进行专项优化。

优化必须由实际 workload 特征触发，例如：

- `N`；
- `E`；
- domain size；
- FH domain size；
- edge domain size；
- MSM scalar count；
- polynomial size；
- working-set bytes；
- GPU memory capacity；
- temporary memory；
- batch size；
- stream size；
- transfer amortization。

允许优化：

- 大图 streaming；
- 大图 chunking；
- resident / streamed hybrid route；
- 显存受限的大对象分批；
- large-domain NTT / quotient；
- large MSM；
- large opening；
- working-set-aware scheduling；
- 内存峰值控制；
- host/device pipeline。

严禁：

```text
if dataset == "ogbn-arxiv"
```

或针对某个数据集固定规模写死逻辑。

---

## 17. 只允许 workload-aware，不允许 dataset-specific

专项优化的统一原则：

> **允许 workload-aware optimization，不允许 dataset-specific optimization。**

所有优化条件必须根据通用执行特征触发，而不是数据集名称。

如果必须使用阈值，阈值必须有通用执行依据，例如：

- graph count；
- N / E；
- domain size；
- MSM size；
- working-set size；
- GPU memory；
- batch amortization；
- transfer amortization；
- kernel launch cost。

不得为了适配某一个正式数据集写 magic number。

---

## 18. 大图性能不能以不可接受的内存风险换取

针对大图的优化必须同时控制：

- GPU 显存峰值；
- host 内存峰值；
- unified memory 压力；
- temporary buffer 峰值；
- working set；
- resident SRS/base cache；
- streaming buffer 数量。

不得为了减少时间而引入明显 OOM 风险或不可控内存增长。

---

## 19. 新增重大优化应尽量保留可消融性

对于对论文性能结果有重要影响的新增执行层优化，如果工程上合理，应尽量保留其可独立比较与验收能力：

- 独立控制；
- 可单独关闭；
- 可记录当前启用状态；
- 可用于后续 ablation。

但不能为了“有开关”而增加不必要的复杂度。

已有已验证有效的执行层优化开关不得无故删除。诊断或 A/B 测量时可比较合法执行路径，但**不得以关闭或回退新优化路径来冒充修复其错误或交付成功**。**已取消的完整 GAT 端到端消融矩阵、其专属运行入口和废弃配置不属于必须保留的执行层优化能力**；对机制级消融和正式主实验仍有依赖的证明实现应保留并复用，不得误删。机制级消融本身不是本文件要求继续增加的新一轮端到端消融。

---

## 20. 性能 profiling 与诊断代码不得污染正式性能

新增 profiling、debug counter、trace logging、统计采样和诊断逻辑必须满足：

- 可以关闭，或其运行开销可以证明为可忽略；
- 不得显著改变 kernel 调度、同步、内存访问模式或缓存行为；
- 不得因为性能统计本身导致 commitment / prove / verify 明显变慢；
- 最终正式实验中不得保留高开销 debug / profiling 模式；
- 性能结论不能建立在 profiling 本身明显改变执行行为的情况下。

---

## 21. 正式工件与轻量测试必须隔离

性能开发测试必须使用 `tests/` 下的临时目录、fixture、专用测试工件与隔离输出路径。不得在正式 `runs/` 或 `results/paper_secure/` 内放置仅供开发诊断的测试结果，也不得修改既有正式证明对象、canonical SRS 或 checkpoint。

不得覆盖、修改或污染：

- 正式 checkpoint；
- trusted SRS；
- 正式配置；
- canonical dataset cache；
- 正式 proof；
- 正式 run manifest；
- `runs/*/formal` 类正式运行目录；
- `results/paper_secure/` 中的正式结果；
- 已经用于论文正式实验的其他 canonical artifact。

测试代码和测试输出必须能够在不影响正式工件的情况下独立删除。

---

## 22. 性能优化阶段禁止大规模、高频与重复验证

**本阶段不运行大规模或大量验证测试。** 禁止完整 Cora/Citeseer/Pubmed/PPI/OGBN-Arxiv 正式 proof、完整 CPU/GPU 十组矩阵、接近正式数据集规模的缩规模伪装测试、正式训练、大型 SRS，以及长时间、高内存、高显存的压力测试。不得以“验收”“回归”“benchmark”或“确保全面”为由扩大执行规模。

正式主实验由项目负责人在之后单独安排。Codex 只执行能够支撑开发正确性判断的少量、短时、低资源、可明确终止的针对性验证。**十分钟级及以上、明显高资源或实际近全量工作负载均不允许 Codex 执行**；十分钟是禁止越过的权限边界，不是推荐的单项测试时长。优先选择秒级至少量分钟级的真实计算；必要时设置合理的诊断超时和资源上限。

**不允许每修改一两个函数便重新运行测试。** 不允许对每个局部改动重复构建全部目标、执行相同 proof/verify、CPU/GPU 矩阵或安全测试，也不允许为了“保证充分”反复运行与改动无关的现有测试。测试多少由风险覆盖决定，不由修改次数、函数数量或“多跑更保险”决定。

为修复具体失败而重测受影响路径是允许的；无新证据、无新增风险的重复测试不允许。性能优化不等于为将来的正式实验提前采集全量数据。

---

## 23. 按完整优化单元集中验证：少而精、覆盖充分

**验证的单位是完整、连贯的优化改动集（optimization unit），不是单个函数或每次代码保存。** 一个优化单元围绕同一已证实的父级瓶颈或紧密关联的计算链，可涉及多个函数、文件、kernel、数据布局、调用方和资源生命周期。先完成该单元需要的实质性实现，再进行一次有组织的针对性验证；多个相互依赖的小修改尽量合并为同一验收批次。

动手前先建立最小“改动—风险—验证”对应关系，识别受影响的调用方、消费者、后端、数据规模和数学不变量，并复用已经通过且仍适用的现有测试证据。编码过程中允许静态检查、编译器诊断、代码审查和必要的局部调试，但不得把这些动作扩展成每改一两个函数就执行一整轮验证。

完整优化单元就绪后，**一次集中验证**应按实际风险选取以下必要覆盖，而不是不加区分全部执行：

| 变更涉及范围 | 优先选用的最小有效验证 | 不能仅凭什么认定通过 |
|---|---|---|
| NTT/FFT、MSM、有限域或多项式内核 | 小型真实输入的精确参考结果或合法代数等式、尺寸/填充边界；触及 CUDA 时与 CPU 路径对照 | 仅 kernel 正常返回或一个数值样例 |
| KZG commitment、quotient、opening、transcript 或其生产执行链 | 小型真实 proof → 规范序列化 → **独立 verifier** 验收；对真正受影响的绑定/声明加入至少一种有效篡改或错误输入拒绝检查 | 仅 prover 返回成功、只有 `VERIFY_OK` 日志，或只测函数耗时 |
| CUDA 内存、异步调度、resource admission、streaming/chunking | 短时可控内存和域规模下的正确结果、同步与错误路径；必要时 CPU 对照和资源拒绝检查 | 仅 CUDA kernel 启动成功或静态显存估算 |
| 多图、稀疏边、索引和数据布局 | 少量有代表性的非均匀度数与多图边界样例、输入输出对应关系与索引越界检查 | 仅单图/规则形状样例通过 |
| 共享执行接口、计时、序列化或 verifier-facing 路径 | 受影响调用链的紧凑集成验证、字段/身份/计时不变量及必要的拒绝行为 | 仅编译成功或普通 smoke test |

覆盖应形成从**关键数学不变量 → 受影响执行路径 → 必要上下游接口**的闭环。对于会影响证明语义的执行代码，至少需要一条真实、独立验证通过的微型 proof/verify 集成路径，并在有关绑定与安全门禁确实受到影响时检验拒绝非法证明。无需因只改代数 kernel 而重复全部无关的训练、模型、SRS、安全用例或数据集矩阵。

修复某个失败时，先重跑失败用例和最小受影响接口；只有依赖或覆盖范围发生变化时才增加验证。全部关联改动完成后，最多组织一次范围受控的整体整合回归，复用之前通过的证据而非重跑整个测试库。**“测试全面”是风险维度覆盖完整，不是样本、用例、轮次或运行时间最大化。**

---

## 24. 开发验证代码统一放在 tests/

所有 test-only 的 benchmark、fixture、回归、负例、性能对照脚本和测试产物仅放在 `tests/` 下，并按当前优化单元复用现有测试工具。除非生产执行本身确实需要新增正式代码，否则不在 `src/`、`include/`、`tools/`、`scripts/`、正式 `configs/` 或项目根目录散落只为测试服务的 helper、日志和工件。

不得为了满足测试数量而复制近似用例、增加临时大测试框架或扩张无关协议测试。必要的测试扩展应以当前代码确实新增的风险为依据；正式机制级实验入口与真正生产工具不属于 test-only，仍按其职责安排目录。

---

## 25. 测试范围以优化单元和实际影响面为准

在每个优化单元开始时确定其输出正确性的可信参考以及最短必要的验证链。典型选择为：

- MSM/NTT 等数学计算：小规模准确性、边界和必要的 CPU/GPU 对照；若修改穿越 PCS 边界，才增加一条真实 commitment/opening 或 proof/verify 集成。
- quotient、opening 或承诺生产链：对应域关系、真实短时 proof/verify，以及针对受影响声明/绑定的定向负例；避免重跑无关组件。
- CUDA residency、streaming、workspace 与 FH coset 准入：资源判定、分块/同步正确性和缩规模但具有相关工作集特征的计算；必要时验证拒绝/合法回退路径，严禁把绕过错误认作成功。
- multi-graph、edge mapping 和 sparse aggregation：若受影响，使用包含图边界、不同度数和合法边映射的小型样例；否则无需运行多图专项测试。
- timing、serialization 或共享接口：检查计时分区、身份绑定、必要的真实 verifier 消费与结果导出；不改变既有正式计时边界。

共享代码影响多个调用方时，应选择能覆盖真正不同语义分支的**最小组合**，而不是遍历每个数据集或每种开关。对没有被改变的安全关系与协议路径，优先使用代码审计和既有有效验证证据，不新增无意义的重复测试。

---

## 26. 优化单元的正确性和正向收益验收

不得因一次编译、一个 smoke test、单个 kernel benchmark 或一次 `VERIFY_OK` 日志就宣布优化完成。一个完整优化单元应同时满足：

1. **协议不变量有证据**：正式 GAT 语义、数学关系、公开/私有输入、ProofSchema、Fiat–Shamir 顺序、KZG、quotient、opening、SRS 绑定、120-bit 安全目标及正式计时口径均未被改写；对执行层改动有可信的实现或代数等价依据。
2. **相关代码确实正确**：受影响源代码能够正确构建；选择的真实数值、域/索引边界和 CPU/GPU 一致性验证通过；触及真实证明路径时，短时真实 proof/serialization/独立 verify 通过，并按实际风险检查非法声明或篡改拒绝。未触及的路径不要求重复跑无关测试。
3. **存在正向性能证据**：使用相同受控输入和计时条件对优化前后执行路径做尽量少的配对测量；结合已知正式 profile 和真实父级调用关系，确认目标热点有实质性改进且未制造明显的新同步、数据搬运或资源瓶颈。不在此阶段跑全量数据集来证明收益，最终正式收益留待项目负责人统一实测。
4. **不引入新的生产风险**：不破坏已有有效优化、CPU 运行、GPU 资源准入、多图/大图通用能力，不污染 checkpoint、SRS、cache、proof、正式 results，也不留下已知的 verifier 失败、OOM 风险或安全绕过。
5. **测试投入受控**：实际只执行与本单元有关的必要短时验证；已运行内容有明确目的和结论，无循环重跑、无大规模测试，测试失败已经按根因局部修复且必要范围复验。

有一项不满足，就不得声称该优化单元已成功。遇到新实现 bug，必须定位并修复根因，不得以关闭优化、回退到旧实现或弱化检查冒充已完成；若候选方案确实无法达到正确性与性能双重门槛，应明确报告阻塞，不得交付带已知错误的实现或虚报正向收益。**这不是要求无限测试，而是要求关键证据闭合。**

---

## 27. 开发测试结果不等于论文正式结果

轻量 microbenchmark、缩规模测试、受控阶段 benchmark 和短时 proof/verify 可作为开发正确性、性能趋势及优化候选筛选证据，但不能作为正式五数据集 CPU/GPU 端到端性能结果，也不得混入 `results/paper_secure/`。

未执行完整主实验就不得声称“正式 commitment/prove/verify 已降低若干秒或百分比”。可以明确报告实测的局部收益与根据正式历史 profile 估计的潜在影响，且二者必须严格区分。最终统一正式实验只能由项目负责人另行安排。

---

## 28. 不允许预防性大范围重构

不得因为“以后可能更快”进行没有真实热点依据的大范围重构。

必须基于：

- 当前真实调用链；
- 当前 profiling；
- 当前 GPU fallback；
- 当前阶段耗时；
- 当前数据搬运和内存行为；

再决定是否修改。

原则：

> **先审计、后修改；先证明值得优化，再投入实现。**

---

## 29. 不允许为了性能扩大协议状态面

执行层优化不得引入新的：

- public input；
- proof field；
- transcript element；
- verifier dependency；
- SRS semantic dependency；
- 正式协议对象；
- 正式安全假设。

性能优化只能改变执行方式，不得改变协议可观察语义。

---

## 30. 性能优化阶段的精简工作流程

每轮围绕一个具有明确性能收益预期的**成组优化单元**，而不是每个函数单独开工、单独测试：

1. 沿正式 CPU/GPU 调用链审计真实 profile，选择父级大热点并分析工作集、数据流与关键数学关系。
2. 明确优化单元的完整修改边界、预期父级收益、性能风险及必须保持的不变量；计划覆盖相关的多个函数/文件/调用方，避免碎片化反复测试。
3. 在不改变 zkGAT 协议方案的前提下完成整个连贯改动；过程使用源码审查与必要的编译器诊断，不进行每修改一两个函数就触发的整套回归。
4. 改动集就绪后，集中进行**一次有针对性的构建与正确性验证**：选择最小但完整的数学、CPU/GPU、资源/边界和必要真实 proof/verify 验证；根据风险选取定向安全负例，不盲目运行所有测试。
5. 用受控、短时、尽量少轮次的前后配对测量确认对应热点的真实正向收益，并结合既有 profile 判断是否值得进入下一优化单元；如果测量噪声妨碍判断，只增加必要复测，不无限重复。
6. 若失败，定位根因、修复当前实现并仅复测受影响路径。达到正确性和性能门槛后记录真实结论，停止微调该点，转向下一个更大热点。
7. 多个相关单元均结束后，执行**一次范围受控的必要集成回归**，确保它们的交互没有破坏正式 zkGAT 的证明、验证和 CPU/GPU 行为；不得变相执行全部正式数据集或大量无关测试。
8. 归纳已实测的局部收益、推断的父级潜力、未覆盖的正式规模风险和仍需负责人后续统一正式测量的事项，不自动启动最终正式实验。

测试计划应追求**最少执行次数下覆盖全部重要风险**；禁止“没测充分就 pass”和“测试越多越安全”两种做法。

---

## 31. 性能优化阶段的停止条件

在证据充分时停止当前优化单元，而不是反复在小函数上试错、回滚或重复跑测试。整体优化阶段满足以下情况即可向项目负责人报告开发闭合：

- 高占比父级热点已经得到有依据的正向优化，或剩余可行优化没有足够高的预计端到端价值；
- 所有已实施改动保持正式 zkGAT 协议、ProofSchema、安全参数、正式模型语义及固定计时边界；
- 按第 23–26 节进行了**次数受控但风险覆盖充分**的短时真实验证，相关集成调用链和 CPU/GPU 语义没有发现未解决错误；
- 无已知 verifier 失败、未处理的安全门禁缺陷、明显同步/数据搬运倒退、不可接受内存风险或既有有效优化被破坏；
- 已提供优化清单、真实局部测量、不能由开发测试确认的全量收益与限制，不以假定的正式大图加速冒充最终论文结果。

满足后**停止优化并汇报**。只有项目负责人明确授权后才进入正式十组结果采集；不得为了“全面验收”由 Codex 启动大型证明或重复运行大量验证。

---

## 32. 最终正式实验只能在性能冻结后执行

仅在项目负责人启动主实验性能优化、且该阶段全部开发闭合之后，才进入由项目负责人另行安排的最终正式实验阶段。此条不授权 Codex 在当前机制级消融完成后自行优化或统一重跑。

最终阶段一次性执行：

- 5 个正式 CPU 全量实验；
- 5 个正式 CUDA 全量实验；
- 共 10 组正式结果；
- 全部要求真实 `VERIFY_OK`；
- 汇总 commitment / prove / verify / proof size；
- 执行最终安全检查与结果聚合。

正式十组实验不是日常性能开发测试。

---

## 33. 最终核心原则

- **协议不动**：正式证明关系、对象、ProofSchema、KZG、Fiat–Shamir、quotient、opening、安全边界、多层多头推理和 verifier 语义均不改变；只优化实现与执行路径。
- **GPU 优先、CPU 保真**：优先处理高收益的 GPU/CPU 大热点，保留已有有效优化，CPU 与 GPU 的数学结果、证明语义和安全等级一致。
- **只优化有实质价值的大热点**：用真实调用链与现有 profile 选择父级或结构性瓶颈；允许通用 workload-aware 处理，禁止 dataset-name-specific 捷径。
- **不挪动正式计时**：`commitment_time_ms`、`prove_time_ms` 和 `verify_time_ms` 的统计边界冻结，Commitment 与 Prove 不重叠；不得转移成本、隐藏预计算或跨独立运行复用状态虚报收益。
- **完整改动后集中测试**：禁止改一两个函数就启动回归，禁止重复无效测试、大规模数据集验证；在优化单元完成后按真实风险选择少量短时但覆盖充分的数学正确性、CPU/GPU、proof/verify、资源/边界与必要负例检查。
- **不以简单 smoke test 放行**：编译通过、单个算子快、控制台 `VERIFY_OK` 或单条正例都不能替代受影响调用链和安全关系的验证。
- **错误必须根因解决**：不得以回退旧路径、关闭新优化、绕过异常、弱化 verifier 或放宽安全条件来冒充 bug 修复或成功优化；无法闭合的候选必须如实报告而非交付缺陷。
- **正式工件和对比实验隔离**：`comparison/` 完全不参与本阶段；所有 test-only 内容位于 `tests/`，不得污染正式 checkpoint、SRS、cache、proof、results。
- **阶段独立**：性能优化只在项目负责人新 Prompt 中启动；开发完成后汇报并停止。最终五个数据集、CPU/GPU 十组全量正式实验由项目负责人之后单独安排。

---

