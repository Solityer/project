# PERFORMANCE_OPTIMIZATION_CONSTRAINTS

## 1. 文档定位

本文档用于约束 zkGAT 主实验在**多层多头正式实现完成之后的性能优化阶段**。

当前正式模型与证明路径已经完成多层多头迁移，性能优化阶段的目标是：

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

以下两份文件属于项目最高权限只读约束文件：

- `GAT-ZKML-多层多头.md`
- `PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md`

性能优化阶段不得：

- 修改；
- 重命名；
- 移动；
- 删除；
- 通过性能优化文档、README、代码注释或当前 prompt 覆盖其约束。

如果当前实现与上述两份文件发生冲突，应修改实现，而不是修改最高权限文档以适配代码。

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

`commitment_time`、`prove_time`、`verify_time` 的正式计时语义、阶段归属和统计边界必须保持不变。

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

## 5. 已有有效优化必须保留

所有已经验证有效的优化点都必须继续保留。

不得：

- 删除已有有效优化；
- 回退已有有效优化；
- 绕开已有有效优化路径；
- 为统一代码路径而取消已经证明有收益的专项优化；
- 因新优化导致旧优化失效而不处理。

包括现有的优化开关、路径和消融能力，例如：

- `fast_msm`
- `parallel_fft`
- `fft_backend_upgrade`
- `fft_kernel_upgrade`
- `trace_layout_upgrade`
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

但如果一个局部阶段变快，而它所对应的正式父级指标没有实际意义上的下降，则不能认为该优化有效。

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

对于对论文性能结果有重要影响的新增执行层优化，如果工程上合理，应尽量：

- 独立控制；
- 可单独关闭；
- 可记录当前启用状态；
- 可用于后续 ablation。

但不能为了“有开关”而增加不必要的复杂度。

已有的优化开关和消融能力不得删除。

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

性能开发测试必须使用：

- 临时目录；
- test fixture；
- 专用测试工件；
- 明确隔离的测试输出路径。

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

## 22. 性能优化阶段禁止长时间完整实验

这是硬性约束。

性能开发过程中不得：

- 运行完整 Cora 正式 proof 作为常规开发测试；
- 运行 Citeseer / Pubmed / PPI / OGBN-Arxiv 完整正式实验；
- 运行 5 CPU + 5 CUDA 十组矩阵；
- 使用“测试”“回归”“benchmark”等名义变相执行完整数据集；
- 使用缩小程度不足、实际接近正式全量 workload 的长时间测试。

性能开发阶段不是生成论文结果的阶段。

---

## 23. 只允许轻量、针对性的测试

每个优化点只允许执行与该修改直接相关的短时验证。

可以使用：

- 单元测试；
- 小规模 synthetic workload；
- 缩规模 graph；
- 缩规模 domain；
- 小规模 MSM；
- 小规模 NTT；
- 阶段级 benchmark；
- component benchmark；
- target-specific regression；
- CPU/GPU 一致性小测试；
- 针对修改阶段的上下游 smoke test。

目标只需要判断：

1. 修改是否正确；
2. 修改是否有效；
3. 修改是否导致相关阶段失败；
4. 修改是否破坏 CPU/GPU 一致性；
5. 修改是否破坏协议和安全约束；
6. 修改是否引入明显新的性能问题。

---

## 24. 所有新增测试代码必须位于 tests/

所有为性能优化新增的测试代码必须统一放在：

```text
tests/
```

不得把临时测试程序散落在：

- `src/`
- `tools/`
- `scripts/`
- 项目根目录
- 其他生产目录

如果测试只为某个优化服务，应明确命名并保持测试范围最小化。

---

## 25. 优化什么，就测试什么

禁止因为修改了一个局部执行路径就运行大范围、长时间、无关测试。

例如：

- 优化 MSM → 测 MSM route / batching / correctness；
- 优化 quotient materialization → 测 quotient 对应阶段；
- 优化 opening → 测 opening；
- 优化 multi-graph batching → 测小型 multi-graph workload；
- 优化 large-graph streaming → 测缩规模 large-workload；
- 修改共享逻辑 → 增加必要的 CPU/GPU 小型一致性回归。

测试范围必须与改动范围匹配。

---

## 26. 优化开发闭合判定

某个优化点不需要通过完整正式数据集才能判断开发闭合。

如果同时满足：

- 修改项正确性测试通过；
- 对应轻量 benchmark 证明性能有效；
- 相关上下游 targeted regression 通过；
- CPU/GPU 一致性未破坏；
- 协议语义未改变；
- proof schema 未改变；
- 安全边界未改变；
- 已有有效优化未回退；
- 没有引入新的明显同步问题；
- 没有引入新的明显数据搬运问题；
- 没有引入不可接受的内存风险；
- 没有引起其他相关阶段失败；
- 没有改变正式计时语义或统计边界；
- 没有污染正式 checkpoint、SRS、cache、proof 或结果目录；
- profiling / debug 逻辑没有成为新的显著性能开销；

则可以判断：

> **当前优化点开发闭合。**

随后应停止继续打磨该点，转向下一个更大的热点。

---

## 27. 轻量测试结果不得作为论文正式实验结果

性能优化阶段所有轻量测试、microbenchmark、stage benchmark、synthetic workload 的结果：

- 仅用于判断优化是否正确和有效；
- 不得作为最终论文实验时间；
- 不得作为正式 5 CPU + 5 CUDA 结果；
- 不得混入 `results/paper_secure` 的最终正式结果。

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

## 30. 当前性能优化阶段的推荐工作顺序

每轮优化应遵循：

1. 审计当前正式 GPU 调用链；
2. 根据真实 profile 找父级大热点；
3. 估计理论端到端收益；
4. 排除低收益小热点；
5. 设计执行层优化；
6. 实现修改；
7. 在 `tests/` 中增加最小必要测试；
8. 运行短时 correctness / regression；
9. 运行轻量性能 benchmark；
10. 判断是否正确、有效、无明显副作用；
11. 如果满足闭合条件，则停止继续深挖；
12. 转向下一个更大的父级或结构性热点。

---

## 31. 性能优化阶段停止条件

当满足以下条件时，应停止继续优化：

- 当前主要父级大热点已经处理；
- 剩余热点对端到端总时间影响不再显著；
- 继续优化只能获得局部或微小收益；
- 进一步优化需要修改协议语义；
- 进一步优化会显著增加复杂度但收益很小；
- multi-graph 和 large-graph 路径没有剩余明显结构性问题；
- 已有优化没有明显回退；
- targeted tests 全部通过。

此时认为：

> **性能开发阶段闭合。**

---

## 32. 最终正式实验只能在性能冻结后执行

只有性能开发阶段全部闭合后，才进入最终实验阶段。

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

性能优化阶段必须始终遵守以下原则：

> **协议不动。**

> **证明语义不动。**

> **安全边界不动。**

> **多层多头正式模型语义不动。**

> **已有有效优化不退。**

> **只优化父级大热点、结构性热点和端到端高占比热点。**

> **优先从执行层和 GPU 正式路径获得性能收益。**

> **允许 multi-graph / large-graph workload-aware 专项优化，但禁止 dataset-specific 优化。**

> **禁止沉迷局部小热点。**

> **禁止性能开发阶段运行长时间完整数据集或变相全量实验。**

> **正式 commitment / prove / verify 的计时语义和统计边界冻结，不允许通过挪计时、隐藏预计算或跨独立运行复用协议状态获得虚假性能收益。**

> **`comparison/` 完全隔离，不属于本阶段分析、修改、运行和推断范围。**

> **两份最高权限约束文档只读，不得因性能优化而修改。**

> **轻量测试必须与正式 checkpoint、SRS、cache、proof 和结果目录隔离。**

> **优化什么就轻量测试什么，所有新增测试代码统一放 `tests/`。**

> **正确、有效、无明显副作用，即认为当前优化点开发闭合。**

> **所有高收益优化闭合并冻结后，再一次性执行最终 5 CPU + 5 CUDA 十组正式实验。**
