# zkGAT 项目背景与执行约束

> 工作区：`/home/pzh/project`。本文件仅适用于 zkGAT 主项目，不包括独立的 `comparison/`。
>
> 最高技术规范：`GAT-ZKML-多层多头.md`。
>
> 本文件负责长期项目约束、任务阶段切换、当前机制级实验以及代码与工件治理。`PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md` 只在项目负责人后续明确下达“主实验性能优化”任务时启用，不对当前机制级实验形成提前优化要求。
>
> Codex 开始任务前必须完整阅读本文件与 `GAT-ZKML-多层多头.md`；若当前任务属于主实验性能优化，还必须完整阅读 `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md`。三份文件均由项目负责人管理，Codex 只读，不得擅自修改、删除或重命名。项目负责人明确要求修订约束文件的情况除外。

---

## 1. 项目范围、优先级及阶段边界

### 1.1 技术与执行优先级

技术规范优先级：

1. `GAT-ZKML-多层多头.md`：正式 zkGAT 模型、协议、密码学关系与安全目标。
2. 本文件：任务范围、实验体系、执行权限、工作区与正式工件约束。
3. `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md`：**仅在主实验性能优化阶段**生效的专项限制。
4. 当前 Codex Prompt：在以上边界内明确本轮操作。

阶段专项文件只有在对应阶段激活后才适用。不得利用低优先级文件覆盖高优先级协议或安全规则；也不得用尚未激活的性能优化任务扩大当前机制级实验的工作范围。

### 1.2 当前与后续任务严格分离

**当前阶段：机制级消融重构。**

本轮唯一目标：去除不再需要的端到端消融专属代码，最大化安全复用其中正确的证明实现，构建并真实跑通两组独立机制级对照实验，完成必要回归，汇报结果，然后停止。

本轮**不得**自行启动主实验的通用性能优化；不得在机制级实验完成后自动转向 MSM、NTT、quotient、opening、GPU 调度等独立优化专项。为机制级实验所必需的正确性修复不构成启动性能优化的授权。

**后续阶段：主实验性能优化。**

只有项目负责人另行发布明确的新 Prompt 才能开始；此时再启用 `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md` 的全部专项条款。机制级消融结果汇报不构成自动进入该阶段的授权。

**再后续的统一正式重跑、论文性能结果收集与性能冻结**由项目负责人另行安排，本轮不代替其作决定。

### 1.3 严格隔离 comparison

`comparison/` 及全部子目录不得分析、修改、删除、运行、清理或用来推断主实验的实际实现；不得将其代码、结果、配置或协议组件引入 zkGAT。必要的外部方案介绍不得导致本项目对 `comparison/` 的运行或改造。

### 1.4 当前实现以源码和真实调用链为准

README、普通注释、旧文档和历史讨论仅可辅助定位，不可代替代码证据。Codex 必须核实真实的配置、trace、挑战、承诺、quotient、opening、prover/verifier、序列化与结果导出调用链。协议规范定义“应是什么”，代码检查证明“实际是什么”；不能把二者混淆。

---

## 2. 固定的 zkGAT 正式主实验

### 2.1 模型、任务与公开输出

研究对象是多层多头 GAT **推理**的零知识证明，不新增训练性能或模型精度研究。正式模型结构固定为：

```text
L = 3
hidden_profile = [8x8,8x8]
K_out = 2
```

即两层隐藏 GAT，每层 8 个注意力头、每头宽度 8；隐藏层多头拼接后的维度为 64；输出层具有两个真实注意力头，并执行当前固定的输出后偏置与多头平均规则。前向、量化、图排序、跨图边界、trace、KZG、Fiat–Shamir、quotient、opening、私有见证保护以及 public logits 绑定均遵守最高技术规范。

不得为了建立机制级实验而改变正式 zkGAT 的模型语义、ProofSchema、Fiat–Shamir 顺序、KZG 安全性、证明对象、验证方独立性或安全参数。现有正式安全目标保持 **120-bit**，不得通过缩小挑战域、减少必要约束或使用已知 trapdoor 降低成本。

### 2.2 五个正式数据集与配置

| 数据集 | `d_in_profile` | `num_classes` | `range_bits` | 正式配置 |
|---|---|---:|---:|---|
| Cora | `[1433,64]` | 7 | 12 | `configs/cora_secure.cfg` |
| Citeseer | `[3703,64]` | 6 | 12 | `configs/citeseer_secure.cfg` |
| Pubmed | `[500,64]` | 3 | 12 | `configs/pubmed_secure.cfg` |
| PPI | `[50,64]` | 121 | 14 | `configs/ppi_secure.cfg` |
| OGBN-Arxiv | `[128,64]` | 40 | 21 | `configs/ogbn_arxiv_secure.cfg` |

完整 zkGAT 的正式结果仍为五个数据集 × CPU/GPU 两个后端，共十组完整推理证明，全部必须真实 `VERIFY_OK`。同一数据集的 CPU/GPU 使用相同正式模型、checkpoint、图、公开输入、config、SRS 和证明语义，仅在合法执行后端上不同。正式模型的 checkpoint、训练种子 11、SRS 规则、range 审计及已有安全门禁保持不变。

正式主实验的 canonical checkpoint：

```text
artifacts/checkpoints/cora_gat/
artifacts/checkpoints/citeseer_gat/
artifacts/checkpoints/pubmed_gat/
artifacts/checkpoints/ppi_gat/
artifacts/checkpoints/ogbn_arxiv_gat/
```

正式 SRS：

```text
artifacts/srs/cora_formal_srs/
artifacts/srs/citeseer_formal_srs/
artifacts/srs/pubmed_formal_srs/
artifacts/srs/ppi_formal_srs/
artifacts/srs/ogbn_arxiv_formal_srs/
```

正式主实验结果统一保持在 `results/paper_secure/`，不允许机制级结果覆盖该目录中的主实验工件。

### 2.3 机制级实验不是完整推理结果

机制级实验可以仿照 VerfCNN 的 Pure Convolution Evaluation，直接构造与目标机制相符的不同规模证明工作负载；不必为每组机制配置执行五个数据集的完整 GAT 推理。独立机制工作负载必须具有合法图、节点/边关系、真实有限域见证及可验证的证明声明，并记录规模来源与参数。机制级测试可以采用真实工作负载尺寸和受控的参数化设置，但不能把测试 fixture 或缩规模结果冒充完整 zkGAT 五数据集正式结果。

本轮不重新设计正式 GAT 网络的 architecture scaling；机制级实验所需的节点数、边数、特征宽度等规模变化是**证明机制的规模评估**，不等同于改变五个正式 GAT 模型的层数、头数或输出结构。

### 2.4 正式 checkpoint、训练与 SRS 的长期不变量

本轮不重新训练或重生成正式 SRS，但后续任何涉及正式主实验的任务均不得破坏以下规则：

- checkpoint 必须来自针对对应正式多层多头 GAT 的真实训练；不得用旧模型权重复制、补层、reshape、导入/导出转换或 synthetic weights 冒充正式训练。训练固定 `seed=11`。正式训练 schedule 不得静默采用 smoke/default 训练参数；尚未明确的正式参数应报告项目负责人决定。
- checkpoint / manifest / loader 必须严格匹配对应正式 config 的层数、头数、维度及参数身份；不匹配直接拒绝，不能放宽校验。
- 五个正式配置使用 §2.2 已登记的唯一 canonical 路径，不另建 `*_new.cfg`、`*_v2.cfg`、`*_final.cfg` 等平行正式配置。
- checkpoint 生成后必须执行真实 forward、量化一致性与整数范围审计。`range_bits` 默认保持 §2.2 数值；若不足，必须报告层/头、超限值、所需位数，由项目负责人决定，Codex 不得自行修改范围参数来促使实验通过。
- SRS 必须通过真实生成、manifest、PK/VK、domain 目录、digest 与 proof-shape 绑定检查判定兼容；不能仅依据名义 capacity 推断兼容，也不能使用已知 trapdoor 或非正式 setup 代替安全 SRS。
- 正式主实验对应的 private model、量化和 public logits 绑定不因机制级消融实验而变更；禁止使用测试性模型、子图或缩减节点/边数替代正式完整数据集证明。

### 2.5 正式主实验报告与身份

正式主实验继续沿用现有 canonical 汇总与安全报告，包括：

```text
results/paper_secure/summary.md
results/paper_secure/latest.csv
results/paper_secure/latest.json
results/paper_secure/security_tests_report.md
```

正式结果必须能追溯数据集、backend、config、checkpoint、SRS、proof、verifier 判定与四项正式指标。不同模型架构或不同数据集的历史证明不得与当前固定的多层多头正式实验混成同一主表。任何正式结果必须真实通过 `VERIFY_OK` 与一致性、安全门禁，不能只依据日志中的单阶段成功宣告完成。

---

## 3. 当前只保留的三类论文实验内容

最终实验体系只包括：

1. **机制一：Sparse Node-to-Edge Feature Verification**，比较随机特征折叠与逐坐标 node-to-edge 特征对应证明。
2. **机制二：Shared Neighborhood Verification**，比较共享邻域证明与 denominator / weighted aggregate 的两套独立邻域证明。
3. **完整 zkGAT End-to-End 主实验**，五个数据集的真实多层多头完整推理证明及外部基线比较（外部基线不属于本文件代码修改范围）。

**不保留端到端消融作为正式实验类别。** 不再执行或维护完整 GAT 的 `zkgat`、`wo_feature_folding`、`wo_shared_neighborhood` 三变体消融矩阵。历史端到端消融数据可以保留作为历史证据，但不进入新机制级主表，也不得混入正式 zkGAT 结果。

### 3.1 机制一：Feature Folding

证明声明必须准确对应现有 zkGAT 的**隐藏层源节点至边特征对应关系**，不得笼统描述为“禁用整个网络的所有折叠”。

- 原始方案：以承诺后随机挑战折叠特征坐标，在绑定源表示的前提下证明 node-to-edge 对应关系。
- 对照方案：对相同源节点特征和边映射，逐坐标执行相应的 node-to-edge 对应证明。
- 必须保持相同输入语义、映射条件、隐私要求、来源 commitment 绑定、安全等级及 verifier 判定目标。
- 两种实现均不得跳过源特征绑定、有效边检查、随机挑战次序或完整的对应关系证明。
- 先审计现有 `wo_feature_folding` 实现；它若包含正确的逐坐标路由与证明实现，应直接复用。仅删除不再需要的完整 GAT 消融包装层。

### 3.2 机制二：Shared Neighborhood

证明声明必须对应同一组目标节点邻域上的 denominator 与 weighted aggregate 两项关系。

- 原始方案：使用承诺后采样的随机线性组合，共享 PSQ 与目标邻域证明关系。
- 对照方案：使用两套独立 PSQ 及对应的目标邻域证明关系。
- 两者输入图结构、邻居分组、边序、值、聚合语义、隐私要求、挑战强度和验证目标一致；不得将独立方案偷换成只验证其中一条关系。
- 对多图 workload 保留 `node_ptr` / `edge_ptr` 与每图内部有效分组语义，禁止跨图拼接错误。
- 先审计现有 `wo_shared_neighborhood` 实现；符合数学语义与安全规范的证明、quotient、opening 及 verifier 逻辑应保留复用。

### 3.3 真实机制证明的边界

机制级实验只执行所比较的**目标证明关系及其必要的证明依赖**；不重复无关的 GAT 模型投影、注意力计算、激活、全局中间值和整个网络的统一证明。

必须先沿生产源码确定两侧各自的 commitment、trace、LogUp/route/PSQ、Fiat–Shamir、quotient、opening、proof serialization 和 verifier 最小闭合依赖。可以构建清楚定义独立证明声明的机制级 proof/verifier，但不得改变原正式 zkGAT proof 语义；机制级证明属于明确隔离的实验协议与工件。

只有确实能够生成**独立、完整、可序列化且由真实 verifier 验收**的机制证明，才可以报告其独立 Commitment、Prove、Verify 和 Proof Size 四项结果。

如受共享商多项式、transcript、opening registry 或安全绑定限制，当前设计不能形成数学上成立的独立机制证明，必须分析并解决这些依赖；不得通过单独计时函数、硬编码校验通过、伪造 proof 或随意相减全局 proof size 冒充机制证明成功。确实无法解决的约束必须如实汇报，不得宣布任务已完成。

### 3.4 公平比较和机制规模

每一组对照使用相同目标声明、输入规模、图与边映射、原始见证、120-bit 安全目标、GPU 设备与后端条件。各变体特有的 witness 生成、折叠、route、PSQ、commitment、quotient、opening、serialization 必须计入实际相关计量范围。

可按机制本身的关键变量测试多种工作负载：节点数 N、边数 E、隐藏特征宽度 d_h；Shared Neighborhood 额外记录目标邻域分布和多图边界。具体参数必须有可行的密码学域与 SRS，并能真实完成验证；不为了制造更大的加速比挑选不公平参数。优先从相同工作负载的配对比较出发。

---

## 4. 正式机制级指标与计时语义

两项机制实验优先输出和正式主实验同名的五项结果：

```text
commitment_time_ms
prove_time_ms
verify_time_ms
proof_size_bytes
VERIFY_OK
```

根据当前 `src/main.cpp` 的在线计时与 wall-clock 分区校验，`commitment_time_ms` 与 `prove_time_ms` 是**互不重叠**的 prover 时间分区；`prove_time_ms` **不包含** `commitment_time_ms`。`prove_time_ms` 对应除承诺外的在线 trace、prover 和证明序列化工作，并受既有计时边界及校验约束；`verify_time_ms` 依照正式路径计入 proof 解析与实际验证的指定范围。后续若真实源码变更导致这一事实不再成立，必须先报告计时语义冲突并按正式计时规范处理，不得悄悄修改本规则或实验口径。正式合计口径保持：

```text
formal_time_sum_ms =
    commitment_time_ms + prove_time_ms + verify_time_ms
```

不得将 Commitment 再计入 Prove，也不得凭习惯改变既有官方 timer 边界。机制级实验应采用相同的不重叠语义；它们仅因**真实工作范围不同**而与全网端到端耗时不同。

`proof_size_bytes` 必须来自该机制 proof 的实际规范序列化字节数，既不能使用全网 proof 冒充机制 proof，也不能去除真实 proof 必需的字段。若对应的是局部归因统计而非独立证明，应使用单独准确命名的增量/诊断指标，不得使用上述完整 proof 指标名称伪装。

如有必要，可另外记录 witness/trace、PSQ/route、commitment MSM、quotient、opening、对象/关系数量等解释性指标，但不能混淆嵌套计时，不得让 profiler 明显改变所报告的正式性能。

---

## 5. 旧端到端消融代码的清理与复用

先进行完整的生产调用链与文件依赖审计，然后才删除。

- 删除确认**仅服务于旧完整 GAT 消融矩阵**的 runner、矩阵聚合、重复配置、专属 CLI 包装和无效文档/产物生成逻辑。
- 尽量保留并复用旧变体中对机制级证明真正必要且正确的 trace、PSQ、特征坐标路由、quotient、commitment、opening、challenge、序列化及 verifier 逻辑。
- 对已被正式 zkGAT 或新机制证明共享调用的代码，禁止为了“清理消融”删除、降级或复制出不一致的替代版本。
- 不在主实验的正式 proof schema、transcript 或验证规则中保留不必要的实验分支；但必须根据真实使用关系决定能否移除，不能凭名称删除。
- 已有历史 `runs/`、证明、日志、SRS、checkpoint、canonical cache 和主实验结果保留，不作通配符式大范围删除。
- 新增机制级实验必须具有明确独立入口、参数验证、独立输出目录、日志、可追溯配置、证明工件与结果汇总；不得与旧端到端消融混用。

---

## 6. 当前已知执行错误的处理范围

旧 `wo_shared_neighborhood / OGBN-Arxiv` 端到端运行曾报告：

```text
CUDA quotient coset route was selected but materialization was not admitted:
target_size=134217728 domain=FH route=cuda_coset_to_host
```

这条原生错误应保留为故障事实，不得将后续 `run_manifest.json` 缺失当成根因，也不得推断它必然是物理 OOM。

若审计确认缺陷发生在**正式 zkGAT 或新机制级实验共用**的执行层，且影响本轮机制正确性、构建或可执行性，则按必要最小范围修复并执行针对性回归。不得为了修复已取消的旧端到端消融而重新运行完整 OGBN-Arxiv，也不得借该错误提前启动全局主实验性能优化。

不得关闭 CUDA 安全检查、跳过被证明关系、伪造结果或把 CPU 退化路径称为 GPU 成功。对于资源拒绝，应找到触发条件并验证计算语义等价的合法执行路径。

---

## 7. 测试、权限与执行

### 7.1 测试职责、范围与真实性

所有 test-only 的单元测试、回归测试、负例测试、fixture、smoke/profiling 辅助程序及测试工件统一放入 `tests/`。真正的**机制级正式实验运行入口和证明实现不是 test-only**，应根据生产职责保留在合理的源代码、脚本、配置与结果目录中，不得机械地全部迁入 `tests/`。

Codex 必须自行完成能力和资源允许范围内的构建与**有针对性的完整正确性验证**。对两项机制分别覆盖两侧真实证明与独立验证、必要的挑战/承诺绑定、非法声明或 proof 篡改拒绝、输入索引与多图边界、序列化和计时分区；当实际实现涉及 CPU/CUDA 共享路径时，须覆盖有代表性的 CPU/GPU 结果一致性。测试对象是机制级证明，不是完整多层多头 GAT 的长时间证明。

测试要按证明关系、执行路径和风险**合并组织**。可复用既有测试，不为每个函数另建一套，也不因为代码微调反复运行相同测试。构建成功、函数返回值正确或简单 smoke test **单独都不足以证明**真实密码学证明正确；但也不需要对未受影响的所有主实验组件重复全面回归。

允许 Codex 在权限和资源允许时执行**预计短于十分钟的真实机制级实验**并报告实际结果；若某正式工作负载预计达到十分钟级或明显高资源，应先完成其余可执行实现和短时验证，再向项目负责人提供直接可执行的完整命令。不得仅因完整 GAT 正式 proof 禁止由 Codex 运行，就拒绝执行合理短时的独立机制 proof。

本轮不得运行完整 Cora/Citeseer/Pubmed/PPI/OGBN-Arxiv GAT 正式证明、五数据集全矩阵、正式训练或大型 SRS 生成。长时间任务由项目负责人决定并执行；不得以“完整验证”“正式回归”为由变相进行大规模测试。正式任务应运行至自然结束或真实失败，不擅自添加固定超时；诊断性短时测试应有合理的超时与资源界限。

### 7.2 后台运行与结果要求

机制级实验入口需要支持 GPU 后端、明确工作负载选择、日志与退出码完整记录。需后台运行时应可脱离 SSH 会话持续执行；禁止并发启动互相争抢 GPU 资源的正式单元，避免污染性能数据。

每个正式机制 proof 需要可追溯至原始变体、参数、config、SRS、可执行文件、验证器输入、证明序列化摘要及真实验证结论。输出路径与主实验的 `results/paper_secure/` 严格隔离；完整主实验结果仍按现有正式路径管理。

只有实际证明、序列化、独立验证和结果检查全部成功后，才可报告 `VERIFY_OK`；仅有 `prove` 或控制台打印成功不等于最终验收成功。

### 7.3 安全与执行规则

- 不读取或泄露 witness 给 verifier；不把裸中间私有值写入证明。
- 不绕过真实 KZG、Fiat–Shamir、随机挑战后绑定、安全门禁、独立验证或协议身份检查。
- 机制原方案与对照方案必须有可核验的具体证明声明，不允许模拟、空壳或预设成功。
- 与本轮无关的正式训练、模型精度、SRS 重构及全局性能优化不做。
- 每项变更以源码证据和实际测试为依据，不作无理由的大范围重构。

### 7.4 后续性能优化的验证权限

本文件规定所有开发任务共同遵守的资源、安全及工件上限；**后续主实验性能优化专用的测试节奏、覆盖判据和验收方法，以单独启用的 `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md` 为准**。机制级实验的对照测试清单不会自动变成性能优化时每次必须重新运行的测试矩阵。

未来性能优化必须在完成一个具有真实调用链边界的成组改动后，按受影响风险集中验证；**禁止每修改一两个函数便启动一轮完整测试，也禁止仅凭一次简单测试就宣布整个优化成功**。在保证关键正确性覆盖的前提下，严格控制测试次数、规模、耗时和 GPU 资源；测试失败时优先定位根因并只复测受影响的范围，禁止无意义的全面重复运行。

---

## 8. 工件、Git 与工作区治理

1. 约束文件与最高技术规范均由项目负责人维护；Codex 在普通实现任务中只读。
2. 不得新建保留 `old/new/backup/legacy/v2/final2/日期型` 等历史副本或平行的正式协议实现；同一职责只保留当前 canonical 实现。
3. 不得滥改正式 schema、format、artifact、manifest、identity、hash/transcript 绑定；必要变更必须沿 producer、serializer、consumer、verifier、validator 闭合。
4. 不得未经许可创建 `TASK_REPORT.md`、`CHANGES.md`、迁移说明、临时计划等额外过程文档。完成报告直接在 Codex 对话中给出。
5. 只允许 `git status`、`git diff`、`git log`、`git ls-files` 等只读 Git 检查；不允许 pull、commit、push、merge、rebase、reset、branch/tag 变更或其他改变 Git 状态的操作。
6. 不得污染或覆盖正式 SRS、checkpoint、数据缓存、主实验配置、proof、manifest、历史运行结果或 `results/paper_secure/`。
7. 清理仅针对能够由真实依赖证明已废弃的端到端消融专属部分；不删除有效的主实验性能优化能力、GPU 热点实现和机制级共享逻辑。

---

## 9. 当前机制级重构的完成标准

只有以下条件全部满足，才可报告本轮完成：

1. 已沿源码厘清旧端到端消融与新机制级实验的完整依赖，正确清理失效部分并复用有效实现。
2. 完整 zkGAT 正式主实验的模型结构、证明语义、安全参数、CPU/GPU 路径、结果导出未被破坏。
3. Feature Folding 与 Per-coordinate 两种机制均有同一明确证明声明、真实 proof、独立真实 verifier 和必要的正确性/负例测试。
4. Shared Neighborhood 与 Independent Neighborhood 两种机制均有同一明确证明声明、真实 proof、独立真实 verifier 和必要的正确性/负例测试。
5. 两组机制级实验具有可运行的合法工作负载、独立入口、GPU 支持、准确的四项正式指标与 `VERIFY_OK`、真实序列化 proof、独立日志和结果汇总。
6. Codex 已实际运行权限内能够完成的构建和真实机制 proof/verify，修复可复现的相关缺陷；对未执行的高资源/长时间项明确标注并提供完整执行指令，不能声称已通过。
7. 对历史工件及正式主实验没有污染，没有遗留已知可以解决的编译、验证、计时或运行入口错误。
8. 汇报具体复用/删除/新增的代码、实际运行输出、四项指标、真实验证状态、测试结论，以及未执行且需用户处理的项目（如有）。

**完成以上工作后立即停止，等待项目负责人新的 Prompt。不得自动进入主实验性能优化阶段。**

---

## 10. 后续性能优化阶段的启用规则

项目负责人单独发布“开始主实验性能优化”等明确新任务之后，才激活 `PERFORMANCE_OPTIMIZATION_CONSTRAINTS.md`。届时该文件负责进一步规定 GPU 优先、父级热点、正向收益、禁止用回退或关闭优化冒充修复、短时测试和正式性能冻结。

本文件关于正式 zkGAT 数学语义、实验身份、数据一致性、可信计量、安全保护、工件隔离和执行权限的长期约束持续有效；仅与“本轮机制级消融”直接绑定的操作范围在下一阶段自动结束，不再阻止负责人明确授权的性能优化。
