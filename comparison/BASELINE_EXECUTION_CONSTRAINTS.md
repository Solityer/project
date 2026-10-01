# BASELINE_EXECUTION_CONSTRAINTS

## 1. 文件目的

本文件定义 `/home/pzh/project/comparison/` 下三个对比实验的统一执行约束、多代理执行方式、工程边界和完成标准，适用于 `comparison/zkGPT/`、`comparison/zkdl/`、`comparison/ezkl/`。三个目录分别代表三套独立的对比实验实现。本文件规定如何组织代理、如何读取与修改代码、如何构建和验证、允许做什么、不允许做什么以及何时视为实现完成；各 baseline 的具体证明关系、密码学组件和 GAT 证明语义，以当前 baseline 自己的技术说明文件、官方原始实现和当前正式代码为准。

## 2. 总体目标

三个 baseline 的目标不是重新设计新的证明系统，也不是将其优化到与主实验相同的工程水平，而是在忠实保留各 baseline 官方原始实现、原生方法、证明体系和执行风格的前提下，仅进行完成统一 GAT 推理任务所必需的最小适配，使每个 baseline 都能够在自己的目录中独立构建、独立运行、独立证明、独立验证并独立产出正式实验结果。

对比实验必须优先保证 baseline 方法身份真实、GAT 任务语义一致、证明关系完整、实验实现彼此独立、正式运行目录级自包含、主实验专项证明机制和专项性能优化不迁移、正式代码干净简洁、正式结果可验证可复现可审计。主实验可以在开发阶段作为只读参考，并允许在本文件规定的边界内将通用实现和基础设施本地化移植到 baseline；但 baseline 的正式构建和正式运行不得依赖主实验目录。

# 3. 最高优先级硬约束

以下规则属于硬约束，任何违反均应视为 baseline 尚未完成。

## 3.1 三个 baseline 必须完全独立

`zkGPT`、`zkdl`、`ezkl` 必须是三套彼此独立的实验实现。禁止一个 baseline 调用或链接另一个 baseline 的源码、库、脚本或二进制，禁止读取另一个 baseline 的中间结果、proof、registry、setup 参数、运行缓存或其他正式产物，也禁止将三者的核心证明实现抽取到公共目录后共同调用。开发过程中可以人工比较不同 baseline 的代码，但正式构建和正式运行不得形成任何跨 baseline 依赖。

## 3.2 每个 baseline 必须目录级自包含

每个 baseline 必须能够在其自身目录中独立完成正式实验。正式运行所需的项目级资源必须位于该 baseline 自己的目录体系内，包括源代码、构建文件、正式运行脚本、配置、数据或该 baseline 自己管理的数据资源、模型/checkpoint、图预处理资源、静态表、协议参数、setup/PP/SRS/PK/VK、model registry、正式 proof、verifier 输入、正式结果、正式日志、结果收集和校验工具等。正式实验运行时不得读取该 baseline 根目录之外的项目级资源，不得依赖仓库根目录主实验资源或其他 baseline 的资源。

“自包含”不要求复制操作系统、CUDA Driver、编译器、Python/Rust/C++ 工具链、系统库或正常第三方软件依赖。本文件所禁止的外部依赖特指 baseline 目录之外的项目级资源和项目级实现依赖。正常使用 baseline 原生所需的 crates、Python packages、Halo2/ezkl 原生依赖、CUDA Runtime、ICICLE 或其他第三方依赖是允许的；如果正式运行需要这些依赖的项目内副本，应按本文件第 20.20 节处理，使其成为 baseline-local 依赖，而不是继续指向主实验目录。

## 3.3 主实验参考、本地化复用与正式依赖边界

本文件所称“主实验”是 `/home/pzh/project` 中 `comparison/` 之外的主实验实现；`comparison/` 下三个 baseline 不属于这里所称的主实验。主实验在开发阶段保持只读，不得被 baseline 正式运行直接依赖。开发阶段允许参考、复制、移植并适配完成 GAT baseline 接入所需的通用实现和基础设施，例如数据与 checkpoint 格式转换、统一 GAT 数值语义、图结构和边界处理、通用资源加载、通用构建接入、CUDA/ICICLE 等第三方依赖的正常接入方式、一般性的 CPU/GPU 调度和设备接入基础设施、通用 field/MSM/reduction 依赖接入、memory/layout handling、serialization、transcript plumbing、runner/collector/validator plumbing 和 GPU execution plumbing，以及与主实验专项优化无关的其他通用工程实现；不要求对已经存在且符合本文件边界的通用能力从零重新实现。上述 field/MSM/reduction、transcript 等内容只允许复用通用工程接入和 baseline 自身所需的正常基础能力，不得替换 baseline 原生 proof primitive，也不得迁移主实验专用的优化实现。

任何从主实验参考、复制或移植的允许内容都必须在当前 baseline 内形成 baseline-local 实现，正式 build/run 不得 include/import/link/call 主实验源码、模块、库、二进制或脚本，不得读取主实验运行后生成的中间文件、结果、缓存或 proof 资源，也不得通过软链接、环境变量、相对路径或绝对路径形成 hidden dependency。允许的是“开发阶段参考并本地化复用通用实现”，禁止的是“正式运行依赖主实验”以及“迁移主实验受禁止的核心协议、专项性能优化或非必要专项安全工程”。复用和适配还必须继续服从第 20 节的官方源码、原生语言、原生构建体系和 `gat_adapter/` 边界。

## 3.4 只做最小必要适配

baseline 的修改范围应严格限制为使其能够正确表达并证明统一 GAT 推理任务所必需的适配，包括 GAT 图数据表示、稀疏边访问、多层多头 attention、hidden head concat、output multi-head、stable Softmax、LeakyReLU、ELU、ExpMap、SignedRoundDiv、图 batch 边界、PPI 多图处理、定点量化链、最终 logits 绑定、必要的 public/private 输入映射，以及将这些关系映射到 baseline 原生证明原语。不得借“适配”为名进行大规模协议重构、证明体系重写或与完成 GAT 无直接关系的工程改造。

## 3.5 必须保留 baseline 原生方法身份

三个 baseline 必须继续使用各自官方原始实现和原生证明体系：zkGPT 保持官方 zkGPT 核心证明组件和组织方式，zkDL 保持官方 zkDL CUDA/C++ 证明体系，ezkl 保持官方 Rust/Cargo、ezkl/Halo2 原生证明和执行路径。禁止将主实验独有的证明协议、承诺体系、lookup 方案、trace 体系或其他专项设计迁移后再冠以 baseline 名称。如果原 baseline 不原生支持某个 GAT 运算，应优先使用该 baseline 已有证明原语进行最小扩展。

## 3.6 禁止迁移主实验专项性能优化

baseline 不得主动吸收主实验为了提升 commitment/prove/verify 性能而开发的专项优化，包括但不限于主实验专用 trace layout、fast MSM、FFT backend/kernel、CUDA kernel、kernel fusion、异步流水、专用 memory layout、commitment pipeline、polynomial batching、专项缓存策略、针对五个数据集手工设计的 shortcut，以及其他仅用于主实验性能提升的专项工程优化。如果某项能力本身就是 baseline 原项目的原生机制，则应正常保留和启用；不得为了让 baseline 更慢而主动删除其原生优化，也不得为了让 baseline 更快而迁移主实验专项优化。

接入或启用 CUDA、ICICLE 等通用第三方执行/密码学加速 backend 本身不视为迁移主实验专项性能优化；允许参考主实验中这类第三方依赖的版本、构建、链接、设备初始化和通用接入方式，并按第 3.3 和 20.20 节本地化到当前 baseline。禁止迁移的是主实验围绕这些依赖自行开发的专用 kernel、MSM/FFT wrapper 或 backend 改造、fusion、cache、pipeline、memory layout 以及其他为主实验性能服务的专项实现。

## 3.7 禁止通过“复制、改名、封装”迁移受禁止的主实验实现

是否属于违规迁移，以实现来源和实质逻辑判断，而不是以文件位置、文件名、namespace、module、wrapper 或接口形式判断。禁止将主实验核心证明协议、主实验专项性能优化或非必要专项安全工程通过复制、改名、namespace/module 重命名、wrapper、拆分、重排、轻度改写或重新封装后移入 baseline。该限制不适用于第 3.3 节允许的通用 GAT 接入逻辑、通用基础设施和正常第三方依赖接入；这些允许内容可以在满足 baseline-local、自包含、最小适配和原生证明体系不变的前提下参考、复制、移植和适配。

## 3.8 禁止非必要的主实验安全工程迁移

除非属于 baseline 原生机制，或不加入就无法满足 baseline 自身声明的正确性、隐私边界和可信验证要求，否则不得主动引入主实验特有或明显超出 baseline 原方法的安全工程设计，例如主实验特有的多层 manifest、artifact digest 绑定、commitment lifecycle、注册协议、masking protocol、transcript wrapper、proof schema wrapper、telemetry/security evidence protocol 等。允许保留 baseline 原论文、原实现或原生证明体系已有的安全机制，并补齐完成 GAT 适配后维持其原有隐私/正确性声明和独立验证所不可缺少的最小安全检查。

## 3.9 自动化开发代理禁止执行任何完整数据集正式实验

自动化开发代理、Codex 或其他代码执行代理不得运行 Cora、CiteSeer、PubMed、PPI、OGBN-Arxiv 任一完整正式数据集的 preprocessing、material preparation、witness generation、commitment、完整 setup、prove、verify、结果收集或其他实际完整计算，也不得通过拆分、分批、分阶段或多个片段累计覆盖完整数据集。完整数据集只允许做静态检查，例如配置、路径、代码分支、文件命名、runner 参数和资源引用关系检查。

开发阶段允许 build、静态检查、最小 unit/integration test、negative/tamper test、小规模 synthetic test、小规模真实数据片段测试、必要的 CPU/GPU 小规模一致性验证和局部正式关系回归，但必须同时遵守第 20.18 节的短时、定向、最小范围验证规则。五个完整数据集的正式长时间实验只允许由用户本人在实现完成后手动运行。

## 3.10 当前阶段不进行系统级性能重构

baseline 的目标是形成可信、公平、完整并能真正运行的正式对比实现，而不是开展新的性能优化研究。只在存在明显错误、不可接受的实现缺陷、运行阻塞或违背 baseline 原生方法时进行必要修改；一旦统一 GAT relation、正式 proof、独立 verifier、正式执行路径、五数据集 runner、结果收集与 validator、baseline 独立性和正式代码洁净度达到要求，就停止无必要的工程重构。

## 3.11 可修复 blocker 不得作为任务结束结论

如果发现缺失接口、adapter、backend glue、资源接入或其他完成正式实验所需功能，只要能够基于 baseline 官方源码、官方 proof primitive、允许的通用基础设施或正常第三方依赖完成最小必要适配，就必须完成修复，不得仅以 `BLOCKED`、`UNSUPPORTED`、`NOT_IMPLEMENTED` 作为任务完成状态。该要求不能被用于重写 baseline 核心证明系统、迁移主实验核心协议或迁移主实验专项性能优化；只有确实超出 baseline 原生证明能力且无法通过允许的最小增量适配解决时，才允许认定为真正不可实现。

## 3.12 当前 baseline 技术笔记读取边界

在规范性说明文件层面，修改某一个 baseline 时只读取本文件和该 baseline 自己的技术笔记：zkDL 只读取 `comparison/zkdl/GAT-ZKML-zkDL.md`，zkGPT 只读取 `comparison/zkGPT/GAT-ZKML-zkGPT.md`，ezkl 只读取 `comparison/ezkl/GAT-ZKML-ezkl.md`；不得读取或使用另外两个 baseline 的技术笔记指导当前实现。该限制不禁止检查当前 baseline 自己的真实源码、构建逻辑、runner、prover、verifier、resource loader 和调用链，也不禁止按照第 3.3 节只读检查主实验中与当前适配直接相关的通用源码和基础设施作为实现参考。README、普通代码注释和历史说明可能未同步，判断当前实现事实时仍以当前真实代码和调用链为准。

## 3.13 约束文件和技术笔记不得被自动修改

除非用户明确要求修改规范，否则自动化开发代理不得修改 `comparison/BASELINE_EXECUTION_CONSTRAINTS.md` 或任何 baseline 的 `GAT-ZKML-*.md`。如果当前实现与规范冲突，应修改实现；不得通过修改规范迁就错误或不完整的代码。

## 3.14 强制代码排版约束

所有 baseline 的新增和修改代码必须采用紧凑排版，并将本节视为硬约束而不是风格建议。只要一个完整语句、函数调用、函数声明、条件表达式、返回语句、变量初始化、容器初始化、简单链式调用或参数列表在所用语言语法上能够写成一行，就必须保持一行；不得因为固定行宽、formatter 默认规则、视觉对齐、参数数量或个人风格把本可单行表达的代码拆成多行。只有语言语法本身要求分行，或代码本身包含必须以块结构表达的多分支、多层控制流、独立语句块等情况时才允许多行；“这一行较长”“拆开更好看”“formatter 建议换行”均不是允许拆行的理由。

同一函数、同一控制流或同一连续逻辑块内部的语句默认紧邻书写，不允许在变量声明、连续赋值、连续调用、连续判断或同一功能步骤之间插入空行。只有函数/方法/类型定义之间、明确的功能分层之间或同一函数内确有独立阶段边界时才允许空一行，并且最多一个空行，禁止连续多个空白行。不得自动格式化整个文件，不得因本次修改重排无关代码，只允许处理本次实际修改区域；如果 formatter 会违反本节规则，不得运行该 formatter 或接受其冲突结果。

## 3.15 探索纪律与停止规则

仓库读取和源码探索必须围绕当前具体实现问题展开，不得在实现前先做无边界的全仓库审计。只要已有证据足以确定当前 blocker、受影响正式链和必要修复，就停止继续扩大探索范围并进入实现；不得为了“更完整”而冻结一份穷尽式 blocker 清单、检查所有理论相关文件、重复审计同一结论、对同一路径进行无具体冲突依据的二次 cross-review，或反复启动新的审查轮次。这里的“一次性修复”是指在当前受影响正式链内同时检查并处理能够提前确认的同类问题和直接后续问题，不等于对整个仓库开展无关全面审计。

如果已经有足够上下文的代理或已有结论能够继续当前工作，应复用已有上下文，不得为了额外信心重复进行相同探索、相同审计或相同验证。并行工作只应用于真正相互独立、职责清晰且不会重复读取或修改同一问题域的任务；并发能力本身不是增加代理数量、审计轮次或验证轮次的理由。

## 3.16 长会话与 baseline 切换刷新规则

长会话中不得只依赖会话开头的记忆。在开始处理一个 baseline 或从一个 baseline 切换到另一个 baseline 时，`root` 必须重新读取本文件和当前 baseline 自己的技术笔记，并确保后续使用的 subagent 获得与当前任务直接相关的最新约束，然后只按需要检查当前实际代码状态并继续工作。刷新约束的目的只是恢复硬规则和当前实现事实，不得借此重新从零分析 baseline，也不得重新阅读与当前 baseline 无关的大量仓库内容或另外两个 baseline 的技术笔记。

## 3.17 范围控制

只做让当前 baseline 的唯一正式执行路径达到用户可运行状态所必需的工作。禁止重做已经正确工作的组件、进行无关优化或无关重构、清理整个仓库、编写新的设计/规范文档、主动更新 README、在正式路径已经代码完成后继续完善未使用的纯 CPU/纯 GPU 替代路径、在正确性实现已经足够后继续开展性能优化，或在没有新证据的情况下重新打开已经确定的设计决策。当前 baseline 达到第 18.1 节实现完成标准后，应停止继续“顺手改进”，等待用户正式运行。

# 4. 统一 GAT 实验语义

## 4.1 三个 baseline 必须证明同一个任务

三个 baseline 更换的是证明系统，不是机器学习任务。正式实验必须保持相同的数据集、图拓扑、节点编号、self-loop 规则、edge ordering、graph batching、PPI 多图边界、GAT 层数、hidden heads、output heads、hidden concat、bias 位置、activation、stable Softmax、定点量化、rounding、静态函数表语义、output head averaging 和最终 public logits。

## 4.2 正式数据集固定

正式比较固定使用 Cora（`cora`）、CiteSeer（`citeseer`）、PubMed（`pubmed`）、PPI（`ppi`）、OGBN-Arxiv（`ogbn-arxiv`）五个完整数据集。禁止使用 toy graph、sampled graph、subgraph、reduced graph、部分节点、部分边或小规模 synthetic input 替代正式结果。

## 4.3 CPU、GPU 与混合执行必须使用相同证明关系

CPU、GPU 或 CPU+GPU 混合执行只允许改变执行设备和 baseline 原生 backend 的实际调度，不得改变 public statement、GAT 数值语义、proof scope、privacy boundary、output policy、模型语义、图语义或量化语义。不得为了让某个 GPU 阶段跑通而削弱证明关系，也不得为不同设备生成数学语义不同的 GAT relation。

# 5. 完整证明要求

## 5.1 必须证明完整 GAT inference

正式 proof 必须覆盖完整 GAT forward relation。禁止只证明矩阵乘法、projection、attention 的一部分、单层、部分节点或部分边，禁止只证明最终输出 hash，禁止由 host 端计算关键中间值但 proof 不约束，禁止将难实现的非线性直接作为可信 public input，禁止用 sampled relation 代替 full relation，禁止让 verifier 读取 witness 后自行重算来代替证明。

## 5.2 最终 logits 必须由 proof 绑定

最终公开输出 `Y` 必须是被证明关系的一部分。proof 必须证明给定正式 public statement 和隐藏 witness，完整 GAT 推理关系成立并得到指定公开输出 `Y`。

## 5.3 verifier 不得信任 proof 自带的输出声明

如果 proof artifact 内包含输出或 instance 信息，verifier 不得将其直接作为可信 public statement 来源。verifier 必须独立获得 public statement 和公开输出 `Y`，再使用 proof 验证该 statement 与 `Y`；proof 内携带的对应信息只能被校验，不能成为唯一可信来源。

# 6. 隐私与验证边界

## 6.1 隐私目标

正式 baseline 应维持统一约定的隐私边界：public graph、public features、public metadata 和 public final logits 可公开；model、execution trace 和 private witness 保持隐藏，除非某 baseline 原论文和统一实验语义明确规定了更具体但不冲突的公开边界。

## 6.2 verifier 禁止读取 witness

正式 verifier 只能读取验证所必需的公开信息，例如 public statement、public output、proof、verification key、public parameters、public model commitment/registry 和必要公开 metadata。verifier 不得读取 private witness、private model weights、private trace、prover 内存、private opening material 或其他仅 prover 应持有的临时秘密。

## 6.3 proof artifact 不得包含原始 witness

正式 proof 不得通过直接打包原始模型参数、原始 private trace、完整 witness 或 verifier 本不应获得的 private auxiliary data 来实现“可验证”。

## 6.4 重复推理场景不得明显破坏既定隐私

如果 baseline 使用长期 model commitment、model registry 或其他可复用对象，则正式实现应遵守该 baseline 自己的原生隐私机制，不得因为 GAT 适配而主动削弱 baseline 原有隐私边界。

# 7. 正式执行设备与结果矩阵

## 7.1 每个 baseline 只要求一套正式执行路径

不再要求为每个 baseline 同时完成纯 CPU 和纯 GPU 两套路径，也不再将 backend 作为正式实验矩阵维度。每个 baseline 只需要一套能够完成正式实验的执行路径，该路径可以是全 CPU、全 GPU 或 CPU+GPU 混合执行；设备分配应遵循 baseline 原生能力和最小必要适配原则，哪个阶段适合 GPU 就使用 GPU，不适合或没有必要使用 GPU 的阶段继续使用 CPU，不得为了形式上的“纯 GPU”或“纯 CPU”进行无意义重写。

最终正式结果矩阵为 `5 datasets × 1 formal execution path = 5 results`，即 Cora、CiteSeer、PubMed、PPI、OGBN-Arxiv 各一个正式结果。正式结果必须如实记录执行方式；如果是混合执行，应能够审计关键正式阶段实际使用 CPU 或 GPU，但不需要额外生成纯 CPU 和纯 GPU 两套重复结果。

## 7.2 GPU 阶段必须是真实 GPU 执行

如果正式执行计划将某个核心证明阶段标记为 GPU，则该阶段必须真实进入 baseline 合法的 GPU backend，不能只检测到 CUDA 设备、只执行与核心 proof 无关的 GPU 操作、仅在预处理使用 GPU 却把该核心阶段标成 GPU，也不能使用 CPU 计算后伪装成 GPU。CPU+GPU 混合执行是预先明确的正式执行计划，不属于 fallback。

## 7.3 禁止 silent fallback

任何被明确配置为 GPU 的正式阶段失败时必须 fail closed：GPU 不可用、backend 未构建、必要 kernel 未执行或 GPU proof 路径异常时，应直接失败，不得静默退回 CPU 后仍记录为 GPU。预先设计并明确记录的混合执行允许某些阶段使用 CPU、另一些阶段使用 GPU，但阶段设备选择必须真实、明确、可审计。

# 8. 正式代码洁净度

## 8.1 正式代码只能包含生产实验逻辑

正式源码和正式执行脚本中只允许保留完成正式实验所必需的生产逻辑。本项目为 GAT adaptation 新增的 unit test、integration test、toy/sample/mock/stub、自测、KAT、tamper/negative/regression test、microbenchmark、profiler harness、临时 correctness check、测试数据生成器和 test-only helper 不得混入正式 production 路径，应进入 baseline-local `tests/`。必要的正式输入校验、协议一致性检查、安全边界检查、backend 真实性记录和 fail-closed 检查属于生产逻辑，可以保留。

## 8.2 禁止通过隐藏开关把测试逻辑留在正式程序中

正式 runner、CLI 和源码不得保留仅服务于开发测试的 `--toy`、`--mock`、`--self-test`、`--test-mode`、`--debug-proof`、`--sample`、`--microbench` 等特殊执行分支。正式生产入口应保持单一、清晰、可追踪。

## 8.3 正式代码不得残留后续自行产生的临时实现

正式生产路径不得保留本项目后续自行产生且已确认无用途的 `_debug`、`_test`、`_legacy`、`_old`、`_tmp`、experimental implementation、废弃 fallback、无调用 dead code、大段注释旧实现、临时 instrumentation、临时日志洪泛、profiler-only counter 或开发阶段临时 CLI 参数。该清理规则不适用于第 20 节保护的官方原始源码、官方原始工程内容和当前有效 GAT adaptation；官方内容不得因“当前未引用”而删除。

## 8.4 本项目新增测试必须物理隔离

本项目新增的测试代码和测试资源必须放入对应 baseline 根目录下的 `tests/`，可包含 unit/integration/correctness/tamper/negative/CPU-GPU consistency/KAT/regression/microbenchmark/profiler/test fixture/synthetic input/test-only helper。官方仓库原本自带的 tests/examples/benchmarks 按第 20.19 节原样保护，不强制迁移。

## 8.5 正式执行链不得依赖 tests

`tests/` 可以调用正式代码进行测试，但正式代码禁止依赖 `tests/`。删除本项目新增的 `tests/` 后，正式 baseline 仍应能够构建，并具备由用户独立运行五个完整正式数据集的能力。测试代码不得生成正式 statement、witness、proof、setup 参数、model registry、public output 或正式结果。

## 8.6 正式结果 validator 不属于测试代码

用于判断正式实验结果是否合格的 final validator 属于正式实验执行链，可以保留在正式 `scripts/` 或对应生产目录中；它的职责是审计正式产物，而不是执行开发测试。

# 9. 推荐目录边界

每个 baseline 应尽量形成清晰的生产/测试边界，可按其原生工程结构使用 `src/`、`scripts/`、`configs/`、`data/`、`models/`、`resources/`、`artifacts/`、`runs/`、`results/`、`tests/` 和 `gat_adapter/` 等目录，不要求三个 baseline 的目录完全一致。核心要求只有两个：正式代码与本项目新增测试代码清楚分离；baseline 正式运行资源不得依赖其目录之外的项目内容。不得为了目录统一破坏官方原始项目结构。

# 10. 构建与正式运行入口

## 10.1 正式入口必须唯一且清晰

每个 baseline 应明确提供正式 build 入口、单 dataset 正式运行入口、五数据集完整正式入口、正式结果 collect 入口和正式结果 validate 入口。正式 runner 可以内部完成 CPU、GPU 或混合设备调度，不要求暴露纯 CPU/纯 GPU 两套完整 backend 入口。不得同时保留多个无法区分“哪个才是正式路径”的后续历史 runner；官方原项目自带脚本按第 20 节保护，但不得与本项目正式 GAT runner 形成歧义。

## 10.2 历史执行代码不得污染正式路径

后续自行新增且已经确认废弃、错误、临时或与当前正式实验无关的代码、脚本、配置和资源应删除，只有仍具有明确测试价值的内容才移动到 `tests/`。该规则不适用于官方原始工程和有效 GAT adaptation；官方内容的保留边界以第 20 节为准。

## 10.3 正式运行不得要求先运行主实验或其他 baseline

每个 baseline 的正式入口必须从该 baseline 自己的输入和资源开始执行，不允许“先运行主实验或其他 baseline 生成某文件，再由当前 baseline 证明”的链式依赖。

# 11. Setup、参数与资源管理

## 11.1 每个 baseline 自己管理自己的证明材料

PP、SRS、proving key、verification key、compiled circuit、model registry、model commitment、ceremony descriptor、static tables 等正式证明材料必须由当前 baseline 自己管理，不得在正式运行时由主实验或其他 baseline 提供。允许按第 3.3 和 20.20 节在开发阶段将合法的通用资源或第三方依赖本地化到当前 baseline。

## 11.2 setup 与正式 inference 必须区分

如果某一步属于一次性 setup，应单独记录，不得无解释地混入每次 inference 的 prove 时间。根据 baseline 原生流程，可分别记录 setup、compile、model registration、commitment、witness generation、prove、verify 等 raw timing；最终论文使用的统一指标按第 12 节和第 20.24 节映射。

# 12. 结果计量要求

## 12.1 正式结果必须能够审计统一四项核心指标

每个正式实验至少应能够审计 dataset、formal execution path、proof generated、verification status、statement identity、proof identity/digest、关键 protocol/backend identity 和正式运行状态。论文最终统一核心指标固定为 `Commitment Time`、`Prove Time`、`Verify Time`、`Proof Size`；同时保留 baseline-native raw timing，保证映射关系明确、稳定、可审计。若 baseline 没有与其他方案完全同名的一一对应原生阶段，不得人为制造不存在的工作，而应保留真实 raw breakdown，并按第 20.24 节定义统一映射。

## 12.2 proof size 只统计 verifier-facing proof

`Proof Size`/`proof_size_bytes` 必须表示 verifier 实际需要接收的 proof 数据大小，不得将 raw witness、dataset、checkpoint、private trace、debug log、profiler output 或 test artifact 计入，也不得为了缩小数字而排除 verifier 实际必须接收的 proof message。

## 12.3 计时口径必须稳定

同一 baseline 的五个正式数据集必须使用一致计时边界，三个 baseline 的论文级计时语义必须按第 20.24 节保持可比。不得针对某个数据集单独改变 timing boundary 获得更好结果，也不得让同一计算阶段重复计入多个论文指标。

# 13. 正式验证与 fail-closed

## 13.1 每个正式结果必须真实验证

单个正式结果只有在 proof 真实生成、independent verifier 真实运行并返回成功、public statement 正确、public output 正确绑定、proof artifact 完整、执行设备记录与真实执行一致、full GAT relation 被证明且 verifier 未读取 witness 时才能记为成功。单个结果的成功不以前五数据集 final validator 已通过为前置条件。

## 13.2 禁止伪造成功状态

禁止手工写 `VERIFY_OK`，禁止 proof 缺失或 verifier 未执行仍记为成功，禁止 verifier 失败后继续输出正式成功结果，禁止 GPU 阶段回退 CPU 后仍记录为 GPU，禁止输入缺失时自动使用 toy data，禁止证明范围不完整时仍记为 full GAT。

## 13.3 正式 validator 必须独立审计五个正式结果

每个 baseline 必须有正式结果 validator。用户本人完成五个完整数据集正式实验后，final validator 应能够检查五个正式数据集结果唯一且齐全、proof 存在、proof digest/size 可核验、verifier 成功、statement identity 一致、full-dataset identity、full-GAT proof scope、formal execution path 与阶段设备身份、public output binding、verifier 不读取 witness、无 silent fallback 和正式结果文件完整。自动化开发阶段可以使用 `tests/` 中专用 fixture 验证 collector/validator 逻辑，但 fixture 不得写入 production `results/` 冒充正式结果，也不得伪造正式 proof、verification success 或 `VERIFY_OK`。

# 14. 正式产物边界

每个 baseline 的正式产物只能写入该 baseline 自己的目录体系，例如 `artifacts/`、`runs/`、`results/`、`logs/`。禁止将正式实验产物写入主实验目录、仓库根目录的主实验结果目录、其他 baseline 目录或依赖外部项目目录才能定位的临时路径；正式脚本不得硬编码开发机器上的绝对项目路径。

# 15. 允许与禁止的修改

## 15.1 允许

允许进行 GAT 必要 relation 适配、baseline 原生 proof primitive 的必要组合、数据/模型格式适配、CPU/GPU/混合正式执行路径修复、correctness 修复、verifier trust boundary 修复、privacy boundary 修复、独立运行所需资源整理、正式 runner 简化、后续自行产生的过时代码清理、正式代码与本项目新增 tests 物理分离、baseline 原生已有优化的正常启用、阻塞完整实验的必要工程修复，以及第 3.3 节允许的主实验通用实现和基础设施的 baseline-local 移植。

## 15.2 禁止

禁止主实验核心协议移植、主实验专项性能优化移植、非必要的主实验专项安全工程移植，禁止通过复制、改名、namespace/module 重命名、wrapper 或重新封装变相迁移上述受禁止实现，禁止把 baseline 改造成主实验的另一种 backend，禁止为追求性能进行大规模非必要重构，禁止降低 proof scope、降低数据规模、改变统一 GAT 数值语义、偷换公开/隐私边界、让 verifier 读取 witness、让 proof 携带原始 private witness、GPU silent fallback、将本项目新增测试代码混入正式实现、baseline 之间共享核心证明实现、正式运行依赖 baseline 目录外的项目资源，以及自动化开发阶段运行或变相运行任何完整数据集正式实验。

# 16. 开发流程约束

后续修改一个 baseline 时，一次只处理一个 baseline，其余两个 baseline 和主实验均视为只读；规范性说明文件只读取本文件和当前 baseline 自己的技术笔记，再检查当前 baseline 的真实源码、build、runner、prover、verifier、resource loader 和调用链，并可按第 3.3 节只读参考主实验中与当前适配直接相关的通用实现。只识别完成统一 GAT 正式实验所需缺口，进行最小必要修改，本项目新增测试全部进入 baseline-local `tests/`，正式代码中不得残留开发测试实现，正式实验不得依赖 `tests/`。

开发代理必须尽可能在用户下一次长时间正式运行之前，通过静态分析、真实配置、已有资源和短时定向检查，一次性发现并修复当前能够提前确认的问题。禁止把“用户长时间运行一次 -> 暴露一个错误 -> 只修一个错误 -> 再让用户长时间运行”作为常规开发方式；用户的正式长时间运行应主要用于取得实验结果，而不是作为开发代理的错误发现工具。达到实现完成标准后停止无必要修改，等待用户运行五个完整正式数据集。

除非用户明确要求，本阶段不得执行 `git status`、`git diff`、`git add`、`git commit`、`git push`、`git branch`、`git checkout`、`git switch`、`git reset`、`git clean` 或其他 Git/版本管理操作。版本管理不属于 baseline 实现、修复、构建和最小验证流程，也不得为了“确认改动”自动运行 Git 命令。

如果定向 build/check 失败，只处理直接失败以及同一受影响正式链中能够静态确认的同类问题，然后仅重跑该失败检查或其最小必要依赖；不得因为一个编译或运行错误启动全仓库审计、CPU/GPU 全面调查或新的审查链。普通 build、link、path、serialization、resource、runner、prover、verifier、adapter 或 backend glue 错误都属于实现任务，不构成切换正式执行方案的理由。只有静态/代码分析已经表明继续解决会要求 materially different protocol、无法获得的强制外部资源，或明显超出当前最小适配目标的实现规模时，才允许认定为真正 blocker。

# 17. 规范层级与实现事实判断

## 17.1 规范层级

工程与执行边界按以下顺序解释：第一，本文件 `BASELINE_EXECUTION_CONSTRAINTS.md` 规定独立性、自包含、最小适配、正式代码洁净度、运行边界和完成标准；第二，当前 baseline 对应的 `GAT-ZKML-*.md` 规定该 baseline 应证明的 GAT relation、密码学组件、公开/隐私边界和协议语义；第三，baseline 官方论文、官方项目和原生实现用于判断方法学身份、原生组件和原生优化是否被忠实保留。当前实现如果与上述规范冲突，应被视为待修复对象，不能因为“代码现在就是这样”就自动成为正确标准。

## 17.2 当前实现事实的判断依据

判断当前 baseline 实际实现状态时，依次优先检查当前正式源码、正式 runner、verifier、正式 validator、正式构建配置和当前正式产物，README、普通代码注释和历史实验记录只能作为辅助信息，不得覆盖当前真实代码事实，也不得覆盖本文件和技术规范。

## 17.3 冲突处理原则

本文件与 README 冲突时以本文件为准，技术笔记与 README 冲突时以技术笔记为准，当前代码与本文件冲突时修改当前代码，当前代码与技术笔记冲突时先确认 baseline 官方方法和统一 GAT 语义再修复实现，官方论文/原实现与主实验专项设计冲突时优先保持 baseline 方法学身份。未经用户明确要求不得修改本文件或技术笔记来迁就当前实现。

# 18. 完成标准

本文件区分“baseline 实现完成”和“baseline 最终正式实验完成”。自动化开发代理只负责达到“baseline 实现完成”；五个完整数据集的正式长时间实验只能由用户本人运行。

## 18.1 Baseline 实现完成标准

独立性方面，baseline 必须能在自身目录中独立构建和启动正式 runner，不依赖主实验、其他 baseline 或 baseline 目录外的项目级运行资源，不存在隐藏绝对路径、软链接或环境变量回指主实验。方法忠实性方面，必须保留官方原生证明体系和原生核心实现，只做 GAT 必要适配，不迁移主实验核心协议、主实验专项性能优化或非必要专项安全工程，不以复制改名封装方式迁移受禁止实现，也不进行无必要的大规模性能重构。

GAT 正确性方面，正式 relation 必须实现统一完整 GAT 数值语义，正式 runner 配置五个完整正式数据集，proof path 覆盖完整 GAT forward，最终 logits 由 proof relation 绑定，CPU/GPU/混合执行使用同一证明关系，并已通过允许范围内的最小局部正确性验证。隐私与验证方面，model/witness/trace 边界正确，verifier 不读取 witness，proof 不包含原始 private witness，public output 由 verifier 独立获得并验证，verifier 可以独立运行。

正式执行方面，只要求一套可用于最终实验的正式执行路径，该路径可以是 CPU、GPU 或 CPU+GPU 混合；如果某阶段声明使用 GPU，则必须通过允许范围内的小规模验证确认真实进入 GPU backend 且无 silent fallback。正式代码方面，本项目新增测试与 production 物理隔离，正式执行链不依赖 `tests/`，后续自行产生的无引用无用途开发残留已清理，同时完整保护官方原始源码、官方工程内容和有效 GAT adaptation。

正式运行与结果工具方面，正式 build、单数据集 runner、五数据集 runner、result collector 和 final validator 均应准备完成，collector/validator 可使用 `tests/` 中专用 fixture 做最小逻辑验证；`Commitment Time`、`Prove Time`、`Verify Time`、`Proof Size` 的 raw 来源和统一映射已定义；全量实验所需 baseline-local 正式资源已准备或具有明确生成入口。达到以上条件后应停止自动化开发阶段，不得继续运行任何完整数据集正式实验。

自动化开发代理达到实现完成标准不要求取得五个完整正式数据集的 runtime PASS，也不要求通过长时间正式运行证明“所有数据集一定成功”。Codex 侧需要达到的状态是必要实现已经存在、当前可提前发现的明显 blocker 已修复、唯一正式执行路径连贯可运行、该逻辑批次允许范围内的最小 build/check 已通过、用户正式运行命令和结果位置已经明确；正式 runtime confirmation、最终结果收集和五数据集 validator 验收属于用户本人。

## 18.2 最终正式实验完成标准

最终正式实验只由用户本人执行。用户完成 Cora、CiteSeer、PubMed、PPI、OGBN-Arxiv 五个完整数据集共五个正式结果后，一个 baseline 只有在五个结果全部齐全、每个 proof 真实生成、independent verifier 真实通过、proof size 可核验、timing 口径统一、full dataset identity 正确、full-GAT proof scope 正确、执行设备身份与正式计划一致、GPU 阶段真实且无 silent fallback、所有正式产物均保存在 baseline 自己目录且 final validator 对五个结果全部通过时，才视为最终正式实验完成。

## 18.3 自动化开发阶段最终汇报

一个 baseline 达到第 18.1 节实现完成标准后，最终汇报保持简短，只说明实际正式执行计划及关键阶段 CPU/GPU 使用情况、主要代码修改、唯一或最小必要检查及结果、仍只能由用户正式运行确认的事项、用户需要执行的正式命令、正式结果位置和 `VERIFY_OK`/等价成功判定方式。不得输出长篇审计叙述，也不得为了生成最终汇报再启动新的 review、测试、benchmark、profiler 或正式运行。

# 19. 最终原则

所有后续开发和判断都应服从以下原则：对比实验的目标是公平、可信地比较不同证明体系，而不是把三个 baseline 改造成主实验的三个变体；每个 baseline 必须保持官方源码基础和自身方法学身份，只完成统一 GAT 推理任务所必需的最小适配；每个 baseline 必须目录级自包含，能够独立构建、运行、证明、验证和产出结果。

主实验在开发阶段可以作为只读参考，并允许将完成 baseline 接入所需的通用实现、基础设施和正常第三方依赖接入方式本地化移植到 baseline，但正式运行不得依赖主实验；主实验专项证明机制、专项性能优化和非必要专项安全工程不得迁移。正式实验只要求每个 baseline 一套可行的 CPU、GPU 或 CPU+GPU 混合路径，最终产生五个完整数据集结果，不要求重复实现纯 CPU/纯 GPU 两套路径。

自动化开发代理禁止运行完整数据集正式实验，应尽可能一次性清理能够提前发现的问题，只执行短时、定向、最小范围验证；可修复 blocker 必须修，不得把 `BLOCKED`、`UNSUPPORTED`、`NOT_IMPLEMENTED` 当作方便的任务结束结论。正式实验代码必须干净紧凑，新增和修改代码严格遵守第 3.14 节排版规则；官方原始源码和官方工程内容按第 20 节完整保护。

# 20. 官方源码、原生语言与增量适配保护约束

本节属于最高优先级硬约束，适用于 `comparison/zkGPT/`、`comparison/zkdl/`、`comparison/ezkl/` 的全部后续实现、修复、重构、清理和正式执行。如果本节与前文关于代码清理、无引用删除、历史代码删除或正式路径精简的一般性规则冲突，以本节为准。

## 20.1 官方原始源码必须作为唯一 baseline 实现基础

每个 baseline 必须以对应官方原始项目源码作为正式实现的唯一基础，后续 GAT 适配必须形成 `official baseline source + minimal GAT adaptation`。禁止阅读或理解 official baseline 后脱离官方源码重新实现一个新的 `*-style` 证明系统并作为正式 baseline；官方论文、技术说明和协议描述只能用于理解和指导适配，不能替代官方源码本身。

## 20.2 禁止更换 baseline 核心原生实现语言

zkGPT 必须保持官方 C++/CUDA 实现体系，zkDL 必须保持官方 CUDA/C++ 实现体系及其 `.cu/.cuh` 结构，ezkl 必须保持官方 Rust/Cargo 和 Rust/Halo2 执行路径。禁止跨语言重写官方 prover、verifier、commitment、proof protocol、circuit/proof backend、核心 algebra/proof primitive 后替代官方实现。Python、Shell 或其他辅助语言仅可用于数据准备、格式转换、resource staging、正式 runner、配置生成、环境检查、结果收集、结果校验、日志整理等外围自动化。

## 20.3 禁止删除、裁剪或替换官方原始源码

一旦官方原始源码进入对应 baseline 目录，即属于受保护实现基础。不得因当前 GAT 路径未调用、代码清洁、减少冗余、工程精简、unused/dead code 等理由删除、裁剪、移动、归档或替换官方 `.cpp/.hpp/.h/.cc/.cu/.cuh/.rs` 等源码，也不得只保留少量官方核心文件后自行重建项目。官方源码即使不是当前 GAT adaptation 的直接修改目标，也不得因一般清理规则删除。

## 20.4 GAT adaptation 必须是基于官方源码的增量扩展

允许在官方现有源码和工程中进行完成 GAT 所必需的最小修改，新增 GAT 专用源码、图数据结构、relation 所需的算术化或 proof primitive 组合、与官方核心模块连接所需接口、正式 runner/config/resource/dataset/model adapter，以及 CPU/GPU/混合正式运行所需的最小适配。禁止从零重写整个 baseline、用新项目替代官方项目、旁路官方 prover/verifier、重新实现完整 commitment/proof protocol 后仅通过接口兼容官方代码，或把官方项目降级为只提供少量 utility 的依赖。

## 20.5 正式核心执行路径必须真实经过官方核心实现

仅保留官方源码在目录中不等于满足要求。正式 build、prover、verifier、commitment/proof primitive 和核心执行路径必须真实建立在官方实现之上，GAT 新增 relation 和适配逻辑必须连接到官方核心证明系统。禁止官方源码形式上存在但正式 proof generation、verification 或 commitment 实际由另一套自建核心系统完成。

## 20.6 官方原生构建体系不得被替换

每个 baseline 的正式构建必须继续以官方项目原生核心构建体系为基础。允许增加 GAT source、target、必要依赖、CPU/GPU build option 和正式实验入口，但禁止放弃官方核心构建体系并新建另一套工程作为正式实现，也禁止让官方构建文件只成为形式 wrapper 而实际核心 proof binary 来自另一套重实现。

## 20.7 wrapper、binding、FFI 或进程封装不得绕过原生实现约束

是否合规以正式计算实际由哪套核心代码完成为判断标准，而不是以调用接口或进程边界判断。wrapper、language binding、FFI、subprocess、RPC、dynamic/shared library adapter、namespace/package/module 重命名、CLI 代理或 service 只有在用于把 GAT adaptation 接入官方核心实现时才允许；如果实质上让另一套自建 prover/verifier/protocol 替代官方实现，则禁止。

## 20.8 禁止跨语言重写官方核心模块后进行“等价替换”

即使数学公式相同、proof transcript 格式相似、commitment scheme 名称相同、verifier 得到相同真假结果、输出 logits 相同、benchmark 接口相同或自建实现声称兼容官方算法，也不得跨语言替换官方核心模块。正式比较的是对应方案的实际官方实现及其必要 GAT 适配，不是协议模拟器。

## 20.9 官方源码和有效 GAT adaptation 不受一般清理删除规则约束

一般清理规则不得应用于官方原始源码以及为当前 GAT adaptation 正式实现所新增、修改或保留的有效源码。一般清理只适用于后续自行新增且已确认错误或废弃的临时实现、debug-only 文件、测试临时副本、错误重实现、重复生成且无正式或测试用途的文件以及与官方源码和当前有效 GAT adaptation 均无关的开发残留。无法确定文件身份时默认不得删除。

## 20.10 已经完成的有效适配不得因重构被整体推翻

已经确认建立在官方源码上的有效 GAT adaptation 不得仅因重构、代码风格统一、目录重排、语言偏好或工程简化而删除并重新实现。允许 correctness 修复和最小维护，但必须继续保持官方源码基础、原生核心语言、原生核心构建体系、原生 proof/prover/verifier 路径和增量适配性质。

## 20.11 禁止统一三个 baseline 的核心工程语言或框架

不得为了统一开发体验把三个 baseline 改成同一种编程语言、同一种 proof framework、同一个公共 prover backend、同一个 verifier framework、同一套自建 protocol abstraction 或共享核心执行框架。可以统一外部输入语义、结果字段和论文计量口径，但不能统一或替换内部核心实现。

## 20.12 对现有非原生重实现的处理原则

如果发现当前 baseline 存在脱离官方源码形成的非原生核心重实现，例如用 Python 重写原本 C++ 核心证明系统、用 Rust 重写 CUDA/C++ 核心系统或用 C/C++ 重写原本 Rust 核心系统，该实现不得继续作为正式 baseline 核心执行基础。正式实现必须回到官方源码并在官方源码上进行 GAT 增量适配；旧实现只能人工参考已经验证过的 GAT 数值语义、数据组织或外围 runner 思路，不得继续承担核心 proof/prover/verifier 计算。

## 20.13 本节与其他规则的冲突处理

本节的官方源码保护、原生语言保护和增量适配要求优先于代码最少、文件最少、单一源码文件、无 unused source、无历史源码、工程目录统一、语言统一、wrapper 简化、自动 dead-code cleanup 或一般性的“无引用即删除”等工程目标。但不得借“保护官方源码”为理由保留后续自行产生的错误重实现、临时文件或明显无关开发垃圾。

## 20.14 最终执行原则

官方源码是基础，不是参考样例；GAT adaptation 是对官方实现的增量扩展，不是对论文协议的重新实现；原生语言、原生构建体系和原生核心证明路径不得被替换；官方源码不得因清理、无引用、dead code、重构或工程简化而删除；已建立在官方源码上的有效 GAT adaptation 不得被任意删除或跨语言重写；外围脚本可以使用辅助语言，但核心 prover、verifier、commitment、proof protocol 和 proof backend 必须继续由对应 baseline 官方原生实现及其最小 GAT 扩展承担。

## 20.15 官方原始项目整体内容保护

“官方原始实现”不仅包括源码文件，还包括构成官方工程的 `CMakeLists.txt`、`Makefile`、`build.rs`、`Cargo.toml`、`Cargo.lock`、`.gitmodules`、官方 build/run/setup 脚本、依赖声明、submodule/vendor 接入配置、配置文件、资源描述、参数文件以及为正常构建、证明、验证或执行保留的其他工程内容。禁止因 cleanup、unused、dead code、目录整理或工程精简删除这些内容，也禁止只保留源码文件却破坏官方项目原有工程结构。允许对这些文件做完成 GAT adaptation 所必需的最小增量修改，但不得整体替换为新的非原生工程。

## 20.16 `*-style` 表述不得解释为允许重新实现

本文件任何 `zkGPT-style`、`zkDL-style`、`ezkl/Halo2`、原生风格或类似表述，仅用于描述对应 baseline 的方法学身份和原生实现特征，绝不表示允许根据论文、README、协议描述或现有行为重新实现一个 style-compatible 系统。正式 baseline 必须直接建立在对应官方原始项目及其原生核心实现上。

## 20.17 GAT 适配实现必须集中隔离

所有为 GAT 对比实验新增的正式核心适配实现必须集中放入 `comparison/zkGPT/gat_adapter/`、`comparison/zkdl/gat_adapter/`、`comparison/ezkl/gat_adapter/`。该目录负责 GAT data/relation/mapping/integration，核心语言继续遵循官方工程：zkGPT 使用 C++/CUDA，zkDL 使用 CUDA/C++，ezkl 使用 Rust；不得为了统一适配目录而统一语言。

### 20.17.1 官方源码只允许最小必要局部修改

GAT 新增主逻辑必须优先放在 `gat_adapter/`，官方源码只允许为接入和正确运行正式 GAT 路径所不可避免的最小、局部、可追踪修改。允许的修改包括 include/import/mod、operation/layer/circuit/command 注册、函数或类型声明、build target/source registration、必要依赖、正式 CLI 入口、CPU/GPU/backend 调用入口、极小参数传递，以及 adapter 无法独立完成时确有必要的 correctness 修复、正式计量埋点、backend 暴露/初始化、pre-synthesis/fail-closed 检查和兼容性修改。能够完整放在 `gat_adapter/` 的新增 GAT 业务逻辑仍必须放在 `gat_adapter/`，不得以 correctness、计量或 backend 接入为名把完整 GAT 主逻辑迁入官方核心源码，也不得形成第二个 GAT 实现位置。

### 20.17.2 禁止将 GAT 适配逻辑散落到官方源码树

禁止把新的 GAT forward、graph traversal、attention、LeakyReLU、stable Softmax、aggregation、ELU、ExpMap、SignedRoundDiv、multi-head concat、output head averaging、PPI graph boundary、GAT witness/trace、GAT relation 和 GAT-specific proof primitive mapping 大量散落到官方核心源码。确有逻辑必须直接位于官方原文件时，只允许第 20.17.1 规定的最小、局部、可追踪修改，不得形成第二个 GAT 主实现位置。

### 20.17.3 `gat_adapter/` 不得成为新的独立证明系统

`gat_adapter/` 只能把统一 GAT relation 接入 baseline 官方核心证明系统，禁止在其中从零实现新的 prover/verifier、重新实现官方 commitment scheme、完整 Sumcheck/GKR/Hyrax/Halo2/Plonkish 等核心协议，或建立与官方 proof backend 平行的自建证明系统。正确关系必须始终是 `gat_adapter/ -> baseline 官方核心证明原语 -> 官方 prover/verifier`。

### 20.17.4 测试代码不得进入 `gat_adapter/`

`gat_adapter/` 属于正式生产实现，本项目新增的 unit/integration/toy/synthetic fixture/tamper/negative/debug/self-test/benchmark/profiler/test-only helper 等测试相关代码和资源必须放入 baseline-local `tests/`。正式执行链不得依赖 `tests/`。

### 20.17.5 外围脚本与正式资源不强制放入 `gat_adapter/`

`gat_adapter/` 主要用于 GAT 正式核心适配源码，正式 runner/collect/validate 继续放在 `scripts/`，正式配置放在 `configs/`，正式数据和模型资源放在 baseline 自己的资源目录，本项目新增测试代码和 fixture 放在 `tests/`。不得为了目录统一破坏官方原始项目结构。

### 20.17.6 适配隔离原则

官方原始项目必须保持可识别、可追踪、可继续使用；新增 GAT 正式核心实现集中在 `gat_adapter/`；官方源码只进行第 20.17.1 允许的最小、局部、可追踪修改；`gat_adapter/` 只能扩展官方证明系统，不能替代官方证明系统；测试代码与正式 GAT adapter 严格物理隔离。

## 20.18 自动化开发阶段必须采用短时、定向、最小范围验证

自动化开发代理只能执行与当前修改直接相关的最小必要验证，原则是“修改什么，就精准检查什么及其必要直接影响路径”。禁止运行整个 test suite、workspace/repository-wide tests、全部 integration/regression tests、全矩阵测试、与当前修改无关的测试、长时间 proving/verification/setup/parameter generation、长时间 compilation/GPU benchmark、profiler、stress test 或其他持续高资源占用任务，也不得为了等待不确定的长任务不断提高 timeout。不得启动一个运行时间不确定的命令来“看看是否很快”。

每个逻辑上完整的修改批次严格执行 stop rule：先完成一批连贯修改并做静态检查；只有运行验证确实必要时，最多执行一次与该批修改直接相关的定向 compile/build/check；该检查通过后，本批次验证立即结束，不得为了“更有信心”追加 smoke、integration、regression、E2E、proof、tamper 或其他测试。如果检查失败，只修复直接失败并重跑该失败检查或其最小必要依赖；已经通过的检查，后续代码没有直接修改其覆盖路径时不得重复执行。成功 build 不构成自动继续 smoke test 的理由，成功 smoke/minimal check 不构成继续 integration 的理由，任何短验证成功都不构成自动启动正式 proof 的理由。 选定正式执行路径达到代码完成状态且该逻辑批次的最小必要检查已经通过后，不存在额外强制的“最终验证阶段”；不得再启动 full-suite、第二轮 review、额外 smoke/integration、benchmark、profiler 或正式 proof 来证明完成。

在把正式长时间运行命令交给用户之前，开发代理必须尽可能通过静态分析、真实配置、现有资源、调用关系和上述短时定向检查，一次性发现并修复当前能够提前确认的问题。不得依赖用户重复进行长时间正式运行来逐个发现本可提前发现的错误。如果某项结论只能通过完整数据集、长时间 setup/proving/verification 或完整矩阵确认，则开发代理应停在“实现完成 + 代码检查完成 + 最小同路径验证完成 + 正式运行入口准备完成”的状态，将完整验证交给用户本人。

## 20.19 官方自带测试、示例、benchmark 与本项目新增测试隔离规则的边界

第 8 节测试隔离要求仅适用于本项目为 GAT adaptation 新增的测试、调试、benchmark、profiling、fixture 和验证代码。官方仓库原本自带的 tests、test vectors、examples、demo、benchmark、microbenchmark、profiler、sample、verification vectors、reference inputs、官方自测脚本、官方回归测试和官方测试资源属于官方原始项目整体内容，必须保持官方原始位置和工程归属，不得因测试隔离规则迁移、删除或改造成正式 GAT production 任务；它们可以原样保留，但不得成为本项目正式 GAT runner、proof generation 或 verifier 的必需依赖。

## 20.20 官方第三方依赖、submodule 与 vendor dependency 的 baseline-local 复制规则

如果某 baseline 官方工程原本声明、要求或能够正常使用某个第三方项目、submodule、vendor dependency、external library 或通用执行 backend，而当前主项目其他位置已经存在可用副本或正常构建环境，允许将该第三方依赖本体、官方版本信息和完成正常接入所需的通用构建/配置方式复制或本地化到 baseline 官方工程预期位置，以恢复 baseline-local、自包含的官方工程。该规则适用于例如 mcl、CUDA、ICICLE 等正常第三方能力，只要其接入不改变 baseline 原生证明体系；使用通用 CUDA/ICICLE backend 本身不属于主实验专项性能优化，也不要求因为 baseline 目录中尚未单独安装就从零重新实现或直接放弃该能力。

复制后依赖必须位于当前 baseline 自己目录，正式 build/run 不得继续读取原主项目路径，不得用软链接或环境变量回指主实验，只能复制第三方 dependency 本体、官方版本信息及其通用构建、链接、初始化和配置接入信息；如果能够确认官方 baseline 对应 dependency revision/tag/commit，应优先保持一致。禁止借此复制主实验 KZG 实现、主实验 MSM/FFT wrapper 或 backend 改造、主实验专用 CUDA kernel、fusion、cache、pipeline、memory layout、commitment pipeline、proof/transcript wrapper、lookup/trace 实现或其他主实验专项安全/性能工程。无法确认某目录是第三方依赖本体、通用接入信息还是主实验二次专项实现时，必须先做来源检查，不得整目录无差别复制。

## 20.21 `gat_adapter/` 为 GAT 新增核心适配实现的强制位置

所有为 GAT 对比实验新增的正式核心适配实现必须集中在对应 baseline 的 `gat_adapter/`；除第 20.17.1 允许的最小必要局部修改外，不得将新的 GAT 核心实现直接写入官方源码树。官方核心文件可以进行 register/include/import/mod/link/build registration/call/expose interface/type/function declaration/必要参数传递，以及 adapter 无法独立完成时完成正式运行所必需的最小 correctness、计量、backend 接入、fail-closed 和兼容性修改，但这些修改必须局部、可追踪，不得承载完整 GAT forward/relation 或形成第二个 GAT 主实现位置。不得因为修改官方文件更方便、可以少写 adapter 或可以减少接口而把新的 GAT 主逻辑直接写入官方核心文件。

## 20.22 canonical dataset、checkpoint 与模型参数必须跨 baseline 保持一致

三个 baseline 必须以统一来源的 canonical dataset、graph structure、node/edge ordering、graph batching、checkpoint、weight、bias、quantization scale、fixed-point parameter、model architecture、GAT numerical semantics、public features 和 public output policy 作为实验基准。禁止 baseline-specific retraining、finetuning、reinitialization、单独修改 weight/bias、改变层数/head 数/hidden dimension、为某个 proof system 选择更容易证明的 checkpoint，或通过改变量化 scale 改变有效模型。

允许将 canonical checkpoint、tensor、数据和模型参数转换为 baseline 自己的文件格式、原生内存布局、field/fixed-point 表示或序列化格式，但必须是 value-preserving 的表示转换，转换前后模型数学语义一致。正式比较始终满足 `same dataset + same graph + same checkpoint + same model parameters + same GAT semantics + different proof system`。

## 20.23 允许必要的内部 padding/alignment，但不得改变真实 GAT 关系

baseline 可以因 MLE、FFT domain、tensor shape、Halo2 region、CUDA kernel alignment 或其他原生证明/执行要求进行 power-of-two、tensor、polynomial domain、row/column、dummy entries/nodes/edges、circuit rows、region 或 backend memory alignment 等内部 padding，但 padding 必须语义中立：不得删除、替换或减少任何真实节点、真实边、attention、aggregation 和 neighborhood，不得改变真实 Softmax normalization domain 或 logits，dummy/padded 元素不得参与真实 attention 归一化或改变真实 aggregation，并且 padding 逻辑必须受到正式 proof relation 正确约束、mask 或隔离。内部 padding 只能解决表示和对齐问题，不得变相形成 sampled/reduced graph、部分节点/边证明或省略真实运算。

## 20.24 三个 baseline 的论文级 timing 口径必须统一

最终论文统一核心指标为 `Commitment Time`、`Prove Time`、`Verify Time`、`Proof Size`。每个 baseline 必须保留 baseline-native raw timing，再按照统一、明确、可审计的论文计时定义映射到上述字段；如果原生阶段无法一一对应，应保留完整 raw breakdown，不得隐藏差异或人为制造不存在的计算阶段。

一次性 setup 不得混入每次 inference 的 `Prove Time`，compile 不得在某一 baseline 计入 prove 而另一 baseline 排除，PK/VK generation 应单独记录，model registration/commitment 如果属于一次性阶段应与每次 inference 区分，witness generation 的归属必须采用统一定义，verifier-facing proof serialization 的边界必须一致，文件系统冷启动、日志打印、CSV/JSON 写入不得选择性计入 crypto timing，同一计算阶段不得重复计入多个 paper-level 字段。GPU timing 必须在必要位置正确同步设备，确保计时结束时对应 GPU computation 已真实完成，禁止利用异步 kernel launch 或后台执行提前停止计时。

## 20.25 官方 upstream provenance、版本与 adaptation 起点必须固定

每个 baseline 开始正式 GAT adaptation 后必须固定官方 upstream source identity，至少记录 official repository、upstream URL、commit/tag/release/archive identity、adaptation 起始快照、官方 submodule、关键第三方 dependency、vendor dependency 来源和版本、哪些内容来自 pristine upstream、哪些属于 GAT adaptation 新增以及哪些官方文件因 GAT 接入做了最小修改。provenance 信息必须保存在当前 baseline 自己目录中，例如 `UPSTREAM_PROVENANCE.md`，不得依赖聊天记录、临时日志或开发代理记忆。provenance 记录必须优先使用仓库中已经存在的来源文件、lock/submodule 信息、vendor metadata、release/archive 标识或其他可直接读取的静态来源，不得为了补齐 provenance 绕过第 16 节的 Git 禁止规则；确实只有 Git 命令才能取得而用户未明确允许时，不执行 Git，并把该来源字段作为需要用户授权后补齐的唯一例外，而不是因此启动额外版本管理操作。

禁止开发过程中静默替换 upstream 版本、未记录地切换 tag/commit、使用来源不明 fork、不同数据集使用不同 upstream 版本、CPU/GPU 阶段基于不同 upstream snapshot，或为解决一个适配问题整体升级官方工程却不记录版本变化。如果必须升级或切换 upstream，应记录切换前版本、切换后版本和原因，并保证新的 GAT adaptation 仍建立在明确可审计的官方来源上。

## 20.26 约束优先级

第 20 节全部属于最高优先级执行约束。发生歧义时，依次坚持官方源码和官方工程保护、原生语言与原生构建体系保护、`gat_adapter/` 集中隔离、canonical dataset/checkpoint/model parameters 一致性、短时定向最小范围测试、官方自带 tests/examples/benchmarks 原始位置保护、baseline 官方第三方依赖的 baseline-local 合法复制、padding 语义中立、论文级 timing 可比和 upstream provenance 固定。

同时必须遵守本文件的正式执行路径规则：不要求 CPU/GPU 两套正式矩阵，每个 baseline 只需一套 CPU、GPU 或 CPU+GPU 混合正式路径并产生五个完整数据集结果；主实验通用实现和基础设施可以按第 3.3 节本地化复用，但主实验核心证明协议、专项性能优化和非必要专项安全工程仍严格禁止迁移。任何自动化开发代理不得利用文件中更宽泛或与本节冲突的措辞规避这些约束。
