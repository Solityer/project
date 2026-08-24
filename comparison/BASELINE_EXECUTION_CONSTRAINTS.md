# BASELINE_EXECUTION_CONSTRAINTS

## 1. 文件目的

本文件定义 `comparison/` 下三个对比实验的统一执行约束、工程边界和完成标准。

适用目录：

- `comparison/zkGPT/`
- `comparison/zkdl/`
- `comparison/ezkl/`

三个目录分别代表三套独立的对比实验实现。

本文件只规定“如何实现、如何运行、允许做什么、不允许做什么以及何时视为完成”。  
各 baseline 具体证明关系、密码学组件和 GAT 证明语义，应以各自协议说明文件和当前正式代码为准。

本文件取代旧版 `BASELINE_EXECUTION_CONSTRAINTS.md`。旧版约束不再作为后续开发依据。


---

## 2. 总体目标

三个 baseline 的目标不是重新设计新的证明系统，也不是将其优化到与主实验相同的工程水平，而是：

> 在忠实保留各 baseline 原生方法、证明体系和执行风格的前提下，仅进行完成统一 GAT 推理任务所必需的最小适配，使每个 baseline 都能够在自己的目录中独立构建、独立运行、独立证明、独立验证并独立产出正式实验结果。

对比实验必须优先保证：

1. baseline 方法身份真实；
2. GAT 任务语义一致；
3. 证明关系完整；
4. 实验实现彼此独立；
5. 不依赖主实验；
6. 不迁移主实验专项优化；
7. 正式代码干净、简洁；
8. 正式结果可验证、可复现、可审计。


---

# 3. 最高优先级硬约束

以下规则属于硬约束。任何违反均应视为 baseline 尚未完成。

## 3.1 三个 baseline 必须完全独立

`zkGPT`、`zkdl`、`ezkl` 必须是三套彼此独立的实验实现。

禁止：

- 一个 baseline 调用另一个 baseline 的源码；
- 一个 baseline 链接另一个 baseline 的库；
- 一个 baseline 调用另一个 baseline 的脚本或二进制；
- 一个 baseline 读取另一个 baseline 的中间结果；
- 一个 baseline 复用另一个 baseline 的 proof、registry、setup 参数或运行缓存；
- 将三者的核心证明实现抽取到公共目录后共同调用。

允许在开发过程中人工比较不同 baseline 的代码，但正式构建和正式运行不得形成任何跨 baseline 依赖。


## 3.2 每个 baseline 必须目录级自包含

每个 baseline 必须能够在其自身目录中独立完成正式实验。

正式运行所需的项目资源必须位于该 baseline 自己的目录体系内，包括但不限于：

- 源代码；
- 构建文件；
- 正式运行脚本；
- 配置文件；
- 数据集或该 baseline 自己管理的数据资源；
- 模型/checkpoint 或该 baseline 自己管理的模型资源；
- 图预处理所需资源；
- 静态表；
- 协议参数描述；
- setup/PP/SRS/PK/VK 等该 baseline 所需资源；
- model registry 或其他注册对象；
- 正式 proof；
- verifier 输入；
- 正式结果；
- 正式日志；
- 结果收集与正式结果校验工具。

正式实验运行时不得读取该 baseline 根目录之外的项目文件。

例如，`comparison/zkGPT/` 的正式实验不得依赖：

- 仓库根目录的数据目录；
- 主实验 checkpoint；
- 主实验生成的 public output；
- 主实验静态表；
- 主实验构建产物；
- 主实验 proof 参数；
- `comparison/zkdl/` 或 `comparison/ezkl/` 中的任何文件。

“自包含”不要求将操作系统、CUDA Driver、编译器、Python/Rust/C++ 工具链、系统库或正常的第三方软件依赖复制到 baseline 目录中。  
本文件所禁止的“外部依赖”，特指 baseline 目录之外的**项目级资源和项目级实现依赖**。

项目级资源包括但不限于：

- 数据集文件；
- checkpoint/model 文件；
- 图预处理结果；
- 静态表；
- public output；
- 配置文件；
- SRS/PP/PK/VK；
- model registry；
- compiled circuit；
- proof 中间产物；
- 主实验或其他 baseline 生成的缓存与结果。

正式实验所使用的上述项目级输入、参数和产物必须由该 baseline 自己管理。

允许正常使用 baseline 原生实现所需的第三方依赖，例如 crates、Python packages、Halo2/ezkl 原生依赖、CUDA Runtime 等，但不得借第三方依赖之名间接调用主实验或其他 baseline 的项目实现。


## 3.3 不允许依赖主实验

主实验只能作为开发阶段的只读参考，用于理解统一的 GAT 数值语义、数据格式和实验口径。

正式 baseline 不得：

- include 主实验源码；
- import 主实验模块；
- 链接主实验库；
- 调用主实验二进制；
- 调用主实验脚本；
- 读取主实验生成的中间文件；
- 读取主实验结果文件作为 proof 输入；
- 使用主实验运行后才能产生的资源；
- 通过软链接、环境变量、相对路径或绝对路径形成隐式依赖；
- 将主实验作为 hidden dependency。

如果某一段通用逻辑确实是 baseline 完成 GAT 适配所必需，应将“最小必要逻辑”独立实现在该 baseline 内，而不是直接依赖主实验。


## 3.4 只做最小必要适配

baseline 的修改范围应严格限制为：

> 使其能够正确表达并证明统一 GAT 推理任务所必需的适配。

允许的必要适配包括但不限于：

- GAT 图数据表示；
- 稀疏边访问；
- 多层多头 attention；
- hidden head concat；
- output multi-head；
- stable Softmax；
- LeakyReLU；
- ELU；
- ExpMap；
- SignedRoundDiv；
- 图 batch 边界；
- PPI 多图处理；
- 定点量化链；
- 最终 logits 绑定；
- 必要的 public/private 输入映射；
- 将上述关系映射到 baseline 原生证明原语。

不得借“适配”为名进行大规模协议重构、证明体系重写或与完成 GAT 无直接关系的工程改造。


## 3.5 必须保留 baseline 原生方法身份

三个 baseline 必须继续使用各自原生证明体系和核心方法。

原则：

> baseline-native components 保留；main-experiment-specific components 不迁移。

例如：

- zkGPT 应保持其 zkGPT-style 的原生证明组件和组织方式；
- zkDL 应保持其 zkDL-style 的原生证明组件和组织方式；
- ezkl 应保持其 ezkl/Halo2 原生证明与执行路径。

禁止将主实验独有的证明协议、承诺体系、lookup 方案、trace 体系或其他专项设计迁移后再冠以 baseline 名称。

如果原 baseline 不原生支持某个 GAT 运算，应优先用该 baseline 自己已有的证明原语进行最小扩展，而不是引入主实验方案。


## 3.6 禁止迁移主实验专项性能优化

baseline 不应主动吸收主实验为了极限性能而开发的专项优化。

包括但不限于：

- 主实验专用 trace layout；
- 主实验专用 fast MSM；
- 主实验专用 FFT backend/kernel；
- 主实验专用 CUDA kernel；
- 主实验专用 kernel fusion；
- 主实验专用异步流水；
- 主实验专用 memory layout；
- 主实验专用 commitment pipeline；
- 主实验专用 polynomial batching；
- 主实验专用缓存策略；
- 针对五个数据集手工设计的特殊 shortcut；
- 其他仅用于提升主实验性能的专项工程优化。

如果某项优化本身就是 baseline 原项目的原生机制，则应保留。

不得为了让 baseline 更慢而主动删除其原生优化；也不得为了让 baseline 更快而将主实验专项优化移植进去。


## 3.7 禁止通过“复制、改名、封装”绕过独立性约束

是否属于主实验迁移，以**实现来源和实质逻辑**判断，而不是以文件所在位置、文件名、namespace、module 名称或接口形式判断。

以下行为仍然视为依赖或迁移主实验：

- 将主实验源码复制到 baseline 目录后继续使用；
- 将主实验代码改名后作为 baseline 自有实现；
- 修改 namespace/module/package 名称后继续使用主实验逻辑；
- 对主实验实现增加 wrapper 后声称为 baseline 原生实现；
- 将主实验专项优化拆分、重排或轻度改写后移植；
- 将主实验协议组件重新封装成“通用组件”后供 baseline 调用。

如果某段主实验代码仅体现统一 GAT 数值语义，而 baseline 确实需要相同功能，应根据该 baseline 的原生工程结构和原生证明原语进行最小、独立实现，而不是复制主实验实现。


## 3.8 禁止非必要的主实验安全工程迁移

baseline 不仅不得被主实验“性能优化过度”，也不得被主实验“安全工程重构过度”。

除非属于 baseline 原生机制，或不加入就无法满足 baseline 自身声明的正确性、隐私边界和可信验证要求，否则不得主动引入主实验特有或明显超出 baseline 原方法的安全工程设计，包括但不限于：

- 主实验特有的多层 manifest 体系；
- 主实验特有的多重 artifact digest 绑定；
- 主实验特有的 commitment lifecycle；
- 主实验特有的注册协议；
- 主实验特有的 masking protocol；
- 主实验特有的 transcript wrapper；
- 主实验特有的 proof schema wrapper；
- 主实验特有的 telemetry/security evidence protocol；
- 仅为模仿主实验安全工程而增加的大量审计封装。

允许保留或补齐以下内容：

1. baseline 原论文、原实现或原生证明体系已有的安全机制；
2. 完成 GAT 适配后，为维持 baseline 原有隐私/正确性声明所必需的机制；
3. 使正式 proof 和独立 verifier 真正可信所不可缺少的最小安全检查。

原则是：

> 只补齐可信正式实验所必需的安全边界，不把 baseline 重构成主实验的安全工程版本。


## 3.9 禁止执行任何全量数据集正式实验

本轮 baseline 实现、修复、测试和验证阶段，禁止由自动化开发代理、Codex 或其他代码执行代理运行任何完整数据集的正式实验。

自动化开发阶段不得以以下任一完整正式数据集作为实际运行输入：

- Cora；
- Citeseer；
- Pubmed；
- PPI；
- ogbn-arxiv。

该禁令适用于任何执行阶段，包括但不限于：

- preprocessing；
- graph/material preparation；
- witness generation；
- commitment；
- setup 中与完整数据集直接相关的运行阶段；
- prove；
- verify；
- result collection 中会触发完整数据集计算的步骤；
- 其他任何以完整正式数据集为实际输入的计算。

无论 CPU 还是 GPU，均不得实际运行完整正式数据集。

完整数据集只允许进行静态检查，例如检查配置、路径、代码分支、文件命名、runner 参数和资源引用关系，不允许执行实际计算。

所有运行验证必须使用独立的小规模代表性输入。

完整五数据集 CPU/GPU 正式结果只允许由用户本人在实现完成后手动运行。

自动化开发阶段允许：

- build；
- 静态检查；
- unit test；
- integration test；
- negative/tamper test；
- 小规模 synthetic test；
- 小规模真实数据片段测试；
- microbenchmark；
- profiler；
- 局部正式关系回归；
- CPU/GPU 小规模一致性验证；
- 不构成完整数据集正式实验的功能验证。

不得通过拆分完整数据集、关闭某些阶段、分批执行或其他方式变相运行完整数据集正式实验。

小规模测试必须使用独立的小规模代表性输入。不得通过连续运行多个数据片段、分批覆盖节点/边、累计执行多个子任务或其他组合方式，最终覆盖任意一个完整正式数据集。

如果验证某项修改必须依赖长耗时完整数据集结果，则应停止在“实现已完成、等待用户全量运行验证”的状态，不得自行执行。

该限制优先于本文件其他关于“最终 10 组结果”的执行性表述。自动化开发代理负责把代码、正式执行链和 validator 准备到可由用户直接运行的状态；最终 5×2 正式结果由用户本人运行后产生。


## 3.10 当前阶段不进行系统级性能重构

baseline 的目标是形成可信、公平、完整的正式对比实现，而不是开展新的性能优化研究。

完成以下条件后，应停止无必要的工程重构：

- GAT 关系正确；
- proof 完整；
- verifier 独立；
- CPU/GPU 路径真实；
- 五个正式数据集的正式 runner 已准备可运行；
- 结果收集逻辑已通过允许范围内的测试验证；
- final validator 已通过 `tests/` 中专用 fixture 的逻辑验证；
- baseline 独立性满足；
- 正式代码清洁。

仅当存在明显错误、不可接受的实现缺陷、运行阻塞或违背 baseline 原生方法时，才进行必要修改。


---

# 4. 统一 GAT 实验语义

## 4.1 三个 baseline 必须证明同一个任务

三个 baseline 更换的是证明系统，不是机器学习任务。

正式实验应保持统一的：

- 数据集；
- 图拓扑；
- 节点编号；
- self-loop 规则；
- edge ordering；
- graph batching；
- PPI 多图边界；
- GAT 层数；
- hidden heads；
- output heads；
- hidden concat；
- bias 位置；
- activation；
- stable Softmax；
- 定点量化；
- rounding；
- 静态函数表语义；
- output head averaging；
- 最终 public logits。


## 4.2 正式数据集固定

正式比较使用以下五个完整数据集：

1. Cora
2. Citeseer
3. Pubmed
4. PPI
5. ogbn-arxiv

正式结果禁止使用以下内容替代完整数据集：

- toy graph；
- sampled graph；
- subgraph；
- reduced graph；
- 仅部分节点；
- 仅部分边；
- 小规模 synthetic input。


## 4.3 CPU/GPU 使用相同证明关系

CPU 与 GPU 仅允许执行后端不同。

以下内容必须保持一致：

- public statement；
- GAT 数值语义；
- proof scope；
- privacy boundary；
- output policy；
- 模型语义；
- 图语义；
- 量化语义。

不得为了 GPU 跑通而削弱证明关系。


---

# 5. 完整证明要求

## 5.1 必须证明完整 GAT inference

正式 proof 必须覆盖完整 GAT forward relation。

禁止：

- 只证明矩阵乘法；
- 只证明 projection；
- 只证明 attention 的一部分；
- 只证明单层；
- 只证明部分节点；
- 只证明部分边；
- 只证明最终输出 hash；
- host 端计算关键中间值但 proof 不约束；
- 将难实现的非线性直接作为可信 public input；
- 用 sampled relation 代替 full relation；
- 让 verifier 读取 witness 后自行重算来代替证明。


## 5.2 最终 logits 必须由 proof 绑定

最终公开输出 `Y` 必须是被证明关系的一部分。

proof 必须证明：

> 给定正式 public statement 和隐藏 witness，完整 GAT 推理关系成立，并得到指定的公开输出 Y。


## 5.3 verifier 不得信任 proof 自带的输出声明

如果 proof artifact 内包含输出或 instance 信息，verifier 不得将其直接作为可信 public statement 来源。

正确逻辑必须是：

1. verifier 独立获得 public statement；
2. verifier 独立获得公开输出 `Y`；
3. verifier 使用 proof 验证该 statement 与 `Y`；
4. proof 内携带的对应信息只能被校验，不能成为唯一可信来源。


---

# 6. 隐私与验证边界

## 6.1 隐私目标

正式 baseline 应维持约定的隐私边界：

- public graph；
- public features；
- public metadata；
- public final logits；
- model hidden；
- execution trace hidden；
- private witness hidden。


## 6.2 verifier 禁止读取 witness

正式 verifier 只能读取验证所必需的公开信息，例如：

- public statement；
- public output；
- proof；
- verification key；
- public parameters；
- public model commitment/registry；
- 必需的公开 metadata。

verifier 不得读取：

- private witness；
- private model weights；
- private trace；
- prover 内存；
- private opening material；
- 仅 prover 应持有的临时秘密。


## 6.3 proof artifact 不得包含原始 witness

正式 proof 不得通过直接打包以下内容来实现“可验证”：

- 原始模型参数；
- 原始 private trace；
- 完整 witness；
- verifier 本不应获得的 private auxiliary data。


## 6.4 重复推理场景不得明显破坏既定隐私

如果 baseline 使用长期 model commitment、model registry 或其他可复用对象，则正式实现应遵守该 baseline 自己的原生隐私机制。

不得因为 GAT 适配而主动削弱 baseline 原有隐私边界。


---

# 7. CPU 与 GPU 正式执行要求

## 7.1 最终正式矩阵

每个 baseline 最终由用户本人运行的正式结果矩阵必须覆盖：

`5 datasets × 2 backends = 10 results`

即：

- 5 个 CPU 结果；
- 5 个 GPU 结果。

自动化开发代理不得运行这些完整数据集正式结果，其职责是保证正式 runner、prover、verifier、collector 和 validator 已准备完成，可由用户直接执行。


## 7.2 GPU 必须是真实 GPU 执行

GPU 结果不得由以下方式伪装：

- 仅检测到 CUDA 设备；
- CPU proof + GPU 标签；
- 只运行一个与核心 proof 无关的 GPU 操作；
- GPU 失败后静默 fallback 到 CPU；
- 只在预处理阶段使用 GPU，但核心 proof 全部运行于 CPU。

如果结果标记为 GPU，则该 baseline 声称支持 GPU 的核心证明计算必须真实进入其 GPU backend。


## 7.3 禁止 silent fallback

任何 backend 失败都必须 fail closed。

如果请求 GPU：

- GPU 不可用 -> 失败；
- GPU backend 未构建 -> 失败；
- GPU kernel 未执行 -> 失败；
- GPU proof 路径异常 -> 失败。

不得自动退回 CPU 后仍将结果记录为 GPU。


---

# 8. 正式代码洁净度

## 8.1 正式代码只能包含生产实验逻辑

正式源码和正式执行脚本中，只允许保留完成正式实验所必需的代码。

禁止在正式代码中混入：

- unit test；
- integration test；
- toy case；
- sample case；
- mock；
- stub；
- debug-only implementation；
- self-test；
- KAT 测试框架；
- tamper test；
- negative test；
- regression test；
- microbenchmark；
- profiler harness；
- 临时性能测试；
- 临时 correctness check；
- 测试专用数据生成器；
- 为开发阶段保留的测试入口。


## 8.2 禁止通过隐藏开关把测试逻辑留在正式程序中

正式 runner、CLI 和源码不得保留仅服务于测试的模式，例如：

- `--toy`
- `--mock`
- `--self-test`
- `--test-mode`
- `--debug-proof`
- `--sample`
- `--microbench`
- 其他仅用于开发验证的特殊执行分支。

必要的正式输入校验、协议一致性检查、安全边界检查和 fail-closed 检查不属于测试代码，应保留在正式实现中。


## 8.3 正式代码不得残留历史、临时或开发阶段实现

正式生产路径中不得保留与当前正式实验无关的历史代码和开发残留，包括但不限于：

- `_debug` 实现；
- `_test` 实现；
- `_legacy` 实现；
- `_old` 实现；
- `_tmp` 实现；
- experimental implementation；
- 已废弃 fallback；
- 无调用 dead code；
- 注释掉的大段旧实现；
- 仅用于旧协议兼容的无效分支；
- 临时 instrumentation；
- 临时日志洪泛；
- profiler-only counter；
- 开发阶段临时 CLI 参数；
- 与当前正式路径无关的备用实现。

如果某个 telemetry、counter、audit field 或 runtime check 是正式实验结果记录、backend 真实性验证或 fail-closed 所必需，则可以保留；否则应删除。

对代码、脚本、配置、资源和历史实现执行强制清理原则：

> 一旦确认某个文件、模块、函数、脚本、配置、分支、资源或实现无引用、无实际用途、不会进入当前正式实验链，或与当前实验无关，应直接删除，不保留“以后可能有用”的冗余。

测试逻辑只有在仍具有明确测试价值时才允许保留，并必须移动到 `tests/`；没有测试价值的历史测试同样删除。

正式代码应保持单一、清晰、可追踪的生产执行路径，不保留无引用、无用途、实验无关、重复或历史冗余实现。


## 8.4 所有测试必须物理隔离

每个 baseline 必须建立独立测试目录：

- `comparison/zkGPT/tests/`
- `comparison/zkdl/tests/`
- `comparison/ezkl/tests/`

所有开发测试文件均应移动到对应 `tests/` 目录。

`tests/` 可以包含：

- unit tests；
- integration tests；
- correctness tests；
- tamper tests；
- negative tests；
- CPU/GPU consistency tests；
- KAT；
- regression tests；
- microbenchmarks；
- profiler drivers；
- test fixtures；
- synthetic inputs；
- test-only scripts；
- test-only helper code。


## 8.5 正式执行链不得依赖 tests

`tests/` 可以调用正式代码进行测试。

正式代码禁止依赖 `tests/`。

正式实验不得通过执行 `tests/` 中的程序来生成：

- 正式 statement；
- 正式 witness；
- 正式 proof；
- 正式 setup 参数；
- 正式 model registry；
- 正式 public output；
- 正式实验结果。

删除整个 `tests/` 目录后，正式 baseline 仍应能够正常构建，并具备由用户独立运行完整正式实验的能力。


## 8.6 正式结果 validator 不属于测试代码

用于判断正式实验结果是否合格的最终 validator 属于正式实验执行链，可以保留在正式 `scripts/` 或对应生产目录中。

其职责是审计正式产物，而不是执行开发测试。


---

# 9. 推荐目录边界

每个 baseline 应尽量形成清晰的生产/测试边界，例如：

```text
comparison/<baseline>/
├── src/                 # 正式源码
├── scripts/             # 正式 build/run/collect/validate 脚本
├── configs/             # 正式配置
├── data/                # 该 baseline 自己管理的正式数据资源
├── models/              # 该 baseline 自己管理的正式模型资源
├── resources/           # 静态表、SRS/PP、ceremony descriptor 等资源
├── artifacts/           # setup、registry、compiled objects 等正式产物
├── runs/                # 单次正式运行产物
├── results/             # 汇总后的正式结果
├── tests/               # 所有测试、调试、benchmark、profiling 代码
├── GAT-ZKML-*.md        # baseline 协议说明
└── ...
```

具体目录名称可以根据 baseline 原生工程结构调整，不要求三个 baseline 完全一致。

核心要求只有两个：

1. 正式代码与测试代码必须清楚分离；
2. baseline 的正式运行资源不得依赖其目录之外的项目内容。


---

# 10. 构建与正式运行入口

## 10.1 正式入口必须唯一且清晰

每个 baseline 应明确提供：

- 正式 build 入口；
- 单 dataset 正式运行入口；
- CPU/GPU backend 选择；
- 五数据集完整矩阵入口；
- 正式结果 collect 入口；
- 正式结果 validate 入口。

不得同时保留多个无法区分“哪个才是正式路径”的历史 runner。


## 10.2 历史执行代码不得污染正式路径

过时、废弃、旧协议、旧实验路径如果已确认不再被当前正式实验引用或使用，应直接删除。

禁止为了保留历史而继续存放无引用、无用途或与当前实验无关的代码、脚本、配置和资源。

只有仍具有明确测试价值的内容才允许移动到 `tests/`；其余无关历史内容全部删除。

不得让历史代码：

- 被正式构建；
- 被正式 runner 引用；
- 与当前正式路径形成歧义；
- 以 legacy/old/tmp/archive 等形式继续占据 production 工程。


## 10.3 正式运行不得要求先运行主实验或其他 baseline

每个 baseline 的正式入口必须从该 baseline 自己的输入和资源开始执行。

不允许出现：

> 先运行主实验 -> 生成某文件 -> baseline 再开始证明

也不允许：

> 先运行另一个 baseline -> 复用其产物 -> 当前 baseline 再运行


---

# 11. Setup、参数与资源管理

## 11.1 每个 baseline 自己管理自己的证明材料

例如但不限于：

- PP；
- SRS；
- proving key；
- verification key；
- compiled circuit；
- model registry；
- model commitment；
- ceremony descriptor；
- static tables。

上述内容不得由主实验提供。


## 11.2 setup 与正式 inference 必须区分

如果某一步属于一次性 setup，应单独记录，不得无解释地混入每次 inference 的 prove 时间。

根据 baseline 原生流程，可分别记录：

- setup；
- compile；
- model registration；
- commitment；
- witness generation；
- prove；
- verify。

最终论文使用哪些指标可以另行统一，但原始实验记录必须足以区分这些阶段。


---

# 12. 结果计量要求

## 12.1 正式结果至少应能够明确获得

每个正式实验至少应能够审计：

- dataset；
- backend；
- proof generated；
- verification status；
- commitment/preparation time（如适用）；
- prove time；
- verify time；
- proof size；
- statement identity；
- proof identity或digest；
- 关键 protocol/backend identity；
- 正式运行状态。


## 12.2 proof size 只统计 verifier-facing proof

`proof_size_bytes` 应表示 verifier 实际需要接收的 proof 数据大小。

不得将以下内容错误计入 proof size：

- raw witness；
- dataset；
- checkpoint；
- private trace；
- debug log；
- profiler output；
- test artifact。

同时，也不得为了缩小数字而排除 verifier 实际必须接收的 proof message。


## 12.3 计时口径必须稳定

同一 baseline 的五个数据集以及 CPU/GPU 结果应使用一致计时口径。

不得针对某个数据集单独改变计时边界以获得更好结果。


---

# 13. 正式验证与 fail-closed

## 13.1 每个正式结果必须真实验证

只有同时满足以下条件，单个正式 cell 才能记为成功：

- proof 真实生成；
- verifier 真实运行；
- verifier 返回成功；
- public statement 正确；
- public output 正确绑定；
- proof artifact 完整；
- backend 与请求一致；
- full GAT relation 被证明；
- verifier 未读取 witness。

单个 cell 的成功不以 5×2 `final validator` 通过为前置条件。

`final validator` 属于矩阵级验收：只有在用户本人完成 5 个数据集 × CPU/GPU 共 10 个正式 cell 后，才用于对完整正式矩阵进行最终审计。


## 13.2 禁止伪造成功状态

禁止：

- 手工写 `VERIFY_OK`；
- proof 缺失仍记为成功；
- verifier 未执行仍记为成功；
- verifier 失败后继续输出正式成功结果；
- GPU 回退 CPU 后仍记为 GPU；
- 输入缺失时自动使用 toy data；
- 证明范围不完整时仍记为 full GAT。


## 13.3 正式 validator 必须独立审计

每个 baseline 应有正式结果 validator。

`final validator` 是矩阵级正式验收工具，不是单个 cell 的成功判定前置条件。

validator 必须在用户完成全量正式实验后能够检查：

- 五个正式数据集；
- CPU/GPU 共 10 个结果；
- 每个 cell 唯一；
- proof 存在；
- proof digest/size 可核验；
- verifier 成功；
- statement identity 一致；
- full-dataset identity；
- full-GAT proof scope；
- backend identity；
- public output binding；
- verifier 不读取 witness；
- 不存在 silent fallback；
- 正式结果文件完整。

自动化开发阶段允许在 `tests/` 中使用专用 fixture 验证 result collector 和 final validator 的逻辑正确性，但必须满足：

- fixture 明确属于测试数据；
- fixture 只能位于测试范围内；
- fixture 不得写入 production `results/` 目录冒充正式结果；
- 不得伪造正式 proof、verification success 或 `VERIFY_OK`；
- 不得为了验证 collector 或 validator 而运行任何完整数据集；
- 对 5×2 正式矩阵的最终 validator 验收只能在用户本人完成全量实验后进行。


---

# 14. 正式产物边界

每个 baseline 的正式产物只能写入该 baseline 自己的目录体系。

建议：

- `artifacts/`
- `runs/`
- `results/`
- `logs/`（如需要）

禁止将正式实验产物写入：

- 主实验目录；
- 仓库根目录的主实验结果目录；
- 其他 baseline 目录；
- 依赖外部项目目录才能定位的临时路径。

正式脚本不得硬编码开发机器上的绝对项目路径。


---

# 15. 允许与禁止的修改

## 15.1 允许

允许进行：

- GAT 必要关系适配；
- baseline 原生 proof primitive 的必要组合；
- 数据/模型格式适配；
- 完整 CPU/GPU execution path 修复；
- correctness 修复；
- verifier trust boundary 修复；
- privacy boundary 修复；
- 独立运行所需的资源整理；
- 正式 runner 简化；
- 过时代码清理；
- 正式代码与 tests 物理分离；
- baseline 原生已有优化的正常启用；
- 阻塞完整实验的必要工程修复。


## 15.2 禁止

禁止进行：

- 主实验协议移植；
- 主实验专项性能优化移植；
- 非必要的主实验专项安全工程移植；
- 通过复制、改名、namespace/module 重命名、wrapper 或重新封装变相迁移主实验实现；
- 将 baseline 改造成主实验的另一种 backend；
- 为追求性能进行大规模非必要重构；
- 降低 proof scope；
- 降低数据规模；
- 改变统一 GAT 数值语义；
- 偷换公开/隐私边界；
- verifier 读取 witness；
- proof 携带原始 private witness；
- GPU silent fallback；
- 测试代码混入正式实现；
- baseline 之间共享核心证明实现；
- 正式运行依赖 baseline 目录外的项目资源；
- 自动化开发阶段运行任何完整数据集正式实验；
- 通过拆分、分阶段或其他方式变相运行完整数据集正式实验；
- 保留已确认无引用、无用途或与正式实验无关的冗余代码和文件。


---

# 16. 开发流程约束

后续修改一个 baseline 时，应遵循：

1. 一次只修改一个 baseline；
2. 其余两个 baseline 视为只读；
3. 主实验视为只读；
4. 先理解该 baseline 原生方法；
5. 只识别完成 GAT 所需的缺口；
6. 进行最小必要修改；
7. 测试代码全部放入该 baseline 的 `tests/`；
8. 正式代码中不得残留测试/调试实现；
9. 使用小规模测试和局部正式关系回归验证正式实现；
10. 正式实验不依赖 tests；
11. 形成完整正式执行链；
12. 完成正式 validator 的实现，并使用 `tests/` 中专用 fixture 验证其逻辑；
13. 不运行任何完整数据集正式实验；
14. 将代码准备到可由用户直接运行全量 CPU/GPU 正式矩阵的状态；
15. 满足实现完成标准后停止无必要修改，等待用户执行全量实验。


---

# 17. 规范层级与实现事实判断

不得把“当前代码是什么样”与“代码应该是什么样”混为一谈。

## 17.1 规范层级

工程与执行边界按以下顺序解释：

1. 本文件 `BASELINE_EXECUTION_CONSTRAINTS.md`：规定独立性、自包含、最小适配、正式代码洁净度、运行边界和完成标准；
2. 各 baseline 对应的 `GAT-ZKML-*.md`：规定该 baseline 应证明的 GAT 关系、密码学组件、公开/隐私边界和协议语义；
3. baseline 原论文、原项目和原生实现：用于判断其方法学身份、原生组件和原生优化是否被忠实保留。

如果当前实现与上述规范冲突，当前实现应被视为待修复对象，而不能因为“代码现在就是这样”就自动成为正确标准。


## 17.2 当前实现事实的判断依据

当需要判断“当前 baseline 实际实现到了什么程度”时，优先检查：

1. 当前正式源码；
2. 当前正式 runner；
3. 当前 verifier；
4. 当前正式 validator；
5. 当前正式构建配置；
6. 当前正式生成的实验产物；
7. README、代码注释和历史说明。

README、旧注释、历史实验记录只能作为辅助信息，不得覆盖当前正式代码事实，也不得覆盖本文件和协议规范。


## 17.3 冲突处理原则

如果出现冲突：

- 本文件与 README 冲突：以本文件为准；
- 协议说明与 README 冲突：以协议说明为准；
- 当前代码与本文件冲突：修改当前代码；
- 当前代码与协议说明冲突：先确认 baseline 原方法和统一 GAT 语义，再修复实现；
- 原论文/原实现与主实验专项设计冲突：baseline 优先保持自身方法学身份，不迁移主实验设计。

不得通过修改规范文件来迁就明显错误或不完整的当前实现。


# 18. 完成标准

本文件区分“baseline 实现完成”和“baseline 最终正式实验完成”。

自动化开发代理只负责达到“baseline 实现完成”；完整数据集正式实验只能由用户本人运行。


## 18.1 Baseline 实现完成标准

一个 baseline 只有同时满足以下全部条件，才视为实现完成，并可交由用户运行全量正式实验。

### 独立性

- [ ] baseline 可在自身目录中独立构建；
- [ ] baseline 正式 runner 可从自身目录启动；
- [ ] 不依赖主实验；
- [ ] 不依赖其他 baseline；
- [ ] 不依赖 baseline 目录之外的项目运行资源；
- [ ] 不存在隐藏绝对路径或软链接依赖。

### 方法忠实性

- [ ] 保留 baseline 原生证明体系；
- [ ] 只做 GAT 必要适配；
- [ ] 未迁移主实验专项协议；
- [ ] 未迁移主实验专项性能优化；
- [ ] 未通过复制、改名、封装等方式变相迁移主实验实现；
- [ ] 未迁移非必要的主实验专项安全工程；
- [ ] 未进行无必要的大规模性能重构。

### GAT 正确性

- [ ] 正式 relation 实现统一完整 GAT 数值语义；
- [ ] 正式 runner 配置为五个完整正式数据集；
- [ ] proof path 设计覆盖完整 GAT forward；
- [ ] 最终 logits 由 proof relation 绑定；
- [ ] CPU/GPU 使用相同证明关系；
- [ ] 已通过允许范围内的小规模/局部正确性验证。

### 隐私与验证

- [ ] model/witness/trace 隐私边界正确；
- [ ] verifier 不读取 witness；
- [ ] proof 不包含原始 private witness；
- [ ] public output 由 verifier 独立获得并验证；
- [ ] verifier 可独立运行。

### CPU/GPU

- [ ] CPU 正式路径已实现并通过允许范围内的局部验证；
- [ ] GPU 正式路径已实现并通过允许范围内的局部验证；
- [ ] GPU 无 silent fallback；
- [ ] 正式 runner 已支持五数据集 CPU/GPU 全矩阵，但未由自动化开发代理执行全量数据集。

### 正式代码洁净度

- [ ] 正式源码无测试代码；
- [ ] 正式 runner 无测试模式；
- [ ] 所有测试位于独立 `tests/`；
- [ ] 正式执行链不依赖 `tests/`；
- [ ] 所有无引用、无用途、与实验无关的文件和实现已删除；
- [ ] 过时正式入口已删除；
- [ ] 不存在 `_debug`、`_legacy`、`_old`、`_tmp` 等历史/临时实现污染；
- [ ] 不存在大段注释旧代码、dead code 或无效 fallback；
- [ ] 不存在仅服务开发阶段的 profiling/instrumentation 残留；
- [ ] 正式生产路径单一、清晰、无歧义。

### 正式运行与结果工具

- [ ] 正式 build 入口明确；
- [ ] 正式单数据集 runner 明确；
- [ ] 正式 CPU/GPU backend 选择明确；
- [ ] 正式五数据集全矩阵 runner 已准备；
- [ ] result collector 已准备，并已通过 `tests/` 中专用 fixture 的逻辑验证；
- [ ] final validator 已准备，并已通过 `tests/` 中专用 fixture 的逻辑验证；
- [ ] 测试 fixture 未写入 production `results/`，且未伪造正式 proof/verification 状态；
- [ ] proof size 和 timing 口径已定义；
- [ ] 全量实验所需 baseline-local 正式资源已准备或有明确生成入口。

达到以上条件后，应停止自动化开发阶段，不得继续运行任何完整数据集正式实验。


## 18.2 最终正式实验完成标准

以下步骤只能由用户本人执行。

用户完成正式：

`5 datasets × 2 backends = 10 results`

之后，一个 baseline 最终正式实验完成还需要满足：

- [ ] 5 个 CPU 完整数据集结果齐全；
- [ ] 5 个 GPU 完整数据集结果齐全；
- [ ] 每个 cell 的 proof 真实生成；
- [ ] 每个 cell 的 independent verifier 真实通过；
- [ ] proof size 可核验；
- [ ] timing 口径统一；
- [ ] full dataset identity 正确；
- [ ] full-GAT proof scope 正确；
- [ ] GPU backend 真实且无 silent fallback；
- [ ] 所有正式结果保存在 baseline 自己目录；
- [ ] final validator 对 10 个结果全部通过。

自动化开发代理不得通过自行运行全量数据集来完成本节检查。


# 19. 最终原则

所有后续开发和判断都应服从以下原则：

> 对比实验的目标是公平、可信地比较不同证明体系，而不是把三个 baseline 改造成主实验的三个变体。

> 每个 baseline 必须保持自身方法学身份，只完成统一 GAT 推理任务所必需的最小适配。

> 每个 baseline 必须目录级自包含，能够独立构建、独立运行、独立证明、独立验证并独立产出结果，不依赖主实验、其他 baseline 或 baseline 目录之外的项目运行资源。

> 主实验专项证明机制和专项性能优化不得迁移到 baseline；baseline 自己原生已有的优化可以正常保留。不得通过复制、改名、重新封装等方式绕过这一边界。

> 不得把主实验的专项安全工程整体迁移到 baseline。只保留 baseline 原生安全机制，并补齐完成可信正式 GAT proof 所必需的最小安全边界。

> 正式实验代码必须绝对干净、简洁。任何测试、调试、benchmark、profiling、toy、mock、KAT、tamper 或回归测试代码都必须物理隔离到独立 `tests/` 目录，并且正式实验执行链不得依赖测试代码；历史、临时、dead code 和无效 fallback 同样不得残留在正式生产路径。

> 自动化开发阶段禁止运行任何完整数据集正式实验。开发代理只负责将 baseline 实现、正式执行链和 validator 准备到可由用户直接运行的状态；五数据集 CPU/GPU 共 10 组长耗时正式结果只能由用户本人运行。

> 一旦确认代码、文件、脚本、配置或资源无引用、无实际用途或与当前正式实验无关，应直接删除；不保留无价值的 legacy、old、tmp、archive 或“以后可能有用”的冗余。

> 当实现完成标准全部满足后，应停止自动化开发和全量执行，等待用户运行正式矩阵。用户完成 10 组正式实验并通过 final validator 后，该 baseline 才视为最终正式实验完成。
