# 对比实验统一执行约束

> 适用范围：`comparison/ezkl/`、`comparison/zkdl/`、`comparison/zkGPT/` 三套独立 GAT-ZKML 对比实验。  
> 文件定位：本文件规定三个 baseline 长期有效的工作区边界、实现原则、主实验引用权限、方法学独立性、正式代码结构、测试隔离、结果目录、验证边界和交付要求。  
> 本文件不记录某一轮临时错误、临时性能数字、失败尝试或执行流水。  
> 本文件默认只读；只有项目负责人明确要求修改时才允许修改。

---

## 0. 规范层级与解释原则

执行任一 baseline 修改时，规范优先级固定为：

1. 项目负责人当前任务中明确、具体的要求；
2. 当前 baseline 根目录中的 `GAT-ZKML-*.md` 协议笔记；
3. 本文件；
4. 当前 baseline 已有代码、原项目结构、原生 proof backend、原生优化与原有构建方式；
5. 主实验内容只作为只读参考和必要通用内容的复制来源。

其中：

- 协议笔记决定“证明什么、公开什么、隐藏什么、怎样证明，以及什么情况下必须标记 unsupported”；
- 本文件决定“在哪里修改、允许借用什么、怎样保持 baseline 独立、正式代码与测试如何隔离、结果写到哪里、做到什么程度应停止”；
- 当前 baseline 已有且符合协议的实现应优先于重新实现；
- 主实验是只读参考资源，不是 baseline 的默认实现模板。

一般性目标，例如“让实验跑通”“完成实现”“修复问题”“做最小优化”，不自动构成突破本文件硬边界的授权。

只有项目负责人明确指出允许突破的具体规则、具体路径或具体行为时，才视为有效例外。

若协议笔记、当前 baseline 能力与本文件之间出现无法自行解决的真实冲突，必须停止冲突部分并报告，不得自行修改协议、主实验或本文件来消除冲突。

---

## 1. 项目目标与当前阶段

三个 baseline 的目标是形成三套：

- 彼此独立；
- 可单独构建；
- 可单独运行；
- 忠实实现各自协议笔记；
- 保留各自原生证明体系；
- 有各自独立的 README；
- 有各自独立的正式结果目录；
- 可用于与主实验进行方法学公平比较

的 GAT-ZKML 对比实验。

当前阶段不是对三个 baseline 开展系统级性能重构，而是：

> **以当前 baseline 为主体，完成协议要求下的最小可运行适配，并把正式实验入口准备到可直接执行的状态。**

当前工作的优先级固定为：

1. 忠实协议；
2. 正式代码结构清晰；
3. 真实证明与真实验证路径完整；
4. baseline 保持独立；
5. 修改范围尽可能小；
6. 保留 baseline 自己已有的原生优化；
7. 不主动把主实验的方法或优化迁移进来；
8. 不把测试、调试或临时代码混入正式代码目录；
9. 在已经满足要求后及时停止修改。

“最小可运行修改”是指：

> 在不改变协议语义、不削弱证明、不改变方法学身份的前提下，优先复用当前 baseline 已有能力，只补充使其忠实实现协议笔记并能够独立运行所必需的最小实现。

“最小”描述的是修改范围，不描述证明强度、安全强度或验证强度。

---

## 2. 当前目录结构与唯一可写目录

三个 baseline 的正式根目录固定为：

| baseline | 根目录 | 协议笔记 |
|---|---|---|
| ezkl | `comparison/ezkl/` | `comparison/ezkl/GAT-ZKML-ezkl.md` |
| zkDL | `comparison/zkdl/` | `comparison/zkdl/GAT-ZKML-zkDL.md` |
| zkGPT | `comparison/zkGPT/` | `comparison/zkGPT/GAT-ZKML-zkGPT.md` |

### 2.1 一次只修改一个 baseline

默认情况下，一次任务只允许选择一个 baseline 作为当前目标。

修改 ezkl 时，唯一允许写入的项目目录是：

`comparison/ezkl/**`

修改 zkDL 时，唯一允许写入的项目目录是：

`comparison/zkdl/**`

修改 zkGPT 时，唯一允许写入的项目目录是：

`comparison/zkGPT/**`

除非项目负责人明确要求修改本文件，否则：

- 主实验全部内容只读；
- 另两个 baseline 目录只读；
- `comparison/BASELINE_EXECUTION_CONSTRAINTS.md` 只读；
- 当前 baseline 之外的其他项目内容只读。

### 2.2 禁止越界写入

处理某个 baseline 时，禁止在当前 baseline 根目录之外：

- 修改文件；
- 新建文件；
- 删除文件；
- 移动或重命名文件；
- 自动格式化后写回；
- 生成新的项目源码、配置或脚本；
- 修改构建入口；
- 为当前 baseline 增加主实验侧适配层；
- 为当前 baseline 改写另一个实验目录。

工具链自身使用的系统临时目录、编译缓存和包缓存不属于项目写入范围，但不得成为 baseline 的隐式运行依赖。

---

## 3. 主实验永久只读

修改任一 baseline 时，主实验内容永久视为只读参考。

允许：

- 阅读；
- 搜索；
- 分析；
- 对照；
- 查看主实验的 GAT 数值语义；
- 查看数据和模型输入方式；
- 查看配置和公共工具；
- 在确有必要时，把最小必要的通用内容复制到当前 baseline 内。

禁止：

- 修改主实验源码；
- 修改主实验头文件；
- 修改主实验配置；
- 修改主实验脚本；
- 修改主实验测试；
- 修改主实验构建文件；
- 为 baseline 给主实验增加接口；
- 为 baseline 给主实验增加条件分支；
- 为 baseline 创建主实验侧 adapter；
- 为 baseline 改写主实验已有执行路径。

唯一允许的借用模式是：

`主实验只读内容 -> 复制到当前 baseline -> 只修改 baseline 内副本`

绝不允许：

`修改主实验 -> 让 baseline 使用修改后的主实验`

---

## 4. 禁止对主实验形成隐式运行依赖

即使主实验文件没有被修改，也不得通过构建或运行时引用规避目录隔离。

当前 baseline 不得：

- 通过相对路径直接把主实验源码编译进来；
- 直接链接主实验生成的库或对象；
- 调用主实验可执行程序替代 baseline 的证明或验证；
- 依赖主实验运行后产生的中间证明结果、临时结果或缓存；
- 使用符号链接或其他路径映射，把主实验源文件伪装成 baseline 本地文件；
- 通过环境变量隐藏对主实验源码或构建产物的依赖；
- 依靠仓库根级构建逻辑注入主实验实现；
- 依赖主实验的 Python 虚拟环境作为 baseline 唯一可运行环境；
- 要求先运行主实验才能得到 baseline 所需的小型配置、证明输入或中间文件。

baseline 所需的源码、证明逻辑、配置、脚本和小型必要资源应位于自身目录中。

对于体积很大的公共只读数据集、模型输入、checkpoint 或第三方资源，可以作为明确的外部输入使用，不要求机械复制；但不得要求这些外部输入先经过主实验程序修改或生成后才能运行 baseline。

---

## 5. 协议笔记是只读技术规范

当前 baseline 的 `GAT-ZKML-*.md` 是该实验的最高技术规范之一。

修改前必须完整阅读对应协议笔记。

协议笔记默认只读，不得为了迁就现有代码、减少工作量、提高性能或绕过 backend 限制而修改。

当实现与笔记冲突时：

1. 优先修正实现；
2. 如果 baseline 原生体系能够忠实补充，则做最小补充；
3. 如果确实无法忠实支持，则按笔记定义显式报告 unsupported；
4. 不得修改笔记把“不支持”重新定义为“支持”。

特别禁止为了跑通而改变：

- GAT 层数语义；
- attention head 语义；
- 隐藏层 concat；
- 输出 attention head；
- 输出 bias 位置；
- 最终多头平均；
- stable Softmax；
- 定点尺度链；
- `SignedRoundDiv`；
- rounding mode；
- 静态表语义；
- 图结构和 batch 语义；
- 公开/私有输入边界；
- 模型绑定方式；
- 最终公开输出；
- 协议笔记规定的安全或隐私边界。

---

## 6. Baseline-native first

每个需要的能力都必须按固定优先级处理。

### 6.1 当前 baseline 已有实现

如果当前 baseline 已经存在满足协议要求的实现：

- 直接复用；
- 不重写；
- 不从主实验复制替代版本；
- 不仅因为主实验更快、更成熟或更熟悉而替换。

### 6.2 经代码级协议检查确认满足要求后直接停止扩展

只有在完整阅读当前协议笔记，并对当前正式代码进行代码级协议覆盖检查后，确认当前实现忠实覆盖协议要求，才能认定“当前 baseline 已经满足要求”。

至少应确认当前正式代码具备：

- 独立构建能力；
- 协议要求的正式 proof generation 路径；
- 协议要求的正式 verification 路径；
- 目标 GAT 实验运行入口；
- 正式结果输出路径；
- 与协议一致的 README；
- 协议要求的关键 relation、公开/私有边界、安全/隐私条件没有被旧实现遗漏或弱化。

以下内容均不能单独作为“已经满足要求、无需修改”的依据：

- 旧 README 中的成功描述；
- 旧 `verified` / `VERIFY_OK` 状态；
- 旧 benchmark 数字；
- 旧结果文件；
- 旧运行日志；
- 旧 runner 曾经能够启动或结束；
- 历史测试曾经通过。

只有协议笔记与当前正式代码逐项对照后确实不存在实现缺口，才应停止继续修改生产实现。

此时应：

- 保留现有正确实现；
- 不从主实验复制任何新方法；
- 不主动增加新的性能优化；
- 只修正与当前要求直接相关的结构、路径或说明问题；
- 报告当前实现已经具备正式运行条件。

不得仅因为“还能更快”“还能重构”“还能统一代码”而继续修改。

### 6.3 baseline 原生机制的最小扩展

如果缺少某个协议必需能力，但可以利用 baseline 自己的原生 proof system、backend 或原项目机制自然实现，则只做该最小扩展。

### 6.4 通用基础设施才允许参考主实验

只有在 baseline 自身确实缺少通用的 GAT/data/config/preprocessing/serialization 基础设施时，才允许参考并复制主实验中最小必要内容。

### 6.5 无法忠实支持时显式 unsupported

如果继续实现必须：

- 引入主实验专有证明方法；
- 改变 baseline 方法学身份；
- 弱化 relation；
- 弱化安全或隐私；
- 改变协议笔记定义的任务；

则不得继续导入该方法，应按协议笔记显式报告：

- `unsupported_exact_relation`；
- `unsupported_privacy_profile`；
- 或协议笔记定义的其他 unsupported 状态。

禁止 silent fallback。

---

## 7. 允许从主实验复制的内容

只有同时满足以下条件时才允许复制：

1. 当前 baseline 自身确实没有等价能力；
2. 协议实现确实需要；
3. 该内容不属于主实验专有证明方法或专有优化；
4. 复制后 baseline 仍然可以独立运行；
5. 只复制最小必要部分。

优先允许的类别包括：

- GAT 数据加载；
- 图数据解析；
- 模型权重读取；
- 与证明系统无关的 GAT 明文 forward/reference 逻辑；
- 必须一致的 graph preprocessing；
- 节点/边编号和 batch 预处理；
- 量化配置解析或生成；
- 静态表通用生成逻辑；
- 输入/输出序列化；
- 数据集配置；
- 模型结构配置；
- 实验输入准备；
- 与密码学后端无关的 utility；
- 为保持相同 GAT 数值语义所必需的通用参考实现。

复制时必须：

- 先确认 baseline 没有等价实现；
- 复制到当前 baseline 内；
- 只修改 baseline 内副本；
- 删除对主实验路径的运行依赖；
- 保留必要的许可证和版权信息；
- 最终说明复制了什么、从哪个源路径复制、为什么必须复制。

如果拟复制内容可能涉及主实验专有证明方法或专有性能优化，应停止复制并等待项目负责人判断。

不得通过路径映射、链接或其他方式规避上述要求。

---

## 8. 禁止迁移主实验专有证明方法和优化

主实验只用于提供只读参考，不用于把主实验的优化体系反向套到 baseline。

除非某项机制本来就是对应 baseline 原论文、原项目或原 backend 的自然组成部分，否则禁止从主实验迁入：

- 主实验专有 KZG 证明结构；
- 主实验 LogUp 组织；
- PSQ；
- CRPC；
- 主实验专用 commitment aggregation；
- 主实验专用 opening aggregation；
- 主实验 proof-local reblinding 体系；
- 主实验专用工作域布局；
- 主实验专用 Lagrange-SRS 热路径；
- 主实验专用 tensor-binding/route 结构；
- 主实验专用批量 MSM/NTT 组织；
- 主实验专用缓存体系；
- 主实验专用 CPU worker/NUMA 调度；
- 主实验专用 GPU kernel；
- 主实验专用 CUDA batching/streaming/buffer reuse；
- 主实验专用 CPU/GPU shape-aware routing；
- 主实验特有的 proof/constraint fusion；
- 为主实验专门设计的跨阶段性能重排；
- 其他明显改变 baseline 原算法、原证明体系或复杂度特征的主实验优化。

如果名称相同的技术本来就在 baseline 中存在，应继续使用 baseline 自己的实现，而不是复制主实验版本。

---

## 9. 不得人为削弱或人为做慢 baseline

方法学独立不意味着故意降低 baseline 性能。

必须保留并正常使用当前 baseline 已经具备的：

- 原生 proof backend；
- 原生 lookup；
- 原生 Sumcheck/GKR/Lasso/FAC4DNN 等机制；
- 原生 batching；
- 原生编译优化；
- 原生多线程；
- 原生 GPU 支持；
- 原生缓存；
- 原生 constraint fusion/circuit squeeze；
- 原项目默认且安全的有效性能路径。

不得为了“公平”而关闭 baseline 自己已有的原生正常优化。

也不得回退当前 baseline 已存在且正确有效的实现，除非它与协议笔记冲突或存在真实正确性/安全问题。

当前阶段不要求主动寻找新的大规模性能优化；只要协议路径正确、完整并具备正式运行条件，就应停止扩展。

---

## 10. 禁止伪运行、弱证明和安全降级

“可运行”必须指代码准备的是正式真实协议路径，不是表面返回成功。

正式实现禁止：

- mock proof；
- fake proof；
- fake verifier；
- verifier 无条件返回成功；
- 固定 Fiat–Shamir challenge；
- 把测试随机性直接当作正式 proving randomness；
- 只做明文重新计算而不执行密码学验证；
- 跳过 commitment；
- 跳过 opening/evaluation proof；
- 跳过 Sumcheck/GKR/Lasso/lookup 的必要检查；
- 只证明部分层；
- 只证明部分 attention head；
- 用 host-side assertion 代替证明约束；
- 利用已知 trapdoor 绕过正常证明流程；
- 为了运行或测速关闭零知识、binding 或必要验证后仍称为正式 profile；
- 把 unsupported profile 自动切换成更弱 profile；
- 复用本应 fresh 的 proof-specific randomness 或 proof；
- 用 replay proof 代替新 proof；
- 通过删除 verifier check 获得性能提升；
- 用预计算 logits 或中间结果绕过正式 GAT 推导；
- 用测试 fixture 作为正式证明结果。

测试专用固定随机输入只能存在于明确的测试路径，不得进入正式 proving path。

任何能力、安全或隐私降级都必须显式失败、显式标记或显式报告。

---

## 11. 三个 baseline 的方法学身份

### 11.1 ezkl

`comparison/ezkl/` 必须继续以 ezkl/Halo2 原生体系为主体。

如果 ezkl 自身 gate、lookup、Halo2 argument 和现有编译路径能够完成某项关系，应直接使用。

不得为了实现 GAT 而把主实验专有证明体系迁入。

### 11.2 zkDL

`comparison/zkdl/` 必须继续以 zkDL-style 的 MLE、Sumcheck、Pedersen/Hyrax、FAC4DNN 和其原生 evaluation 体系为主体。

不得用主实验证明框架替代这些机制。

### 11.3 zkGPT

`comparison/zkGPT/` 必须继续以 zkGPT 原生 GKR、Lasso、Hyrax、矩阵乘 Sumcheck、advice、constraint fusion/circuit squeeze 等机制为主体。

不得以主实验证明框架替代这些机制。

具体安全边界、adaptation、repair 和 unsupported 条件，以各自 `GAT-ZKML-*.md` 为准。

---

## 12. 代码修改必须最小、局部

除协议实现、目录修正和独立运行确有必要外，不进行与当前任务无关的工程重构。

默认禁止：

- 大规模目录重组；
- 全项目重命名；
- 全文件格式化；
- 大面积无关注释更新；
- 顺手清理无关技术债；
- 替换整个构建系统；
- 大范围升级依赖；
- 更换密码学库；
- 更换曲线或有限域；
- 重写已经可用的模块；
- 为统一三个 baseline 而新建公共 framework；
- 为代码风格一致而让三个 baseline 共享新的实现层；
- 仅为了追求进一步性能而开展与“跑通协议”无关的大范围优化。

依赖和工具链调整必须满足：

- 只发生在当前 baseline 内；
- 只处理运行当前协议所必需的问题；
- 不升级无关依赖；
- 不因为环境问题而改变协议；
- 不因为“新版更方便”而整体替换当前 baseline 框架。

如果外部依赖、编译器、CUDA 或第三方组件缺失，应作为环境问题报告；不得通过弱化协议或假实现绕过。

---

## 13. 正式代码、测试代码与实验产物必须严格隔离

这是三个 baseline 的硬结构约束。

### 13.1 正式目录只放正式实验内容

正式源码、正式证明后端、正式运行脚本、正式配置和正式结果目录中，只允许存在正式实验运行所需要的内容。

典型正式目录包括：

- `src/`
- `backend/`
- `scripts/`
- `configs/`
- 正式构建文件和构建配置
- `results/`

这些位置不得混入仅用于测试、调试、负例或临时验证的实现。

### 13.2 测试内容必须集中到独立 `tests/`

如果当前 baseline 存在测试、negative、debug、fixture、fault-injection、测试 helper 或其他仅用于开发验证的内容，则必须集中放入当前 baseline 自己的 `tests/`：

- `comparison/ezkl/tests/`
- `comparison/zkdl/tests/`
- `comparison/zkGPT/tests/`

如果当前 baseline 完全不存在此类测试内容，则不要求为了形式完整创建空的 `tests/` 目录。

所有仅用于以下目的的代码或文件必须进入当前 baseline 的 `tests/`：

- 单元测试；
- 集成测试；
- negative test；
- fault injection；
- mock 输入；
- 测试 fixture；
- test helper；
- test runner；
- 调试专用 runner；
- 测试数据生成；
- 临时 assertion 检查；
- 协议实现的测试性篡改；
- proof/public input/witness 的故意破坏测试；
- 仅用于测试的 benchmark helper；
- 仅用于开发验证的诊断脚本；
- 测试输出解析；
- 临时测试日志。

如果确有需要，可以在 `tests/` 下按实际内容使用：

```text
tests/
  unit/
  integration/
  negative/
  fixtures/
  helpers/
  tmp/
```

没有对应内容时不要为了形式完整创建空的复杂层级。

### 13.3 正式目录禁止出现测试专用实现

正式 `src/`、`backend/`、`scripts/`、`configs/` 和正式构建入口中禁止保留：

- `test_*` 或 `*_test` 类型的测试 runner；
- negative-test runner；
- debug-only runner；
- mock prover / mock verifier；
- fault-injection 代码；
- 专门篡改 proof 的工具；
- 临时诊断脚本；
- 测试 fixture；
- 仅用于本轮开发的 helper；
- 测试结果文件；
- 测试日志。

如果当前正式目录中已有纯测试、负例或调试内容，应在不改变正式实验语义的前提下移动到当前 baseline 的 `tests/`。

纯历史、失效、无执行价值且不再服务正式实现或当前有效测试的代码，必须从当前正式实验工程中移除。

处理原则固定为：

- 正式运行必需的代码 -> 保留在正式目录；
- 测试、negative、debug、fixture、fault-injection 内容 -> 移入当前 baseline 的 `tests/`；
- 已失效 adapter、旧实现备份、历史 debug runner、废弃实验实现、无执行价值的 archive 代码 -> 删除；
- 不允许仅因为“以后可能参考”就在当前正式实验工程中长期保留 `archive/invalid_*`、`old_*`、`deprecated_*`、`legacy_debug_*` 等失效代码目录。

正常的 README、LICENSE、协议笔记以及仍有正式用途的设计文档不属于上述“失效代码”，不得为了清理代码而误删。

### 13.4 正式代码不得依赖 `tests/`

正式 build / prepare / prove / verify / run / result-export 路径不得：

- import `tests/` 中的模块；
- include `tests/` 中的头文件；
- source `tests/` 中的脚本；
- 读取测试 fixture 作为正式输入；
- 依赖测试运行后产生的中间文件。

原则上删除整个 `tests/` 目录后，正式 baseline 仍应具备完整的构建和正式实验执行路径。

### 13.5 正式协议检查不是测试代码

不得为了目录整洁把真正属于协议的验证逻辑错误移动到 `tests/`。

判断标准固定为：

- 正式 proof / verification 必须执行的逻辑 -> 正式代码；
- 只用于判断正式代码是否正确的逻辑 -> `tests/`。

例如，正式 verifier 的约束检查、commitment/opening verification、Sumcheck/GKR/Lasso verification、正式 public-input binding 都属于正式代码。

### 13.6 测试产物不得写入正式结果目录

测试运行产生的：

- 临时 proof；
- 被篡改的 proof；
- negative-test 输出；
- 测试日志；
- fixture 输出；
- debug dump；
- 临时 profile；

必须写入 `tests/tmp/` 或其他明确的测试目录。

不得写入当前 baseline 的任何正式结果目录或正式运行 artifact 目录。

### 13.7 构建系统中的测试 target 与内嵌测试代码

正式构建配置可以声明指向 `tests/` 的可选测试 target，但必须满足：

- 测试 target 不是默认正式 build 的必需组成；
- 测试 target 不得被正式 run / prove / verify 路径自动触发；
- 正式代码不得依赖测试 target 的输出；
- 测试源码本身必须位于当前 baseline 的 `tests/`；
- 删除或不构建测试 target 时，正式 build / run / prove / verify 路径仍然成立。

对于当前正式源码中已经存在的纯测试模块或测试专用入口，如果它们只服务测试，应移出正式源码并放入 `tests/`。典型情况包括：

- Rust 中只用于测试的 `#[cfg(test)]` 模块；
- Python 正式模块中只用于测试的 helper、fixture 或测试入口；
- C/C++/CUDA 正式源码目录中的测试专用 `main`；
- debug self-test；
- negative/fault-injection 专用逻辑。

但是，真正属于正式协议或正式 verifier 的检查不得因为名称中含有 `check`、`self-check`、`verify` 等字样就被误判为测试代码。

判断标准仍然是：

- 正式证明或验证必须执行的逻辑 -> 保留在正式代码；
- 仅用于验证正式代码是否正确的逻辑 -> 放入 `tests/`。

---


## 14. `scripts/` 只保留正式实验脚本

每个 baseline 的正式 `scripts/` 目录只用于正式实验生命周期。

允许保留的正式脚本包括：

- build；
- dependency/setup preparation；
- 正式输入准备；
- 单 dataset/backend 的正式运行入口；
- prove；
- verify；
- 正式全量运行入口；
- 正式结果收集；
- 正式结果导出；
- 与正式运行直接相关的环境检查。

不属于正式实验生命周期的测试、negative、debug、fault-injection、临时诊断脚本必须移入 `tests/`。

正式全量 runner 可以保留，因为它属于正式实验入口；但普通代码修改阶段不得因为它存在就自动执行全量实验。

正式 build 脚本不得默认自动触发大规模测试、negative suite 或长时间 benchmark。

---

## 15. 每个 baseline 必须有独立 README

以下文件必须存在：

- `comparison/ezkl/README.md`
- `comparison/zkdl/README.md`
- `comparison/zkGPT/README.md`

README 必须描述当前正式实现，而不是历史实现。

每份 README 至少包含：

1. baseline 的用途和方法学身份；
2. 当前协议笔记的正确路径；
3. 正式目录结构；
4. 主要依赖和环境要求；
5. 构建方式；
6. 单 dataset/backend 的正式运行方式；
7. 正式 proof generation / verification 路径，以及它们是独立命令还是由端到端 runner 统一完成；
8. 正式全量实验入口；
9. 正式结果目录；
10. CPU/GPU 支持范围；
11. 已知环境限制；
12. 协议要求下存在的 `unsupported_*` 状态；
13. 如存在测试内容，说明测试入口以及它们在当前 baseline `tests/` 中的位置；
14. 当前代码是否只是“已准备待验证”，还是已经由项目负责人完成正式实验验证。

### 15.1 README 不得保留过时成功声明

当前代码被重新修改后，旧实现产生的：

- `verified`；
- `VERIFY_OK`；
- “全部数据集已通过”；
- 旧 benchmark 数字；
- 旧完整运行状态；

不得继续被描述成“当前修改后代码已经验证成功”。

如果历史结果需要保留，应明确标记为历史材料，不作为当前实现的验证依据。

在项目负责人重新完成正式运行之前，README 应使用准确表述，例如：

> 当前正式实现已经按协议完成代码适配并准备好运行；正式实验结果等待项目负责人统一执行后更新。

不得提前宣称当前代码已经完成正式验证。

---

## 16. 每个 baseline 必须在自身目录内保存正式结果

三个 baseline 的正式结果、正式运行 artifact 和结果汇总文件必须完全保存在各自实验根目录内部。

不强制三个 baseline 使用相同的结果目录名称或相同的内部层级。应优先保留当前 baseline 自己清晰、合理且仍然有效的原生结果组织，不得仅为了目录名称统一进行无意义重构。

例如：

- ezkl 可以使用 `comparison/ezkl/results/`；
- zkGPT 可以使用 `comparison/zkGPT/results/`；
- zkDL 可以继续使用 `comparison/zkdl/zkdl_gat_outputs/results/`，或其他完全位于 `comparison/zkdl/` 内、且由当前正式代码明确管理的结果结构。

硬性要求只有：

- 所有正式结果和正式运行 artifact 均位于当前 baseline 根目录内部；
- 当前 baseline 的 README 明确写出正式结果位置；
- 不把结果写入主实验或另一个 baseline；
- 不依赖 baseline 根目录之外的结果目录才能完成正式运行；
- 不为了表面统一结果路径而破坏当前 baseline 原有清晰结构。

### 16.1 正式结果与运行中间产物

每个 baseline 可以在自身根目录内保留正式运行所需的中间 artifact 目录，例如：

- `runs/`
- `artifacts/`
- `outputs/`
- `zkdl_gat_outputs/`
- 其他当前 baseline 原生且明确的正式输出目录。

这些目录必须：

- 位于当前 baseline 内；
- 只服务当前 baseline；
- 不读取其他 baseline 的结果；
- 不读取主实验证明结果；
- 不与测试产物混用。

最终用于论文或对比汇总的 baseline 结果必须能够由当前 baseline 自己的正式运行产物生成，并保存在当前 baseline 根目录内部的明确正式结果位置。

### 16.2 旧结果处理

旧正式结果不得被修改成新的数字来冒充当前结果。

如果保留旧结果：

- 必须与新正式结果区分；
- 不得由 README 将其描述为当前修改后实现的验证证据；
- 不得被当前正式 validator 自动当作新运行成功；
- 后续项目负责人重新正式运行后，再根据新的正式运行结果更新当前状态。

测试结果永远不得进入任何正式结果目录。

---

## 17. 独立构建与独立运行

每个 baseline 最终都必须能够作为自己的实验工程运行。

至少满足：

- 有自己的构建配置；
- 有自己的源码和 proof implementation；
- 有自己的必要配置；
- 有自己的正式运行入口；
- 有自己的 proof generation 路径；
- 有自己的 verification 路径；

真实 proof generation 和真实 verification 必须存在，但不强制拆成两个独立命令。允许：

- 一个正式端到端 runner 按 baseline 原生顺序完成 prepare/prove/verify；
- 或者分别提供 prove / verify 命令。

不得仅为了统一三个 baseline 的接口形式而拆分、重排或重写本来正确的原生执行结构。
- 有自己的 README；
- 如存在测试内容，则有自己独立的 `tests/`；
- 有自己明确的正式结果位置，且结果位于当前 baseline 根目录内部；
- 不要求修改主实验；
- 不要求构建主实验后才能证明或验证；
- 不要求先运行另一个 baseline；
- 不把主实验程序当作 proving/verifying 子进程。

允许依赖标准系统工具链、明确声明的第三方库以及显式传入的只读数据/模型输入。

“独立运行”不要求把操作系统、CUDA、Rust/C++ 工具链或巨型公共数据复制进 baseline，但要求 baseline 自己的代码和证明路径不存在对其他实验实现的隐式依赖。

### 17.1 必须支持单实验正式入口

正式全量 runner 不能是唯一入口。

每个 baseline 至少应能够选择：

- 一个 dataset；
- 一个当前支持的 backend/device；
- 一套正式配置；

运行对应的正式实验路径。

这样项目负责人后续可以先执行单实验确认，再决定是否运行完整实验矩阵。

---

## 18. 公平比较与实验语义

不得通过改变任务本身来让某个 baseline 更容易运行或更快。

除协议笔记明确允许外，不得为某个 baseline 单独改变：

- 数据集；
- 样本选择规则；
- 图拓扑；
- 节点/边排序语义；
- 模型结构；
- 层数；
- attention head 数；
- hidden dimension；
- 类别数；
- 模型权重；
- 量化尺度；
- rounding mode；
- static-table 语义；
- public/private policy；
- 最终 logits 定义；
- 目标安全或隐私 profile。

如果某个 baseline 不能忠实支持目标 relation，应报告 unsupported，而不是修改比较任务。

不得把主实验的证明结果、运行结果、耗时或成功标志当作 baseline 自己的结果。

不得把另一个 baseline 的结果复制过来作为当前 baseline 的结果。

---

## 19. CPU / GPU 边界

baseline 如果原生支持 GPU，可以继续使用原生 GPU 路径。

如果 baseline 当前只有 CPU 路径，则先保持 CPU 路径；当前阶段不要求仅为了与主实验一致而额外移植 GPU 实现。

禁止仅为了让 baseline 更快而复制主实验的：

- GPU kernel；
- CUDA pipeline；
- GPU MSM/NTT；
- 主实验设备缓存；
- 主实验专用调度体系。

如果某项新增 GPU 能力只能通过主实验专有优化完成，则默认不新增，除非项目负责人明确授权。

任何 GPU 正式运行都必须来自真实 GPU 执行；发生 CPU fallback 时必须如实说明，不得把 fallback 结果标记成 GPU 正式结果。

---

## 20. 计时与结果真实性

当前阶段的首要任务是把 baseline 的正式协议代码准备正确，而不是围绕计时做额外性能工程。

正式实验执行时：

- 保持当前 baseline 自己已有且合理的计时边界；
- 不为了得到更好的数字移动工作；
- 不把原本属于 proving/verifying 的工作偷偷移到不计时阶段；
- 不通过复用 fresh-proof 必需工作制造更快结果；
- setup/preprocessing、proof generation、verification 和 proof size 应按各自实际边界区分；
- 不手工改写实验输出制造结果；
- 不用 toy/smoke 结果冒充正式数据集结果；
- 不把更弱 profile 的结果标记成严格 profile；
- 正式结果只能来自当前 baseline 的真实正式运行路径。

如果 baseline 自然计时边界与其他 backend 不完全相同，应如实保留，不为了表面统一而改变其原生执行结构。

正式大规模性能实验和最终对比结果由项目负责人决定何时执行。

---

## 21. 当前代码修改阶段的测试与验证边界

本阶段的目标是：

> **由开发执行方把三个 baseline 的正式实现、构建入口、运行入口、proof/verify 路径、README 和结果目录准备正确；正式实验和长时间验证由项目负责人统一执行。**

因此普通代码修改阶段不得把“完成任务”变成“反复跑实验”。

### 21.1 开发执行方允许进行的短时检查

在修改过程中，只允许执行明显低成本、用于发现基本工程错误的检查，例如：

- Python 语法检查；
- Python import 检查；
- shell 语法检查；
- 配置解析；
- `--help`；
- 路径解析检查；
- CMake configure；
- 明显短时的小目标编译；
- 不触发正式 proof 的入口检查；
- 不触发大规模运行的 dependency 检查。

这些检查只服务于判断代码是否存在明显语法、路径、构建或入口错误。

### 21.2 开发执行方默认不得执行的任务

除非项目负责人明确要求，本阶段不得执行：

- 五个正式数据集的完整实验矩阵；
- CPU/GPU 全组合完整实验；
- 长时间 proof generation；
- 长时间 verification；
- 全量 benchmark；
- 大规模 profiling；
- negative-test 全套；
- 大规模参数生成；
- 重新训练模型；
- 长时间 setup；
- `run_all` 一类完整正式运行；
- `run_until_complete` 一类自动循环到全部完成的流程；
- 为了“更放心”重复运行已经无新信息的测试；
- 任何明显高资源、长时间操作。

### 21.3 正式实验由项目负责人统一执行

开发执行方完成代码修改后：

- 不需要为了证明“全部可运行”自行跑完整正式实验；
- 不得把旧结果当成本轮验证；
- 不得因为尚未执行正式 proof 就改用 mock 或弱证明；
- 必须把真实正式运行命令准备正确；
- 必须在 README 中写清楚正式运行方式；
- 必须在最终报告中列出项目负责人后续应执行的正式验证命令；
- 必须明确说明哪些正式实验命令本轮没有执行。

项目负责人后续运行正式实验时，再根据实际结果判断是否需要进一步修复。

### 21.4 如项目负责人明确要求单次 smoke run

只有项目负责人明确要求时，开发执行方才可以运行一个低成本、能够真实进入目标 proof/verification 路径的单实例 smoke run。

若该单实例已经成功：

- 立即停止；
- 不自动扩展到其他 dataset；
- 不自动扩展到 CPU/GPU 全组合；
- 不自动运行完整 negative suite；
- 不重复执行同一成功实验。

### 21.5 不允许无限制反复测试

任何阶段都必须遵守：

- 不以“多测几次更放心”为理由重复执行已经成功且没有新信息的同一测试；
- 不为了刷新结果反复运行；
- 不形成“改一个函数 -> 大量测试 -> 再改一点 -> 再大量测试”的循环；
- 多个相关修改应集中完成后再做必要检查；
- 测试失败时先定位并完成实质修改，再重新运行受影响检查；
- 如果失败原因已经明确是环境问题，不反复执行相同失败命令；
- 不为了验证 test/helper 本身再建立新的测试链；
- 不主动增加 benchmark、profiler 或额外测试工具，除非项目负责人明确要求。

---

## 22. 正式代码准备完成与正式实验验证必须区分

不得混淆以下两个状态：

### 22.1 代码准备完成

可以表示：

- 协议所需实现已经完成；
- 正式 build/run/prove/verify 入口已经准备；
- 结果目录已经准备；
- README 已更新；
- 仅完成短时工程检查；
- 正式长时间实验尚未由项目负责人执行。

### 22.2 正式实验验证完成

只有在项目负责人实际运行相应正式 proof / verification / benchmark 后，才能根据真实结果声明正式验证状态。

因此：

- “代码已完成”不等于“实验已 verified”；
- “能够编译”不等于“正式证明已成功”；
- “旧结果成功”不等于“当前修改后的代码成功”；
- 未实际运行的命令不得写成已通过；
- README 和最终报告必须准确区分“已准备”与“已验证”。

---

## 23. 长时间和高资源操作

长时间、高资源、最终正式数据集实验不应在普通代码修改过程中反复执行。

因此修改 baseline 时：

- 先完成所有不依赖长时间操作即可完成的代码工作；
- 先完成配置、参数、脚本、路径和运行入口；
- 用短时工程检查排除明显错误；
- 把正式实验命令准备完整；
- 由项目负责人统一决定何时开始正式长时间实验。

不得因为后续可能需要大规模实验，就提前停止当前仍能完成的代码工作。

也不得为了“确认得更充分”在开发阶段反复执行长时间实验。

---

## 24. 依赖、第三方组件与环境边界

当前 baseline 已有框架、第三方组件和依赖关系默认保持。

除非运行协议确实需要，否则不得：

- 整体替换 baseline 框架；
- 切换到另一套实现；
- 大范围更新第三方依赖；
- 更换密码学库；
- 更换曲线、有限域或 proof backend；
- 为了方便开发而改用与当前 baseline 方法学不同的实现。

如果某项依赖调整确实是运行所必需：

- 只能在当前 baseline 内处理项目侧配置；
- 只改必要部分；
- 不扩散到主实验或另一个 baseline；
- 最终说明为什么必须调整以及是否影响证明、安全、隐私或性能口径。

如果外部依赖、编译器、CUDA、系统库、Python/Rust/C++ 环境或第三方组件缺失：

- 如实报告环境问题；
- 不反复执行相同失败命令；
- 不通过弱化协议绕过；
- 不通过 fake/mock 路径冒充正式实现已成功。

---

## 25. 已有结果和实验材料保护

已有结果可以保留，但必须正确区分历史状态与当前状态。

必须遵守：

- 不手工修改旧结果制造新的“当前结果”；
- 不删除旧结果仅为了保留更好的数字；
- 不把旧结果作为当前修改后代码的成功证明；
- 不把测试输出写入正式结果目录；
- 新正式结果只能由后续正式运行生成；
- 所有正式结果留在当前 baseline 根目录内部明确的正式结果位置；
- 当前 baseline 之外的实验结果不得被修改或引用成自己的结果。

临时测试产物如果没有长期价值，可以在当前 baseline 的 `tests/tmp/` 中完成任务后清理。

---

## 26. Fail-closed 与停止条件

出现以下情况时必须停止对应实现并报告：

- 需要修改主实验才能继续；
- 需要修改另一个 baseline 才能继续；
- 需要修改协议笔记才能让现有实现“匹配”；
- 只能通过弱化 relation 才能运行；
- 只能通过删除密码学检查才可运行；
- 只能通过主实验专有 proof technique 才能实现；
- 目标 privacy/security profile 无法由当前 backend 支持；
- 环境或依赖缺失且无法在当前 baseline 内合理处理；
- 发现真实 soundness/privacy/correctness 问题，需要先确定修复策略；
- 无法判断某项拟复制内容究竟属于通用基础设施还是主实验专有方法；
- 正式路径与协议预期存在无法解释的矛盾；
- 当前 baseline 已经满足本阶段代码准备要求，不再需要额外修改。

报告问题优先于制造一个表面可运行但不合规的结果。

---

## 27. 修改前的执行流程

每次修改某个 baseline 前，按以下顺序：

1. 明确当前目标 baseline 和唯一可写目录；
2. 完整阅读本文件；
3. 完整阅读当前 baseline 的 `GAT-ZKML-*.md`；
4. 阅读当前 baseline 的 README、构建配置、正式源码、正式 scripts 和 configs；如存在 `tests/`，再阅读其中的测试内容；
5. 全面搜索旧目录层级、错误绝对路径和目录移动后失效的父目录计算；
6. 对当前正式代码进行代码级协议覆盖检查，判断与协议笔记之间的真实差距；
7. 旧 README、旧结果、旧成功状态不能替代该检查；只有当前正式代码确实符合协议时才直接复用；
8. 如果存在缺失能力，优先寻找 baseline-native 最小实现路径；
9. 只有缺失项属于通用基础设施时，才查看主实验对应内容；
10. 在复制前确认不会引入主实验专有证明方法或优化；
11. 检查正式代码目录中是否混入纯测试/debug/negative 内容；
12. 检查 README 是否引用当前正确协议路径、构建方式和结果路径；
13. 所有写操作保持在当前 baseline 内。

禁止一开始就照主实验结构重构 baseline。

### 27.1 协议、实际代码与文字说明的可信等级

判断当前 baseline 的真实实现状态时，统一采用以下优先级：

1. 当前 baseline 的 `GAT-ZKML-*.md` 是协议与安全边界的最高规范；
2. 当前实际正式生产代码路径和构建/运行逻辑用于判断“现在到底实现了什么”；
3. README、源码注释、历史说明、旧脚本描述和测试文字只作为辅助理解材料。

因此：

- 不得仅凭 README、注释或历史说明认定某项能力已经实现或尚未实现；
- 如果文字说明与当前实际代码行为不一致，应以协议笔记和实际正式代码为准进行修正；
- 不得为了匹配过时注释或旧 README 而修改本来正确且符合协议的实现；
- 不得仅因为历史文字提到某项功能、优化或 TODO，就自动扩大本轮修改范围；
- README 必须在本轮最后更新为与当前正式代码一致。

---

## 28. 修改后的执行流程

完成一个 baseline 的代码修改后，按以下顺序：

1. 检查本轮所有项目写入是否位于当前 baseline；
2. 确认主实验、另两个 baseline、本文件和协议笔记未被本轮违规修改；
3. 如果存在测试内容，检查正式代码和测试代码是否已经隔离到当前 baseline 的 `tests/`；
4. 检查正式 `scripts/` 中不存在纯测试、negative、debug-only runner；
5. 如果存在测试内容，检查测试输出不会写入任何正式结果目录；
6. 如果存在 `tests/`，检查正式代码不依赖 `tests/`；
7. 检查当前 baseline 根目录存在并更新了 `README.md`；
8. 检查当前 baseline 根目录内部存在明确的正式结果输出位置；
9. 检查至少有一个可选择单 dataset/backend 的正式运行入口；
10. 检查真实 proof generation 和 verification 路径完整；允许由一个正式端到端 runner 顺序完成，不要求人为拆分命令；
11. 执行本文件允许的短时语法、配置、路径、入口或构建检查；
12. 不执行长时间 proof、全量数据集、完整 CPU/GPU 矩阵或其他高资源实验；
13. 汇总本轮实际修改、必要依赖调整、主实验复制内容和 unsupported 状态；
14. 给出项目负责人后续正式验证命令；
15. 停止，不自行扩展测试范围。

---

## 29. 最终交付报告

每完成一个 baseline 的修改，报告至少包括：

### 29.1 Target

- 当前 baseline；
- 当前 baseline 根目录。

### 29.2 Changed files

列出本轮：

- 新增文件；
- 修改文件；
- 删除文件；
- 每个文件为什么需要改变。

### 29.3 Formal/test separation

说明：

- 哪些内容属于正式代码；
- 哪些旧测试/debug/negative 内容被移动到 `tests/`；
- 是否删除了无执行价值的历史测试或失效代码；
- 正式代码是否完全不依赖 `tests/`。

### 29.4 Main-experiment copies

如果没有复制，写明 `none`。

如果有复制，列出：

- source path；
- destination path；
- reason；
- type；
- 是否涉及证明系统逻辑。

正常情况下不应复制主实验专有证明系统逻辑。

### 29.5 Protocol coverage

说明本轮实现或修复了协议笔记中的哪些要求。

### 29.6 Unsupported capabilities

如果存在 unsupported，必须明确列出。

### 29.7 Build / short checks

只说明实际执行的短时检查。

不得把未运行的正式 proof / verify 写成已通过。

### 29.8 Formal run commands

列出项目负责人后续应执行的实际正式入口，包括：

- build；
- 单 dataset/backend 正式运行；
- 正式 proof / verification 路径；
- 正式结果导出；
- 完整正式实验入口。

如果当前 baseline 原生流程由一个正式端到端 runner 顺序完成 proof generation 和 verification，则直接列出该 runner，不要求人为拆分成独立 prove / verify 命令。

如果当前 baseline 原生提供独立 prove / verify 命令，则按实际接口分别列出。

明确标注这些命令是否在本轮执行。

### 29.9 Result location

明确写出当前 baseline 实际使用的正式结果路径。

该路径可以保留当前 baseline 自己合理的原生组织，不要求三个 baseline 使用相同目录名称；但所有正式结果必须位于当前 baseline 根目录内部，不得写到当前 baseline 之外。

### 29.10 README

确认当前 baseline 的 README 已与当前正式代码、协议路径、正式运行入口和结果目录一致。

### 29.11 Environment limitations

如存在 CUDA、依赖、编译器、内存或第三方组件问题，单独说明。

### 29.12 Boundary statement

明确说明：

> 本轮所有项目写入均限制在当前 baseline 目录；主实验和其他 baseline 未被本轮修改。

---

## 30. 本阶段最终验收门

一个 baseline 的“代码修改阶段”只有同时满足以下条件，才能视为完成。

### Gate A — Workspace isolation

本轮所有项目修改只发生在当前 baseline 目录。

### Gate B — Main experiment integrity

主实验没有被本轮修改，也没有为了 baseline 增加 adapter 或兼容路径。

### Gate C — Protocol fidelity

没有改变或弱化 `GAT-ZKML-*.md` 定义的 relation、安全和隐私边界。

### Gate D — Methodological independence

没有把主实验专有证明方法或专有性能优化迁入 baseline。

### Gate E — Baseline-native preservation

优先使用并保留 baseline 自己已有的证明体系和有效原生优化。

### Gate F — Formal path completeness

正式 proof generation 和 verification 路径完整存在，不是 mock、fake、partial 或 host-only 替代。

本阶段不要求开发执行方通过长时间正式实验来证明该 Gate 已运行成功，但必须确保正式路径没有被测试替代。

### Gate G — Independent build/run

baseline 不依赖被修改的主实验、主实验程序、主实验构建产物或另一个 baseline 才能运行。

### Gate H — Formal/test isolation

如当前 baseline 存在测试、negative、debug、fixture 等内容，则它们已经与正式代码严格隔离到当前 baseline 的 `tests/`；正式代码不依赖 `tests/`。如果完全不存在测试内容，则不要求为了形式创建空 `tests/`。

### Gate I — README completeness

当前 baseline 有独立 README，且 README 描述的是当前正式实现。

### Gate J — Result locality

当前 baseline 的正式结果和正式运行 artifact 只写入自身根目录内部的明确正式输出位置；不强制统一目录名称，不得写到当前 baseline 之外。

### Gate K — Short-check boundary

开发执行方只执行必要的短时检查，没有自行运行长时间、大规模或完整正式实验。

### Gate L — Honest status

未执行的正式实验没有被写成已通过；旧结果没有被当作当前代码的验证结果；unsupported 和环境限制均已如实报告。

任何 Gate 未通过，都不能把该 baseline 的代码修改阶段标记为完成。

---

## 31. 核心判断树

```text
当前 baseline 已经实现所需能力？
    是
      -> 直接复用。
      -> 不用主实验替换。

    否
      -> 能否用 baseline 自己的原生 proof/backend 机制做最小扩展？

         能
           -> 只在当前 baseline 内做最小扩展。

         不能
           -> 缺失的是否只是通用 GAT / data / config /
              preprocessing / serialization 基础设施？

              是
                -> 从只读主实验复制最小必要内容到当前 baseline。
                -> 只修改 baseline 内副本。
                -> 去除对主实验实现的运行依赖。

              否
                -> 是否必须引入主实验专有证明方法、
                   安全捷径或专有性能优化？

                   是
                     -> 不导入。
                     -> 按协议笔记报告 unsupported。

                   不确定
                     -> 停止并由项目负责人判断。
```

正式代码与测试：

```text
这个逻辑是否是正式 build/prove/verify/run 必须执行的？
    是
      -> 放在正式代码路径。

    否，只用于检查正式代码是否正确
      -> 如果需要保留，放在当前 baseline 的 tests/。
      -> 如果已经失效且无当前测试价值，删除。

正式代码是否依赖 tests/？
    是
      -> 结构不合规，必须解除依赖。

测试输出是否写进任何正式结果目录或正式运行 artifact 目录？
    是
      -> 结构不合规，必须改到 tests/tmp/。
```

执行验证：

```text
当前任务是否由项目负责人明确要求运行正式实验？
    否
      -> 只执行短时工程检查。
      -> 准备正式运行命令。
      -> 不运行长时间 proof / verify / full matrix。

    是
      -> 只执行负责人明确要求的范围。
      -> 达到目标后立即停止，不自动扩展。
```

---

## 32. 最终不可突破原则

1. **主实验永远不可修改，只允许查看和必要复制通用内容。**
2. **修改哪个 baseline，所有项目写操作只能发生在该 baseline 根目录。**
3. **当前正式目录为 `comparison/ezkl/`、`comparison/zkdl/`、`comparison/zkGPT/`，旧目录层级不得继续作为正式依赖。**
4. **三个 baseline 必须保持彼此独立。**
5. **协议笔记不可为了代码而修改。**
6. **当前 baseline 已经能直接实现的能力，不得用主实验替换。**
7. **优先 baseline-native；主实验复制只用于真正缺失的通用基础设施。**
8. **不得把主实验专有证明方法和优化体系反向移植到 baseline。**
9. **不得人为关闭 baseline 自己已有的正常原生优化。**
10. **最小修改不等于弱证明，不允许 mock、fake、跳过、partial proof 或静默降级。**
11. **无法忠实支持时明确 unsupported，而不是改变任务或伪装支持。**
12. **每个 baseline 必须有自己的 README。**
13. **每个 baseline 的正式结果必须保存在自身根目录内部，结果目录名称不强制统一。**
14. **正式代码与测试代码必须严格隔离；如果存在测试/negative/debug 内容，必须集中进入当前 baseline 自己的 `tests/`。**
15. **正式代码不得依赖 `tests/`，测试输出不得进入任何正式结果目录。**
16. **正式 `scripts/` 只保留正式实验生命周期脚本，不混入测试 runner。**
17. **正式构建配置可以声明可选测试 target，但正式 build/run/prove/verify 不得依赖或自动触发这些测试 target。**
18. **旧实验成功状态不得作为当前修改后代码的验证结果。**
19. **开发执行方本阶段只做必要短时检查，不运行长时间 proof、全量数据集、CPU/GPU 全组合、完整 benchmark 或大规模 profiling。**
20. **正式实验和最终性能结果由项目负责人统一运行和确认。**
21. **未执行的正式命令不得写成已通过。**
22. **所有主实验复制必须最小、必要、可说明来源与用途。**
23. **正确性、协议忠实性、安全性、方法学独立性和代码结构清晰性优先于便利、代码复用和性能。**
24. **达到本阶段代码准备要求后应停止修改，不自行扩大测试或优化范围。**
