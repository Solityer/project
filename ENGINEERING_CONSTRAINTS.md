# Engineering Constraints

## 1. General

- 项目语言必须是 C++。
- 构建系统必须是 CMake。
- 目标环境是 Ubuntu 22.04。
- 代码必须模块化，不允许把大部分逻辑堆在单文件中。
- 头文件与源文件必须合理分离。
- 代码命名、目录组织、依赖关系必须清晰。
- 不允许生成伪代码、空壳框架、占位类、占位函数。
- 不允许使用 TODO、placeholder、stub、mock implementation 充当正式实现。
- 不允许把关键实现工作留给用户手工补全。
- 最终结果必须直接写入当前项目目录，而不是只在聊天中输出代码文本。

## 2. Delivery Standard

首轮交付结果必须是完整项目，而不是骨架工程。必须满足：

- 可以编译；
- 可以运行；
- 有清晰入口；
- 有基础测试；
- 有 README；
- 有配置文件；
- 有数据加载逻辑；
- 有 prove / verify 的最小闭环；
- 有最小可运行示例；
- 有真实落地到文件系统中的工程文件。

允许部分底层密码学组件采用“可运行参考实现”，但不允许未实现。

## 3. Protocol Fidelity

所有实现必须严格对齐 `GAT-ZKML流程.md` 中的定义、对象、阶段划分、挑战顺序、承诺顺序、证明结构和 verifier 检查逻辑。

不得擅自改变以下内容：

- 公开对象与动态 witness 的分层；
- 工作域设计；
- challenge 生成顺序；
- commitment 顺序；
- quotient identity 的构造和检查方式；
- lookup / LogUp 的语义；
- zkMaP 的语义；
- PSQ / CRPC 的语义；
- 单点 / 批量 / 外点评值 opening 的语义；
- verifier 的重建顺序；
- proof shape；
- verify 最终检查逻辑。

若主规格存在工程实现歧义，必须：

1. 采用最保守、最忠实于原文流程的实现；
2. 在 README 和代码注释中明确写出该工程取舍；
3. 不允许静默改协议。

## 4. Architecture Requirement

项目至少必须清晰拆分为以下模块：

- 配置与参数
- 有限域与基础代数
- 曲线/群抽象
- KZG 承诺与 opening
- Transcript / Fiat-Shamir
- 工作域与多项式
- lookup / LogUp
- zkMaP
- PSQ / CRPC
- 图数据加载与预处理
- GAT 明文前向
- witness / trace 构造
- prover
- verifier
- 日志与导出
- 测试

## 5. Performance Discipline

实现必须具备基本工程纪律：

- 热路径对象尽量使用原生域元素 / 原生曲线点类型；
- 不要在主循环中频繁进行 string 与域元素、bytes 与曲线点之间的来回转换；
- 静态对象（如 SRS、VK、表承诺、模型承诺等）应考虑集中初始化和缓存；
- 不得为了追求“写起来快”而破坏模块边界；
- 不得让底层密码学库 API 直接污染整个上层协议代码。

## 6. Dataset Support

必须首先支持以下真实数据集格式：

- `data/cora/`
- `data/citeseer/`
- `data/pubmed/`

它们均采用经典 Planetoid / citation-network 文件格式。

此外必须内置：

- 一个 toy dataset
- 一个最小 smoke test
- 一个小图 correctness test

## 7. Testing

至少应包含以下测试：

1. 数据加载测试；
2. toy graph 前向测试；
3. witness / trace 构造一致性测试；
4. transcript / challenge 顺序测试；
5. commitment / opening 最小测试；
6. prove / verify 最小闭环测试；
7. 若可行，加入若干 verifier reject case。

## 8. Documentation

最终项目必须提供：

- 项目目录说明；
- 模块职责说明；
- Ubuntu 22.04 依赖安装说明；
- CMake 构建说明；
- 数据准备说明；
- 运行命令；
- 测试命令；
- 日志与导出说明；
- 中间变量 dump 说明；
- 常见错误排查；
- 复现实验步骤。

## 9. Comment Quality Requirement

关键模块、关键类、关键函数、关键数据结构和关键流程代码必须写必要注释。

注释必须说明：

- 该模块或函数对应主规格中的哪一步；
- 它处理的是哪类 witness、约束、challenge 或证明对象；
- 输入输出的协议语义；
- 它在 prove 或 verify 流程中的作用；
- 若采用工程化取舍，取舍点是什么。

禁止空泛注释，例如：

- “初始化变量”
- “处理数据”
- “执行计算”

禁止关键函数无注释。

## 10. File-Generation Requirement

生成项目时，必须直接在当前项目目录中创建和写入文件，而不是只在聊天中展示目录树、代码片段或文件内容。

聊天中的输出应主要用于：

- 简要汇报已完成模块；
- 说明已创建或更新的关键文件；
- 说明下一步要继续生成的部分；
- 解释必要的工程取舍。

## 11. Non-Negotiable Rule

以下行为一律禁止：

- 只给框架，不给完整实现；
- 只给部分文件；
- 只给单文件 demo；
- 用伪代码替代实现；
- 把关键逻辑推迟到“后续再补”；
- 为了迎合库接口而修改协议流程；
- 略过 README、测试、CMake、配置和脚本；
- 只在聊天中展示代码而不落地到文件系统。