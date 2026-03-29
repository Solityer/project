# Dependency Note

## 1. Default Crypto Backend

本项目默认使用 `mcl` 作为以下底层能力的实现后端：

- 有限域运算
- 椭圆曲线群运算
- 双线性配对
- KZG 多项式承诺相关底层对象

官方仓库：
- GitHub: https://github.com/herumi/mcl

## 2. Integration Requirements

生成的项目代码必须通过项目内部封装层接入 `mcl`，不允许在上层业务逻辑中散落直接调用 `mcl` 原始 API。

至少必须抽象出以下模块边界：

- `field`
- `curve`
- `kzg`
- `transcript`

上层协议模块只能依赖项目内部定义的抽象接口、类型封装或适配层，不允许直接把 `mcl` 具体类型扩散到整个工程。

## 3. Architecture Constraint

不得为了迎合 `mcl` 的接口风格而静默修改 `GAT-ZKML流程.md` 中定义的协议流程、对象语义、challenge 顺序、proof shape 或 verifier 检查逻辑。

实现优先级必须是：

1. 忠实复现 `GAT-ZKML流程.md`
2. 保证工程可编译、可运行、可测试
3. 在上述前提下合理接入 `mcl`

## 4. KZG-Related Requirement

项目中与 KZG 相关的以下部分必须有清晰、独立、可运行的实现，不允许只留空壳接口：

- commitment
- 单点 opening
- 批量 opening
- 外点评值相关 opening / folding witness
- verifier 侧对应检查逻辑

如果某些底层细节采用参考实现，也必须是真正可运行、可测试、可接入完整 prove / verify 流程的参考实现，而不是示意代码。

## 5. Build and README Requirement

生成结果中必须满足以下要求：

1. `README.md` 中必须明确写出 Ubuntu 22.04 下的 `mcl` 安装或接入方式；
2. `CMakeLists.txt` 中必须体现 `mcl` 的接入方式；
3. 至少支持以下任一接入方案：
   - 系统预装
   - `third_party/mcl` 子目录
   - `git submodule` 方式
4. 必须说明该项目当前采用的是哪一种默认接入策略。

## 6. Performance and Engineering Requirement

实现时必须注意以下工程约束：

- 热路径对象尽量使用原生域元素 / 原生曲线点类型；
- 不要在主循环中频繁进行字符串与域元素、字节与曲线点之间的来回转换；
- 静态对象（如 SRS、VK、模型承诺、表承诺等）应考虑缓存或集中初始化；
- 密码学底层封装必须服务于上层协议模块化，而不是反过来绑架整体工程结构。

## 7. File-Generation Requirement

`mcl` 的接入方案必须直接体现在当前项目目录中生成的工程文件里，包括但不限于：

- `CMakeLists.txt`
- README 中的依赖安装说明
- 相关 crypto backend 抽象与实现文件

不允许只在聊天中口头说明依赖策略而不落实到工程文件。

## 8. Non-Negotiable Rule

第一轮输出的项目必须：

- 可编译
- 可运行
- 可测试
- 可用于论文实验复现

不允许因为依赖接入问题而退化成空壳工程、接口工程或伪代码工程。