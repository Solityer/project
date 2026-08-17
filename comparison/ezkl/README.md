# ezkl GAT comparison baseline

本目录保留 ezkl/Halo2 上游源码，并按 `GAT-ZKML-ezkl.md` 要求评估 GAT comparison relation。比较方法要求证明电路忠实约束完整 relation；宿主侧预计算、未受约束输入或替代协议均不能作为正式路线。

## 当前状态

整体状态为 `unsupported_exact_relation`。当前 ezkl/Halo2 集成缺少完整受约束的 sparse routing、exact group max、协议要求的四处 `SignedRoundDiv`、完整多层多头执行，以及同一电路内的 ZKG `H(W)` 绑定。因此 CPU 与 GPU 正式路径均为 `unsupported`。

此前的 native adapter、GAT 专用 CLI、runner、collector、validator、配置与派生结果建立在错误 comparison relation 上，现已删除或禁用。当前没有正式 run、prove 或 verify 入口，也没有可发布的实验结论。

## 目录结构

- `src/`：ezkl/Halo2 上游正式源码；不再暴露错误的 GAT comparison 命令。
- `configs/capabilities.json`：机器可读的 relation 与 backend 支持状态。
- `scripts/build.sh`：仅构建 release 版 ezkl CLI 的正式 build 入口。
- `tests/`：上游测试代码与测试资源；正式代码不依赖该目录。
- `results/`、`runs/`：未来忠实实现完成后重新实验的保留位置，目前仅含空占位文件。

构建依赖为仓库锁定的 Rust 工具链、`Cargo.lock` 中锁定的依赖及本机 C/C++ linker。构建入口不会安装或更新依赖：

```bash
scripts/build.sh
```

代码和实验逻辑已整理；正式实验结果等待未来忠实实现后重新运行。
