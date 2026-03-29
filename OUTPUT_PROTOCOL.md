# Output Protocol

仓库默认采用“直接落地文件”的工作模式：

1. 协议改动直接写入 `include/` 与 `src/`；
2. 正式配置直接写入 `configs/`；
3. 测试直接写入 `tests/`；
4. 主入口文档直接写入 `README.md` 与 `docs/`；

正式输出口径以 full-graph 为准：

- `is_full_dataset = true`
- `notes` 中包含 `dataset_scope=full`
- proof 必须可生成
- verifier 必须通过

可选 GPU backend 通过构建选项与环境变量切换：

- CMake: `-DGATZK_ENABLE_CUDA_BACKEND=ON`
- Runtime: `GATZK_ALGEBRA_BACKEND=cuda`
