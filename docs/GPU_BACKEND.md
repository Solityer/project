# GPU Backend

第一版 GPU backend 目标：

1. 不改协议对象、不改 transcript 顺序、不改 verifier；
2. 给 algebra vector ops 和 packed eval backend 增加 CPU/CUDA 双后端切换；
3. 通过 `GATZK_ALGEBRA_BACKEND=cuda` 选择 CUDA backend；
4. 通过 `-DGATZK_ENABLE_CUDA_BACKEND=ON` 编译 CUDA shim。

当前代码落点：

- `include/gatzk/algebra/vector_ops.hpp`
- `src/algebra/vector_ops.cpp`
- `src/algebra/vector_ops_cuda.cu`
- `include/gatzk/algebra/eval_backend.hpp`
- `src/algebra/eval_backend.cpp`
- `src/algebra/eval_backend_cuda.cu`
- `CMakeLists.txt`

当前限制：

- pairing 和 verifier 仍是 CPU / mcl
- `FieldElement` 仍基于 mcl `Fr`
- CUDA 路径当前是第一版 algebra backend shim，主要目的在于打通后端切换和数据布局接口，不改变协议正确性

正确性检查方式：

1. 同一配置分别跑 `cpu` 和 `cuda`
2. 对比 `VERIFY_OK`
3. 对比 `proof_size_bytes`
4. 对比 challenge 重放和 verifier 接受结果
