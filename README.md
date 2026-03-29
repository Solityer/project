# GAT-ZKML Full-Graph Mainline

正式协议规范：

- [`GAT-ZKML_终稿_修订版.md`](./GAT-ZKML_终稿_修订版.md)

正式入口配置：

- [`configs/cora_full.cfg`](./configs/cora_full.cfg)
- [`configs/citeseer_full.cfg`](./configs/citeseer_full.cfg)
- [`configs/pubmed_full.cfg`](./configs/pubmed_full.cfg)

不再作为正式入口维护的旧口径：

- `toy`
- `smoke`
- `cora_benchmark`

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## CPU Run

```bash
./build/gatzk_run --config configs/cora_full.cfg --benchmark-mode single
./build/gatzk_run --config configs/citeseer_full.cfg --benchmark-mode single
./build/gatzk_run --config configs/pubmed_full.cfg --benchmark-mode single
```

建议检查日志字段：

- `dataset=cora|citeseer|pubmed`
- `is_full_dataset=true`
- `notes` 中包含 `dataset_scope=full`
- `prove_time_ms`
- `verify_time_ms`
- `proof_size_bytes`
- 最终输出 `VERIFY_OK`

## CUDA Algebra Backend

默认仍为 CPU。可选 CUDA backend 只覆盖 algebra / eval / trace 热路径接入，不覆盖 pairing、verifier 和 mcl 曲线主路径。

构建：

```bash
cmake -S . -B build-cuda -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build-cuda -j
```

运行：

```bash
GATZK_ALGEBRA_BACKEND=cuda ./build-cuda/gatzk_run --config configs/cora_full.cfg --benchmark-mode single
```

当前 CUDA 路径是第一版 backend shim，协议对象、transcript 顺序和 verifier 结果必须与 CPU 一致。

## Tests

```bash
ctest --test-dir build --output-on-failure
```

## Docs

- [`docs/FULL_GRAPH_MAINLINE.md`](./docs/FULL_GRAPH_MAINLINE.md)
- [`docs/GPU_BACKEND.md`](./docs/GPU_BACKEND.md)
- [`docs/LEGACY_STATUS.md`](./docs/LEGACY_STATUS.md)
