# Cora Full Mainline

正式入口只保留 `configs/cora_full.cfg`。

当前状态已经去掉了 `cora_full` 对 synthetic 参数的 silent fallback。现在 `configs/cora_full.cfg`
只能读取真实 checkpoint bundle；如果 bundle 架构与当前单头协议模型不兼容，程序会直接失败，
而不是偷偷退回 seed 造参。

Python 侧真实 checkpoint / reference forward 依赖：

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-export.txt
```

数据准备：

```bash
.venv/bin/python scripts/prepare_planetoid.py \
  --data-root data \
  --dataset cora \
  --cache-root data/cache
```

从原始 `../GAT` checkpoint 导出真实参数：

```bash
.venv/bin/python scripts/export_gat_checkpoint.py \
  --checkpoint-prefix ../GAT/pre_trained/cora/mod_cora.ckpt \
  --output-dir artifacts/checkpoints/cora_gat
```

跑原始 GAT reference forward 并导出中间对象：

```bash
.venv/bin/python scripts/gat_reference.py \
  --checkpoint-dir artifacts/checkpoints/cora_gat \
  --data-root data \
  --dataset cora \
  --output-dir runs/cora_full/reference
```

构建与测试：

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

正式 `cora_full` 入口校验：

```bash
./build/gatzk_run --config configs/cora_full.cfg
```

当前这条命令会因真实 bundle 为 `8` 个隐藏 attention heads 加输出 attention head 而失败退出。
这是新的硬闸门：正式入口不再允许用 synthetic 参数伪装成“真实 checkpoint 闭环”。
