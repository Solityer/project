# GAT-ZKML：图注意力网络的零知识证明系统

## 项目简介

本项目实现了多层多头图注意力网络（Graph Attention Network，GAT）的零知识证明系统。基于 KZG 多项式承诺和 Fiat-Shamir 哈希链转录，在标准 BLS12-381 曲线上为 GAT 推理过程生成可验证的简洁零知识证明。

协议形式化规范见 [GAT-ZKML-多层多头.md](./GAT-ZKML-多层多头.md)。所有实现均与该规范保持严格一致，不支持任何语义变体。

---

## 支持的数据集与模型结构

| 数据集 | 节点数 | 边数 | 隐藏层 | 多头结构 |
|--------|--------|------|--------|----------|
| Cora | 2,708 | 13,264 | 2 层 | 多头拼接+均值 |
| Citeseer | 3,327 | 12,431 | 2 层 | 多头拼接+均值 |
| Pubmed | 19,717 | 108,365 | 2 层 | 多头拼接+均值 |
| PPI | 56,944 | 850,576 | 2 层 | 多头拼接+均值 |
| ogbn-arxiv | 169,343 | 1,335,586 | 2 层 | 多头拼接+均值 |

---

## 当前最新最快实验结果

以下为各数据集在 **warm** 模式下的官方基准结果（见 `runs/benchmarks/summary.md`）：

| 数据集 | 状态 | 承诺时间(ms) | 证明时间(ms) | 验证时间(ms) | 证明大小(B) |
|--------|------|-------------|------------|------------|-----------|
| Cora | 已完成 | 1,217 | 3,815 | 292 | 37,283 |
| Citeseer | 已完成 | 2,554 | 9,138 | 292 | 37,291 |
| Pubmed | 已完成 | 6,350 | 18,702 | 315 | 37,288 |
| PPI | 已完成 | 3,890 | 14,086 | 4,151 | 9,048 |
| ogbn-arxiv | 已完成 | 见最新 | 见最新 | 见最新 | 9,049 |

最新 ogbn-arxiv 结果请查阅 `runs/benchmarks/latest.json` 与 `runs/ogbn_arxiv_full/warm/run_manifest.json`。

---

## 从零开始到最新结果的完整运行步骤

### 1. 构建项目

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### 2. 准备数据集

各数据集的检查点（checkpoint）文件存放于 `artifacts/checkpoints/` 目录下。ogbn-arxiv 数据集通过以下脚本准备：

```bash
python scripts/prepare_ogbn_arxiv.py
python scripts/train_ogbn_arxiv_gat.py
```

训练完成后，使用 `scripts/import_ppi_bundle.py` 导入 PPI bundle。

> **PPI 参数说明**：PPI 模型参数来自上游真实参数来源（`/home/pzh/GAT-for-PPI`），通过 `import_ppi_bundle.py` 导入后方可用于 formal / benchmark 运行主线。上游训练脚本不属于本项目维护范畴，仅作为参数来源使用。

### 3. 运行基准测试

```bash
# 单数据集 warm 模式运行
./build/gatzk_run --config configs/cora_full.cfg --warm

# 导出全量基准表
python scripts/export_benchmark_table.py
```

### 4. 运行验证测试

```bash
./build/gatzk_tests
```

---

## 主要性能热点

ogbn-arxiv 的主要耗时热点（优化前后）：

- `dynamic_domain_convert_ms`：FH 域多项式在 τ 处重心插值权重的计算
- `domain_open_FH_ms`：FH 域多项式在开放点处的批量求值
- `quotient_t_fh_ms`：FH 约束商多项式在 τ 处的求值

当前优化：FH 域（规模 2^25）的重心权重计算已并行化，大幅降低上述三项热点的耗时。

---

## 形式化协议说明

本系统严格遵循 [GAT-ZKML-多层多头.md](./GAT-ZKML-多层多头.md) 中定义的协议：

- **7 个工作域**：FH、edge、in、d_h、cat、C、N
- **Fiat-Shamir 转录顺序**：元数据 → FH → 各层隐藏头 → 拼接阶段 → 输出头 → α_quot → 域开放点
- **LogUp 查找协议**：证明特征检索 P_H[i,j] = T_H[I_i, j]
- **KZG 多项式承诺**：基于 BLS12-381 曲线，使用 MCL 库

---

## 目录结构

```
├── GAT-ZKML-多层多头.md       # 形式化协议规范（只读）
├── configs/                   # 各数据集运行配置
├── artifacts/checkpoints/     # 模型检查点文件
├── runs/                      # 基准测试结果
│   ├── benchmarks/            # 汇总表与 latest.json
│   └── */warm/run_manifest.json
├── scripts/                   # 数据准备与导出脚本
├── src/                       # 实现代码
│   ├── algebra/               # 多项式与域运算
│   ├── crypto/                # KZG 承诺
│   ├── protocol/              # 证明器与验证器
│   └── model/                 # GAT 前向推理
└── tests/                     # 测试套件
```
