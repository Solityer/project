# zkGAT 两项机制性消融实验

## 当前矩阵

完整注册矩阵有 47 组配置，重复 3 次并对每组运行 Original/Control，共 141 对、282 个证明单元。正式命令支持按机制和输入范围分别启动；每个会话只记录所选配置和对应计划行。

| 机制 | 数据集与配置 | 组数 |
|---|---|---:|
| Feature Folding | Cora、Citeseer、PubMed、PPI、OGBN-Arxiv；D=1..8，G=1 | 40 |
| Shared Neighborhood | 五个数据集；D=8，G=1 | 5 |
| 附录：Feature Folding + Shared Neighborhood | PPI 全部 295 个图；D=8，G=295 | 2 |

五数据集主矩阵中，PPI 只取正式缓存中的原始 component 0（N=1546、E=17699、G=1）。附录使用全部 component ID 0..294，不筛图、不裁边，并保留 296 个完整 node_ptr/edge_ptr 边界。其余数据集使用各自完整图。FF 的 D=1..8 是同一真实 8 维隐藏特征的前缀宽度；Shared Neighborhood 固定 D=8。数据、模型、预处理、量化、安全参数、SRS 和 CUDA backend 由对应正式配置绑定。

这里的 `main` 是**机制消融主表输入范围**：四个完整非 PPI 数据集及 PPI 原始组件 0；它不是 zkGAT 主实验。三个独立子矩阵如下：

| 单独运行 | 选择参数 | 配置组 | 3 次重复的证明单元 |
|---|---|---:|---:|
| Feature Folding 主表 | `--mechanism feature_folding --input-scope main` | 40 | 240 |
| Shared Neighborhood 主表 | `--mechanism shared_neighborhood --input-scope main` | 5 | 30 |
| 全量 PPI 附录（两种机制） | `--mechanism all --input-scope ppi_full` | 2 | 12 |

省略选择参数时仍运行完整 47 组矩阵。`--input-scope ppi_full` 只选全量 PPI 附录，不能用 PPI component 0 代替。

## 预检和正式启动

对准备运行的每个子矩阵，先用相同选择参数执行有界 metadata-only 预检。例如：

```bash
cd /home/pzh/project
python3 scripts/run_mechanism_ablation.py preflight --device 0 --mechanism feature_folding --input-scope main
python3 scripts/run_mechanism_ablation.py preflight --device 0 --mechanism shared_neighborhood --input-scope main
python3 scripts/run_mechanism_ablation.py preflight --device 0 --mechanism all --input-scope ppi_full
```

每次预检只核对被选中的配置组、数据集及其 SRS；它还检查对应 checkpoint manifest、缓存元数据、原始边数量与端点范围、PPI 图边界、SRS manifest 声明的 120-bit 安全级别、域和所选实例容量，并确认 trusted material 文件存在。它不运行 native `prepare-real`，不生成 proof 输入 bundle，不读取或重新计算大型 SRS key 文件内容哈希，不装载 SRS，也不运行 Prove、Verify 或安全账本审计。预检产生的 `results.csv` 只列出所选矩阵的计划单元，均标为“未运行”；`summary.md` 明确写明这不是实验结果。

预检通过后，按需要一次启动一条正式命令。`launch` 会在后台运行，每个 `--output-root` 都是独立的新目录；同一设备一次只启动一个会话，如需重跑请使用新的目录名。

```bash
python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 3 \
  --mechanism feature_folding --input-scope main \
  --output-root results/mechanism_ablation_final/ff

python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 3 \
  --mechanism shared_neighborhood --input-scope main \
  --output-root results/mechanism_ablation_final/sn

python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 3 \
  --mechanism all --input-scope ppi_full \
  --output-root results/mechanism_ablation_final/ppi_full
```

`launch` 在指定输出根目录下创建新的独立 session，不覆盖已有结果。它会执行完整输入准备、SRS trusted material 哈希验证、CUDA Prove、独立 CPU Verify 和逐单元安全检查。启动命令先打印 `launcher_dir` 和 `runner.log`；等待日志出现 `MECHANISM_SESSION_STARTED ... output=...` 后，将该 `output` 路径填入 `SESSION`：

```bash
LAUNCHER_DIR=<LAUNCHER_STARTED 输出中的 launcher_dir>
grep 'MECHANISM_SESSION_STARTED' "$LAUNCHER_DIR/runner.log"
SESSION=<runner.log 中 output= 后的 session 路径>
cat "$SESSION/status.json"
tail -f "$SESSION/session.log"
```

会话结束后，主要阅读文件只有中文 `summary.md` 和 Excel 可打开的 UTF-8 BOM `results.csv`。CSV 每个所选配置/重复/版本占一行；未运行或未完成单元明确标记状态，不填入 0。在线承诺/证明/验证时间与完整 Prover/Verifier 子进程墙钟时间分列。汇总统计只纳入同一配置和重复序号下 Original、Control 均成功、初次独立 Verify 通过且安全账本有效的配对；保存率和 C/O 倍数逐对计算再汇总，负保存率照实保留。

逐单元日志、Proof、Statement、Prover/Verifier 回执、输入元数据、状态、source/build/config/SRS 清单及审计文件保留在 session 子目录中，供复核使用。`summary.json`、`status.json`、`results_manifest.json` 和必要 provenance 文件是机器状态及审计证据，不是额外的用户阅读报表。预检与正式运行都创建新的 session，不清理旧输出。

## 历史结果

早期 session `results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/` 使用旧的 11 组矩阵（包含 D=1/2/4 的子集及 PPI component 0+9 的 G=2 组）。它只作为历史 artifact 保留，不代表当前 47 组矩阵，也不会被新 runner 覆写或并入新结果。旧 session 中的日志、Proof/Statement、回执和报告均留在原位置。
