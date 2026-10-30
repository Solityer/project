# zkGAT 两项机制性消融实验

## 矩阵与重复次数

47 组配置定义不变。每组运行 Original 和 Control；重复次数 `R` 决定每组各运行几次。

| 机制 | 数据集与配置 | 组数 | 单次重复单元 | R=3 单元 |
|---|---|---:|---:|---:|
| Feature Folding | Cora、Citeseer、PubMed、PPI、OGBN-Arxiv；D=1..8，G=1 | 40 | 80 | 240 |
| Shared Neighborhood | 五个数据集；D=8，G=1 | 5 | 10 | 30 |
| 附录：两种机制 | PPI 全部 295 个图；D=8，G=295 | 2 | 4 | 12 |
| **合计** |  | **47** | **94** | **282** |

因此，R=1 是 47 对/94 个单元，R=3 是 141 对/282 个单元。runner 的默认重复次数仍为 3；要运行 94 单元矩阵，必须显式传入 `--repetitions 1`。`main` 表示机制消融主表输入：四个非 PPI 数据集的完整图和 PPI 原始 component 0，不是 zkGAT 主实验。`--input-scope ppi_full` 只选择全量 PPI 附录，不能用 component 0 代替。

主矩阵的 PPI 输入使用原始 component 0（N=1546、E=17699、G=1）。全量 PPI 附录使用 component ID 0..294，不筛图、不裁边，并保留完整的节点和边界指针。Feature Folding 的 D=1..8 是同一真实 8 维隐藏特征的前缀宽度；Shared Neighborhood 固定 D=8。数据、模型、预处理、量化、安全参数、SRS 和 CUDA backend 由对应正式配置绑定。

## 预检与正式启动

正式 `launch` 会执行真实输入准备和运行所需的严格检查。单独的 metadata-only 预检是可选的；它不生成 proof 输入，也不运行 Prove 或 Verify，因此不必在直接启动前重复执行。下面的 `RUN_ID` 为每次启动创建新输出根，避免覆盖既有结果：

```bash
cd /home/pzh/project
RUN_ID="$(date +%Y%m%dT%H%M%S%N)-$$"
python3 scripts/run_mechanism_ablation.py launch --device 0 --repetitions 1 \
  --mechanism all --input-scope all \
  --output-root "results/mechanism_ablation_final/all_47rep1_${RUN_ID}"
```

`launch` 在后台运行，使用独占 GPU 检查；不要在同一设备上并行启动其他计算任务。runner 不会代替用户终止外部 GPU 进程。若需分开运行子矩阵，应按顺序启动并为每次运行指定新的 `--output-root`：Feature Folding 主表用 `--mechanism feature_folding --input-scope main`（R=1 时 80 单元），Shared Neighborhood 主表用 `--mechanism shared_neighborhood --input-scope main`（10 单元），全量 PPI 附录用 `--mechanism all --input-scope ppi_full`（4 单元）。不要将三个会话同时提交到同一 GPU。

启动输出会给出 `launcher_dir`。将该实际路径填入以下命令，再从 `runner.log` 读取会话目录：

```bash
LAUNCHER_DIR=/absolute/path/printed/as/launcher_dir
grep 'MECHANISM_SESSION_STARTED' "$LAUNCHER_DIR/runner.log"
SESSION=/absolute/path/from/the/output/path
cat "$SESSION/status.json"
tail -f "$SESSION/session.log"
```

`tail -f` 是只读日志跟随；按 Ctrl-C 只会停止跟随，不会停止后台 runner。不要用同一个输出根重启已启动或已结束的会话。

## 结果与验证状态

每个正式单元执行一次 CUDA Prove，随后执行一次独立 CPU Verify；只有 Verify 成功及安全账本检查通过后才计为已完成。正常完成时，读取会话下的中文 `summary.md` 和 UTF-8 BOM `results.csv`。CSV 每个配置、重复和版本占一行，记录 Original/Control 对比、原生在线 Commit/Prove/Verify 时间、完整子进程时间、proof 字节和证据路径。未运行或未完成单元明确标记状态，不以 0 代替。无需在单元成功后再次运行 Verify；`audit --reverify` 会重复验证所有 proof，不属于正常完成流程。

本工作区已完成一轮正式矩阵：94/94 单元独立验证通过，47/47 配置对完成，R=1。结果位于 [`../results/mechanism_ablation_final/all_47rep1_20261025/summary.md`](../results/mechanism_ablation_final/all_47rep1_20261025/summary.md) 和 [`../results/mechanism_ablation_final/all_47rep1_20261025/results.csv`](../results/mechanism_ablation_final/all_47rep1_20261025/results.csv)。PPI component 0 的 Feature Folding D=8 配对密码学验证通过，但 Original Prove 曾与外部 GPU 工作重叠约 35 秒，因此该配对保留在结果中并排除于性能比较之外；其余 46 对用于性能统计。

PubMed Feature Folding D=1 Original 曾有一次 Verify 拒绝。该次尝试及回执保留；之后在同一 proof 上执行 verify-only 重试并通过，没有重做 Prove。此前通过的单元也未重跑。

## 中断后的续跑限制

runner 拒绝已存在的输出根，普通 `launch` 不会自动跳过已成功单元，也不会为任意中断会话生成可续跑的 provenance closure。CLI 提供的 `continue` 只适用于已有外层 `aggregate_provenance/segment_000_original_closure.json` 和父会话 `provenance_segments/segment_000_original/snapshot_manifest.json` 等闭合证据的会话；常规 `launch` 不会创建这些文件。因此不要把 `continue` 当作通用恢复命令，也不要以新的完整矩阵重跑覆盖已有部分结果。保留中断会话、日志和产物，先依据其状态确认哪些单元已经通过，再确定受支持的续跑路径。

## 构建

若需构建机制 runner，沿用 `build_cuda` 中已配置的 CUDA Release 设置，只构建 `gatzk_mechanism` 目标：

```bash
cmake --build build_cuda --target gatzk_mechanism --parallel 2
```

无需重新生成 SRS 或重建主实验入口。机制回归测试目标与正式实验启动相互独立。

## 历史结果

早期 session `results/mechanism_ablation_final/mechanism_20261010T063825Z_9fe0a93c236b/` 使用旧的 11 组矩阵（包含 D=1/2/4 子集及 PPI component 0+9 的 G=2 组）。该 session 仅作为历史 artifact 保留，不代表当前 47 组矩阵，也不会被新 runner 覆写或并入新结果。
