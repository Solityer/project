# 主实验修复、性能与收敛报告

日期：2026-07-26（Asia/Shanghai）

## 结论边界

本轮完成了 Cora 启动异常的真实根因修复、公共 SRS 资源预检修复、ogbn-arxiv 包络
修复、正式 CLI fail-closed、canonical proof 工件保存、阶段性能候选取舍、CPU/CUDA
构建与受控回归、五套配置启动预检和仓库清理。

本轮没有运行 Cora、Citeseer、Pubmed、PPI 或 ogbn-arxiv 的完整正式 proof，也没有运行
五数据集乘 CPU/GPU 的十组矩阵。Citeseer、Pubmed、PPI 和 ogbn-arxiv 的 independently
trusted SRS 尚未发布，因此当前不能声称这四个正式行已具备完整运行条件。

## `vector::reserve` 根因与修复

观测位置是：

```text
[progress] context:static:V_h0_W_col0:start
fatal: vector::reserve
```

调试器在 `std::__throw_length_error` 处捕获的公共正式调用链是：

```text
gatzk_run
  -> protocol::build_context
  -> add_static_commitment
  -> crypto::KZG::commit
  -> make_commitment
  -> commit_polynomial_point
  -> g1_multi_scalar_mul_profiled
  -> msm_execute_chunked
  -> msm_execute_single / MCL MSM
  -> std::vector::reserve
```

旧实现先计算：

```text
task_count = min(worker_count * 2, count)
chunk_size = ceil(count / task_count)
begin = task * chunk_size
end = min(count, begin + chunk_size)
count_for_task = end - begin
```

当有效 scalar 数不能被 ceil-sized chunk 整除时，尾部 task 可出现 `begin >= count`。
此时无符号 `end - begin` 下溢为巨大 `size_t`，MCL/Pippenger 把它当作输入长度并在
`reserve` 抛出 `length_error`。所以异常不是 Cora 内存不足、SRS 文件太小、配置要求裁剪
或模型维度过大。

`src/crypto/curve.cpp` 已改为 quotient/remainder 平衡分块：

```text
base = count / task_count
larger = count % task_count
begin = task * base + min(task, larger)
end = begin + base + (task < larger)
```

由于 `task_count <= count`，每个 task 都是非空合法区间。`tests/test_main.cpp` 新增 257
和 1009 个 dense scalar 的非整除回归，逐点与单线程 MCL 结果比较。标准测试通过。

修复后的 Cora 正式 Context 在调试器中完成全部 public/static commitment，包括 h0 至 h7
的所有 W 列、attention 参数、输出 W/bias，并到达
`build_protocol_compiler_artifact` 断点。该检查在 trace/prove 前停止，不是完整 Cora
实验。

已排除的假设：

- 不是通过捕获异常继续；
- 不是通过裁剪数据或缩小维度；
- 不是 Cora 专用分支；
- 不是 SRS 重生或增大内存后“掩盖”；
- 不是静态承诺内容、顺序或 KZG 方程改变。

## 正式资源与工件修复

`src/protocol/prover.cpp` 的 quotient materializer 对七域都使用 `4|D|` coset，但旧 Context
预检只要求 `2|FH|` coefficient powers。该检查已改为 `4|FH|`，使次数不足在 Context
加载期 fail closed，而不是进入证明后才失败。

ogbn-arxiv 的最大 coefficient object 长度是 21,675,904，固定 edge domain 是
8,388,608，每个 chunk 可用 8,388,607 个位置，因此需要 3 个 chunk。包络中的旧值 2
已修为 3，并由当前 SRS compiler 通过。

五套精确域和 `D_max` 的 compiler preflight 均成功：

| 数据集 | `D_max` | `FH` | `edge` | `in,d,cat,C,N` |
| --- | ---: | ---: | ---: | --- |
| Cora | 16,777,215 | 4,194,304 | 32,768 | 2048,16,128,8,4096 |
| Citeseer | 67,108,863 | 16,777,216 | 32,768 | 4096,16,128,8,4096 |
| Pubmed | 67,108,863 | 16,777,216 | 262,144 | 512,16,128,8,32768 |
| PPI | 16,777,215 | 4,194,304 | 1,048,576 | 64,16,16,128,65536 |
| ogbn-arxiv | 134,217,727 | 33,554,432 | 8,388,608 | 256,16,16,64,262144 |

Cora 新编译 artifact 与已发布 SRS 内 artifact 字节级相同。其余 compiler artifact
内部 canonical `artifact_hash`：

- Citeseer: `f88a76d5de4b237e9ad622358801baf888f30519e68ac304168f33c0e85bf93b`
- Pubmed: `18434a28e2e08aaf000c79e7251ecb2a843e73b85519b109646e22a672a9e6b6`
- PPI: `ff30d1e15b3d76fa9685ea20adb5bc8de245e419c311f3f68278b986b358cae1`
- ogbn-arxiv: `115aaa7c9f00ffbdd87bfae25abb9bb622503dec85b9cbd98b6ada99ac619bde`

四份 artifact 保存在 `artifacts/srs/compiler_artifacts/`。它们是 setup 输入，不是 SRS。

## 唯一生产路径与正式 CLI

`gatzk_run` 仍是唯一正式入口。unit/controlled benchmark 调用同一生产 Context、Trace、
Prover、KZG、serializer、parser 和 Verifier；阶段 benchmark 直接调用生产函数，没有
复制协议算法。

本轮入口修复：

- 未知参数直接失败；
- 所有带值选项的缺失值直接失败；
- `on/off` 非法值直接失败；
- CPU/CUDA 每项优化记录实际状态、来源和适用性；
- CPU 中 CUDA 项记录 `not_applicable`；
- CUDA kernel 计数只覆盖 measured run，不混入 warmup；
- 请求 CUDA 但 measured run 为零 kernel 时直接失败；
- `proof.bin` 保存真实 canonical bytes，长度必须等于 `proof_size_bytes`；
- manifest 的 `proof_hash` 改为 `proof.bin` 的 SHA-256。

受控 CPU 直跑产生 `VERIFY_OK`，`proof.bin` 为 250,448 bytes，文件 SHA-256 与 manifest
均为 `689d460bc6ba629bbe3b31f72b2c3c46355cce16f7798035602f7724e8725f4a`。
该值只属于本次随机化 unit proof，不是正式数据集 proof size。

## 性能结果与权衡

稳定冻结的三次受控生产路径证据：

| 指标 | 基线中位数 | 保留实现中位数 | 绝对变化 | 比例 |
| --- | ---: | ---: | ---: | ---: |
| commitment | 1682.144 ms | 976.702 ms | -705.442 ms | -41.937% |
| prove | 2143.149 ms | 1823.338 ms | -319.811 ms | -14.922% |
| verify | 76.633 ms | 78.102 ms | +1.469 ms | +1.917% |
| 三项合计 | 3897.541 ms | 2876.125 ms | -1021.416 ms | -26.207% |
| wall | 4110.572 ms | 3088.730 ms | -1021.842 ms | -24.859% |

接受 verify 小幅增加的原因是：增加仅 1.469 ms，远小于 commitment/prove 的
1025.253 ms 下降；三项合计净降 26.207%；parser、verifier 语义、接受条件和计时边界
均未改变；未把大规模 prover 工作转移给 verifier。

本轮当前源码的三次生产路径诊断三项总和为 3517.402、2898.343、2953.748 ms，全部
完成 canonical serialization、strict parse 和 verify。主机同时存在无关两核持续负载，
第一组是明显干扰离群值，因此这些数字只证明当前源码没有丢失既有效果，不替代上表的
稳定基线。

当前主要热点：

- dynamic commitments：稳定记录约 749.159 ms / 26.048%；
- quotient non-commitment：约 1048.971 ms / 36.472%；
- 其中 `t_edge`：约 940.858 ms / 32.713%，仍是最大单核；
- domain + external openings：约 401.395 ms / 13.956%；
- non-commit trace：约 279.192 ms / 9.707%。

保留项包括持久 MSM worker pools、2x 大任务分块、动态 commitment batching、严格的
proof-local pointer/semantic cache、point-group opening 索引与自适应调度、mixed
pairing 和受限内部 claim reuse。

本轮测量后回滚的 32-task `t_edge` overdecomposition：局部样本有收益，但三次完整受控
生产路径三项总和为 3004.489、2949.553、3113.691 ms，external opening 出现明显长尾，
没有稳定净收益，源码已恢复 20-worker 策略。此前 custom Pippenger、OpenMP outer split、
fixed-base table、多种 opening scheduler、t_edge prebound/compiled-expression 方案和
coset scratch reuse 仍保持删除，没有无证据重试。一次 CUDA `native` 架构构建候选虽能
编译，但 weighted-evaluation 内核运行失败，已完整回滚；恢复的 CUDA 构建重新通过
90-kernel 精确比对。

CPU 端到端受控证据满足稳定比较；CUDA 正式热点 benchmark 在 NVIDIA GB10 上执行
90 个 kernel，所有 CPU/CUDA 输出精确一致。unit 输入低于 size-aware 阈值，CUDA
`gatzk_run` 因 measured run 零 kernel 按设计失败，因此本轮没有可接受的 CUDA
端到端受控时间，不能用 CPU 数字代替。四个代表性正式 SRS 缺失且完整 Cora GPU 被本轮
禁止运行，这是未闭合限制。

### 第三阶段续轮新增结果

本续轮先用当前源码重新冻结五次 CPU 基线，中位数为 commitment 999.433 ms、
prove 1845.469 ms、verify 76.099 ms、三项合计 2924.897 ms。此前 26.207% 不计入
本续轮新增成果。

锁定证据：HEAD `18d92aa1f5c532f57f68b7ec173db8c06cec3d70`，代码/配置树
（`CMakeLists.txt`、`include/`、`src/`、`tools/`、`tests/`、`configs/`、
`scripts/`）SHA-256
`475f913a32b209f8c82d67c6f5cd33fbeda1fcba53a9b892e8e13be5b29e1993`，
CPU/CUDA `gatzk_run` SHA-256 分别为
`05d73247e704fc085646ade04559b1341cec66d8c6e9d1ed42f6bad8ebe651ed` 和
`53580a10ab3f184b39f9729ab376e86cbea74ab63d3183be601d09e929eb7ddc`。
硬件为 10 Cortex-X925 + 10 Cortex-A725、121 GiB RAM、NVIDIA GB10/driver
580.142；两套均为 Release + native-architecture 构建。

保留两组公共生产改动：

1. 同域 proof-local model reblind 在对象数不少于 4 时调用正式
   `KZG::commit_batch`；1 至 3 个对象严格保留原逐对象顺序。8 列、domain
   128/2048/4096 的 commitment stage 中位分别从 34.500/57.290/120.680 ms
   降至 28.583/47.778/96.732 ms，即 -17.151%/-16.604%/-19.846%。3 列、
   domain 16 的 batch 反而 +5.182%，因此没有让小组进入默认 batch。
2. CUDA 的单点、补零列和多旋转 domain evaluation 直接读取 canonical 列及真实
   长度；rotation 由 kernel 索引完成，多点合为一次 3-D launch 和一次归约，并以
   256 MiB 限制 partial storage。domain 65536、32 列、3 点的 full/padded
   中位从 4.591/4.948 ms 降至 1.596/1.567 ms，即 -65.236%/-68.331%；
   单点 padded 从 1.462 降至 0.734 ms，即 -49.795%。

CUDA 最终扩展 shape 回归在 NVIDIA GB10 上真实执行 129 个 kernel，所有结果 exact
match。padded 三点在 `32768 x 64`、`262144 x 8`、`1048576 x 2` 和
`4194304 x 1` 的中位分别是 1.585、2.631、3.242、6.825 ms，未出现随 domain
增长的异常放大。固定可比子集的 kernel 数由 175 降为 105；扩展套件额外 24 个
kernel 来自四种 shape 检查。

续轮 clean CPU canary 的 commitment/prove/verify/合计中位为
1013.382/1839.767/79.591/2921.514 ms，相对新基线分别
+1.396%/-0.309%/+4.589%/-0.116%，wall 仍为 3.20 s。unit 模型的同域 reblind
组少于 4，因此该 canary 按设计不触发新增 batch；它证明未转移计时和整链无显著回退，
不能冒充正式规模 CPU 总收益。之后另一用户 `zkdl_cpu_cli` 与编辑器进程提高主机负载的
样本已排除，没有用来制造正负结论。

本续轮还测量并完整删除了 dynamic `t_edge` scheduling/temp reuse、shifted-base、
共享 opening inverse workspace、LTO、`-mcpu=native`、降低 commitment threshold、
capacity-weighted pinned range、12-worker medium MSM 和 16-worker `t_edge` cap。
其中 16-worker 在五种局部正式形态汇总有收益，但七组完整受控 proof 中 `t_edge`
仅从 949.160 降至 948.036 ms，三项合计反而从 2926.495 升至 2936.719 ms，故回滚。

当前 CPU 大头 `t_edge`、dynamic MSM、opening witness、FFT/coset、内存与线程调度均
已通过局部和整链候选闭环；未保留只在极小输入变快或把长尾/内存转移到其他阶段的方案。
CUDA field-domain 路径取得新的稳定大幅下降；curve MSM、KZG witness 和 pairing 因
现有依赖没有 production GPU curve backend 而仍在 CPU，这只限制这些具体阶段。
由于 unit 低于 CUDA dispatch threshold、完整 Cora 被禁止，本报告仍不声称 CUDA
端到端 commitment/prove/verify 已测量。

## 仓库收敛清理

已删除：

- 五个重复旧正式配置；
- 独立 PPI representative secure 配置/SRS 包络；其两图边界测试现在直接加载
  `ppi_secure.cfg` 后只缩小 `batch_graphs`，继续调用生产 loader/normalizer；
- 过度复杂且设置固定 timeout 的历史矩阵 runner、runner-only test 和耦合导出器；
- 一次性 LReLU range 诊断程序及 CMake target；
- Cora v1 SRS 副本，并把保留 KZG benchmark 更新到 v2；
- 460 个历史 build log、失败候选、raw proof、profiling、旧审计与重复性能文件；
- 历史运行日志、旧 GPU 非正式输出和 Python cache。

保留：

- 两份最高只读规范；
- 唯一生产代码、五套 secure 配置和 unit security fixture；
- 五套正式数据 cache/checkpoint；
- Cora v2 SRS、unit SRS 和四份待 setup compiler artifact；
- 安全回归、strict parser/verifier 测试和直接调用生产函数的阶段 benchmark；
- 数据/cache/checkpoint/SRS 可复现生成工具和可选结果整理脚本；
- 本报告与压缩后的 Phase 3 性能记录。

## 构建与验证

- CPU Release 全构建：通过；
- CUDA Release 全构建：通过；
- CPU `ctest`：2/2 通过；
- CUDA `ctest`：2/2 通过；
- `GATZK_RUN_UNIT_SECURITY_TESTS=1 GATZK_RUN_PPI_BOUNDARY_TESTS=1
  build/gatzk_tests`：通过；
- 257/1009 非整除 MSM 回归：通过；
- CPU direct unit `gatzk_run`：`VERIFY_OK`；
- CUDA production hotspot benchmark：扩展套件 129 kernels，全部 exact match；
- CPU build 请求 CUDA：fail closed；
- CUDA unit measured run 零 kernel：fail closed；
- unknown/missing/invalid CLI negative checks：全部按预期失败；
- `--export-tag` 非单一安全路径分量：fail closed；
- 五套 SRS compiler preflight：通过；
- Cora正式 Context 到 compiler artifact：通过，未进入 trace/prove；
- Citeseer/Pubmed/PPI/ogbn-arxiv：data、model、config、domain 均完成生产加载，随后在缺失
  SRS manifest 处按预期失败；
- `git diff --check`：通过；
- 两份只读规范 SHA-256 未变化。

## 当前未完成事项

1. Citeseer、Pubmed、PPI、ogbn-arxiv 的 independently trusted SRS 尚未发布。README
   给出了精确 setup 输入、目标目录和命令。
2. `static_scan_no_tau.py --require-all-configured-srs` 在四套 SRS 安装前应当失败；安装后
   必须重新执行并通过。
3. 五个完整正式数据集的 CPU/GPU 十行没有运行，正式结果、规模趋势和 GPU 端到端收益
   尚未验收。
4. `t_edge` 和 dynamic commitments 仍是 CPU 大热点；本续轮已对 evaluator、
   materialization、MSM worker、commit batch、opening、FFT/coset 和调度候选做了
   局部与整链闭环，未找到新的稳定 CPU 总收益。未来若引入生产级 GPU curve
   MSM/KZG/pairing，必须重新走协议等价、安全回归和 CPU/GPU 分别测量。

正式命令、日志、结果读取、消融和 SRS 安装步骤见 `README.md`。
