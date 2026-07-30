# 主实验第三阶段性能优化进展报告

日期：2026-07-30（Asia/Shanghai）

## 当前状态

本轮已在唯一生产路径 `gatzk_run` 上完成代码分析、成组修改、CPU/CUDA Release
构建、定向测试、完整安全回归、生产 stage benchmark、quotient profiling 和一次
当前源码的 Cora CPU 完整证明。该 Cora 证明生成 canonical `proof.bin` 并
`VERIFY_OK`。

第三阶段尚未收敛。已经完成并必须保留的是部分正确性、file-backed 生命周期、内存、
quotient 和 verify 优化；commitment 尚无可信下降，当前完整 Cora prove 仅下降
4.33%，三项合计没有显著净收益。commitment、prove 和总体时间的大幅优化目标仍未
完成，当前不得标记第三阶段完成。

两份最高规范 `GAT-ZKML-多层多头.md` 和
`PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md` 未被修改、重命名、移动或删除。
模型语义、PPI 多图边界、量化与 SignedRoundDiv、证明关系、KZG/SRS、proof schema、
Fiat–Shamir transcript、canonical serialization、strict parser、公开/私有定义和
Verifier 接受逻辑均未改变。

## 本轮识别并闭环的主要问题

| 问题 | 根因 | 正式修改与位置 | 结论 |
| --- | --- | --- | --- |
| file-backed FH 对象丢失 blinding tail | 私有 FH 多项式移入 `spilled_polynomials` 后，quotient materializer 先命中公开 lazy identity 分支，只重建语义列，绕过已注册的独立 `Z_FH` blinder | `src/protocol/prover.cpp` 的 spilled 对象优先解析、`materialize_spilled_lazy_feature_coset`、point-eval memoization 和 debug self-check spill 读取 | 修复共同生命周期错误；exact quotient self-check 和完整安全回归通过。PPI/ogbn-arxiv 原 `P_h0_R_src` opening material 错误所涉及的正式机制已局部闭环 |
| FH lazy 对象直接全量重载会放大 I/O/FFT | 正确性直修会为 table/query/m/R 各自重载整列并做大 coset 变换 | 从真实公开依赖只构造一次 canonical semantic coset，再为每个私有对象加入其独立 blinder；Query 使用两对象 blinder 差值 | 正确性不变，t_FH prewarm 从直修的 53.904 s 降至 34.139–38.049 s |
| quotient 文件映射产生同步写回和重复冷读 | 临时、已 unlink 的 coset 文件仍执行同步 `msync(MS_SYNC)`，并在消费前整段 `MADV_DONTNEED` | `src/protocol/prover.cpp` 保留 `MAP_SHARED`、顺序访问和消费后前缀回收，删除同步写回和首次整段丢页 | exact self-check 通过；不再把临时 quotient 数据强制写回后立即重读 |
| `t_FH`/`t_edge` 每点重复求边界分母逆 | 每个 quotient 点分别计算 Lagrange 边界逆元，规模随 coset 点数放大 | `include/gatzk/protocol/quotients.hpp`、`src/protocol/quotients.cpp` 和 `src/protocol/prover.cpp`：8192 点块内批量求逆并向生产 evaluator 传入预计算边界 | 新增直接公式/完整 `evaluate_t_fh` 等价回归；exact self-check 通过 |
| 相同语义 coset 被独立物化 | 多个关系对象具有相同 canonical evaluation，但独立 blinding 使它们不能直接 alias | `src/protocol/prover.cpp`：共享 canonical coset，只对每个对象追加自己的 lazy blinder；未 blinding 对象才允许直接 alias | 完整安全形态的 t_edge 临时文件由 2270 个/9,521,070,080 bytes 降至 1690 个/7,088,373,760 bytes |
| public evaluation backend 启动即复制全部列 | Prover/Verifier registry 构造时把所有公开多项式完整复制成全局 interleaved buffer；大域上造成高峰内存、复制和缓存压力 | `include/gatzk/algebra/eval_backend.hpp`、`src/algebra/eval_backend.cpp`：只索引 canonical 列，实际命中的小 subset 才惰性交织；CPU/CUDA fallback 直接读 canonical 值 | 完整安全回归峰值 RSS 从 25,899,732 KiB 降至 17,250,228 KiB，下降 33.40% |
| lazy backend 的静态 registry 持有悬空非 owning 指针 | backend 改为读 canonical 列后，跨 proof 的静态 cache 可能越过 Context/Trace 生命周期 | `src/protocol/prover.cpp`、`src/protocol/verifier.cpp`：registry 改为 proof/verification-local；构造只建索引 | 消除生命周期风险；直接正式 proof verifier 仍通过 |
| 正式 proof 的 verifier-only benchmark 被 unit 限制误拦截 | `proof_codec_stage_bench` 对 generate/parse/verify 都要求小配置 | `tools/benchmarks/proof_codec_stage_bench.cpp`：只在 generate 限制小配置，parse/verify 直接使用生产 parser/verifier | 可对既有正式 canonical proof 做独立当前源码验证，不形成第二条 runner |

此前已经保留且本轮继续保护的公共生产优化包括：

- `KZG::commit_batch`、持久 bounded MSM workers、proof-local semantic cache；
- point-group opening、单点 folded witness、mixed pairing；
- fixed-length Taylor shift 与精确 cyclic-tail correction；
- LogUp denominator 合并批量求逆与 accumulator reuse；
- CUDA canonical 列、多 rotation 单次 launch 的 field-domain 路径。

## 性能结果

### 冻结前完整 CPU 基线

这些是用户给定且目录中 canonical 工件可核对的完整结果。它们属于优化前源码；对于
当前源码应标记为 `stale-for-current-source`，但仍是本轮前后比较基线。

| 数据集 | commitment | prove | verify | proof bytes | 状态 |
| --- | ---: | ---: | ---: | ---: | --- |
| Cora | 34.849 s | 159.022 s | 1.199 s | 1,462,631 | `VERIFY_OK` |
| Citeseer | 78.635 s | 429.439 s | 3.373 s | 1,493,238 | `VERIFY_OK` |
| Pubmed | 113.707 s | 1020.030 s | 12.653 s | 2,743,639 | `VERIFY_OK` |
| PPI | — | — | — | — | 旧二进制失败；不是 `VERIFY_OK` |
| ogbn-arxiv | — | — | — | — | 旧二进制失败；不是 `VERIFY_OK` |

### 当前源码 Cora 完整 CPU 结果

命令：

```text
GATZK_RUN_PROGRESS=1 build/gatzk_run \
  --config configs/cora_secure.cfg --compute-backend cpu \
  --export-tag phase3_final_optimized_20260730
```

产物：
`runs/cora_secure/phase3_final_optimized_20260730/warm/`

| 指标 | 优化前 | 当前源码完整运行 | 表面变化 |
| --- | ---: | ---: | ---: |
| commitment | 34.849 s | 43.858 s | +25.85% |
| prove | 159.022 s | 152.137 s | -4.33% |
| verify | 1.199 s | 0.713 s | -40.56% |
| 三项合计 | 195.070 s | 196.708 s | +0.84% |
| `proof_size_bytes` | 1,462,631 | 1,462,616 | 仅记录 |

状态为 `VERIFY_OK`，计时口径仍是 `exclusive_commit_prove_verify_v2`。

该运行期间主机同时存在三个不属于本项目的重负载：两个训练任务分别长期占用约一个
CPU 核和 2.2/8.3 GiB RSS，其中一个占用 GPU；另有一个 CPU 测试任务。系统已有约
2.1 GiB swap。当前进程自身没有发生 swap，但 commitment 的 FH 部分由 3.600 s
升至 8.650 s、dynamic finalize 由 0.007 s 升至 1.694 s，opening/trace 也出现
明显长尾。因此这次运行可以冻结正确性和工件，不能作为受控性能上限；报告不使用
模型估算值替代正式结果，也不把 commitment 的表面回退归因于保留代码。

### 同源 stage/profiling 证据

| 生产阶段 | 旧同形态证据 | 当前源码证据 | 变化 |
| --- | ---: | ---: | ---: |
| 完整安全形态 t_edge prewarm + evaluate | 44.339 s | 29.075 s | -34.43% |
| t_FH prewarm | 42.758 s | 38.049 s | -11.01% |
| Cora `quotient_build_ms` | 126.253 s | 102.171 s | -19.08% |
| Cora `quotient_t_edge_ms` | 55.326 s | 41.185 s | -25.56% |
| Cora `quotient_t_fh_ms` | 57.545 s | 47.497 s | -17.46% |

当前完整安全回归中的 t_FH 分解为 prewarm 38.049 s、evaluate 6.109 s、
coefficients 5.493 s；exact hidden-coefficient self-check 通过。Taylor shift
在 degree 32769 的稳定 warm 时间为 40.753–41.592 ms；LogUp steady 为
3.010–3.037 ms。

当前 verifier 对优化前 Cora canonical proof 的独立三次生产验证中位数为
512.707 ms，均 `verify_ok=true`；相对原完整运行的 1,199.368 ms 下降 57.25%。
其中 metadata 14.888 ms、transcript 101.482 ms、domain opening 8.452 ms、
registry 42.889 ms、quotient 313.063 ms、external opening 28.428 ms。

commitment 本轮没有可接受的新整链降幅。现有 3610 个动态 commitment 已使用公共
batch、semantic reuse、sparse/tail MSM 和 bounded workers；跨 transcript barrier
合并对象会改变协议，已测 custom Pippenger、OpenMP outer split、fixed-base table、
更低 batch threshold 和不同 worker caps 均无稳定净收益。现有 CUDA/MCL 依赖也没有
可审计的生产 G1 MSM/KZG witness/pairing backend，因此不把“新写一套 GPU 曲线密码
实现”列作本阶段可保留候选。

## 内存、临时磁盘和 CPU/CUDA

- 完整安全形态峰值 RSS：25,899,732 KiB → 17,250,228 KiB，下降 33.40%。
- 同形态 t_edge file-backed coset：9,521,070,080 → 7,088,373,760 bytes，
  下降 25.55%；当前还通过 lazy blinder view 避免 5,146,411,008 bytes 物化。
- t_FH 当前保持 9 个 file-backed coset、4,831,838,208 bytes。临时文件创建后
  立即 unlink，并在消费后按前缀 `MADV_DONTNEED`，正常或异常退出不留下正式依赖。
- 清理了 402 个旧顶层 trace spill 和 3 个中断 proof spill，共
  73,488,179,904 bytes（68.44 GiB）。这些是可重建临时数据，不是 proof、SRS、
  数据、模型或正式基线。
- CPU 路径仍承担 G1 MSM、KZG witness、pairing、NTT/IFFT、quotient
  materialization；CUDA 路径仍只加速既有 Fr/domain 热点，没有静默 CPU
  fallback。CUDA Release 构建和安全回归通过；因 GPU 被其他任务占用，本轮 CUDA
  stage 数字不作为性能证据，也没有运行完整 CUDA 数据集。

## 保留、回滚和清理

本轮保留：

- spilled private FH 优先解析和独立 blinder correction；
- semantic coset reuse、lazy blinder view 和消费后页回收；
- t_FH/t_edge 批量边界求逆；
- canonical-column lazy evaluation backend；
- proof-local Prover/Verifier backend registry；
- FH boundary 等价回归，以及允许既有正式 proof 直接走生产 verifier 的 stage
  benchmark 修正。

本轮测量后回滚或删除：

- 泛化的 t_FH 全标签 prewarm 分组：没有稳定收益；
- 直接为每个 spilled lazy FH 对象全量重载/FFT：正确但 prewarm 达 53.904 s，
  被 semantic+blinder 实现替代；
- 已删除的完整 identity materializer、fused feature-table 旧 helper、
  selector boolean quotient coset context cache；
- 静态 Prover/Verifier backend registry；
- 已确认倒退的 8 GiB opening worker-budget scheduler、32-task t_edge、
  custom Pippenger、OpenMP MSM split、fixed-base table、coset scratch reuse
  等历史方案没有重新引入。

旧 SRS 命名目录没有删除：它们包含不同 manifest/ceremony provenance，属于潜在有效
基线而不是可证明无用的缓存；当前五套配置只引用无后缀的正式目录。未修改、生成或替换
任何正式 SRS。

## 已通过的验证

- CPU Release 全构建：通过；
- CUDA Release 全构建：通过；
- CPU：
  `GATZK_RUN_UNIT_SECURITY_TESTS=1 GATZK_RUN_PPI_BOUNDARY_TESTS=1
  build/gatzk_tests`：通过；
- CUDA 同一组 unit/PPI boundary：通过；
- CPU 完整安全回归，启用 unit、PPI boundary、full security 和 exact quotient
  self-check：通过，wall 11:00.57，峰值 RSS 17,250,228 KiB；
- Taylor shift、LogUp、t_edge evaluator、quotient transform、KZG opening
  生产 stage benchmarks：通过；
- 当前生产 verifier 直接验证优化前 Cora canonical proof：三次均通过；
- 当前源码 Cora CPU 完整路径：`VERIFY_OK`；
- `python3 tools/security/static_scan_no_tau.py --require-all-configured-srs`：通过；
- 主实验变更的 whitespace 检查：通过。

## PPI 与 ogbn-arxiv 的状态

PPI 和 ogbn-arxiv 没有在本轮重新完整运行。旧二进制的
`Taylor shift is missing source opening material: P_h0_R_src` 不能计为当前源码
结果。共同 file-backed opening material 生命周期和同机制正式路径已经修复，构建、
定向测试、安全回归、PPI 多图边界和 exact quotient 检查均通过；两者当前状态仍是：

> 局部验证通过，最终全量待冻结版本验证。

不得写成已经 `VERIFY_OK`。

## 尚未完成的主要热点

当前仍耗时的阶段是 dynamic commitment/KZG MSM、t_edge/t_FH 正式 quotient 方程、
FH opening witness 和 verifier quotient evaluation。需要继续对以下生产路径做
架构级优化：

- commitment 的对象组织、同基 MSM、跨对象 base reuse、批处理和 CPU/CUDA 分工；
- t_edge/t_FH 的 coset 表示、重复 NTT/IFFT、跨关系公共子表达式和流水执行；
- FH opening witness 的折叠、系数生命周期、MSM 输入组织和并行流水；
- 当前完整结果中的 trace/opening 长尾及总体资源竞争。

第 10 项当前结论为：**仍存在可实施且可能产生显著整链收益的架构级方向，第三阶段
不得结束。**

第 11 项当前结论为：**主实验保持唯一正式实现和唯一正式执行路径。**
`src/main.cpp` 是唯一生产入口；测试和 stage benchmark 直接调用 `src/` 中生产函数，
没有替代 Trace/Prover/Verifier/KZG、数据集专用路径或隐藏性能分支。

## 冻结后最终完整验证命令

先在无其他重负载、内存和 GPU 空闲的同一台机器上重新构建并完成门禁：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DGATZK_BUILD_TESTS=ON -DGATZK_ENABLE_CUDA_BACKEND=OFF
cmake --build build -j10
cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release \
  -DGATZK_BUILD_TESTS=ON -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j10

GATZK_RUN_UNIT_SECURITY_TESTS=1 \
GATZK_RUN_PPI_BOUNDARY_TESTS=1 \
GATZK_RUN_FULL_SECURITY_TESTS=1 \
build/gatzk_tests
python3 tools/security/static_scan_no_tau.py --require-all-configured-srs
```

CPU 五行应顺序运行：

```bash
GATZK_RUN_PROGRESS=1 build/gatzk_run --config configs/cora_secure.cfg --compute-backend cpu --export-tag phase3_frozen_cpu
GATZK_RUN_PROGRESS=1 build/gatzk_run --config configs/citeseer_secure.cfg --compute-backend cpu --export-tag phase3_frozen_cpu
GATZK_RUN_PROGRESS=1 build/gatzk_run --config configs/pubmed_secure.cfg --compute-backend cpu --export-tag phase3_frozen_cpu
GATZK_RUN_PROGRESS=1 build/gatzk_run --config configs/ppi_secure.cfg --compute-backend cpu --export-tag phase3_frozen_cpu
GATZK_RUN_PROGRESS=1 build/gatzk_run --config configs/ogbn_arxiv_secure.cfg --compute-backend cpu --export-tag phase3_frozen_cpu
```

CUDA 五行也应顺序运行：

```bash
GATZK_RUN_PROGRESS=1 build_cuda/gatzk_run --config configs/cora_secure.cfg --compute-backend cuda_hotspots --export-tag phase3_frozen_cuda
GATZK_RUN_PROGRESS=1 build_cuda/gatzk_run --config configs/citeseer_secure.cfg --compute-backend cuda_hotspots --export-tag phase3_frozen_cuda
GATZK_RUN_PROGRESS=1 build_cuda/gatzk_run --config configs/pubmed_secure.cfg --compute-backend cuda_hotspots --export-tag phase3_frozen_cuda
GATZK_RUN_PROGRESS=1 build_cuda/gatzk_run --config configs/ppi_secure.cfg --compute-backend cuda_hotspots --export-tag phase3_frozen_cuda
GATZK_RUN_PROGRESS=1 build_cuda/gatzk_run --config configs/ogbn_arxiv_secure.cfg --compute-backend cuda_hotspots --export-tag phase3_frozen_cuda
```

每行只有 `benchmark.txt` 与 `run_manifest.json` 同时记录
`status=VERIFY_OK`、`full_dataset=true`、canonical proof 长度/hash 一致时才可进入最终
矩阵。PPI 和 ogbn-arxiv 在这些命令成功之前必须继续保持“最终全量待验证”。
