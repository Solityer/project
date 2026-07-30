# zkGAT 正式主实验

本仓库的唯一生产入口是编译后的 `gatzk_run`。协议、模型语义和安全边界以只读文件
`GAT-ZKML-多层多头.md` 与 `PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md` 为最高规范。
小规模回归、阶段 benchmark 与正式数据共用 `src/` 中同一套 Context、Trace、Prover、
Verifier、KZG、canonical serializer 和 strict parser。

第三阶段尚未收敛。部分正确性、file-backed 生命周期、内存、quotient 和 verify
优化已经完成并保留，但 commitment 尚无可信下降，当前完整 Cora prove 仅下降
4.33%，三项合计没有显著净收益；commitment、prove 和总体时间的大幅优化仍在进行。
当前源码 Cora CPU 完整证明为 `VERIFY_OK`。五套配置所引用的正式 SRS 当前均存在，
并已通过严格静态绑定扫描。Citeseer、Pubmed 的既有完整
`VERIFY_OK` 结果属于优化前源码；PPI、ogbn-arxiv 的共同 file-backed opening
material 生命周期错误已局部验证修复，但两者仍必须标记为“冻结版本最终全量待验证”。
本轮没有运行 CPU/GPU 十组矩阵。精确证据和受共享主机负载污染的计时边界见
`MAIN_EXPERIMENT_FINAL_REPORT.md` 与 `PHASE3_PERFORMANCE_OPTIMIZATION.md`。

## 构建

CPU Release：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGATZK_BUILD_TESTS=ON \
  -DGATZK_ENABLE_CUDA_BACKEND=OFF
cmake --build build -j10
```

CUDA Release：

```bash
cmake -S . -B build_cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGATZK_BUILD_TESTS=ON \
  -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j10
```

CUDA 后端是同一协议中的 size-aware Fr 热点后端。KZG MSM、pairing、NTT/IFFT、
quotient materialization 和 opening witness 仍明确在 CPU 上执行。请求
`cuda_hotspots` 时，CUDA 构建、运行时或设备缺失会直接失败；测量区间内零 CUDA
kernel 也会失败，不会把 CPU fallback 标成 GPU 结果。

## 五个正式数据集的直接命令

CPU：

```bash
build/gatzk_run --config configs/cora_secure.cfg --compute-backend cpu --export-tag cpu_default
build/gatzk_run --config configs/citeseer_secure.cfg --compute-backend cpu --export-tag cpu_default
build/gatzk_run --config configs/pubmed_secure.cfg --compute-backend cpu --export-tag cpu_default
build/gatzk_run --config configs/ppi_secure.cfg --compute-backend cpu --export-tag cpu_default
build/gatzk_run --config configs/ogbn_arxiv_secure.cfg --compute-backend cpu --export-tag cpu_default
```

GPU：

```bash
build_cuda/gatzk_run --config configs/cora_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_default
build_cuda/gatzk_run --config configs/citeseer_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_default
build_cuda/gatzk_run --config configs/pubmed_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_default
build_cuda/gatzk_run --config configs/ppi_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_default
build_cuda/gatzk_run --config configs/ogbn_arxiv_secure.cfg --compute-backend cuda_hotspots --export-tag cuda_default
```

`gatzk_run` 不设置固定超时；进程会运行到正常结束、真实错误或用户中断。不写 tag 时
输出到 `runs/<dataset>_secure/warm/`；上面的命令分别写入
`runs/<dataset>_secure/cpu_default/warm/` 和
`runs/<dataset>_secure/cuda_default/warm/`，避免两个后端互相覆盖。一般形式
`--export-tag <tag>` 会写入 `runs/<dataset>_secure/<tag>/warm/`。
tag 必须是最多 96 个 ASCII 字母、数字、点、下划线或连字符组成的单一路径分量，
且不能是 `.` 或 `..`；非法路径会直接报错。

每次成功运行会在控制台输出一行：

```text
dataset=... backend=... status=VERIFY_OK commitment_time_ms=... prove_time_ms=... verify_time_ms=... proof_size_bytes=...
```

运行目录中的正式真源包括：

- `proof.bin`：canonical serialized proof；
- `benchmark.txt`：五个交付指标；
- `run_manifest.json`：计时拆分、后端、优化实际状态、工件 hash 和安全审计状态；
- `metadata.txt`、`proof_block_order.txt`、`commitment_order.txt`；
- `dynamic_commitment_object_profile.csv`：内部 commitment 热点明细。

`run_manifest.json` 的 `proof_hash` 是 `proof.bin` 的 SHA-256，
`proof_size_bytes` 必须与 `proof.bin` 文件长度完全一致。

## 默认优化与消融

Release 构建默认开启：

- `fast_msm`
- `parallel_fft`
- `fft_backend_upgrade`
- `fft_kernel_upgrade`
- `trace_layout_upgrade`
- `fast_verify_pairing`

CUDA 的 `cuda_trace_hotspots` 与 `cuda_field_hotspots` 由
`--compute-backend cuda_hotspots` 选择；CPU 运行中二者记录为 `not_applicable`。
`run_manifest.json` 的 `optimization_flags` 对每项记录实际状态、来源和当前后端适用性。

正常正式运行无需写任何 `--xxx on`。消融时只显式关闭待测项，例如：

```bash
build/gatzk_run --config configs/cora_secure.cfg --compute-backend cpu \
  --fast-msm off --export-tag ablation_fast_msm_off
```

其余可关闭参数是 `--parallel-fft off`、`--fft-backend-upgrade off`、
`--fft-kernel-upgrade off`、`--trace-layout-upgrade off` 和
`--fast-verify-pairing off`。带 `off` 的目录只能作为消融结果，不能与默认完整优化结果
混合统计。未知参数、缺失值和非法值都会直接报错。

## 长时间运行与结果读取

启用阶段进度并保留日志：

```bash
GATZK_RUN_PROGRESS=1 build/gatzk_run \
  --config configs/cora_secure.cfg --compute-backend cpu \
  --export-tag cpu_default \
  2>&1 | tee cora_cpu.log
```

另一个终端可用 `tail -f cora_cpu.log` 查看进度，用 `ps` 或 `top` 查看 CPU、内存和
进程状态。不要为正式主程序包装固定 timeout。

读取正式摘要：

```bash
cat runs/cora_secure/cpu_default/warm/benchmark.txt
jq '{
  status,
  commitment_time_ms,
  prove_time_ms,
  verify_time_ms,
  formal_time_sum_ms,
  proof_size_bytes,
  optimization_flags,
  gpu_kernel_invocation_count
}' runs/cora_secure/cpu_default/warm/run_manifest.json
sha256sum runs/cora_secure/cpu_default/warm/proof.bin
```

可选的 `scripts/export_benchmark_table.py` 只整理已经存在的严格 manifest，不运行协议、
不改变指标，也不是单数据集运行依赖。十个正式运行全部完成后，可分别生成 CPU/GPU
统一结果：

```bash
python3 scripts/export_benchmark_table.py \
  --output-dir results/paper_secure/cpu \
  --run cora=runs/cora_secure/cpu_default/warm \
  --run citeseer=runs/citeseer_secure/cpu_default/warm \
  --run pubmed=runs/pubmed_secure/cpu_default/warm \
  --run ppi=runs/ppi_secure/cpu_default/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_secure/cpu_default/warm

python3 scripts/export_benchmark_table.py \
  --output-dir results/paper_secure/cuda \
  --run cora=runs/cora_secure/cuda_default/warm \
  --run citeseer=runs/citeseer_secure/cuda_default/warm \
  --run pubmed=runs/pubmed_secure/cuda_default/warm \
  --run ppi=runs/ppi_secure/cuda_default/warm \
  --run ogbn-arxiv=runs/ogbn_arxiv_secure/cuda_default/warm
```

输出为各目录中的 `latest.json`、`latest.csv` 和 `summary.md`；脚本只接受
`warm + full-dataset + verified` manifest。

## 正式工件状态与 SRS

五个数据集的 cache、checkpoint bundle 和配置引用的正式 SRS 均已由当前生产路径
读取；五套最大包络已由当前 `gatzk_srs_setup compile` 验证。正式状态如下：

| 数据集 | 配置 | data/cache | checkpoint | compiler artifact | trusted SRS |
| --- | --- | --- | --- | --- | --- |
| Cora | `configs/cora_secure.cfg` | 就绪 | 就绪 | 匹配 | 就绪 |
| Citeseer | `configs/citeseer_secure.cfg` | 就绪 | 就绪 | 匹配 | 就绪 |
| Pubmed | `configs/pubmed_secure.cfg` | 就绪 | 就绪 | 匹配 | 就绪 |
| PPI | `configs/ppi_secure.cfg` | 就绪 | 就绪 | 匹配 | 就绪 |
| ogbn-arxiv | `configs/ogbn_arxiv_secure.cfg` | 就绪 | 就绪 | 匹配 | 就绪 |

正式运行前执行：

```bash
python3 tools/security/static_scan_no_tau.py --require-all-configured-srs
```

该扫描已在 2026-07-30 当前工作区通过。它检查静态安全禁项和工件绑定，不等价于
完整数据集证明通过，也不能把 PPI/ogbn-arxiv 写成已经 `VERIFY_OK`。旧命名 SRS
目录具有不同 manifest/ceremony provenance，未作为缓存删除；当前正式配置只引用
`artifacts/srs/<dataset>_formal_srs`。

## 测试与 benchmark

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build_cuda --output-on-failure
GATZK_RUN_UNIT_SECURITY_TESTS=1 build/gatzk_tests
GATZK_RUN_UNIT_SECURITY_TESTS=1 GATZK_RUN_PPI_BOUNDARY_TESTS=1 build/gatzk_tests
build_cuda/gatzk_cuda_hotspots_stage_bench
```

`gatzk_proof_codec_stage_bench` 调用生产 Context、Trace、Prover、canonical codec、
strict parser 和 Verifier，只限制输入规模；`gatzk_msm_stage_bench`、
`gatzk_kzg_opening_stage_bench`、`gatzk_t_edge_eval_stage_bench` 与
`gatzk_quotient_transform_stage_bench` 直接调用相应生产函数，不复制协议实现。

## 保留脚本

单数据集正式运行不依赖任何 Python 或 Shell 脚本。保留脚本仅用于工件重建、检查或
可选结果整理：

- `prepare_planetoid.py`、`prepare_ppi.py`、`prepare_ogbn_arxiv.py`：重建正式数据 cache；
- `planetoid_utils.py`、`ogbn_arxiv_utils.py`：上述数据工具的共享加载/规范化逻辑；
- `train_planetoid_gat.py`、`train_ogbn_arxiv_gat.py`：可复现训练/加载 checkpoint；
- `export_gat_checkpoint.py`、`export_torch_gat_bundle.py`、`import_ppi_bundle.py`：
  生成 C++ 生产加载器使用的 checkpoint bundle；
- `checkpoint_reader_compat.py`：旧 TensorFlow checkpoint 读取兼容层，仅供导出工具使用；
- `check_planetoid_parity.py`：训练 checkpoint 与导出 bundle 的可选一致性检查；
- `export_benchmark_table.py`：可选结果整理；
- `fetch_mcl.sh`：缺少 vendored MCL 时的可选依赖获取工具。

## 性能证据边界

优化前完整 CPU 基线为：

| 数据集 | commitment | prove | verify | 状态 |
| --- | ---: | ---: | ---: | --- |
| Cora | 34.849 s | 159.022 s | 1.199 s | `VERIFY_OK` |
| Citeseer | 78.635 s | 429.439 s | 3.373 s | `VERIFY_OK` |
| Pubmed | 113.707 s | 1020.030 s | 12.653 s | `VERIFY_OK` |

这些结果属于优化前源码，对当前源码是 `stale-for-current-source`。当前源码 Cora
完整 CPU 路径也已 `VERIFY_OK`：43.858/152.137/0.713 s，canonical proof 为
1,462,616 bytes。该次运行与两个训练任务和一个 CPU 测试任务重叠，因此正确性与工件
有效，时间不能当作受控冻结上限。

同源 production profiling 显示 Cora quotient build 从 126.253 s 降至 102.171 s
（-19.08%），t_edge/t_FH 分别下降 25.56%/17.46%；当前 verifier 对既有 Cora
canonical proof 的三次直接验证中位为 512.707 ms（均通过）。完整安全形态峰值 RSS
从 25,899,732 KiB 降至 17,250,228 KiB（-33.40%），t_edge 临时 coset 从
9,521,070,080 bytes 降至 7,088,373,760 bytes（-25.55%）。

commitment 没有可接受的新整链降幅；当前完整运行中的表面回退与明显外部负载长尾同时
出现，不作为代码结论。保留/回滚候选、各阶段原始值、PPI/ogbn-arxiv 边界和冻结后
十行最终验证命令见两份报告。
