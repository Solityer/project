# zkGAT Secure Main Experiment

本仓库实现完整图节点分类的正式 zkGAT 协议。协议与执行约束以只读文件
`GAT-ZKML-多层多头.md` 和 `PROJECT_CONTEXT_AND_EXECUTION_CONSTRAINTS.md`
为准；`comparison/` 不属于主实验构建、测试或运行依赖。

## 当前状态（2026-07-22）

**协议实现处于候选冻结状态；第三阶段本轮有界正式路径性能目标已达到，但五个数据集的正式性能验收仍未执行。**
当前 proof schema、transcript、proof wire format、parser 与 verifier 已统一为全局
point-group opening：每个不同点独立派生非零 challenge，同点 claim 只对应一个 owner
和一个 witness。完整数据集正式性能基线仍标记为待重建；旧协议时代的性能预算和速度门槛不再参与
运行有效性、恢复、正确性或安全性判断。第三阶段当前证据、回滚项和真实边界见
[`PHASE3_PERFORMANCE_OPTIMIZATION.md`](PHASE3_PERFORMANCE_OPTIMIZATION.md)。此状态不代表
第一阶段最终全量验收已通过，也不授权自动生成论文正式结果。

当前没有可信的完整数据集性能基线，旧协议运行数字不再在 README 中作为当前证据展示。
用户启动任何正式行前，必须用当前代码重新核验对应数据集 SRS 的 manifest、hash、
degree bound 和 schema 绑定。

本轮在同一小型 SRS、相同 20 核绑定和 `exclusive_commit_prove_verify_v2` 口径下冻结
3 次原始基线并复测 3 次最终状态。代表性正式调用链的三项时间总和中位数由
46612.391 ms 降至 3715.614 ms（-92.029%），wall time 中位数由 48876.405 ms
降至 3926.298 ms（-91.967%）。该有界入口完整执行 commitment、prove、canonical
serialization、strict parse 和 verify，但不是任何完整数据集的论文性能结果。

## 正式协议边界

- 公开输入是图结构和节点特征；最终 logits 是公开输出。
- 模型参数、PK、trace、attention/softmax、lookup、route、PSQ、folding、私有多项式求值和 blinding randomness 均不是 verifier 输入。
- route terminal 已由 proof 内 `S_route` commitment/reference、正式 quotient identity、点群 opening 与 verifier 方程闭环；旧裸标量通道已删除。
- strict parser 只接受 canonical schema/registry/order/length；proof 往返不携带 witness polynomial、PK 或私有模型。
- verifier-only 路径只消费公开上下文、VK、proof 和公开 metadata，不读取 prover 私有状态。
- 旧单输出协议仅保留 fail-closed 拒绝门，不存在兼容接受路径。
- `proof_size_bytes` 是 strict canonical wire bytes 的精确长度，不包含 SRS、缓存、日志、checkpoint 或数据集。

正式计时 schema 是 `exclusive_commit_prove_verify_v2`。`commitment_time_ms` 仅统计
proof-specific commitment point 生成及必要同步；`prove_time_ms` 统计从 trace/witness
生成到内存中 canonical serialization 的其余全部 prover 工作；`verify_time_ms` 从
canonical bytes 的 strict parse 开始，到 verifier 接受或拒绝结束。三者互斥；上下文、
SRS/PK/VK 加载、I/O、工件导出、审计和 runner 开销只进入 wall-time 对账。

## 代码结构与执行后端

主要模块边界如下：

- `src/protocol/schema.cpp`：schema/ledger/关系注册；标签构造已拆到 `schema_labels.cpp`；
- `src/protocol/trace.cpp`：正式 witness、lookup/route/PSQ/zkMaP 组装；transcript/schema 收尾核对为独立内部阶段；
- `src/protocol/prover.cpp`：quotient、opening 与 proof 组装；运行工件导出已拆到 `artifact_export.cpp`；
- `src/protocol/proof_codec.cpp`：strict wire serializer/parser；
- `src/protocol/verifier.cpp`：verifier-only 验证；
- `src/algebra/vector_ops.cpp` 与 CUDA translation units：唯一 CPU/CUDA 执行路由边界。

`--compute-backend cpu` 与 `--compute-backend cuda_hotspots` 共享同一协议、schema、
transcript 和 verifier 接受条件。CUDA 当前仅覆盖 barycentric weights、packed weighted
evaluation 和 size-aware lookup histogram；KZG MSM、pairing、小域及未迁移阶段明确记录为
CPU fallback。请求 CUDA 后端而运行时不可用时会 fail closed。

vendored MCL 现在由正式 CMake Release 规则以 `-O3 -DNDEBUG` 构建；此前其静态库遗漏
优化级别。动态 commitment 使用每个 proof 内的精确语义列复用并为每个对象独立加入
zero-domain blinder，quotient 的相同语义列只共享只读 base coset material。两类缓存均不
进入 proof，不跨 proof、challenge、owner、witness 或 opening group。

第二阶段只进行了结构移动、引用确认后的死代码/重复路径清理和构建项清理。CPU/CUDA
主程序及测试二进制均已编译链接；阶段内未运行 Cora/PPI 正式 proof。toy 数据、toy
配置、toy 入口及旧开发/中止/重复 SRS 已清理；最小 `unit_security` 安全夹具及其 SRS
继续保留到最终跨后端验收。

## 构建

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target gatzk_run gatzk_tests -j

cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release \
  -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda --target gatzk_run gatzk_tests -j

# 仅 CUDA 配置提供该有界热点 benchmark
cmake --build build_cuda --target gatzk_cuda_hotspots_stage_bench -j
```

## 配置与 SRS

正式配置使用数据集专属 SRS id，不再依赖旧共享主 SRS：

- `configs/cora_secure.cfg` → `cora_formal_srs_v2`；
- `configs/ppi_representative_secure.cfg` → `ppi_representative_srs_v2`；
- `configs/ppi_secure.cfg` → `ppi_formal_srs_v2`；
- `configs/citeseer_secure.cfg` → `citeseer_formal_srs_v2`；
- `configs/pubmed_secure.cfg` → `pubmed_formal_srs_v2`；
- `configs/ogbn_arxiv_secure.cfg` → `ogbn_arxiv_formal_srs_v2`。

这些 v2 正式 SRS 尚未生成；旧 v1 物料仍保留，但不再被任何正式配置引用，因而不能
绕过新 opening schema 绑定。只有 `unit_security_srs_v1` 已按 setup-only 流程重生并与
当前 schema、compiler artifact 和 manifest 一致。

当前调度不生成 PPI 全量、Citeseer、Pubmed 或 ogbn-arxiv 大型 SRS，也不得运行五个
数据集 × CPU/GPU 的十组正式实验。旧 v1 envelope compiler artifact 已清理；需要生成
v2 正式 SRS 时必须先按 setup-only compiler 流程重新产生，不能复用旧 schema 物料。

## 验证状态与后续顺序

第三阶段本轮已完成冻结前后各 3 次的有界正式调用链、strict parser/verifier、扩展安全
单元回归、CPU/CUDA Release 全构建与 CUDA 精确一致性测试；`private_plain_value_count=0`
且 `verifier_reads_witness=false`。本轮 75% 目标已超过，但未执行任何完整数据集证明。
最终统一验收仍包括
Cora、Citeseer、Pubmed、PPI、ogbn-arxiv 的 CPU/GPU 正式行，并且只能由项目负责人决定
何时运行。建议逐行串行执行并先完成缺失 SRS 的生成和严格扫描；当前未测量的资源与
硬件边界统一维护在第三阶段性能记录的“Remaining real boundaries”一节。

上述跨后端验收全部完成前，不得宣称第一阶段或三个阶段全部通过，也不得把小输入或
stage benchmark 的数据写成正式数据集结果。

静态安全扫描默认检查正式源代码和所有已经物化的配置 SRS；未获准生成的 SRS 会明确
列为 skipped。只有所有数据集 SRS 均已生成时才使用严格全配置模式：

```bash
python3 tools/security/static_scan_no_tau.py
python3 tools/security/static_scan_no_tau.py --require-all-configured-srs
```

扫描会检查禁止的 setup 辅助材料、已存在 SRS 的 manifest/hash/schema 一致性，但扫描
通过本身不等价于正式 proof 或零知识证明验收。

## 正式结果导出边界

`results/paper_secure/` 的旧协议结果已清理，不是当前候选冻结实现的证据。
在项目负责人明确授权最终十组实验、所有运行通过当前 strict gates 且结果重新生成前，
不得调用正式结果导出流程或据此发布性能数字。
