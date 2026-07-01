# MAIN_EXPERIMENT_INVARIANTS.md

本文件定义 `/home/pzh/project` 中 zkGAT secure 主实验的长期不可变约束。

任何 Codex、脚本、自动化进程在修改主实验源码、runner、实验脚本、manifest、结果、README 或报告之前，必须先读取并遵守本文件。

本文件不是一次性 prompt，而是项目级长期约束文件。

除非用户明确要求修改本文件，否则不得修改本文件。

本版本由用户明确要求更新，核心调整为：

1. 主实验执行优先级改为 GPU-first，先推进 GPU 路径，后补齐 CPU 路径。
2. 最低接受预算由原 near_2_2x 放宽为 accept_3x。
3. 不得把外部进程、comparison 进程或资源干扰作为最终阻塞理由。
4. 不再大量生成审计、导出、转换、汇总、临时脚本或无关报告文件。
5. 正式 runner 已有的 latest.csv、latest.json、summary.md、run_manifests 可以保留；禁止额外新建大量导出/整理脚本。
6. 跑通结果后直接输出 10 行结果摘要；后续结果整理由用户自行完成。
7. 项目目录最终应尽可能只保留与主实验相关的源码、配置、必要测试、必要结果和必要运行材料。
8. 工程优化策略采用“累计工程优化包”，禁止单点小修小补、没过就回滚。
9. 保留所有 GAT 语义、证明语义、安全边界、KZG/SRS 语义和完整数据集规则。
10. 测试必须限定为主实验相关测试，不得误入、读取或运行 `comparison/**`。
11. 清理文件时禁止使用全仓库破坏性清理命令，只能删除明确识别的本轮临时文件、重复文件或无关中间文件。

如果后续任务 prompt 与本文件冲突，以更严格的规则为准。

---

## 1. 适用范围

本文件只适用于 zkGAT secure 主实验。

最终主实验必须完成 10 行 official full-dataset 结果。

执行优先级为 GPU-first，顺序如下：

```text
GPU cora
GPU citeseer
GPU pubmed
GPU ppi
GPU ogbn-arxiv
CPU cora
CPU citeseer
CPU pubmed
CPU ppi
CPU ogbn-arxiv
```

最终结果可以由现有正式 runner 自动写入：

```text
results/paper_secure/latest.csv
results/paper_secure/latest.json
results/paper_secure/summary.md
results/paper_secure/run_manifests/
```

如果现有 runner 已经具备这些输出能力，可以保留这些主实验正式结果文件。

禁止的是新增临时导出脚本、转换脚本、汇总脚本和无关中间文件；不是禁止正式 runner 生成既有的 `latest.csv`、`latest.json`、`summary.md`、`run_manifests/`。已有正式结果输出可以保留，但不得为整理结果额外新建大量文件。

最终 Codex 回复必须直接输出 10 行结果摘要，包括：

```text
backend
dataset
VERIFY 状态
proof_core_total_ms
budget_status
proof_size_bytes
GPU evidence 简要状态
```

`latest.csv` 和 `latest.json` 如果存在，只能包含最终 10 行正式完整数据集结果。

不得把 diagnostic、failed、timeout、partial、stale、smoke、toy、subgraph 或 comparison 结果写入 final latest。

---

## 2. 只读实现依据

以下文件是主实验语义、投稿实现路线和安全边界依据。Codex 必须读取并遵守，但不得修改：

```text
MAIN_EXPERIMENT_INVARIANTS.md
GAT-ZKML-多层多头-投稿实现最终版.md
zkGAT最终重构修改清单-投稿实现最终版.md
项目整体逻辑与方法流程.md
```

这些文件中的 GAT 多层多头语义、KZG/SRS 语义、零知识证明边界、数据集规则和实验门槛均不得被工程优化修改。

如果代码实现与这些文件冲突，应修代码，而不是修改这些约束文件。

---

## 3. 主实验与 comparison 隔离

`comparison/**` 专用于对比实验和 baseline 实验。

主实验不得：

```text
读取 comparison/**
运行 comparison/**
修改 comparison/**
清理 comparison/**
删除 comparison/**
引用 comparison/** 作为主实验证据
使用 comparison 结果
复制 comparison manifest
把主实验结果写入 comparison/**
把 comparison 结果写入 results/paper_secure/**
```

不得为了主实验直接清理、删除或修改 comparison 文件。

不得把外部进程、comparison 进程或资源干扰作为最终阻塞理由，也不得因此停止主实验任务。若资源紧张导致 timeout、OOM 或性能不达标，应继续做工程优化、重跑对应 official row，而不是结束任务或要求用户决策。

不得主动 kill comparison 进程或无关外部进程。

不得把资源干扰下的 failed、timeout、partial 或 diagnostic row 写成 final result。

runner timeout 只能 kill 当前 runner 自己创建的子进程树。

禁止使用模糊杀进程命令：

```text
pkill
killall
fuser -k
xargs kill
```

---

## 4. 禁止修改模型、协议、证明方案和安全语义

不得修改：

```text
GAT 模型语义
multi-layer multi-head GAT forward
hidden attention
ELU
LReLU
lookup
head concat
output heads
output bias
output head average
node classification
edge sort
route
PSQ
batch semantics
zkGAT 协议
证明方案
证明关系
verifier equation
proof schema
transcript order
challenge label/order
commitment label/order
opening label/order
public metadata fields
public/private boundary
KZG commitment semantics
KZG opening semantics
SRS/VK security model
dataset definition
quantization semantics
domain semantics
```

允许做工程优化，让同一个证明关系计算得更快、更稳。

禁止改变“证明了什么”。

---

## 5. 必须保持的协议语义

每一行有效 final result 必须保持：

```text
protocol_name = zkgat
proof_task = node_classification_full_gat
privacy_policy = model_and_trace_hidden
input_privacy = public_graph_public_features
output_mode = public_logits
opening_mode = domain_grouped_batch_opening
binding_mode = zk_subproof_for_crpc
verifier_reads_witness = false
private_plain_value_count = 0
proof_payload_audit_status = passed
reference_consistency_passed = true
tamper_tests_passed = true
```

如果任一字段不满足，该 row 无效，不得作为 final result。

---

## 6. 数据集规则

最终主实验必须使用完整数据集：

```text
cora
citeseer
pubmed
ppi
ogbn-arxiv
```

禁止：

```text
toy dataset
smoke dataset
subgraph
node sampling
edge sampling
feature reduction
dataset-name shortcut
stale preprocessed data
```

不得减少：

```text
nodes
edges
features
heads
layers
proof domain
datasets
```

PPI 必须保持 multi-graph 语义。

OGBN-arxiv 必须保持 large-graph 语义。

不得因为 PPI 或 OGBN-arxiv 的规模、domain size、SRS material、lookup capacity、raw span、LReLU table size、OOM、GPU OOM 或 timeout 结束任务。这类问题必须视为工程实现问题继续修复。

---

## 7. SRS / KZG 规则

禁止：

```text
runtime tau
srs_evaluation_point
srs_evaluation_scalar
prover_aux trapdoor scalar
commitment_eval shortcut
commit_precomputed fake commitment
single-point commitment
polynomial.evaluate(tau) as commitment
generator * polynomial evaluation as KZG commitment
fake KZG
fake SRS
fake opening
fake witness
verifier reading witness
```

verifier 必须保持 VK-only。

KZG commitment 必须使用合法的 SRS-based MSM。

KZG opening 必须使用合法的 SRS-based quotient/witness opening。

不得替换 SRS/PK/VK identity，除非用户明确批准。

如果只是补齐缺失的 derived material，必须满足：

```text
仍属于同一 SRS identity
仍在同一 trusted setup boundary 内
srs_id / setup transcript / VK / security semantics 不变
有 hash 证据
verifier semantics 不变
```

---

## 8. Proof bytes 规则

fresh proof 或工程路径变化可能导致：proof bytes、proof hash、commitment bytes、opening proof bytes、challenge scalar values 发生变化。这是允许的。

但以下内容不得改变：

```text
proof relation
proof scheme
verifier equation
proof schema
transcript order
commitment/opening label order
metadata fields
public/private boundary
KZG semantics
```

不要求 proof bytes 与旧 run 完全一致。要求 proof semantics 与旧主实验语义一致。

---

## 9. GPU-first 规则

本版本主实验执行优先级为 GPU-first。

优化、canary、预验证和 official row 推进顺序优先为：

```text
GPU cora
GPU citeseer
GPU pubmed
GPU ppi
GPU ogbn-arxiv
CPU cora
CPU citeseer
CPU pubmed
CPU ppi
CPU ogbn-arxiv
```

GPU row 必须提供真实 GPU evidence：

```text
gpu_backend_requested = true
gpu_runtime_loaded
gpu_device
gpu_kernel_invocation_count
gpu_used_for
gpu_fallback_stages
gpu_fallback_reason
gpu_acceleration_status
cpu_gpu_reference_consistency_passed
```

允许记录 GPU fallback。

但不得把 CPU fallback 冒充 GPU acceleration。

复制 CPU result 不能算有效 GPU row。

GPU 路径不是替代 CPU 路径。GPU 路径推进后，必须补齐 CPU 5 行 official results。

---

## 10. 性能指标和 3x 预算

主性能指标是：

```text
proof_core_total_ms = commitment_ms + prove_ms + verify_ms
```

不得用 `total_wall_clock_ms` 替代 `proof_core_total_ms`。

final row 必须包含：

```text
commitment_ms 或 commitment_time_ms
prove_ms 或 prove_time_ms
verify_ms 或 verify_time_ms
proof_core_total_ms
budget_status
```

如果 proof-core 字段缺失，必须先修 runner/exporter 的已有逻辑，不得新建大量导出脚本。

旧基准单位：ms。

```csv
backend,dataset,baseline_proof_core_total_ms,target_2x,accept_3x
CPU,cora,2238.275,4476.550,6714.825
CPU,citeseer,1984.201,3968.402,5952.603
CPU,pubmed,8839.821,17679.642,26519.463
CPU,ppi,11275.126,22550.252,33825.378
CPU,ogbn-arxiv,84410.517,168821.034,253231.551
GPU,cora,1638.292,3276.584,4914.876
GPU,citeseer,1998.596,3997.192,5995.788
GPU,pubmed,4331.885,8663.770,12995.655
GPU,ppi,6455.733,12911.466,19367.199
GPU,ogbn-arxiv,55014.665,110029.330,165043.995
```

接受标准：

```text
优先完成：proof_core_total_ms <= target_2x
最低接受：proof_core_total_ms <= accept_3x
不接受：proof_core_total_ms > accept_3x
```

任一 final matrix row 超过 `accept_3x` 时：

```text
不得作为最终有效结果
继续针对该 row 所在路径做工程优化
修复后重新跑对应 official row
最终 10 行均必须 <= accept_3x
```

不要求因为单行超过预算而删除所有已完成有效 row，但最终输出时只能承认全部 10 行均满足 `accept_3x` 的结果。

统一命名只允许使用：

```text
target_2x
accept_3x
COMPLETED_TARGET
COMPLETED_ACCEPTED_3X
```

不得再使用：

```text
near_2x
near_2_2x
near accepted
near threshold
COMPLETED_NEAR_ACCEPTED
```

---

## 11. Runner 执行规则

runner 必须真实执行：

```text
per-row budget
subprocess hard timeout
只 kill 当前 runner 自己创建的子进程树
timeout row 不得作为 final result
failed row 不得作为 final result
failure reason 记录在现有日志或终端输出中
```

不得通过修改 runner gate、timeout、budget、exporter 来伪造通过。

不得手写 latest。

不得把 failed、timeout、partial、diagnostic 结果写成 final result。

---

## 12. 错误 cache 禁止规则

不得使用：

```text
failed-run cache
timeout-run cache
diagnostic-run cache
partial-run cache
proof-specific persistent cache
dynamic commitment persistent cache
quotient polynomial persistent cache
quotient point persistent cache
coset persistent cache
coeff persistent cache
chunked MSM point persistent cache
runs/performance_cache/** 作为 final path 依赖
```

不得通过多轮 timeout run 逐步填 cache 来通过 canary。

如果发现这些路径进入 final path，必须禁用或隔离；不得删除未知用户文件。

---

## 13. 累计工程优化包规则

本主实验性能优化必须采用“累计工程优化包”策略。

当前性能问题是多个主热路径瓶颈叠加，不得继续执行：

```text
改一个小点 -> 跑 canary -> 没过 -> 回滚 -> 再猜一个小点
```

必须改为：

```text
分批实现工程优化包
build/主实验相关测试正确性通过后保留
继续叠加下一个工程优化包
阶段性跑 clean-cold-cache canary
先推进 GPU 路径
再补齐 CPU 路径
最终完成 10 行 official results
```

不要把某一个 canary 是否一次通过，作为单个优化包是否保留的唯一标准。

只要一个优化包满足以下条件，就先保留并继续叠加：

```text
build 通过
ctest 或项目实际存在的主实验相关测试通过，且不得进入、读取或运行 comparison/**
不改变 proof relation / verifier equation / transcript/order / public-private boundary / KZG-SRS 语义
不依赖 failed-run / timeout-run / diagnostic-run / partial-run cache
不触碰 comparison/**
没有明显性能倒退
属于主实验真实热路径
对 GPU 和 CPU 路径都有意义，或至少优先改善 GPU-first 路径且不损害 CPU 路径
对 PPI / OGBN 也有意义
```

只有出现以下情况才回滚对应优化包：

```text
编译失败且无法直接修复
主实验相关测试或正确性检查失败
verifier / proof payload / transcript / public-private boundary 有风险
明显性能倒退
只是局部小修小补，不能进入结构性优化包
依赖 failed-run cache 或 timeout partial artifact
修改 runner gate / timeout / budget / exporter 来伪造通过
```

---

## 14. 不要择优只做一个优化包

不要只选择一个“看起来最优”的优化包后就停止。

凡是属于主实验真实热路径、能够减少以下任一项，并且不改变证明语义、不依赖错误 cache、不触碰 `comparison/**` 的工程优化，都必须持续实现并尽量保留：

```text
计算量
内存峰值
重复 basis conversion
真实 MSM 数量
完整 domain pass
巨大临时 vector
CPU/GPU 大热点开销
host-device 往返
重复 vector allocation
SRS slice 重复访问
```

实现方式不是无序乱改，而是分批推进：

```text
每个批次围绕一个明确热路径或一组强相关热路径
每个批次必须保持 diff 可解释、可回滚、可验证
build/主实验相关测试通过且不破坏证明语义的优化包应保留
不要因为一个优化包完成、一个瓶颈下降、或一次 canary 仍 timeout 就停止
只要 10 行 official results 还没有完成，就继续推进下一个必要的可优化热路径
一旦当前优化已经足以推进下一实验阶段，就立即进入下一阶段，不得为了“还能继续优化”无限停留
最终目标是累计完成所有必要工程优化，而不是只选一个“最优”方向
```

---

## 15. 必须持续推进的工程优化包

必须持续推进的优化包包括但不限于：

### 15.1 GPU-first backend 优化包

优先处理 GPU official path 中仍然明显落在 CPU fallback 或 host-side 大热点的部分。

允许做：

```text
增强真实 GPU path 覆盖
优化大规模 field vector kernel
优化 coset / packed weighted evaluation
优化 quotient materialization 后端
优化 MSM/FFT 热路径中能安全进入 GPU 或减少 host-side 开销的部分
减少 host-device 往返
减少 GPU path 中多余的 CPU fallback 包装、重复转换和重复 allocation
```

必须满足：

```text
GPU backend 真实 loaded
GPU kernel count > 0
GPU row 不能复制 CPU 结果
CPU/GPU reference consistency 通过
不改变 proof relation / verifier equation / transcript/order / public-private boundary
```

### 15.2 t_edge / t_FH transform 优化包

目标：减少重复 IFFT / coset / NTT / materialization，减少重复 basis conversion。

允许做：

```text
合并重复 dependency
删除真正无用的 materialization
避免重复 evaluation-basis -> coefficient-basis -> coset
将安全可复用的域内计算合并到同一工程路径
优化 edge/FH domain 上的大规模 field vector 处理
减少完整 full-domain transform 次数
```

不得猜证明等式。

不得改变约束语义、verifier equation、commitment/opening label order。

### 15.3 quotient materialization 优化包

目标：减少完整 domain pass 和巨大临时 vector。

允许做：

```text
融合 dependency evaluation、constraint accumulation、division、quotient output
对 t_edge / t_FH 做 streaming 或 block processing
降低 peak memory
避免重复扫描 full domain
避免重复 vector allocation
在同一次 official row 内完成，不跨 run 缓存
```

禁止跨 timeout run 保存 quotient poly、coset、coeff、commitment point 或 partial artifact。

### 15.4 KZG / MSM 优化包

目标：减少真实 MSM 次数和 MSM 输入构造成本。

允许做：

```text
batch 内去重后再 MSM
过滤 zero / sparse / constant-tail scalar
减少重复 scalar conversion
减少重复 vector allocation
按 domain / basis / size 分组
优化 SRS slice 访问
改善 batch partition
```

禁止跨 run MSM chunk point cache。

禁止 fake KZG、fake opening、commitment shortcut。

### 15.5 CPU 路径补齐优化包

GPU 路径推进后，必须回到 CPU 路径补齐优化。CPU 不是可跳过路径。

允许做：

```text
复用不改变语义的通用工程优化
优化 CPU field vector、coset、quotient materialization、MSM/FFT 热路径
减少重复 conversion、allocation 和 full-domain pass
保证 CPU rows 也满足 3x 预算和正确性要求
```

---

## 16. 严禁局部小修小补

局部小修小补不得作为本轮优化方向。

以下都属于禁止路线：

```text
alias 小 patch
selector closed-form 小 patch
单个 label 派生
单个临时变量替换
单个 worker 参数
单纯调 worker 数、线程数、worker cap
只让 timeout 日志停点后移一点的 HOTPATH patch
只影响 Cora 的特判
只做 micro benchmark，不改善 official path
```

只有当这些改动是某个工程优化包的一部分，并且能实际减少主热路径计算量，才允许保留。

worker cap 不能作为独立优化。只有在代码和日志明确显示过度并发导致 MSM、FFT、coset 或 quotient materialization 发生内存带宽竞争时，才允许把 worker cap 作为阶段化调度或批处理优化的一部分。

---

## 17. OOM / crash 规则

以下是实现或资源管理问题，不是外部阻塞：

```text
host OOM
GPU OOM
CUDA OOM
SIGKILL
allocation failure
process crash
```

发生后必须：

```text
不得作为 final result
修复 memory / streaming / chunking 实现
修复后重新跑对应 official row
```

不得把 OOM/crash/SIGKILL 说成 external blocker。

---

## 18. 文件生成与目录清理规则

本版本不要求 Codex 大量写审计文件、导出脚本、汇总脚本或临时报告。

禁止为了本轮任务新建大量以下文件：

```text
临时 export 脚本
临时 result converter
临时 benchmark wrapper
临时 audit 脚本
临时 profile 脚本
重复 summary 文件
无关 markdown 报告
无关 json/csv 中间文件
大段历史过程记录
```

优先使用仓库已有 runner、已有测试、已有构建流程。

如果必须临时写小工具，只允许满足以下条件：

```text
只服务于当前工程修复
不进入 final path
不作为正式结果来源
完成后删除
不得长期留在项目目录
```

最终清理本轮产生的无关、重复、临时文件。可以删除已经不再需要的临时脚本、临时 profile、临时 diagnostic、中间 json/csv、临时 markdown 和废弃 patch 文件。

清理前必须先确认文件是本轮新产生的临时文件、重复文件或无关中间文件。

禁止使用全仓库破坏性清理命令，例如：

```text
git clean -fdx
rm -rf results/**
rm -rf runs/**
rm -rf comparison/**
find . -delete
对未知目录执行批量 rm -rf
```

清理只能针对明确识别的临时文件路径。不得批量删除未知文件、用户原有文件、正式结果、正式日志、数据集、SRS/PK/VK、源码、配置和测试。

不得删除：

```text
final results
当前仍需复现或定位问题的最小必要日志
runner 正式日志
主实验源码
主实验配置
主实验正式结果
必要测试
必要 SRS/PK/VK material
数据集文件
只读投稿实现笔记
MAIN_EXPERIMENT_INVARIANTS.md
comparison/**
用户原有文件
```

项目目录最终应尽可能只保留与主实验相关的源码、配置、必要测试、必要结果、必要运行材料和最小必要证据。

---

## 19. 验证门槛

每个保留优化包至少必须通过：

```text
build 受影响目标
ctest 或项目实际存在的主实验相关测试，且不得进入、读取或运行 comparison/**
项目实际存在的必要检查
proof semantics 不变
```

最终 10 行结果前必须通过：

```text
build CPU
build CUDA
CPU 主实验相关测试
CUDA 主实验相关测试
项目实际存在的必要检查
reference consistency
tamper tests
```

如果项目中不存在某个 static scan 脚本或检查流程，不得发明不存在的 static scan。

不得把外部进程、comparison 进程或资源干扰作为最终阻塞理由。若资源紧张导致 timeout、OOM 或性能不达标，应继续工程优化和重跑，而不是停止任务。

---

## 20. Final matrix 顺序

final matrix 固定顺序为 GPU-first：

```text
1 GPU cora
2 GPU citeseer
3 GPU pubmed
4 GPU ppi
5 GPU ogbn-arxiv
6 CPU cora
7 CPU citeseer
8 CPU pubmed
9 CPU ppi
10 CPU ogbn-arxiv
```

任一 row 失败或超过 3x 预算，不得作为 final result。

修复后重新跑对应 row 或完整 matrix。

partial、failed、timeout、diagnostic row 不得作为 final result。

---

## 21. Final matrix 启动条件

不得过早启动完整 final matrix。

final matrix 启动前至少满足：

```text
GPU cora 已通过 clean-cold-cache official canary 或同等 official row
至少一个 GPU 中等数据集已通过 canary 或 official row
CPU cora 已通过 clean-cold-cache official canary 或同等 official row
至少一个 CPU 中等数据集已通过 canary 或 official row
build CPU 通过
build CUDA 通过
CPU 主实验相关测试通过
CUDA 主实验相关测试通过
无 forbidden proof-specific cache 进入 final path
comparison/** untouched
proof relation unchanged
verifier equation unchanged
transcript/order unchanged
public/private boundary unchanged
KZG/SRS semantics unchanged
```

未满足前只允许运行：

```text
targeted stage benchmark
diagnostic profile
representative canary
单行 official diagnostic row
```

diagnostic 不得进入 final result。

---

## 22. 最终完成条件

最终完成必须同时满足：

```text
10 行 final official rows 存在
顺序符合 GPU-first final matrix
全部 full_dataset=true
全部 VERIFY_OK
全部 fresh
无 timeout row
无 stale row
无 diagnostic row
无 partial row
全部 proof_core_total_ms <= accept_3x
GPU rows 有真实 GPU evidence
CPU rows 全部完成
proof relation unchanged
proof scheme unchanged
verifier equation unchanged
proof schema unchanged
transcript order unchanged
commitment/opening label order unchanged
metadata fields unchanged
verifier_reads_witness=false
private_plain_value_count=0
proof payload audit passed
reference consistency passed
tamper tests passed
comparison/** untouched
no failed-run cache in final path
项目目录已清理无关、重复、临时文件
```

唯一允许的最终状态：

```text
FINAL_STATUS: COMPLETED_TARGET
FINAL_STATUS: COMPLETED_ACCEPTED_3X
```

禁止作为最终状态：

```text
FINAL_STATUS: FORMAL_CONSTRAINT_CONFLICT
FINAL_STATUS: BLOCKED_EXTERNAL
FINAL_STATUS: BLOCKED
FINAL_STATUS: USER_DECISION_REQUIRED
FINAL_STATUS: NOT_READY_PERFORMANCE
FINAL_STATUS: PARTIAL
FINAL_STATUS: DIAGNOSTIC_ONLY
FINAL_STATUS: COMPLETED_NEAR_ACCEPTED
```

---

## 23. 最终回复格式

如果全部 10 行达到 2x target：

```text
FINAL_STATUS: COMPLETED_TARGET
final_matrix_order:
- GPU cora
- GPU citeseer
- GPU pubmed
- GPU ppi
- GPU ogbn-arxiv
- CPU cora
- CPU citeseer
- CPU pubmed
- CPU ppi
- CPU ogbn-arxiv
results:
- backend,dataset,verify_status,proof_core_total_ms,budget_status,proof_size_bytes,gpu_evidence
proof_equivalence:
- proof relation unchanged: yes
- proof scheme unchanged: yes
- verifier equation unchanged: yes
- transcript/order unchanged: yes
- label order unchanged: yes
gpu_evidence:
- GPU rows have real GPU evidence: yes
cleanup:
- temporary/unrelated files cleaned: yes
tests_passed:
- build CPU
- build CUDA
- CPU main-experiment tests
- CUDA main-experiment tests
- reference consistency
- tamper tests
```

如果全部 10 行未全部达到 2x，但全部 <= 3x：

```text
FINAL_STATUS: COMPLETED_ACCEPTED_3X
final_matrix_order:
- GPU cora
- GPU citeseer
- GPU pubmed
- GPU ppi
- GPU ogbn-arxiv
- CPU cora
- CPU citeseer
- CPU pubmed
- CPU ppi
- CPU ogbn-arxiv
accepted_3x_rows:
- ...
results:
- backend,dataset,verify_status,proof_core_total_ms,budget_status,proof_size_bytes,gpu_evidence
proof_equivalence:
- proof relation unchanged: yes
- proof scheme unchanged: yes
- verifier equation unchanged: yes
- transcript/order unchanged: yes
- label order unchanged: yes
gpu_evidence:
- GPU rows have real GPU evidence: yes
cleanup:
- temporary/unrelated files cleaned: yes
tests_passed:
- build CPU
- build CUDA
- CPU main-experiment tests
- CUDA main-experiment tests
- reference consistency
- tamper tests
```

不得输出其他 final status。

---

## 24. 后续短 prompt 模板

后续任务 prompt 可以简化为：

```text
你现在在 /home/pzh/project 工作。

先读取并严格遵守 MAIN_EXPERIMENT_INVARIANTS.md。
不得触碰 comparison/**。
不得使用 failed-run / timeout-run / diagnostic-run / partial-run proof-specific cache。
不得修改模型、协议、证明关系、verifier equation、transcript/order、public/private boundary、KZG/SRS 语义。
不得单点小修小补。
必须采用累计工程优化包策略。
必须 GPU-first，先推进 GPU 路径，再补齐 CPU 路径。
最低接受预算为 accept_3x。
不要把外部进程、comparison 进程或资源干扰作为最终阻塞理由。
不要新增导出/汇总脚本；使用现有 runner 和现有输出机制。
测试限定为主实验相关测试，不得进入、读取或运行 comparison/**。
跑通后在最终回复中直接列出 10 行结果摘要，用户后续自行整理。
最终清理项目目录中无关、临时、重复文件，只保留主实验必要内容和最小必要证据。
清理前必须确认文件是本轮新产生的临时文件、重复文件或无关中间文件。
禁止 git clean -fdx、rm -rf results/**、rm -rf runs/**、rm -rf comparison/**、find . -delete 或对未知目录执行批量 rm -rf。
任务：继续 zkGAT secure 主实验，完成 GPU-first 5 数据集 × CPU/GPU 共 10 行 official results。
最终状态只能是 COMPLETED_TARGET 或 COMPLETED_ACCEPTED_3X。
```
