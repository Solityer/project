# MAIN_EXPERIMENT_INVARIANTS.md

本文件定义 `/home/pzh/project` 中 zkGAT secure 主实验的不可变约束。

任何 Codex、脚本、自动化进程在修改主实验源码、runner、实验脚本、manifest、结果、README 或报告之前，必须先读取并遵守本文件。

本文件不是一次性 prompt，而是项目级长期约束文件。

除非用户明确要求修改本文件，否则不得修改本文件。

如果后续任务 prompt 与本文件冲突，以更严格的规则为准。

---

## 1. 适用范围

本文件只适用于 zkGAT secure 主实验。

最终主实验必须产出 10 行 official full-dataset 结果：

```text
CPU cora
GPU cora
CPU citeseer
GPU citeseer
CPU pubmed
GPU pubmed
CPU ppi
GPU ppi
CPU ogbn-arxiv
GPU ogbn-arxiv
```

最终结果必须写入：

```text
results/paper_secure/latest.csv
results/paper_secure/latest.json
results/paper_secure/summary.md
results/paper_secure/run_manifests/
```

`latest.csv` 和 `latest.json` 只能包含最终 10 行正式完整数据集结果。

不得把 diagnostic、failed、timeout、partial、stale、smoke、toy、subgraph 或 comparison 结果写入 final latest。

---

## 2. 主实验与对比实验双向隔离

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
kill 对比实验进程
```

主实验不能干扰对比实验。

对比实验也不能污染主实验的 official timing、final matrix、结果文件、缓存和资源环境。

在运行 official timing、official diagnostic row 或 final matrix row 之前，必须检查是否有 comparison 进程或其他重负载进程正在占用 CPU/GPU/IO/内存资源。

如果存在明显资源干扰：

```text
不得运行 official timing
不得运行 final matrix row
只能等待资源合适、降低非正式任务并发，或只做轻量检查/报告工作
必须把资源干扰记录到 performance_path_equivalence_audit.md
```

如果 official row 与 comparison 重负载资源干扰重叠，该 row 无效，只能归档为 diagnostic，不得进入 latest。

主实验只允许写入：

```text
results/paper_secure/**
results/performance_engineering/**
runs/** 中主实验相关目录
```

不得使用全局进程清理命令误杀第二进程，例如：

```text
pkill
killall
fuser -k
xargs kill
```

runner timeout 只能 kill 当前 runner 自己创建的子进程树。

---

## 3. 禁止修改模型、协议、证明方案和安全语义

不得修改：

```text
GAT 模型语义
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

## 4. 必须保持的协议语义

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

如果任一字段不满足，该 row 无效，不得进入 latest。

---

## 5. GAT 语义不得改变

不得改变：

```text
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
public graph
public features
hidden model
hidden trace
public logits
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

PPI 必须保持 multi-graph 语义。

OGBN-arxiv 必须保持 large-graph 语义。

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

fresh proof 或工程路径变化可能导致：

```text
proof bytes
proof hash
commitment bytes
opening proof bytes
challenge scalar values
```

发生变化。

这是允许的。

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

不要求 proof bytes 与旧 run 完全一致。

要求 proof semantics 与旧主实验语义一致。

---

## 9. GPU 规则

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

---

## 10. 性能指标

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

如果 proof-core 字段缺失，必须先修 runner/exporter，再启动 final matrix。

---

## 11. 旧主实验预算

旧基准单位：ms。

```csv
backend,dataset,baseline_proof_core_total_ms,target_2x,near_2_2x
CPU,cora,2238.275,4476.550,4924.205
CPU,citeseer,1984.201,3968.402,4365.242
CPU,pubmed,8839.821,17679.642,19447.606
CPU,ppi,11275.126,22550.252,24805.277
CPU,ogbn-arxiv,84410.517,168821.034,185703.137
GPU,cora,1638.292,3276.584,3604.242
GPU,citeseer,1998.596,3997.192,4396.911
GPU,pubmed,4331.885,8663.770,9530.147
GPU,ppi,6455.733,12911.466,14202.613
GPU,ogbn-arxiv,55014.665,110029.330,121032.263
```

接受标准：

```text
优先完成：proof_core_total_ms <= target_2x
近似接受：proof_core_total_ms <= near_2_2x
不接受：proof_core_total_ms > near_2_2x
```

任一 final matrix row 超过 `near_2_2x` 时：

```text
立即中止 matrix
不得继续后续 row
不得导出 latest
已完成 row 全部降级为 diagnostic archive
修复后必须从第 1 行重新跑完整 final matrix
```

---

## 12. Runner 预算执行规则

runner 必须真实执行：

```text
per-row budget
subprocess hard timeout
只 kill 当前 runner 自己的子进程树
timeout 后中止 matrix
completed-but-over-budget 后中止 matrix
matrix 失败后不导出 latest
failed row 归档
failure reason 记录
```

必须 dry-run 验证：

```text
timeout row 会中止 matrix
VERIFY_OK 但 proof_core 超预算的 row 也会中止 matrix
```

如果只实现 timeout kill，但没有实现 completed-over-budget abort，禁止启动 final matrix。

---

## 13. OOM / crash 规则

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
中止 matrix
不得导出 latest
归档日志
修复 memory / streaming / chunking 实现
修复后从第 1 行重新跑 final matrix
```

不得把 OOM/crash/SIGKILL 说成 external blocker。

---

## 14. OGBN 规则

不得因为 OGBN-arxiv 的 LReLU table size、raw span、SRS material、domain size 或 lookup capacity 结束任务。

这类问题必须视为工程问题继续修复。

修复方向包括：

```text
检查 quant/preprocess/checkpoint 是否一致
检查旧主实验 domain/table policy
在同一 SRS identity 内修复 derived material
修复 table/domain regression
使用 streaming/chunking/memory pool
使用 large-graph GPU pipeline
```

禁止逃生方式：

```text
跳过 OGBN
缩小 OGBN
改 quantization 缩小 raw range
改 proof relation
改 verifier equation
改 transcript
使用 witness-dependent fake table
未经用户批准替换 SRS/PK/VK identity
以 FORMAL_CONSTRAINT_CONFLICT 结束
以 BLOCKED_EXTERNAL 结束
```

---

## 15. 必须生成或更新的审计文件

必须生成或更新：

```text
results/performance_engineering/initial_checkpoint.md
results/performance_engineering/dataset_integrity_audit.md
results/performance_engineering/dataset_integrity_audit.json
results/performance_engineering/srs_pk_vk_integrity_audit.md
results/performance_engineering/srs_pk_vk_integrity_audit.json
results/performance_engineering/baseline_budget_2x.md
results/performance_engineering/baseline_budget_2x.json
results/performance_engineering/performance_path_equivalence_audit.md
results/performance_engineering/five_dataset_bottleneck_matrix.md
results/performance_engineering/final_matrix_preflight_budget_check.md
results/performance_engineering/final_matrix_preflight_budget_check.json
results/performance_engineering/failed_runs/runner_budget_enforcement_dry_run.md
results/paper_secure/optimization_profile_report.md
results/paper_secure/gpu_backend_report.md
results/paper_secure/baseline_comparison_report.md
results/paper_secure/negative_optimization_report.md
results/paper_secure/cleanup_audit_report.md
```

---

## 16. 验证门槛

必须通过：

```text
build CPU
build CUDA
CPU ctest
CUDA ctest
static scan
full security regression
proof-equivalence audit
reference consistency
tamper tests
export validation
```

official timing 不得与以下任务并发：

```text
full security regression
另一个 official row
heavy benchmark
large build
CUDA benchmark
dataset preprocessing
long profiler
comparison-heavy resource interference
```

---

## 17. Final matrix 顺序

final matrix 固定顺序：

```text
1 CPU cora
2 GPU cora
3 CPU citeseer
4 GPU citeseer
5 CPU pubmed
6 GPU pubmed
7 CPU ppi
8 GPU ppi
9 CPU ogbn-arxiv
10 GPU ogbn-arxiv
```

任一 row 失败，必须立即中止 matrix。

修复后必须从第 1 行重新跑。

partial final matrix 不得复用。

---

## 18. 最终完成条件

最终完成必须同时满足：

```text
10 行 final official rows 存在
全部 full_dataset=true
全部 VERIFY_OK
全部 fresh
无 timeout row
无 stale row
无 diagnostic row
无 partial row
全部 proof_core_total_ms <= near_2_2x
GPU rows 有真实 GPU evidence
dataset integrity passed
SRS/PK/VK integrity passed
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
static scan passed
full security regression passed
latest.csv/latest.json 只含 10 行
comparison/** untouched
comparison process not killed
main official timing not polluted by comparison-heavy resource interference
```

唯一允许的最终状态：

```text
FINAL_STATUS: COMPLETED_TARGET
FINAL_STATUS: COMPLETED_NEAR_ACCEPTED
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
```

---

## 19. 最终回复格式

如果全部 10 行达到 2x target：

```text
FINAL_STATUS: COMPLETED_TARGET
final_results_path:
- results/paper_secure/latest.csv
- results/paper_secure/latest.json
process_isolation:
- comparison untouched: yes
- comparison process not killed: yes
- main timing not polluted by comparison-heavy interference: yes
proof_equivalence:
- proof relation unchanged: yes
- proof scheme unchanged: yes
- verifier equation unchanged: yes
- transcript/order unchanged: yes
- label order unchanged: yes
tests_passed:
- build CPU
- build CUDA
- CPU ctest
- CUDA ctest
- static scan
- full security regression
- export validation
```

如果存在 near accepted，但全部 <= 2.2x：

```text
FINAL_STATUS: COMPLETED_NEAR_ACCEPTED
final_results_path:
- results/paper_secure/latest.csv
- results/paper_secure/latest.json
near_budget_rows:
- ...
process_isolation:
- comparison untouched: yes
- comparison process not killed: yes
- main timing not polluted by comparison-heavy interference: yes
proof_equivalence:
- proof relation unchanged: yes
- proof scheme unchanged: yes
- verifier equation unchanged: yes
- transcript/order unchanged: yes
- label order unchanged: yes
tests_passed:
- build CPU
- build CUDA
- CPU ctest
- CUDA ctest
- static scan
- full security regression
- export validation
```

不得输出其他 final status。

---

## 20. 后续短 prompt 模板

后续任务 prompt 可以简化为：

```text
你现在在 /home/pzh/project 工作。

先读取并严格遵守 MAIN_EXPERIMENT_INVARIANTS.md。
在 results/performance_engineering/initial_checkpoint.md 中记录该文件 hash。
不得修改 MAIN_EXPERIMENT_INVARIANTS.md。
不得触碰 comparison/**。
不得 kill comparison 进程。
不得在 comparison-heavy resource interference 下运行 official timing。

任务：继续 zkGAT secure 主实验，完成 5 数据集 × CPU/GPU 共 10 行 official results，写入 results/paper_secure/latest.csv 和 latest.json。

最终状态只能是 COMPLETED_TARGET 或 COMPLETED_NEAR_ACCEPTED。
```
