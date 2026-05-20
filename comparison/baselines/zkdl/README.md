jjjjjjjjjjjjjjjjjjjj'j# zkDL-GAT baseline 对比实验目录

本目录是 zkDL baseline 的 GAT 适配实验目录，用于在同一批五个全量数据集、同一套数据/cache/checkpoint/config 输出口径下，对比 zkDL proof system 与主仓库方案。这里对比的是证明方案，不是把主仓库方案混入 zkDL。

## 协议边界

- 本适配严格以 `/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 为最高协议边界。
- 本轮修改未改变该文件规定的 GAT 前向语义：隐藏层多头 attention + ELU，输出层仍是 GAT attention head，输出头先聚合再加偏置并做算术平均。
- 本轮修改未混入主仓库 GAT-ZKML 证明方案；没有搬入 KZG proof schema、quotient polynomial、verifier equation、opening proof 规则、transcript challenge 语义或 proof object layout。
- 未达到 Level A 时，`latest.csv/json` 保持 `blocked + NA`。
- 所有 `partial_formal` 只作为 supporting evidence，不能写入 official 四项。

## 当前目录结构

- zkDL 原 baseline 核心：`bls12-381.cuh`、`fr-tensor.cuh`、`g1-tensor.cuh`、`commitment.cuh`、`proof.cuh`、`zkfc.cuh`、`zkmatmul.cuh`、`zkprod.cuh`、`zkrelu.cuh`。
- GAT 适配入口：`gat_zkdl.cu`，只保留 CLI 解析和显式调度；默认旧 `--run-all` 已禁用，避免 mixed debug/official 结果。
- shared/common 工具：`gat_modules/common.cuh`，包含 timing、路径、格式化和 CUDA buffer 辅助。
- 数据与配置加载：`gat_modules/data_cache.cuh`，负责 official full dataset cache、特征、边和标签加载。
- checkpoint / manifest / cache 加载：`gat_modules/model_bundle.cuh`，负责 checkpoint bundle、manifest 与 GAT 参数解析。
- public topology / edge / group context：`gat_modules/data_cache.cuh` 与 `gat_modules/formal_sparse_utils.cuh`，负责归一化图、`src/dst`、topology hash、group pointer 与 padded shape。
- GAT forward trace 构造：`gat_modules/gpu_forward.cuh`，保留 CUDA forward kernel 与 trace 构造。
- shared formal 工具：`gat_modules/formal_common.cuh`，负责 field 编码、commitment registry、proof size 与 proof_debug 精简。
- CPU relation / gadget：`gat_modules/cpu_forward.cuh`、`gat_modules/cpu_formal_backend.cuh` 与 `gat_modules/run_dispatch.cuh`，提供 CPU trace、host Fr/G1/Commitment/opening、CPU relation prover/verifier 与 runner/export。
- GPU relation / gadget：`gat_modules/formal_sparse_utils.cuh` 与 `gat_modules/formal_softmax_runners.cuh`，负责 sparse/grouped、denominator/inverse/alpha 与 multi-column weighted aggregation partial formal runner。
- CPU/GPU benchmark timing：CPU probe 与 GPU partial formal 分别写入 `runs/<dataset>/cpu/warm/` 和 `runs/<dataset>/gpu/warm/`，不共享 timing row。
- proof_size / artifact / evidence 管理：`gat_modules/formal_common.cuh`、`gat_modules/formal_linear_trace.cuh` 与 `update_zkdl_gat_results.py`；proof_size 只统计 proof 传输对象。
- latest / summary / coverage / formal_partial_debug / debug_level_b 导出：`update_zkdl_gat_results.py` 是当前正式同源导出入口。
- selftest：`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。
- zkDL relation/gadget primitive：`formal_relations.cuh` 与 `formal_*_selftest.cu`。
- 复现实验配置：`configs/zkdl_gat_full_datasets.cfg`。
- 结果同源导出：`update_zkdl_gat_results.py`。
- CPU official 结果：`zkdl_gat_outputs/results/cpu/`。
- GPU official 结果：`zkdl_gat_outputs/results/gpu/`。
- GPU partial evidence：`zkdl_gat_outputs/evidence/` 与 `zkdl_gat_outputs/runs/<dataset>/gpu/warm/`。

## CPU/GPU 口径

- CPU official path 当前存在独立结果目录，但 official 仍为 `blocked + NA`；本轮已实现 CPU-only relation backend，并在 Cora/Citeseer full dataset 上真实完成 sparse/grouped proof generation 与 verification。
- CPU Pubmed full dataset 已尝试并进入 CPU proof/verify，但 host G1 opening replay 超过交互窗口未完成；PPI/ogbn-arxiv 因该性能 blocker 未继续盲跑。
- GPU path 当前不是完整 GPU GAT proof。CUDA forward trace 真实参与 witness 生成，zkDL 原生 CUDA-backed Fr/G1 commitment/opening 真实参与 relation-level prover；verifier 主要是 host transcript/opening 重放。
- CPU 结果不会复制 GPU 结果；GPU kernel timing 不写入 official proof/verify timing；GPU partial formal 只写入 `formal_partial_debug.*`。
- CPU/GPU artifact 与 result 已分目录：`results/cpu/`、`results/gpu/`、`runs/<dataset>/cpu/warm/`、`runs/<dataset>/gpu/warm/`。

## official latest 规则

基础字段保持为 `dataset,status,blocker,benchmark_mode,commitment_time_ms,prove_time_ms,verify_time_ms,proof_size_bytes`。只有 full dataset + complete formal proof generation + complete formal verification 成功后才填四项指标；当前 CPU/GPU 均未达到，因此均为 `NA`。

## 已审计并迁移的 proof-system-neutral 工程优化

- 已迁移：模块化组织；一次加载 dataset/cache/checkpoint；复用 public topology、`src/dst`、topology hash、padded shape 与 hidden layer0/head0 trace；CPU/GPU result/artifact 分目录；relation-level batch runner；`commitment_block_size=64`；multi-column compressed relation；compact artifact；`latest/summary/coverage/formal_partial_debug/debug_level_b` 同源导出；proof_size 只统计 relation proof 传输对象，不包含 witness/tensor/debug dump。
- 不能迁移：主仓库 KZG proof schema、quotient 约束、verifier equation、opening proof 规则、transcript challenge 语义、proof object layout，因为这些是主仓库 proof-system-specific 内容。
- 理论上可迁移但尚未完成：zkDL 自己的 cross-relation batched opening/verifier、lookup/range primitive、max/comparison primitive、rescale/range primitive、GPU buffer 复用、pinned memory、failure 后自动缩 batch、所有层/头 single-process full runner。

## 当前 supported relation

`commitment/evaluation binding`、`transcript binding`、CPU-only host Fr/G1/Commitment relation backend、`HadamardRelationProof`、`MatmulRelationProof`、`LinearRelationProof`、`SparseSourceRoutingRelationProof`、`SparseDestinationRoutingRelationProof`、一维 `GroupedSumRelationProof` 与 `MultiColumnGroupedSumRelationProof`。这些是 primitive 或 relation-level 支持，不等于完整 GAT proof。

## 当前 partial_formal relation

CPU：Cora/Citeseer full dataset hidden layer0/head0 sparse routing 与 grouped sum 已通过 CPU-only relation proof/verify；Pubmed CPU full-dataset attempt 因 host opening replay timeout 未完成。GPU：五个全量数据集 hidden layer0/head0 已验证 sparse routing、grouped sum、denominator grouped sum、inverse Hadamard、inverse destination routing、alpha multiplication、alpha_times_denominator、multi-column sparse routing、multi-column alpha Hadamard 与 8/8 feature multi-column grouped sum weighted aggregation。projection/dot、hidden concat、output bias add、output-head average、public logits binding、fixed-to-rounded decomposition 与 residual bounded range 只能作为 supporting evidence。

## 当前 blocked relation

`target-node softmax group`、`softmax group max/shift`、`ExpMap lookup/range`、`LeakyReLU/ELU lookup-range`、所有层/头 weighted aggregation、`output aggregation`、`output rescale`、real rescale chain 与 rounding sign/tie-breaking 仍 blocked。当前没有把 sparse GAT 改成 dense attention，没有跳过 output attention head，没有用 host equality/debug comparison/witness comparison 冒充 verifier。

## sparse/grouped evidence

| dataset | nodes | edges | status | commitment_block_size | commitment_ms | prove_ms | verify_ms | proof_size_bytes |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| cora | 2708 | 13264 | PASS | 64 | 214.651 | 1058.992 | 837.638 | 26885 |
| citeseer | 3327 | 12431 | PASS | 64 | 159.714 | 924.190 | 586.782 | 26921 |
| pubmed | 19717 | 108365 | PASS | 64 | 2057.175 | 1382.246 | 6988.849 | 28919 |
| ppi | 56944 | 850576 | PASS | 64 | 8091.479 | 4329.571 | 49075.976 | 30140 |
| ogbn-arxiv | 169343 | 1335586 | PASS | 64 | 23067.313 | 6871.957 | 83059.013 | 31259 |

## CPU-only sparse/grouped evidence

| dataset | nodes | edges | status | commitment_ms | prove_ms | verify_ms | proof_size_bytes | relation scope |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| cora | 2708 | 13264 | partial_formal | 13.960 | 45.593 | 26257.734 | 13017 | SparseSourceRouting;SparseDestinationRouting;GroupedSum |
| citeseer | 3327 | 12431 | partial_formal | 103.438 | 43.651 | 36045.114 | 13053 | SparseSourceRouting;SparseDestinationRouting;GroupedSum |
| pubmed | 19717 | 108365 | blocked_after_attempt_timeout | NA | NA | NA | NA | host G1 opening replay timeout |
| ppi | NA | NA | not_run_after_pubmed_timeout | NA | NA | NA | NA | deferred after Pubmed CPU timeout |
| ogbn-arxiv | NA | NA | not_run_after_pubmed_timeout | NA | NA | NA | NA | deferred after Pubmed CPU timeout |

## denominator / inverse / alpha / weighted aggregation evidence

| dataset | nodes | edges | weighted columns | denom | inverse | alpha_from_inverse | alpha_times_denom | weighted aggregation | commitment_block_size | commitment_ms | prove_ms | verify_ms | proof_size_bytes |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cora | 2708 | 13264 | 8/8 | PASS | PASS | PASS | PASS | PASS | 64 | 1507.717 | 8244.618 | 4734.040 | 280185 |
| citeseer | 3327 | 12431 | 8/8 | PASS | PASS | PASS | PASS | PASS | 64 | 1422.316 | 8429.009 | 4301.289 | 280325 |
| pubmed | 19717 | 108365 | 8/8 | PASS | PASS | PASS | PASS | PASS | 64 | 11304.127 | 12535.613 | 35135.303 | 1447807 |
| ppi | 56944 | 850576 | 8/8 | PASS | PASS | PASS | PASS | PASS | 64 | 80826.597 | 43087.596 | 261444.942 | 9857494 |
| ogbn-arxiv | 169343 | 1335586 | 8/8 | PASS | PASS | PASS | PASS | PASS | 64 | 162922.338 | 75629.638 | 406350.312 | 20182539 |

## 复现命令

```bash
make all
./formal_opening_selftest
./formal_sumcheck_selftest
./formal_relation_selftest
./gat_zkdl --formal-cpu-backend-probe --dataset cora --output-root zkdl_gat_outputs
./gat_zkdl --formal-cpu-backend-probe --dataset citeseer --output-root zkdl_gat_outputs
# Pubmed CPU full-dataset attempt is expected to expose the current host opening replay bottleneck.
./gat_zkdl --formal-cpu-backend-probe --dataset pubmed --output-root zkdl_gat_outputs
for dataset in cora citeseer pubmed ppi ogbn-arxiv; do
  ./gat_zkdl --formal-sparse-grouped --dataset "$dataset" --formal-commitment-block-size 64 --output-root zkdl_gat_outputs
  ./gat_zkdl --formal-softmax-aggregation --dataset "$dataset" --formal-commitment-block-size 64 --output-root zkdl_gat_outputs
done
python3 update_zkdl_gat_results.py
```

`--formal-cpu-backend-probe` 现在会运行 CPU-only relation-level backend；Cora/Citeseer 已产出 CPU proof/verify，Pubmed 记录 timeout failure evidence。它仍不是 CPU official full GAT proof，不能填 official 四项。

## 当前是否 Level A

不是 Level A。最具体 blocker 是缺 zkDL-native lookup/range/max/rescale primitive、CPU/GPU batched opening/verifier、all-layer/all-head runner、output aggregation、final logits binding 与完整 proof object；CPU Pubmed 还暴露 host G1 opening replay timeout。

## 下一步最小有效动作

先在 zkDL proof primitive 内实现选择矩阵 lookup/table/range 与 max/comparison gadget，并把 ExpMap、LeakyReLU、ELU 接入 verifier；并行实现 zkDL-native batched opening/verifier。随后补 output aggregation、rescale/rounding 公开契约和所有层/头 full runner。
