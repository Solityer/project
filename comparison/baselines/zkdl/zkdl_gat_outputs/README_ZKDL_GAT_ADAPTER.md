# zkDL-GAT baseline

本目录是 zkDL baseline 的 GAT 正式对比实验目录。实验对象固定为五个全量数据集：`cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`，缺一不可；不能用前三个数据集替代完整实验。

## 实验目的

在同一批数据/cache/checkpoint/config 口径下，用 zkDL 自身 proof system 适配 GAT，并与主仓库方案做论文对比。这里比较的是证明方案，不比较 CUDA forward kernel timing，也不把 partial formal 写成 official。

## 协议边界

`/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 是最高协议边界。本目录只使用 zkDL baseline 自身的 Hyrax-style tensor commitment、opening、transcript、sumcheck 与 relation proof。

## 与主仓库 proof scheme 的边界

禁止迁移主仓库 KZG proof schema、quotient polynomial、verifier equation、opening proof rule、transcript challenge 语义和 proof object layout。本轮修改没有引入这些内容。

## 目录结构

- zkDL baseline 核心：`bls12-381.cuh`、`fr-tensor.cuh`、`g1-tensor.cuh`、`commitment.cuh`、`proof.cuh`、`formal_relations.cuh`。
- GAT 数据/config/checkpoint/cache 加载：`gat_modules/data_cache.cuh`、`gat_modules/model_bundle.cuh`、`configs/zkdl_gat_full_datasets.cfg`。
- public topology / group context：`gat_modules/formal_sparse_utils.cuh`。
- CPU forward / trace：`gat_modules/cpu_forward.cuh`。
- GPU forward / trace：`gat_modules/gpu_forward.cuh`。
- CPU formal backend：`gat_modules/cpu_formal_backend.cuh`。
- GPU formal backend：`gat_modules/formal_softmax_runners.cuh`。
- CPU formal runner：`gat_modules/cpu_formal_runner.cuh` 调用 CPU-only relation backend。
- GPU formal runner：`gat_modules/gpu_formal_runner.cuh` 调用 GPU trace 与 relation gadget。
- GPU output binding runner：`gat_modules/output_binding_runner.cuh` 在真实 full graph output trace 上生成 output-head average / public logits MLE relation artifact。
- full-chain scope runner：`gat_modules/full_chain_scope.cuh` 为每个 dataset/device 记录完整协议范围、已有 partial artifact 和未闭合协议子句；它不写 official manifest。
- CLI 调度：`gat_modules/run_dispatch.cuh` 只做 dataset/device 选择，`gat_zkdl.cu` 只保留薄 CLI 入口。
- runner 侧 artifact/evidence/proof_debug/RESUME 写入辅助：`gat_modules/artifact_writer.cuh` 集中管理 CPU/GPU partial 运行产物，避免 runner/backend 内联文件格式。
- result/export 管理：`update_zkdl_gat_results.py` 从真实 artifact 同源导出 latest、summary、coverage、README，并刷新最终 per-device RESUME。
- 五数据集调度脚本：`run_zkdl_gat_experiments.py` 负责自测、per-dataset/per-device 执行与 exporter。
- selftest：`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。

## CPU path

`--formal-cpu-chain` 使用 CPU-only host Fr/G1/Commitment/opening，不依赖 CUDA proof backend。当前可生成并验证 hidden layer0/head0 full graph/full edge set 的 sparse routing、grouped sum、denominator、inverse、inverse routing、alpha consistency relation proof。`--formal-full-chain-scope --device cpu` 会额外记录 CPU 全链协议范围和缺失子句，但不会写 official manifest。CPU verifier 不读取 witness，proof object 不包含完整 tensor。

## GPU path

`--formal-sparse-grouped`、`--formal-softmax-aggregation` 与 `--formal-output-binding` 使用 GPU Fr/G1 commitment/opening 与 CUDA trace，host 侧重放 verifier。`--formal-output-binding` 证明 output-head average 与 public logits MLE 的线性关系；`--formal-full-chain-scope --device gpu` 会额外记录 GPU 全链协议范围和缺失子句，但不会写 official manifest。GPU kernel timing 不写入 official proof/verify timing；host replay verifier 也不会被写成 GPU full verifier。

## CPU/GPU 结果口径

CPU/GPU 的 result、artifact、proof_debug、RESUME 均位于 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，互不覆盖、互不冒充。official 只允许 full dataset + complete proof generation + complete verification 写入四项指标。

## 如何判断结果有效

runner 只写本次真实运行产生的 artifact；artifact_writer 统一 runner 侧 artifact/evidence/proof_debug/RESUME 格式；full-chain scope runner 记录完整协议门禁但不写 official manifest；exporter 只读 artifact/evidence 并刷新最终结果表与 RESUME；`latest/summary/formal_partial_debug/coverage` 同源导出。旧 probe、旧 schedule、旧 debug、backend 不清楚的结果不作为有效实验结论。

## 构建和运行

```bash
make clean && make all
./run_zkdl_gat_experiments.py --dataset all --device all --output-root zkdl_gat_outputs --commitment-block-size 64
```

也可以按 dataset/device 分阶段执行；每个 dataset/device 的 relation runner 之后会执行 full-chain scope 记录。任一 dataset/device 失败后，runner 会继续调度剩余项，最后统一导出 result 表。

## result 文件说明

- `results/<device>/latest.csv|json`：只含 official 四项；未 full verified 时为 `blocked + NA`。
- `results/<device>/summary.md`：official 摘要与五数据集状态。
- `results/<device>/formal_partial_debug.csv|json`：relation-level partial formal 或 dataset-specific failure evidence。
- `results/<device>/debug_level_b.csv|json`：debug_only，不进入 official。
- `results/coverage_matrix.*`：按 CPU/GPU 区分 supported、partial_formal、debug_only、blocked。

## supported relation

当前支持 zkDL-native commitment/opening、transcript/sumcheck、HadamardRelationProof、MatmulRelationProof、LinearRelationProof、SparseSourceRoutingRelationProof、SparseDestinationRoutingRelationProof、GroupedSumRelationProof、MultiColumnGroupedSumRelationProof。GPU 路径新增真实 GAT output-head average / public logits MLE binding 的 LinearRelationProof artifact。

## partial_formal relation

CPU partial formal 覆盖 hidden layer0/head0 sparse + denominator/inverse/alpha。GPU partial formal 当前五个数据集都有 sparse/grouped artifact；denominator/inverse/alpha、weighted aggregation 与 output binding 覆盖范围以 coverage_matrix 为准。lookup/range/max/comparison 已有 selftest 级 tamper rejection，但尚未接入 full GAT table/context。

## blocked relation

blocked 项包括 target-node max/shift、ExpMap lookup/range table、LeakyReLU/ELU lookup-range table、all-layer/all-head proof runner、output attention aggregation、bias/average/public logits binding、real rescale chain、rounding sign/tie-breaking public contract 和 clean verifier。full-chain scope 已记录这些子句，但尚未生成可 official 的 proof object。

## 当前五数据集状态

| dataset | CPU 状态 | GPU 状态 | CPU artifact | GPU artifact |
| --- | --- | --- | --- | --- |
| cora | partial_formal | partial_formal | zkdl_gat_outputs/runs/cora/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt | zkdl_gat_outputs/runs/cora/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| citeseer | partial_formal | partial_formal | zkdl_gat_outputs/runs/citeseer/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt | zkdl_gat_outputs/runs/citeseer/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| pubmed | partial_formal | partial_formal | zkdl_gat_outputs/runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt | zkdl_gat_outputs/runs/pubmed/gpu/warm/formal_artifacts/zkdl_sparse_grouped_relations.proof.txt |
| ppi | partial_formal | partial_formal | zkdl_gat_outputs/runs/ppi/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt | zkdl_gat_outputs/runs/ppi/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| ogbn-arxiv | partial_formal | partial_formal | zkdl_gat_outputs/runs/ogbn-arxiv/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt | zkdl_gat_outputs/runs/ogbn-arxiv/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |

## official 结果表

| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| cpu | cora | blocked | warm | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| cpu | citeseer | blocked | warm | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| cpu | pubmed | blocked | warm | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| cpu | ppi | blocked | warm | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| cpu | ogbn-arxiv | blocked | warm | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| gpu | cora | blocked | warm | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| gpu | citeseer | blocked | warm | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| gpu | pubmed | blocked | warm | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| gpu | ppi | blocked | warm | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| gpu | ogbn-arxiv | blocked | warm | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |

## CPU 当前状态

CPU path 已实现 `--formal-cpu-chain`，并按五个全量数据集逐一运行。`--formal-full-chain-scope --device cpu` 记录每个数据集的 full-chain 范围、hidden/output head 数和 missing clauses。当前五个数据集均有 full graph/full edge set 的 relation-level `partial_formal` artifact；CPU official 仍为 NA，因为 full GAT proof object、all-layer/all-head proof、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding 尚未闭合。

## GPU 当前状态

GPU path 当前五个全量数据集都有 relation-level `partial_formal` sparse/grouped artifact；cora、citeseer、ppi、ogbn-arxiv 还有 denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation artifact。`--formal-output-binding` 在真实 output trace 上生成 output-head average / public logits MLE binding artifact；`--formal-full-chain-scope --device gpu` 记录每个数据集的 full-chain 范围、hidden/output head 数和 missing clauses。pubmed 当前只保留 sparse/grouped artifact，不把旧 softmax 失败日志或缺失 artifact 当作有效结论。GPU verifier 当前是 host replay，不写成 GPU full verifier；GPU official 仍为 NA。

## Level A 状态

当前不是 Level A。CPU/GPU official latest 仍为 `blocked + NA`，因为还没有完整 GAT proof generation + complete verification。

## 已迁移的 proof-system-neutral 优化

已迁移或保留：模块化组织、一次加载 dataset/cache/checkpoint、public topology/context 复用、trace 复用、CPU/GPU 分目录、artifact_writer 集中写入、full-chain scope 门禁、output binding public MLE、relation-level batch、multi-column grouped sum、CPU commitment block-size 调度、CPU commitment 按行并行与零/一标量跳过、compact artifact、proof_size 只统计 proof object、同源导出、五数据集失败不中断的结果口径。

## 不能迁移的主仓库优化

不能迁移：KZG batch opening、quotient 约束、verifier equation、opening proof rule、transcript challenge 语义、proof object layout、任何 protocol/schema 专属约束。

## 下一步最小有效动作

把 selftest 级 lookup/range/max/comparison 接入 full GAT 的 VK_static/table/context；随后把 hidden/output 所有层头、output aggregation、bias、average、public logits binding 与 real rescale/rounding 串成单一 zkDL proof object。
