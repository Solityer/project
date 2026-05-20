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
- prover/verifier runner：`gat_modules/run_dispatch.cuh`，`gat_zkdl.cu` 只保留清晰 CLI 入口。
- result/export/artifact/resume 管理：`update_zkdl_gat_results.py` 从真实 artifact 同源导出。
- selftest：`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。

## CPU path

`--formal-cpu-chain` 使用 CPU-only host Fr/G1/Commitment/opening，不依赖 CUDA proof backend。当前可生成并验证 hidden layer0/head0 full graph/full edge set 的 sparse routing、grouped sum、denominator、inverse、inverse routing、alpha consistency relation proof。CPU verifier 不读取 witness，proof object 不包含完整 tensor。

## GPU path

`--formal-sparse-grouped` 与 `--formal-softmax-aggregation` 使用 GPU Fr/G1 commitment/opening 与 CUDA trace，host 侧重放 verifier。GPU kernel timing 不写入 official proof/verify timing；host replay verifier 也不会被写成 GPU full verifier。

## CPU/GPU 结果口径

CPU/GPU 的 result、artifact、proof_debug、RESUME 均位于 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，互不覆盖、互不冒充。official 只允许 full dataset + complete proof generation + complete verification 写入四项指标。

## 如何判断结果有效

runner 只写本次真实运行产生的 artifact；exporter 只读 artifact/evidence；`latest/summary/formal_partial_debug/coverage` 同源导出。旧 probe、旧 schedule、旧 debug、backend 不清楚的结果不作为有效实验结论。

## 构建和运行

```bash
make clean && make all
./formal_opening_selftest
./formal_sumcheck_selftest
./formal_relation_selftest
for dataset in cora citeseer pubmed ppi ogbn-arxiv; do
  ./gat_zkdl --formal-cpu-chain --dataset "$dataset" --output-root zkdl_gat_outputs
  ./gat_zkdl --formal-sparse-grouped --dataset "$dataset" --output-root zkdl_gat_outputs
  ./gat_zkdl --formal-softmax-aggregation --dataset "$dataset" --output-root zkdl_gat_outputs
done
./update_zkdl_gat_results.py
```

## result 文件说明

- `results/<device>/latest.csv|json`：只含 official 四项；未 full verified 时为 `blocked + NA`。
- `results/<device>/summary.md`：official 摘要与五数据集状态。
- `results/<device>/formal_partial_debug.csv|json`：relation-level partial formal 或 dataset-specific failure evidence。
- `results/<device>/debug_level_b.csv|json`：debug_only，不进入 official。
- `results/coverage_matrix.*`：按 CPU/GPU 区分 supported、partial_formal、debug_only、blocked。

## supported relation

当前支持 zkDL-native commitment/opening、transcript/sumcheck、HadamardRelationProof、MatmulRelationProof、LinearRelationProof、SparseSourceRoutingRelationProof、SparseDestinationRoutingRelationProof、GroupedSumRelationProof、MultiColumnGroupedSumRelationProof。

## partial_formal relation

CPU partial formal 覆盖 hidden layer0/head0 sparse + denominator/inverse/alpha。GPU partial formal 覆盖 hidden layer0/head0 sparse + denominator/inverse/alpha + weighted aggregation。lookup/range/max/comparison 已有 selftest 级 tamper rejection，但尚未接入 full GAT table/context。

## blocked relation

blocked 项包括 target-node max/shift、ExpMap lookup/range table、LeakyReLU/ELU lookup-range table、all-layer/all-head runner、output attention aggregation、bias/average/public logits binding、real rescale chain、rounding sign/tie-breaking public contract。

## 当前五数据集状态

| dataset | CPU 状态 | GPU 状态 | CPU artifact | GPU artifact |
| --- | --- | --- | --- | --- |
| cora | partial_formal | partial_formal | zkdl_gat_outputs/runs/cora/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt | zkdl_gat_outputs/runs/cora/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| citeseer | partial_formal | partial_formal | zkdl_gat_outputs/runs/citeseer/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt | zkdl_gat_outputs/runs/citeseer/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| pubmed | failed_with_evidence | partial_formal | zkdl_gat_outputs/runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_formal_chain.timeout.txt | zkdl_gat_outputs/runs/pubmed/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| ppi | failed_with_evidence | partial_formal | zkdl_gat_outputs/runs/ppi/cpu/warm/formal_artifacts/zkdl_cpu_formal_chain.timeout.txt | zkdl_gat_outputs/runs/ppi/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| ogbn-arxiv | failed_with_evidence | partial_formal | zkdl_gat_outputs/runs/ogbn-arxiv/cpu/warm/formal_artifacts/zkdl_cpu_formal_chain.timeout.txt | zkdl_gat_outputs/runs/ogbn-arxiv/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |

## CPU 当前状态

CPU path 已实现 `--formal-cpu-chain`，并按五个全量数据集逐一尝试。Cora/Citeseer 达到 relation-level `partial_formal`；Pubmed/PPI/ogbn-arxiv 均有各自的 `failed_with_evidence` timeout artifact。CPU official 仍为 NA。

## GPU 当前状态

GPU path 已有五个全量数据集的 relation-level `partial_formal` artifact，覆盖 sparse/grouped、denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation。GPU verifier 当前是 host replay，不写成 GPU full verifier；GPU official 仍为 NA。

## Level A 状态

当前不是 Level A。CPU/GPU official latest 仍为 `blocked + NA`，因为还没有完整 GAT proof generation + complete verification。

## 已迁移的 proof-system-neutral 优化

已迁移或保留：模块化组织、一次加载 dataset/cache/checkpoint、public topology/context 复用、trace 复用、CPU/GPU 分目录、relation-level batch、multi-column grouped sum、compact artifact、proof_size 只统计 proof object、同源导出、五数据集失败不中断的结果口径。

## 不能迁移的主仓库优化

不能迁移：KZG batch opening、quotient 约束、verifier equation、opening proof rule、transcript challenge 语义、proof object layout、任何 protocol/schema 专属约束。

## 下一步最小有效动作

把 selftest 级 lookup/range/max/comparison 接入 full GAT 的 VK_static/table/context；随后把 hidden/output 所有层头、output aggregation、bias、average、public logits binding 与 real rescale/rounding 串成单一 zkDL proof object。
