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
- protocol metadata / VK_static / VK_model / table context / quant rounding contract：`gat_modules/protocol_context.cuh`，其中 table range 会按 full forward trace 的真实 `m_dst-z_edge` 自动提升并进入 public binding。
- CPU forward / trace：`gat_modules/cpu_forward.cuh`。
- GPU forward / trace：`gat_modules/gpu_forward.cuh`。
- full proof object：`gat_modules/full_proof_object.cuh` 写 `full_proof_manifest.json`、`full_verifier_manifest.json` 与不含完整 witness/tensor 的 proof object 摘要。
- full verifier：`gat_modules/full_verifier.cuh` 校验 all-layer/all-head 子证明、public metadata/table/quant binding、`missing_relations=[]`、`verifier_reads_witness=false` 与 `witness_included_in_proof=false`。
- full-chain runner：`gat_modules/full_chain_runner.cuh` 与 `gat_modules/cpu_full_chain_runner.cuh` 遍历 hidden/output 所有 head，并把子证明写入 full manifest。
- relation runner：`projection_runner.cuh`、`rescale_rounding_runner.cuh`、`attention_dot_runner.cuh`、`range_comparison_runner.cuh`、`expmap_lookup_runner.cuh`、`nonlinear_lookup_runner.cuh`、`softmax_lookup_runner.cuh`、`output_attention_runner.cuh`、`hidden_concat_runner.cuh`。
- CLI 调度：`gat_modules/run_dispatch.cuh` 只做 dataset/device 选择，`gat_zkdl.cu` 只保留薄 CLI 入口。
- result/export 管理：`update_zkdl_gat_results.py` 只从 full proof manifest + full verifier manifest 导出 official latest/summary，并保留 partial/debug 分表。
- 五数据集调度脚本：`run_zkdl_gat_experiments.py` 负责自测、per-dataset/per-device 执行与 exporter。
- selftest：`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。

## CPU path

`--formal-full-chain --device cpu` 已覆盖五个全量数据集的完整 GAT proof generation + verification。CPU path 使用 zkDL-native CPU commitment/opening/relation verifier，不读取 witness，不把完整 tensor 写入 proof object。PPI 与 ogbn-arxiv 使用较小 CPU commitment block 以降低 host G1 inner recursion；结果仍由 full manifests 通过 official gate 导出。

## GPU path

`--formal-full-chain --device gpu` 已覆盖五个全量数据集的完整 GAT proof generation + verification。GPU path 使用 zkDL-native Fr/G1 commitment/opening 与 CUDA trace，verification 走 full proof object 的 clean verifier path；GPU kernel timing 不冒充 proof/verify timing。

## CPU/GPU 结果口径

CPU/GPU 的 result、artifact、proof_debug、RESUME 均位于 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，互不覆盖、互不冒充。official 只允许 full dataset + complete proof generation + complete verification 写入四项指标。

## 如何判断结果有效

official 只从 `full_proof_manifest.json` 与 `full_verifier_manifest.json` 导出，且必须同时满足 `proof_generated=true`、`verified=true`、`missing_relations=[]`、`verifier_reads_witness=false`、`witness_included_in_proof=false`。旧 probe、旧 schedule、debug-only 和 relation-level partial 不进入 official。

## 构建和运行

```bash
make all
./gat_zkdl --formal-full-chain --device gpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device cpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64
python3 update_zkdl_gat_results.py
```

五个数据集可以按 dataset/device 分阶段执行。PPI/GPU 与 ogbn-arxiv/GPU 使用 `8192` block 运行；PPI/CPU 与 ogbn-arxiv/CPU 使用 `64` block 运行。最终只以 exporter 导出的 official latest 为准。

## result 文件说明

- `results/<device>/latest.csv|json`：只含 official 四项。
- `results/<device>/summary.md`：只展示 official 摘要。
- `results/<device>/formal_partial_debug.csv|json`：relation-level partial formal，保留调试用途，不进入 official。
- `results/<device>/debug_level_b.csv|json`：debug_only，不进入 official。
- `results/coverage_matrix.*`：记录 supported、partial_formal、debug_only、official gate 等覆盖信息。
- `runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_chain_public_metadata.json`：记录 full-chain public metadata、VK_static/VK_model、table context、quant/rounding contract；它不是 proof object。
- `runs/<dataset>/<device>/warm/full_proof_manifest.json` 与 `full_verifier_manifest.json`：official 唯一入口；必须 `proof_generated=true`、`verified=true` 且 `missing_relations=[]` 才会写 latest。

## 已闭合 relation

full proof object 已包含 hidden/output projection matmul、real rescale/rounding、rounded trace binding、attention source/destination dot、LeakyReLU lookup/range、target-node max/shift comparison、ExpMap lookup/range、denominator grouped sum、denominator nonzero、inverse soundness、alpha consistency、alpha * message、weighted aggregation、hidden concat、ELU lookup/range、output attention bias、output head average 和 public logits binding。Verifier 只读取 public metadata、commitments、proof object 与必要 public table/quant metadata。

## official 结果表

| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| cpu | cora | ok | warm | 635063.747 | 959127.647 | 785188.254 | 7162889 |  |
| cpu | citeseer | ok | warm | 485544.394 | 695846.112 | 823660.929 | 6831181 |  |
| cpu | pubmed | ok | warm | 838669.938 | 972933.76 | 3358514.479 | 17747551 |  |
| cpu | ppi | ok | warm | 6053234.188 | 944898.332 | 6432377.9 | 30728091 |  |
| cpu | ogbn-arxiv | ok | warm | 2294748.217 | 1261421.929 | 11015509.922 | 42072342 |  |
| gpu | cora | ok | warm | 201935.494 | 315047.541 | 749597.5 | 6959845 |  |
| gpu | citeseer | ok | warm | 359076.448 | 300987.887 | 956138.021 | 6628137 |  |
| gpu | pubmed | ok | warm | 632869.764 | 388140.447 | 3364287.373 | 17538459 |  |
| gpu | ppi | ok | warm | 1950148.59 | 3090128.024 | 494271.239 | 2496711 |  |
| gpu | ogbn-arxiv | ok | warm | 2518575.277 | 3869816.174 | 1403228.972 | 3301922 |  |

## official gate

CPU/GPU 五个全量数据集均已通过 official gate。每个 `full_proof_manifest.json` 均为 `proof_generated=true`、`verified=true`、`missing_relations=[]`、`witness_included_in_proof=false`；每个 `full_verifier_manifest.json` 均为 `verified=true`、`verifier_reads_witness=false`。

## 已迁移的 proof-system-neutral 优化

已迁移或保留：模块化组织、一次加载 dataset/cache/checkpoint、public topology/context 复用、public metadata/VK_static/VK_model/table context/rounding contract 组织、trace 复用、CPU/GPU 分目录、artifact_writer 集中写入、full-chain scope 门禁、output binding public MLE、relation-level batch、multi-column grouped sum、CPU commitment block-size 调度、CPU commitment 按行并行与零/一标量跳过、compact artifact、proof_size 只统计 proof object、同源导出、五数据集失败不中断的结果口径。

## 不能迁移的主仓库优化

不能迁移：KZG batch opening、quotient 约束、verifier equation、opening proof rule、transcript challenge 语义、proof object layout、任何 protocol/schema 专属约束。

## 下一步最小有效动作

当前 official 表可直接用于 zkDL-GAT baseline 对比。后续若重跑，请保持 exporter gate 不放宽，并按 CPU/GPU 分目录保留 manifest 与 latest。
