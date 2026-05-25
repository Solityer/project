# zkDL-GAT baseline

本目录是 zkDL baseline 的 GAT 正式对比实验目录。实验对象固定为五个全量数据集：`cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`。CPU 与 GPU 是两条独立 official 路径，结果分别写入 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，再由 exporter 从 full proof/verifier manifest 导出 latest 与 summary。

## 实验目的

本目录在同一批主仓库 data/cache/checkpoint/config 口径下，用 zkDL 自身 proof system 适配 GAT 前向证明，并导出可与主仓库结果对齐的 official 四项指标：`commitment_time_ms`、`prove_time_ms`、`verify_time_ms`、`proof_size_bytes`。

## 协议边界

`/home/pzh/project/comparison/GAT-ZKML-zkDL.md` 是最高协议边界。本目录只使用 zkDL baseline 自身的 Hyrax-style tensor commitment、opening、transcript、sumcheck 与 relation proof；证明范围固定为 GAT 前向计算证明。

## 与主仓库 proof scheme 的边界

本目录不迁移主仓库 KZG proof schema、quotient polynomial、verifier equation、opening proof rule、transcript challenge 语义和 proof object layout。主仓库只提供数据、checkpoint、config 与结果口径参照。

## 工程优化迁移边界

可以迁移的内容只限 proof-system-neutral 工程优化，包括数据/cache/checkpoint 读取、全量 dataset/device 调度、CPU/GPU 目录隔离、计时字段分列、manifest gate、结果表导出、固定 dataset 顺序和复现脚本组织。禁止迁移的内容包括主仓库 KZG 承诺布局、quotient polynomial、verifier equation、opening proof rule、transcript challenge 语义、proof object layout 以及任何 proof-system-specific 字节结构。

## 目录结构

- zkDL 核心：`bls12-381.cuh`、`fr-tensor.cuh`、`g1-tensor.cuh`、`commitment.cuh`、`proof.cuh`、`formal_relations.cuh`。
- GAT 加载与前向：`gat_modules/data_cache.cuh`、`gat_modules/model_bundle.cuh`、`gat_modules/cpu_forward.cuh`、`gat_modules/gpu_forward.cuh`。
- 协议公共上下文：`gat_modules/protocol_context.cuh` 生成 public metadata、VK_static/VK_model、table context 与 quant/rounding contract。
- relation proof：`projection_runner.cuh`、`rescale_rounding_runner.cuh`、`attention_dot_runner.cuh`、`range_comparison_runner.cuh`、`expmap_lookup_runner.cuh`、`nonlinear_lookup_runner.cuh`、`softmax_lookup_runner.cuh`、`output_attention_runner.cuh`、`hidden_concat_runner.cuh`、`full_output_binding_runner.cuh`。
- full-chain runner：`gat_modules/full_chain_runner.cuh` 和 `gat_modules/cpu_full_chain_runner.cuh` 遍历 hidden/output 全部 head 并写 full proof/verifier manifest。
- proof object 与 gate：`gat_modules/full_proof_object.cuh` 写 proof object 摘要、`full_proof_manifest.json`、`full_verifier_manifest.json`；`gat_modules/full_verifier.cuh` 执行 clean verifier gate。
- CLI 与调度：`gat_zkdl.cu` 是薄 CLI，`gat_modules/run_dispatch.cuh` 只分发 `--formal-full-chain`。
- 全量调度：`run_zkdl_gat_experiments.py` 可按 dataset/device 调度 full-chain 运行，并在末尾调用 exporter。
- 结果导出：`update_zkdl_gat_results.py` 只从 full proof/verifier manifest 导出 official latest/summary，并同步重写本 README。
- 构建与自测：`Makefile`、`formal_opening_selftest.cu`、`formal_sumcheck_selftest.cu`、`formal_relation_selftest.cu`。

## CPU/GPU official 路径

`--formal-full-chain --device cpu` 使用 CPU commitment/opening/relation verifier；`--formal-full-chain --device gpu` 使用 CUDA trace 与 zkDL-native Fr/G1 commitment/opening。两条路径都写入 `zkdl_gat_outputs/runs/<dataset>/<device>/warm/`，CPU/GPU 结果互不覆盖。

## full-chain 产物

每个 dataset/device 成功运行后必须保留以下文件：

- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/full_proof_manifest.json`：proof 侧 manifest，记录 `protocol_id`、`proof_scope`、`is_full_dataset`、`proof_generated`、`verified`、`missing_relations`、proof object 路径、relation proof 摘要，以及 `commitment_time_ms`、`prove_time_ms`、`proof_size_bytes`。
- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/full_verifier_manifest.json`：verifier 侧 manifest，记录 clean verifier 是否接受，并显式保留 `verifier_reads_witness=false`、`witness_included_in_proof=false`、`verify_time_ms`。
- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_proof_object.txt`：zkDL full-chain proof object 摘要和子证明摘要，不包含完整 witness 或完整 tensor。
- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_chain_public_metadata.json`：public metadata、VK_static/VK_model、table context、topology、quant/rounding contract。

## official gate

exporter 的 official gate 在 `update_zkdl_gat_results.py` 中实现，只接受同时满足以下条件的行：

- proof manifest 与 verifier manifest 均存在且 JSON 格式有效。
- `protocol_id=GAT-ZKML-zkDL`。
- `proof_scope=zkdl_gat_full`。
- `benchmark_mode=warm`。
- manifest 中的 `dataset`、`device` 与当前导出的 dataset/device 一致。
- `is_full_dataset=true`。
- `relation_proofs_generated=true`。
- proof manifest 与 verifier manifest 均满足 `proof_generated=true`、`verified=true`。
- proof manifest 与 verifier manifest 均满足 `missing_relations=[]`。
- `verifier_reads_witness=false`。
- proof manifest 与 verifier manifest 均满足 `witness_included_in_proof=false`。
- proof object 文件与 public metadata 文件真实存在。
- `proof_payload_version=2` 且 `proof_size_definition=deterministic_relation_payload_sum_v2`。
- `proof_size_bytes` 必须等于 `relations[].proof_size_bytes` 的确定性求和。

partial、debug、relation-level evidence、旧失败日志和临时诊断产物不能进入 official；exporter 不从这些文件回填 official 四项指标。

## 结果文件

- `zkdl_gat_outputs/results/latest.csv|json`：CPU/GPU 合并 official 表。
- `zkdl_gat_outputs/results/summary.md`：CPU/GPU 合并摘要。
- `zkdl_gat_outputs/results/cpu/latest.csv|json` 与 `summary.md`：CPU official 分表。
- `zkdl_gat_outputs/results/gpu/latest.csv|json` 与 `summary.md`：GPU official 分表。
- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_proof_object.txt`：proof metadata 与子证明摘要，不包含完整 witness/tensor。
- `zkdl_gat_outputs/runs/<dataset>/<device>/warm/formal_artifacts/zkdl_full_chain_public_metadata.json`：public metadata、VK_static/VK_model、table context 与 quant/rounding contract。

## 完整复现实验流程

完整复现需要覆盖五个全量数据集的 CPU/GPU 两条路径，共十个 dataset/device 组合。全量实验耗时较长；如果只需要检查当前目录仍可构建、可运行、可导出 official，可先运行单个 Cora CPU/GPU 示例，但这不代表完整五数据集复现。

推荐的完整流程如下：

```bash
make all
./formal_opening_selftest
./formal_sumcheck_selftest
./formal_relation_selftest
python3 run_zkdl_gat_experiments.py --dataset all --device all --output-root zkdl_gat_outputs --commitment-block-size 64 --skip-selftests
python3 update_zkdl_gat_results.py
```

`run_zkdl_gat_experiments.py` 的等价显式命令清单如下。每条命令都会运行一个完整 dataset/device 的 full-chain proof，不会调用 partial/debug runner。

```bash
./gat_zkdl --formal-full-chain --device cpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device cpu --dataset citeseer --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device cpu --dataset pubmed --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device cpu --dataset ppi --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device cpu --dataset ogbn-arxiv --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device gpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device gpu --dataset citeseer --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device gpu --dataset pubmed --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device gpu --dataset ppi --output-root zkdl_gat_outputs --formal-commitment-block-size 64
./gat_zkdl --formal-full-chain --device gpu --dataset ogbn-arxiv --output-root zkdl_gat_outputs --formal-commitment-block-size 64
python3 update_zkdl_gat_results.py
```

## 单数据集单设备复现

单个 dataset/device 的模板命令如下：

```bash
./gat_zkdl --formal-full-chain --device <cpu|gpu> --dataset <cora|citeseer|pubmed|ppi|ogbn-arxiv> --output-root zkdl_gat_outputs --formal-commitment-block-size 64
python3 update_zkdl_gat_results.py
```

单独运行某个 dataset/device 只验证该组合的复现能力。完整实验必须覆盖 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv` 的 CPU/GPU 十个组合。

## Cora 示例

GPU 路径：

```bash
./gat_zkdl --formal-full-chain --device gpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64
```

CPU 路径：

```bash
./gat_zkdl --formal-full-chain --device cpu --dataset cora --output-root zkdl_gat_outputs --formal-commitment-block-size 64
```

参数含义如下：

- `--formal-full-chain`：运行完整 zkDL-GAT proof chain，而不是 partial relation 或 debug 入口。
- `--device`：选择 `cpu` 或 `gpu` official 路径；CPU/GPU 产物分别写入独立目录。
- `--dataset`：选择全量数据集名称，可用值为 `cora`、`citeseer`、`pubmed`、`ppi`、`ogbn-arxiv`。
- `--output-root`：指定输出根目录；CLI 要求该目录留在当前 zkDL baseline 目录内。
- `--formal-commitment-block-size`：指定 commitment/opening 相关分块参数，用于控制 full-chain proof 过程中的资源开销；当前 official 复现使用 `64`。

## 结果导出

运行或重跑任意 dataset/device 后，用同源 exporter 重新导出 official 表：

```bash
python3 update_zkdl_gat_results.py
```

exporter 会重新生成 `zkdl_gat_outputs/results/latest.csv`、`zkdl_gat_outputs/results/latest.json`、`zkdl_gat_outputs/results/summary.md`，以及 `results/cpu/`、`results/gpu/` 下的分表。它只从 full proof/verifier manifest 读取数据；不读取 partial/debug 产物。

## manifest 检查

可以用下面的命令检查 Cora CPU/GPU 是否满足 official gate 的核心字段：

```bash
python3 - <<'PY'
import json
from pathlib import Path

root = Path('zkdl_gat_outputs/runs')
for device in ['gpu', 'cpu']:
    run = root / 'cora' / device / 'warm'
    proof = json.loads((run / 'full_proof_manifest.json').read_text())
    verifier = json.loads((run / 'full_verifier_manifest.json').read_text())
    proof_object = Path(proof['proof_object_path'])
    public_metadata = run / 'formal_artifacts' / 'zkdl_full_chain_public_metadata.json'
    print(device, {
        'proof_generated': proof['proof_generated'] and verifier['proof_generated'],
        'verified': proof['verified'] and verifier['verified'],
        'missing_relations': proof['missing_relations'] + verifier['missing_relations'],
        'verifier_reads_witness': verifier['verifier_reads_witness'],
        'witness_included_in_proof': proof['witness_included_in_proof'] or verifier['witness_included_in_proof'],
        'proof_object_exists': proof_object.exists(),
        'public_metadata_exists': public_metadata.exists(),
    })
PY
```

## 指标口径

- `commitment_time_ms` 来自 full proof manifest 中所有 relation commitment 时间之和。
- `prove_time_ms` 来自 full proof manifest 中 proof generation 时间之和。
- `verify_time_ms` 来自 full verifier manifest；该值表示 clean verifier 重放 full-chain proof 的时间。
- `proof_size_bytes` 使用 `deterministic_relation_payload_sum_v2` 口径：每个 relation 的 `proof_size_bytes` 来自 zkDL relation proof payload 的结构化 `serialized_size_bytes()`，full proof size 只等于固定 relation 顺序下的 `relations[].proof_size_bytes` 求和。
- `proof_size_bytes` 不统计 witness、完整 tensor、debug dump、日志、manifest 文件文本长度、proof object 文件文本长度、README、路径、时间戳、`generated_at` 或机器环境信息。
- range/max 与 rescale 的位宽来自 public quant/table contract；同一代码、同一参数、同一 dataset/device 下，`proof_size_bytes` 应稳定。
- CUDA/CPU forward timing 只服务于运行日志与诊断，不写入 official 四项。
- Cora 或其他单数据集重跑后 timing 可能自然变化；如果论文需要 prover 总耗时，可以使用 `commitment_time_ms + prove_time_ms`，但 official 表保留分列，不覆盖原字段。
- official 表只能由真实 full-chain 运行后的 manifest 通过 exporter 生成，不能手写。

## official 结果表

| device | dataset | status | benchmark_mode | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| cpu | cora | ok | warm | 708671.239 | 926476.54 | 740723.891 | 8210717 |
| cpu | citeseer | ok | warm | 485544.394 | 695846.112 | 823660.929 | 8218353 |
| cpu | pubmed | ok | warm | 838669.938 | 972933.76 | 3358514.479 | 19011475 |
| cpu | ppi | ok | warm | 6053234.188 | 944898.332 | 6432377.9 | 30962995 |
| cpu | ogbn-arxiv | ok | warm | 2294748.217 | 1261421.929 | 11015509.922 | 42090763 |
| gpu | cora | ok | warm | 203114.562 | 346084.162 | 750609.4 | 8007673 |
| gpu | citeseer | ok | warm | 359076.448 | 300987.887 | 956138.021 | 8015309 |
| gpu | pubmed | ok | warm | 632869.764 | 388140.447 | 3364287.373 | 18802383 |
| gpu | ppi | ok | warm | 1950148.59 | 3090128.024 | 494271.239 | 2731615 |
| gpu | ogbn-arxiv | ok | warm | 2518575.277 | 3869816.174 | 1403228.972 | 3320343 |

## selftest / Gate

Gate 1 为 commitment/opening 自测，Gate 2 为 sumcheck 自测，Gate 3 为 relation proof/verifier 自测；日志位于 `zkdl_gat_outputs/runs/selftest/`。这些 Gate 不替代 full-chain official gate，只证明基础 zkDL 后端可用。
