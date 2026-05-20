# zkDL-GAT blocker 证据

- 本目录只使用 zkDL baseline 自身的 commitment、opening、transcript、sumcheck 与 relation proof。没有引入主仓库 GAT-ZKML 的 KZG proof schema、quotient polynomial、verifier equation、opening proof 规则或 proof object layout。
- CPU-only backend 已实现到 relation-level：`gat_modules/cpu_forward.cuh` 构造 CPU trace，`gat_modules/cpu_formal_backend.cuh` 提供 host Fr/G1/Commitment/opening 与 CPU transcript/sumcheck prover/verifier，`gat_modules/run_dispatch.cuh` 负责 CPU runner/export。
- CPU Cora/Citeseer full dataset hidden layer0/head0 sparse/grouped relation 已真实 proof generation + verification；artifact 分别位于 `runs/cora/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt` 与 `runs/citeseer/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt`。
- CPU Pubmed full dataset 已进入 CPU proof/verify attempt，但 host G1 opening replay 未在交互窗口内完成；failure evidence 位于 `runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_backend_attempt.timeout.txt`。
- PPI/ogbn-arxiv CPU proof 未继续盲跑：Pubmed 已暴露 CPU host opening replay 性能 blocker，较大数据集等待 batched opening/verifier 后再跑。
- GPU partial path 当前真实状态：CUDA forward trace 参与 witness 生成；zkDL 原生 Fr/G1 commitment/opening kernel 参与 relation-level prover；verification 主要由 host 重放 transcript/opening 检查；仍不是完整 GPU GAT proof/verify。
- 五个全量数据集 hidden layer0/head0 sparse/grouped relation 已通过 proof/verify。
- 五个全量数据集 hidden layer0/head0 denominator/inverse/alpha 与 8/8 feature weighted aggregation 已通过 relation-level proof/verify。
- 上述 evidence 仍只是 supporting evidence：softmax max/shift、ExpMap、所有层/头 weighted aggregation、output aggregation、LeakyReLU/ELU、rescale 或 final logits chain 仍未完整证明。
- `formal_relations.cuh` 当前只有 Hadamard/Matmul/Linear/SparseRouting/GroupedSum/MultiColumnGroupedSum relation；没有 selection-matrix lookup/table commitment primitive、bit-decomposition range primitive、Bmax max/comparison relation 或 denominator nonzero/inverse soundness gadget。
- ExpMap、LeakyReLU 与 ELU 仍 blocked：当前缺少 zkDL-native public table 或 table commitment、selection matrix one-hot/range 约束与 verifier-side table binding。
- output aggregation 已做代码级可表达性检查：现有 Sparse/Grouped/MultiColumnGrouped/Hadamard/Linear 可表达 `alpha * message` 与按 destination grouped sum，但输出层还缺 output-head alpha 的 max/ExpMap/range 来源、输出 bias/rescale、head average 与 public logits binding 的同一 proof object。
- all-layer/all-head runner 仍 blocked：hidden layer0/head0 trace 已接 relation proof；其余 head/layer 与 output attention head 尚缺完整 per-head trace manifest、relation scheduling 与 cross-relation opening aggregation。
- real rescale chain 与 rounding sign/tie-breaking 仍 blocked：当前 manifest/checkpoint 没有逐 relation 公开 `rho/delta/rem_bits/tie_policy` 字段，不能用 host equality、residual range 或 debug comparison 伪造 rounding proof。
- `results/cpu/latest.*` 与 `results/gpu/latest.*` 只写 official 四项；当前 CPU/GPU 均未 full verified，所以保持 NA。
- `debug_level_b.*` 不作为 formal benchmark；旧 debug artifact 已清理。
