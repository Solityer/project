# zkDL-GAT CPU 继续点

- 数据集: `ogbn-arxiv`
- 当前状态: `partial_formal`；official 仍为 `blocked + NA`。
- artifact: `zkdl_gat_outputs/runs/ogbn-arxiv/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt`
- full-chain scope: `zkdl_gat_outputs/evidence/full_chain_scope_cpu_ogbn-arxiv.json`；missing clauses `13`。
- full proof manifest: `NA`；relation_proofs_generated `NA`；proof_generated `NA`；verified `NA`。
- 五个全量数据集缺一不可；该文件只描述当前 dataset/device，不可替代其他行。
- Level A blocker: full GAT proof object、all-layer/all-head、output aggregation、real rescale/rounding 仍未闭合。
