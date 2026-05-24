# zkDL-GAT CPU 继续点

- 数据集: `pubmed`
- 当前状态: `ok`。
- official latest: `commitment=838669.938`，`prove=972933.76`，`verify=3358514.479`，`proof_size=17747551`。
- artifact: `zkdl_gat_outputs/runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt`
- full-chain scope: `zkdl_gat_outputs/evidence/full_chain_scope_cpu_pubmed.json`；missing clauses `13`。
- full proof manifest: `zkdl_gat_outputs/runs/pubmed/cpu/warm/full_proof_manifest.json`；relation_proofs_generated `True`；proof_generated `True`；verified `True`。
- 五个全量数据集缺一不可；该文件只描述当前 dataset/device，不可替代其他行。
- full verifier manifest 已由 official exporter gate 校验；verifier 不读取 witness，proof object 不包含完整 tensor。
