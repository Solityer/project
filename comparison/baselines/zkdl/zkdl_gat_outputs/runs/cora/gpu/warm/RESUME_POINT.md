# zkDL-GAT GPU 继续点

- 数据集: `cora`
- 当前状态: `ok`。
- official latest: `commitment=201935.494`，`prove=315047.541`，`verify=749597.5`，`proof_size=6959845`。
- artifact: `zkdl_gat_outputs/runs/cora/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt`
- full-chain scope: `zkdl_gat_outputs/evidence/full_chain_scope_gpu_cora.json`；missing clauses `13`。
- full proof manifest: `zkdl_gat_outputs/runs/cora/gpu/warm/full_proof_manifest.json`；relation_proofs_generated `True`；proof_generated `True`；verified `True`。
- 五个全量数据集缺一不可；该文件只描述当前 dataset/device，不可替代其他行。
- full verifier manifest 已由 official exporter gate 校验；verifier 不读取 witness，proof object 不包含完整 tensor。
