# zkDL-GAT GPU 继续点

- 数据集: `pubmed`
- 当前状态: `ok`。
- official latest: `commitment=632869.764`，`prove=388140.447`，`verify=3364287.373`，`proof_size=17538459`。
- artifact: `zkdl_gat_outputs/runs/pubmed/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt`
- full-chain scope: `zkdl_gat_outputs/evidence/full_chain_scope_gpu_pubmed.json`；missing clauses `13`。
- full proof manifest: `zkdl_gat_outputs/runs/pubmed/gpu/warm/full_proof_manifest.json`；relation_proofs_generated `True`；proof_generated `True`；verified `True`。
- 五个全量数据集缺一不可；该文件只描述当前 dataset/device，不可替代其他行。
- full verifier manifest 已由 official exporter gate 校验；verifier 不读取 witness，proof object 不包含完整 tensor。
