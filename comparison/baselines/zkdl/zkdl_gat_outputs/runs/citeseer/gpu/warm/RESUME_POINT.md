# zkDL-GAT GPU 继续点

- 数据集: `citeseer`
- 当前状态: `partial_formal`，不是 Level A。
- GPU 参与范围: CUDA forward trace + zkDL 原生 CUDA-backed Fr/G1 commitment/opening；verifier 主要为 host transcript/opening 重放。
- official latest: `blocked + NA`。

## sparse/grouped relation

- relation: `SparseSourceRoutingRelationProof`, `SparseDestinationRoutingRelationProof`, `GroupedSumRelationProof`
- 证明范围: hidden layer0/head0 full graph / full edge set，edges `12431`
- 结果: `PASS/PASS/PASS`
- proof artifact: `/home/pzh/project/comparison/baselines/zkdl/zkdl_gat_outputs/runs/citeseer/gpu/warm/formal_artifacts/zkdl_sparse_grouped_relations.proof.txt`
- evidence: `zkdl_gat_outputs/evidence/sparse_grouped_relation_citeseer.json`

## softmax/aggregation partial relation

- relation: `denominator grouped sum`, `inverse Hadamard`, `alpha multiplication`, `multi-column weighted aggregation`
- 证明范围: hidden layer0/head0 full graph / full edge set，feature columns `8/8`
- 结果: denom `PASS`, inverse `PASS`, alpha `PASS`, weighted `PASS`
- proof artifact: `/home/pzh/project/comparison/baselines/zkdl/zkdl_gat_outputs/runs/citeseer/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt`
- evidence: `zkdl_gat_outputs/evidence/softmax_aggregation_relation_citeseer.json`

## Level A blocker

- softmax max/shift、ExpMap lookup/range、LeakyReLU/ELU lookup-range、output aggregation、real rescale chain 与 rounding sign/tie-breaking 仍未闭合。
- 下一最小有效动作: 在 zkDL primitive 内实现 lookup/range/max/rescale 后，把 hidden/output 所有层头接入同一个完整 GAT proof object。
