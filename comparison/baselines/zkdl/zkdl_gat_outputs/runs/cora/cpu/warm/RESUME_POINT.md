# zkDL-GAT CPU 继续点

- 数据集: `cora`
- official latest: `blocked + NA`。
- 当前状态: `partial_formal`，不是 Level A。
- CPU 参与范围: CPU-only host Fr/G1/Commitment/opening + CPU relation prover/verifier。
- 已验证 relation: `SparseSourceRoutingRelationProof`, `SparseDestinationRoutingRelationProof`, `GroupedSumRelationProof`。
- proof artifact: `/home/pzh/project/comparison/baselines/zkdl/zkdl_gat_outputs/runs/cora/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt`
- evidence: `zkdl_gat_outputs/evidence/cpu_relation_partial_cora.json`
- timing: commitment `13.960` ms, prove `45.593` ms, verify `26257.734` ms, proof_size `13017` bytes。
- blocker: CPU denominator/inverse/alpha、lookup/range/max、output aggregation、all-layer/all-head 与 rescale/rounding 仍未闭合。
- log: `zkdl_gat_outputs/runs/cora/cpu/warm/cpu_backend_probe.log`
- 下一最小有效动作: 先做 CPU batched opening/verifier，再接 CPU denominator/inverse/alpha 与 full GAT runner。
