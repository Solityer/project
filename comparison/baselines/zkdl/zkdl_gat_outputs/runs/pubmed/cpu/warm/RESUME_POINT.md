# zkDL-GAT CPU 继续点

- 数据集: `pubmed`
- official latest: `blocked + NA`。
- 当前状态: `blocked_after_attempt_timeout`。
- CPU 参与范围: CPU-only host Fr/G1/Commitment/opening + CPU relation prover/verifier 已进入 full-dataset attempt。
- timeout evidence: `zkdl_gat_outputs/runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_backend_attempt.timeout.txt`
- blocker: host G1 opening replay 未在交互窗口内完成；需要 zkDL-native batched opening/verifier 或更快 CPU MSM/opening replay。
- log: `zkdl_gat_outputs/runs/pubmed/cpu/warm/cpu_backend_probe.log`
- 下一最小有效动作: 先做 CPU batched opening/verifier，再接 CPU denominator/inverse/alpha 与 full GAT runner。
