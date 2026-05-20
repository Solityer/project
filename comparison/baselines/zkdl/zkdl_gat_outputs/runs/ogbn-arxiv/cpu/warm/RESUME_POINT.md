# zkDL-GAT CPU 继续点

- 数据集: `ogbn-arxiv`
- official latest: `blocked + NA`。
- 当前状态: `not_run_after_pubmed_timeout`。
- evidence: `zkdl_gat_outputs/runs/ogbn-arxiv/cpu/warm/formal_artifacts/zkdl_cpu_backend_probe.blocker.txt`
- blocker: Pubmed 已暴露 CPU host opening replay timeout；较大数据集等待 batched opening/verifier 后运行。
- log: `zkdl_gat_outputs/runs/ogbn-arxiv/cpu/warm/cpu_backend_probe.log`
- 下一最小有效动作: 先做 CPU batched opening/verifier，再接 CPU denominator/inverse/alpha 与 full GAT runner。
