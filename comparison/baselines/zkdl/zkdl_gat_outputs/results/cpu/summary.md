# zkDL-GAT CPU latest official

当前没有 Level A official 行；partial/debug 数字不进入 official latest。

| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cora | blocked | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| citeseer | blocked | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| pubmed | blocked | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| ppi | blocked | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |
| ogbn-arxiv | blocked | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison 接入和 real rescale/rounding；official 四项保持 NA。 |

五个全量数据集的非 official 状态：

| dataset | formal/debug state | artifact |
| --- | --- | --- |
| cora | partial_formal | zkdl_gat_outputs/runs/cora/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt |
| citeseer | partial_formal | zkdl_gat_outputs/runs/citeseer/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt |
| pubmed | failed_with_evidence | zkdl_gat_outputs/runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_formal_chain.timeout.txt |
| ppi | failed_with_evidence | zkdl_gat_outputs/runs/ppi/cpu/warm/formal_artifacts/zkdl_cpu_formal_chain.timeout.txt |
| ogbn-arxiv | failed_with_evidence | zkdl_gat_outputs/runs/ogbn-arxiv/cpu/warm/formal_artifacts/zkdl_cpu_formal_chain.timeout.txt |
