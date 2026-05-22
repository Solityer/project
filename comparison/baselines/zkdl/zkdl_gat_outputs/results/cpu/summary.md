# 最新正式结果

device: `cpu`；partial/debug 数字不进入 official latest。

| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | 阻塞说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cora | 阻塞 | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison proof 接入和 real rescale/rounding proof；official 四项保持 NA。 |
| citeseer | 阻塞 | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison proof 接入和 real rescale/rounding proof；official 四项保持 NA。 |
| pubmed | 阻塞 | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison proof 接入和 real rescale/rounding proof；official 四项保持 NA。 |
| ppi | 阻塞 | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison proof 接入和 real rescale/rounding proof；official 四项保持 NA。 |
| ogbn-arxiv | 阻塞 | NA | NA | NA | NA | cpu_full_gat_proof_incomplete：CPU 路径已有 CPU-only host Fr/G1/Commitment/opening 与 hidden layer0/head0 relation prover/verifier 实现尝试；未完成 full GAT proof object、all-layer/all-head、output aggregation、lookup/range/max/comparison proof 接入和 real rescale/rounding proof；official 四项保持 NA。 |

五个全量数据集的非 official 状态：

| dataset | formal/debug state | artifact |
| --- | --- | --- |
| cora | partial_formal | zkdl_gat_outputs/runs/cora/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt |
| citeseer | partial_formal | zkdl_gat_outputs/runs/citeseer/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt |
| pubmed | partial_formal | zkdl_gat_outputs/runs/pubmed/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt |
| ppi | partial_formal | zkdl_gat_outputs/runs/ppi/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt |
| ogbn-arxiv | partial_formal | zkdl_gat_outputs/runs/ogbn-arxiv/cpu/warm/formal_artifacts/zkdl_cpu_relation_partial.proof.txt |
