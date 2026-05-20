# zkDL-GAT GPU latest official

当前没有 Level A official 行；partial/debug 数字不进入 official latest。

| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | blocker |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cora | blocked | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped、denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation partial formal evidence，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| citeseer | blocked | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped、denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation partial formal evidence，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| pubmed | blocked | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped、denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation partial formal evidence，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| ppi | blocked | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped、denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation partial formal evidence，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |
| ogbn-arxiv | blocked | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped、denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation partial formal evidence，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding public contract 与完整 verifier；official 四项保持 NA。 |

五个全量数据集的非 official 状态：

| dataset | formal/debug state | artifact |
| --- | --- | --- |
| cora | partial_formal | zkdl_gat_outputs/runs/cora/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| citeseer | partial_formal | zkdl_gat_outputs/runs/citeseer/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| pubmed | partial_formal | zkdl_gat_outputs/runs/pubmed/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| ppi | partial_formal | zkdl_gat_outputs/runs/ppi/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| ogbn-arxiv | partial_formal | zkdl_gat_outputs/runs/ogbn-arxiv/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
