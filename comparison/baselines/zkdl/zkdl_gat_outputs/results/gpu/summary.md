# 最新正式结果

device: `gpu`；partial/debug 数字不进入 official latest。

| 数据集 | 状态 | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | 阻塞说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| cora | 阻塞 | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding proof 与完整 verifier；official 四项保持 NA。 |
| citeseer | 阻塞 | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding proof 与完整 verifier；official 四项保持 NA。 |
| pubmed | 阻塞 | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding proof 与完整 verifier；official 四项保持 NA。 |
| ppi | 阻塞 | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding proof 与完整 verifier；official 四项保持 NA。 |
| ogbn-arxiv | 阻塞 | NA | NA | NA | NA | gpu_full_gat_proof_incomplete：GPU 路径已有 zkDL-native Fr/G1 commitment/opening、sparse/grouped partial formal evidence；denominator/inverse/alpha 与 hidden layer0/head0 weighted aggregation 覆盖范围以 formal_partial_debug 和 coverage_matrix 为准，但缺完整 all-layer/all-head、output attention aggregation、ExpMap/ELU lookup-range、target-node max/shift、real rescale/rounding proof 与完整 verifier；official 四项保持 NA。 |

五个全量数据集的非 official 状态：

| dataset | formal/debug state | artifact |
| --- | --- | --- |
| cora | partial_formal | zkdl_gat_outputs/runs/cora/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| citeseer | partial_formal | zkdl_gat_outputs/runs/citeseer/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| pubmed | partial_formal | zkdl_gat_outputs/runs/pubmed/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| ppi | partial_formal | zkdl_gat_outputs/runs/ppi/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
| ogbn-arxiv | partial_formal | zkdl_gat_outputs/runs/ogbn-arxiv/gpu/warm/formal_artifacts/zkdl_softmax_aggregation_relations.proof.txt |
