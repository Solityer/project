# zkDL-GAT CPU 继续点

- 数据集: `ppi`
- 当前状态: `failed_with_evidence`；official 仍为 `blocked + NA`。
- artifact: `zkdl_gat_outputs/runs/ppi/cpu/warm/formal_artifacts/zkdl_cpu_formal_chain.timeout.txt`
- 五个全量数据集缺一不可；该文件只描述当前 dataset/device，不可替代其他行。
- Level A blocker: full GAT proof object、all-layer/all-head、output aggregation、real rescale/rounding 仍未闭合。
