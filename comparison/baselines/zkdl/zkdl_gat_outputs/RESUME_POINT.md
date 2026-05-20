# zkDL-GAT 当前继续点

- 当前状态: `not_Level_A`。
- Gate 1/2/3 最新执行结果: `PASS`；日志在 `zkdl_gat_outputs/runs/selftest/*_latest.log`。
- CPU official path: `blocked + NA`；CPU-only relation backend 已实现并在 Cora/Citeseer sparse/grouped full dataset 通过 proof/verify，但未达到完整 GAT proof。
- CPU Pubmed: full-dataset proof/verify attempt 因 host G1 opening replay timeout 未完成；PPI/ogbn CPU 等待 batched opening/verifier 后运行。
- GPU official path: `blocked + NA`，原因是当前只有 relation-level partial formal evidence，不是完整 GAT proof/verify。
- GPU partial evidence: 五个全量数据集 hidden layer0/head0 sparse/grouped、denominator/inverse/alpha 与 8/8 feature weighted aggregation。
- 结果路径: `zkdl_gat_outputs/results/cpu/` 与 `zkdl_gat_outputs/results/gpu/`；顶层 `results/latest.*` 只是聚合索引。
- 下一最小有效动作: 实现 zkDL lookup/range/max primitive 与 batched opening/verifier；随后补 output aggregation、rescale/rounding 公开契约和所有层/头 proof object。
