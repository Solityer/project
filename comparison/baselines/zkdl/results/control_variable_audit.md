# Control Variable Audit

Both CPU and GPU routes use the same five full dataset configs under `comparison/baselines/zkdl/configs` and read the same main-repository data/checkpoint sources in read-only mode.

| dataset | cpu_status | gpu_status | cpu_public_equivalent | gpu_public_equivalent | cpu_route | gpu_route |
| --- | --- | --- | --- | --- | --- | --- |
| cora_full | blocked | blocked | True | True | zkdl_gat_full_forward_cpu | zkdl_gat_full_forward_gpu |
| citeseer_full | blocked | blocked | True | True | zkdl_gat_full_forward_cpu | zkdl_gat_full_forward_gpu |
| pubmed_full | blocked | blocked | True | True | zkdl_gat_full_forward_cpu | zkdl_gat_full_forward_gpu |
| ppi_full_formal | blocked | blocked | True | True | zkdl_gat_full_forward_cpu | zkdl_gat_full_forward_gpu |
| ogbn_arxiv_full | blocked | blocked | True | True | zkdl_gat_full_forward_cpu | zkdl_gat_full_forward_gpu |

The GPU primitive probe is recorded only as backend availability evidence. It is not copied into full GAT proof metrics and cannot make a full GAT row verified.