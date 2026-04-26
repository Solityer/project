# Native Compressed CSR Attention Report

- Route: `ezkl_gat_native_compressed_csr_attention_full`
- Native primitive: `GraphSegmentAttentionCompressedCSR` / `csr_segment_attention_compressed_v1`
- Row model: CSR edge rows, row-compressed projection and aggregation, segment max selector/state machine, exact `Sum_i * Inv_i = 1`, and batched formal lookup table accounting.
- Formal tables: deterministic from `range_bits`, `quant_cfg_id`, and `static_table_id`; no witness-specific table is generated.

| Dataset | status | block_reason | previous rows | compressed rows | reduction | required_logrows | feasible | table_hash |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Cora | blocked | ezkl_compressed_csr_backend_unavailable | 252496464 | 2117846 | 119.22324097219533 | 22 | True | fnv1a64:f2933ff0aa727bba |
| Citeseer | blocked | ezkl_compressed_csr_backend_unavailable | 792557483 | 2218463 | 357.25521813976616 | 22 | True | fnv1a64:f2933ff0aa727bba |
| Pubmed | blocked | ezkl_compressed_csr_backend_unavailable | 657323900 | 16143874 | 40.71661485960557 | 24 | True | fnv1a64:f2933ff0aa727bba |
| PPI | blocked | ezkl_compressed_csr_backend_unavailable | 315476272 | 43372416 | 7.273661490289127 | 26 | True | fnv1a64:f2933ff0aa727bba |
| ogbn-arxiv | blocked | ezkl_compressed_csr_backend_unavailable | 381668318 | 59159176 | 6.451548919477851 | 26 | True | fnv1a64:dca099df18983198 |

## Required Artifacts

- Cora: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- Citeseer: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- Pubmed: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- PPI: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- ogbn-arxiv: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
