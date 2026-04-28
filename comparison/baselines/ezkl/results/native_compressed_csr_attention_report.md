# Native Compressed CSR Attention Report

- Route: `ezkl_gat_native_compressed_csr_attention_full`
- Native primitive: `GraphSegmentAttentionCompressedCSR` / `csr_segment_attention_compressed_v1`
- Row model: CSR edge rows, row-compressed projection and aggregation, segment max selector/state machine, quantized inverse lookup, alpha rescale, and batched formal lookup table accounting.
- Formal tables: deterministic from `range_bits`, `quant_cfg_id`, and `static_table_id`; no witness-specific table is generated.

| Dataset | status | block_reason | previous rows | compressed rows | reduction | required_logrows | feasible | table_hash |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Cora | verified |  | 252496464 | 2079934 | 121.39638276983789 | 21 | True | fnv1a64:5e858053087e95d4 |
| Citeseer | verified |  | 792557483 | 2178539 | 363.8022927292098 | 22 | True | fnv1a64:3a73fd481a173b1f |
| Pubmed | verified |  | 657323900 | 16025572 | 41.01718802923228 | 24 | True | fnv1a64:bb4ce3403958d2d3 |
| PPI | verified |  | 315476272 | 29591968 | 10.660875005001357 | 25 | True | fnv1a64:d1359d7cd1406d38 |
| ogbn-arxiv | verified |  | 381668318 | 45611736 | 8.367765655751406 | 26 | True | fnv1a64:d53cd96062d77fac |

## Required Artifacts

- Cora: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- Citeseer: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- Pubmed: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- PPI: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
- ogbn-arxiv: run_manifest.json=true, compressed_row_estimate.json=true, formal_tables.json=true, formal_table_hashes.json=true, graph_attention_spec.json=true, model_info.json=true
