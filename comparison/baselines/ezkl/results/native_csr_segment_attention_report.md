# Native CSR Segment Attention Report

- Route: `ezkl_gat_native_csr_segment_attention_full`
- Native primitive: `GraphSegmentAttention` / `csr_segment_attention_v1`
- Layout: CSR/ragged dst segments over `src[E]`, `dst[E]`, `segment_ptr[N+1]`, `edge_ptr[G+1]`, and `node_ptr[G+1]`.
- Exact inverse attempt: native witness inverse column with `Sum_i * Inv_i = 1`; no epsilon reciprocal is used.

| Dataset | status | block_reason | failure_step | rows | required_logrows | max_logrows |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Cora | blocked | ezkl_native_csr_compile_resource_boundary | compile_ms | 252496464 | 28 | 26 |
| Citeseer | blocked | ezkl_native_csr_compile_resource_boundary | compile_ms | 792557483 | 30 | 26 |
| Pubmed | blocked | ezkl_native_csr_compile_resource_boundary | compile_ms | 657323900 | 30 | 26 |
| PPI | blocked | ezkl_native_csr_compile_resource_boundary | compile_ms | 315476272 | 29 | 26 |
| ogbn-arxiv | blocked | ezkl_native_csr_compile_resource_boundary | compile_ms | 381668318 | 29 | 26 |

## Artifact Contract

- Cora: spec=True, csr=True, tensor_payloads=True
- Citeseer: spec=True, csr=True, tensor_payloads=True
- Pubmed: spec=True, csr=True, tensor_payloads=True
- PPI: spec=True, csr=True, tensor_payloads=True
- ogbn-arxiv: spec=True, csr=True, tensor_payloads=True

## Forced max-logrows=30 Compile Attempt

| Dataset | status | block_reason | rows | required_logrows | max_logrows | lookup_status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Cora | blocked | ezkl_native_lookup_tables_missing_from_quant_cfg | 252496464 | 28 | 30 | blocked_without_quant_cfg_bound_finite_lookup_tables |
| Citeseer | blocked | ezkl_native_lookup_tables_missing_from_quant_cfg | 792557483 | 30 | 30 | blocked_without_quant_cfg_bound_finite_lookup_tables |
| Pubmed | blocked | ezkl_native_lookup_tables_missing_from_quant_cfg | 657323900 | 30 | 30 | blocked_without_quant_cfg_bound_finite_lookup_tables |
| PPI | blocked | ezkl_native_lookup_tables_missing_from_quant_cfg | 315476272 | 29 | 30 | blocked_without_quant_cfg_bound_finite_lookup_tables |
| ogbn-arxiv | blocked | ezkl_native_lookup_tables_missing_from_quant_cfg | 381668318 | 29 | 30 | blocked_without_quant_cfg_bound_finite_lookup_tables |
