# Native Full Experiment Report

| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | block_reason |
| --- | --- | --- | --- | --- | --- |
| Cora | verified | True | True | True |  |
| Citeseer | verified | True | True | True |  |
| Pubmed | verified | True | True | True |  |
| PPI | verified | True | True | True |  |
| ogbn-arxiv | verified | True | True | True |  |

## Final Logits Binding

| Dataset | binding_exists | binding_status | shape | public_instance_length | layout | columns | chunk_size | order | no_precomputed_logits_bypass |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Cora | True | public_instance_bound_to_derived_final_logits | [2708, 7] | 18956 | chunked_columns | 1 | 18956 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | True |
| Citeseer | True | public_instance_bound_to_derived_final_logits | [3327, 6] | 19962 | chunked_columns | 1 | 19962 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | True |
| Pubmed | True | public_instance_bound_to_derived_final_logits | [19717, 3] | 59151 | chunked_columns | 1 | 59151 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | True |
| PPI | True | public_instance_bound_to_derived_final_logits | [56944, 121] | 6890224 | chunked_columns | 2 | 4194304 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | True |
| ogbn-arxiv | True | public_instance_bound_to_derived_final_logits | [169343, 40] | 6773720 | chunked_columns | 2 | 4194304 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | True |

No output-binding workaround, no precomputed logits, no subset, no shard, and no sample route was used.

## Quantized Inverse Preflight Evidence

| Dataset | step | gate | region | node | degree | policy | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok | relation |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Cora |  |  |  | 0 | 4 | quantized_inverse_lookup | 63 | 4 | 4 | 252 | True | True |  |
| Citeseer |  |  |  | 0 | 2 | quantized_inverse_lookup | 32 | 8 | 8 | 256 | True | True |  |
| Pubmed |  |  |  | 0 | 6 | quantized_inverse_lookup | 96 | 3 | 3 | 288 | True | True |  |
| PPI |  | None | None | 0 | 1 | quantized_inverse_lookup | 16 | 16 | 16 | 256 | True | True | None |
| ogbn-arxiv |  | None | None | 0 | 290 | quantized_inverse_lookup | 1014 | 0 | 0 | 0 | True | True | None |

## Native Constraint Failure Evidence

| Dataset | gate | region | row | values | remaining_blocker |
| --- | --- | --- | ---: | --- | --- |
