# Native Full Experiment Report

| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete | manifest_strict_check |
| --- | --- | --- | --- | --- | --- |
| Cora | verified | true | true | true | passed |
| Citeseer | verified | true | true | true | passed |
| Pubmed | verified | true | true | true | passed |
| PPI | verified | true | true | true | passed |
| ogbn-arxiv | verified | true | true | true | passed |

## Final Logits Binding

| Dataset | binding_exists | binding_status | shape | public_instance_length | layout | columns | chunk_size | order | no_precomputed_logits_bypass |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | --- | --- |
| Cora | true | public_instance_bound_to_derived_final_logits | [2708, 7] | 18956 | chunked_columns | 1 | 18956 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | true |
| Citeseer | true | public_instance_bound_to_derived_final_logits | [3327, 6] | 19962 | chunked_columns | 1 | 19962 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | true |
| Pubmed | true | public_instance_bound_to_derived_final_logits | [19717, 3] | 59151 | chunked_columns | 1 | 59151 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | true |
| PPI | true | public_instance_bound_to_derived_final_logits | [56944, 121] | 6890224 | chunked_columns | 2 | 4194304 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | true |
| ogbn-arxiv | true | public_instance_bound_to_derived_final_logits | [169343, 40] | 6773720 | chunked_columns | 2 | 4194304 | final_logits_row_major_node_then_class_chunked_columns_column_then_row | true |

No GPU baseline, output-binding workaround, precomputed logits, subset, shard, or sample route was used.
