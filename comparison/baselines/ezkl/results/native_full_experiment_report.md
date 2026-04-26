# Native Full Experiment Report

| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | block_reason |
| --- | --- | --- | --- | --- |
| Cora | blocked | False | False | final_logits_derivation_incomplete |
| Citeseer | blocked | False | False | final_logits_derivation_incomplete |
| Pubmed | blocked | False | False | final_logits_derivation_incomplete |
| PPI | blocked | False | False | native_keygen_resource_failure |
| ogbn-arxiv | blocked | False | False | native_keygen_resource_failure |

## Final Logits Binding

| Dataset | binding_exists | binding_status | shape | public_instance_length | order | no_precomputed_logits_bypass |
| --- | --- | --- | --- | ---: | --- | --- |
| Cora | True | public_instance_bound_to_assigned_final_logits | [2708, 7] | 18956 | final_logits_row_major_node_then_class | True |
| Citeseer | True | public_instance_bound_to_assigned_final_logits | [3327, 6] | 19962 | final_logits_row_major_node_then_class | True |
| Pubmed | True | public_instance_bound_to_assigned_final_logits | [19717, 3] | 59151 | final_logits_row_major_node_then_class | True |
| PPI | False |  | None | None | None | None |
| ogbn-arxiv | False |  | None | None | None | None |

No output-binding workaround, no precomputed logits, no subset, no shard, and no sample route was used.

## Quantized Inverse Preflight Evidence

| Dataset | step | gate | region | node | degree | policy | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok | relation |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Cora | prove_time_ms | native.full_model.final_logits.derivation | native.full_model.output.head_0.projection | 0 | 4 | quantized_inverse_lookup | 63 | 4 | 4 | 252 | True | True | strict_public_input_equivalent now has a row-major public instance binding for final_logits[N,C], but strict verification also requires those final logits to be derived from complete quantized output-head projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, bias add, and K_out average relations over the real hidden activations |
| Citeseer | prove_time_ms | native.full_model.final_logits.derivation | native.full_model.output.head_0.projection | 0 | 2 | quantized_inverse_lookup | 32 | 8 | 8 | 256 | True | True | strict_public_input_equivalent now has a row-major public instance binding for final_logits[N,C], but strict verification also requires those final logits to be derived from complete quantized output-head projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, bias add, and K_out average relations over the real hidden activations |
| Pubmed | prove_time_ms | native.full_model.final_logits.derivation | native.full_model.output.head_0.projection | 0 | 6 | quantized_inverse_lookup | 96 | 3 | 3 | 288 | True | True | strict_public_input_equivalent now has a row-major public instance binding for final_logits[N,C], but strict verification also requires those final logits to be derived from complete quantized output-head projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, bias add, and K_out average relations over the real hidden activations |
| PPI | setup_ms | native.full_model.keygen | native.full_model.setup.create_keys | 0 | 1 | quantized_inverse_lookup | 16 | 16 | 16 | 256 | True | True | graph-gat-setup must complete KZG key generation and write pk.key/vk.key after quantized inverse and alpha preflight succeeds |
| ogbn-arxiv | setup_ms | native.full_model.keygen | native.full_model.setup.create_keys | 0 | 290 | quantized_inverse_lookup | 1014 | 0 | 0 | 0 | True | True | graph-gat-setup must complete KZG key generation and write pk.key/vk.key after quantized inverse and alpha preflight succeeds |
