# Native Full Experiment Report

| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | block_reason |
| --- | --- | --- | --- | --- |
| Cora | blocked | False | False | native_full_gat_forward_constraints_incomplete |
| Citeseer | blocked | False | False | native_full_gat_forward_constraints_incomplete |
| Pubmed | blocked | False | False | native_full_gat_forward_constraints_incomplete |
| PPI | blocked | False | False | native_keygen_resource_failure |
| ogbn-arxiv | blocked | False | False | native_keygen_resource_failure |

No output-binding workaround, no precomputed logits, no subset, no shard, and no sample route was used.

## Quantized Inverse Preflight Evidence

| Dataset | step | gate | region | node | degree | policy | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok | relation |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Cora | prove_time_ms | native.public_output_binding | native.full_model.layer_1.output.public_output_binding | 0 | 4 | quantized_inverse_lookup | 63 | 4 | 4 | 252 | True | True | strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, ELU, output average |
| Citeseer | prove_time_ms | native.public_output_binding | native.full_model.layer_1.output.public_output_binding | 0 | 2 | quantized_inverse_lookup | 32 | 8 | 8 | 256 | True | True | strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, ELU, output average |
| Pubmed | prove_time_ms | native.public_output_binding | native.full_model.layer_1.output.public_output_binding | 0 | 6 | quantized_inverse_lookup | 96 | 3 | 3 | 288 | True | True | strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, ELU, output average |
| PPI | setup_ms | native.full_model.keygen | native.full_model.setup.create_keys | 0 | 1 | quantized_inverse_lookup | 16 | 16 | 16 | 256 | True | True | graph-gat-setup must complete KZG key generation and write pk.key/vk.key after quantized inverse and alpha preflight succeeds |
| ogbn-arxiv | setup_ms | native.full_model.keygen | native.full_model.setup.create_keys | 0 | 290 | quantized_inverse_lookup | 1014 | 0 | 0 | 0 | True | True | graph-gat-setup must complete KZG key generation and write pk.key/vk.key after quantized inverse and alpha preflight succeeds |
