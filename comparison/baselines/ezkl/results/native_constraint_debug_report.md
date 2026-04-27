# Native Constraint Debug Report

- Route: `ezkl_gat_native_compressed_csr_attention_full`
- Diagnostic policy: every native debug scope runs MockProver before KZG prove; failures write `native_constraint_failure.json` and `.log`.
- No output-binding workaround, precomputed logits, subset, shard, sample, digest, folding, or identity route is used.

## Cora Scope Ladder

| Scope | status | setup_ms | prove_ms | verify_ms | proof_size | verify_passed | block_reason | failure gate | failure region | row |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| level0_mul | missing |  |  |  |  | False |  |  |  |  |
| inverse_only | missing |  |  |  |  | False |  |  |  |  |
| segment_sum_inverse | missing |  |  |  |  | False |  |  |  |  |
| segment_max_only | missing |  |  |  |  | False |  |  |  |  |
| lrelu_lookup_only | missing |  |  |  |  | False |  |  |  |  |
| exp_lookup_only | missing |  |  |  |  | False |  |  |  |  |
| elu_lookup_only | missing |  |  |  |  | False |  |  |  |  |
| compressed_projection_only | missing |  |  |  |  | False |  |  |  |  |
| compressed_aggregation_only | missing |  |  |  |  | False |  |  |  |  |
| single_head_first_layer | missing |  |  |  |  | False |  |  |  |  |
| first_hidden_layer_all_heads | missing |  |  |  |  | False |  |  |  |  |
| hidden_only | missing |  |  |  |  | False |  |  |  |  |
| output_only | missing |  |  |  |  | False |  |  |  |  |
| output_projection_only | missing |  |  |  |  | False |  |  |  |  |
| output_attention_only | missing |  |  |  |  | False |  |  |  |  |
| output_aggregation_only | missing |  |  |  |  | False |  |  |  |  |
| output_bias_add_only | missing |  |  |  |  | False |  |  |  |  |
| output_average_only | missing |  |  |  |  | False |  |  |  |  |
| final_logits_public_binding_only | missing |  |  |  |  | False |  |  |  |  |
| output_full_derivation | missing |  |  |  |  | False |  |  |  |  |
| full_model | missing |  |  |  |  | False |  |  |  |  |
| full_model_current | verified | 34249.261 | 20726.156 | 566.251 | 1280198 | True |  |  |  |  |

## Full-Dataset Quantized Inverse Preflight Evidence

| Dataset | node | degree | policy | sum_fp32 | inv_fp32 | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Cora | 0 | 4 | quantized_inverse_lookup | 3.9303939786677424 | 0.25442742010788516 | 63 | 4 | 4 | 252 | True | True |
| Citeseer | 0 | 2 | quantized_inverse_lookup | 1.9950954385290314 | 0.5012291545998884 | 32 | 8 | 8 | 256 | True | True |
| Pubmed | 0 | 6 | quantized_inverse_lookup | 5.995988523252744 | 0.16677817112590357 | 96 | 3 | 3 | 288 | True | True |
| PPI | 0 | 1 | quantized_inverse_lookup | 1 | 1 | 16 | 16 | 16 | 256 | True | True |
| ogbn-arxiv | 0 | 290 | quantized_inverse_lookup | 63.34412927528665 | 0.015786782633858767 | 1014 | 0 | 0 | 0 | True | True |
