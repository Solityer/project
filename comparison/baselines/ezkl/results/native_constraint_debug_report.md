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
| full_model | missing |  |  |  |  | False |  |  |  |  |
| full_model_current | blocked | 142970.888 | 15035.652 |  |  | False | final_logits_derivation_incomplete | native.full_model.final_logits.derivation | native.full_model.output.head_0.projection | 0 |

## Current Full-Model Blocker

- gate_name: `native.full_model.final_logits.derivation`
- region_name: `native.full_model.output.head_0.projection`
- row_offset: `0`
- relation: strict_public_input_equivalent now has a row-major public instance binding for final_logits[N,C], but strict verification also requires those final logits to be derived from complete quantized output-head projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, bias add, and K_out average relations over the real hidden activations
- actual_values: `['public_instance_binding=assigned_and_constrained_to_final_logits', 'public_instance_order=final_logits_row_major_node_then_class', 'public_instance_length=18956', 'final_logits_hash=fnv1a64:28269ee5bd574b96', 'scope=full_model', 'full_gat_forward_in_ezkl_proof=false', 'strict_public_input_equivalent=false', 'run_dir=.']`
- attempted_fix: implemented row-major final_logits[N,C] assignment, K_out average gate rows, output head projection/attention/bias_add diagnostic regions, and public instance equality constraints without enabling unsound strict flags
- remaining_blocker: replace diagnostic output-head rows with complete output-layer witness assignment over real hidden activations and checkpoint tensors before strict_public_input_equivalent can be true

## Full-Dataset Quantized Inverse Preflight Evidence

| Dataset | node | degree | policy | sum_fp32 | inv_fp32 | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Cora | 0 | 4 | quantized_inverse_lookup | 3.9303939786677424 | 0.25442742010788516 | 63 | 4 | 4 | 252 | True | True |
| Citeseer | 0 | 2 | quantized_inverse_lookup | 1.9950954385290314 | 0.5012291545998884 | 32 | 8 | 8 | 256 | True | True |
| Pubmed | 0 | 6 | quantized_inverse_lookup | 5.995988523252744 | 0.16677817112590357 | 96 | 3 | 3 | 288 | True | True |
| PPI | 0 | 1 | quantized_inverse_lookup | 1 | 1 | 16 | 16 | 16 | 256 | True | True |
| ogbn-arxiv | 0 | 290 | quantized_inverse_lookup | 63.34412927528665 | 0.015786782633858767 | 1014 | 0 | 0 | 0 | True | True |
