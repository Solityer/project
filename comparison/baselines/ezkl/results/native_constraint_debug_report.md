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
| full_model_current | blocked | 2.295 |  |  |  | False | native_full_exact_inverse_quantization_conflict | native.exact_inverse | native.full_model.layer_0.head_0.softmax.exact_inverse | 0 |

## Current Full-Model Blocker

- gate_name: `native.exact_inverse`
- region_name: `native.full_model.layer_0.head_0.softmax.exact_inverse`
- row_offset: `0`
- relation: strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness
- actual_values: `['sum_q=63', 'inv_q=4', 'sum_q_times_inv_q=252', 'quant_cfg_id=fp32_bundle_export']`
- attempted_fix: C++ adapter now executes the real full-dataset first hidden attention head and checks the formal exact inverse relation before Rust witness assignment
- remaining_blocker: A compliant strict route needs a quant_cfg_id that defines exact inverse-compatible downstream alpha/aggregation/ELU semantics; replacing this with rounded fp32 reciprocal or output binding would violate the formal route

## Full-Dataset Preflight Evidence

| Dataset | node | degree | sum_fp32 | inv_fp32 | sum_q | inv_q | product |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 0 | 4 | 3.9303939786677424 | 0.25442742010788516 | 63 | 4 | 252 |
| Citeseer | 0 | 2 | 1.9950954385290314 | 0.5012291545998884 | 32 | 8 | 256 |
| Pubmed | 0 | 6 | 5.995988523252744 | 0.16677817112590357 | 96 | 3 | 288 |
| PPI | 0 | 1 | 1 | 1 | 16 | 16 | 256 |
| ogbn-arxiv | 0 | 290 | 63.34412927528665 | 0.015786782633858767 | 1014 | 0 | 0 |
