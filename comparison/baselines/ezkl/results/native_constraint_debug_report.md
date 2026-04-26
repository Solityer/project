# Native Constraint Debug Report

- Route: `ezkl_gat_native_compressed_csr_attention_full`
- Diagnostic policy: every native debug scope runs MockProver before KZG prove; failures write `native_constraint_failure.json` and `.log`.
- No output-binding workaround, precomputed logits, subset, shard, sample, digest, folding, or identity route is used.

## Cora Scope Ladder

| Scope | status | setup_ms | prove_ms | verify_ms | proof_size | verify_passed | block_reason | failure gate | failure region | row |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | ---: |
| level0_mul | verified | 51.910 | 42.657 | 5.564 | 8545 | True |  |  |  |  |
| inverse_only | verified | 48.840 | 41.319 | 5.414 | 8580 | True |  |  |  |  |
| segment_sum_inverse | verified | 325.261 | 207.890 | 9.083 | 8651 | True |  |  |  |  |
| segment_max_only | verified | 323.916 | 184.307 | 11.614 | 8625 | True |  |  |  |  |
| lrelu_lookup_only | verified | 43.408 | 36.419 | 6.080 | 8594 | True |  |  |  |  |
| exp_lookup_only | verified | 50.070 | 44.690 | 6.525 | 8572 | True |  |  |  |  |
| elu_lookup_only | verified | 42.576 | 36.844 | 6.781 | 8579 | True |  |  |  |  |
| compressed_projection_only | verified | 47.965 | 36.367 | 5.588 | 8581 | True |  |  |  |  |
| compressed_aggregation_only | verified | 48.926 | 37.589 | 5.074 | 8554 | True |  |  |  |  |
| single_head_first_layer | verified | 1057.009 | 632.487 | 19.029 | 8756 | True |  |  |  |  |
| first_hidden_layer_all_heads | verified | 7743.635 | 4181.520 | 124.170 | 8774 | True |  |  |  |  |
| hidden_only | verified | 7945.855 | 4416.195 | 114.662 | 8782 | True |  |  |  |  |
| output_only | verified | 1068.895 | 589.447 | 17.681 | 8478 | True |  |  |  |  |
| full_model | blocked | 123566.101 | 14246.984 |  |  | False | native_full_gat_forward_constraints_incomplete | native.public_output_binding | native.full_model.layer_1.output.public_output_binding | 0 |

## Current Full-Model Blocker

- gate_name: `native.public_output_binding`
- region_name: `native.full_model.layer_1.output.public_output_binding`
- row_offset: `0`
- relation: strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, exact inverse, aggregation, ELU, output average
- actual_values: `['diagnostic proof binds CSR/table/metadata scoped relations only', 'scope=full_model', 'full_gat_forward_in_ezkl_proof=false', 'strict_public_input_equivalent=false', 'run_dir=.']`
- attempted_fix: replaced the artificial full compressed CSR failure with real Halo2 MockProver/KZG diagnostic scopes and preserved strict_public_input_equivalent=false for non-full semantics
- remaining_blocker: full quantized GAT row assignment for public logits remains unimplemented; marking strict=true would be unsound
