# Native Full Experiment Report

| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | block_reason |
| --- | --- | --- | --- | --- |
| Cora | blocked | False | False | native_full_exact_inverse_quantization_conflict |
| Citeseer | blocked | False | False | native_full_exact_inverse_quantization_conflict |
| Pubmed | blocked | False | False | native_full_exact_inverse_quantization_conflict |
| PPI | blocked | False | False | native_full_exact_inverse_quantization_conflict |
| ogbn-arxiv | blocked | False | False | native_full_exact_inverse_quantization_conflict |

No output-binding workaround, no precomputed logits, no subset, no shard, and no sample route was used.

## Exact-Inverse Preflight Evidence

| Dataset | step | gate | region | node | degree | sum_q | inv_q | product | relation |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cora | setup_ms | native.exact_inverse | native.full_model.layer_0.head_0.softmax.exact_inverse | 0 | 4 | 63 | 4 | 252 | strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness |
| Citeseer | setup_ms | native.exact_inverse | native.full_model.layer_0.head_0.softmax.exact_inverse | 0 | 2 | 32 | 8 | 256 | strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness |
| Pubmed | setup_ms | native.exact_inverse | native.full_model.layer_0.head_0.softmax.exact_inverse | 0 | 6 | 96 | 3 | 288 | strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness |
| PPI | setup_ms | native.exact_inverse | native.full_model.layer_0.head_0.softmax.exact_inverse | 0 | 1 | 16 | 16 | 256 | strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness |
| ogbn-arxiv | setup_ms | native.exact_inverse | native.full_model.layer_0.head_0.softmax.exact_inverse | 0 | 290 | 1014 | 0 | 0 | strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness |
