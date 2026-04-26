# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Verified ezkl full-dataset rows: 0/5.
- Strict public input equivalent is true only for rows with a verified full native GAT proof.
- No output-binding workaround, precomputed logits, identity, digest, folding, subset, shard, or sample route was used.

## Dataset Status

| Dataset | status | route_id | strict | full GAT forward in proof | block_reason |
| --- | --- | --- | --- | --- | --- |
| Cora | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_exact_inverse_quantization_conflict |
| Citeseer | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_exact_inverse_quantization_conflict |
| Pubmed | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_exact_inverse_quantization_conflict |
| PPI | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_exact_inverse_quantization_conflict |
| ogbn-arxiv | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_exact_inverse_quantization_conflict |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 2.295 | 1207.818 | blocked | 153.379 | blocked | 90755 | blocked |
| Citeseer | 453.773 | 1.759 | 1418.744 | blocked | 153.837 | blocked | 90763 | blocked |
| Pubmed | 2962.933 | 1.504 | 5604.074 | blocked | 175.454 | blocked | 90760 | blocked |
| PPI | 3563.973 | 1.739 | 7111.447 | blocked | 424.710 | blocked | 21360 | blocked |
| ogbn-arxiv | 18071.562 | 1.322 | 66540.600 | blocked | 417.380 | blocked | 21361 | blocked |

## Blocked/Failed Reasons

- Cora: blocked / native_full_exact_inverse_quantization_conflict; step=setup_ms; gate=native.exact_inverse; region=native.full_model.layer_0.head_0.softmax.exact_inverse; row=0; relation=strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness; values=sum_q=63, inv_q=4, sum_q_times_inv_q=252, quant_cfg_id=fp32_bundle_export; node=0; degree=4; sum_fp32=3.9303939786677424; inv_fp32=0.25442742010788516; sum_q=63; inv_q=4; product=252
- Citeseer: blocked / native_full_exact_inverse_quantization_conflict; step=setup_ms; gate=native.exact_inverse; region=native.full_model.layer_0.head_0.softmax.exact_inverse; row=0; relation=strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness; values=sum_q=32, inv_q=8, sum_q_times_inv_q=256, quant_cfg_id=fp32_bundle_export; node=0; degree=2; sum_fp32=1.9950954385290314; inv_fp32=0.5012291545998884; sum_q=32; inv_q=8; product=256
- Pubmed: blocked / native_full_exact_inverse_quantization_conflict; step=setup_ms; gate=native.exact_inverse; region=native.full_model.layer_0.head_0.softmax.exact_inverse; row=0; relation=strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness; values=sum_q=96, inv_q=3, sum_q_times_inv_q=288, quant_cfg_id=fp32_bundle_export; node=0; degree=6; sum_fp32=5.995988523252744; inv_fp32=0.16677817112590357; sum_q=96; inv_q=3; product=288
- PPI: blocked / native_full_exact_inverse_quantization_conflict; step=setup_ms; gate=native.exact_inverse; region=native.full_model.layer_0.head_0.softmax.exact_inverse; row=0; relation=strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness; values=sum_q=16, inv_q=16, sum_q_times_inv_q=256, quant_cfg_id=fp32_bundle_export; node=0; degree=1; sum_fp32=1; inv_fp32=1; sum_q=16; inv_q=16; product=256
- ogbn-arxiv: blocked / native_full_exact_inverse_quantization_conflict; step=setup_ms; gate=native.exact_inverse; region=native.full_model.layer_0.head_0.softmax.exact_inverse; row=0; relation=strict native route requires exact finite-field Sum_i * Inv_i = 1, but fp32_bundle_export supplies a scale-16 rounded reciprocal witness; values=sum_q=1014, inv_q=0, sum_q_times_inv_q=0, quant_cfg_id=fp32_bundle_export; node=0; degree=290; sum_fp32=63.34412927528665; inv_fp32=0.015786782633858767; sum_q=1014; inv_q=0; product=0
