# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Verified ezkl full-dataset rows: 0/5.
- Strict public input equivalent is true only for rows with a verified full native GAT proof.
- No output-binding workaround, precomputed logits, identity, digest, folding, subset, shard, or sample route was used.

## Dataset Status

| Dataset | status | route_id | strict | full GAT forward in proof | block_reason |
| --- | --- | --- | --- | --- | --- |
| Cora | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_gat_forward_constraints_incomplete |
| Citeseer | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | not_run_because_cora_full_blocked_at_native_public_output_binding |
| Pubmed | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | not_run_because_cora_full_blocked_at_native_public_output_binding |
| PPI | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | not_run_because_cora_full_blocked_at_native_public_output_binding |
| ogbn-arxiv | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | not_run_because_cora_full_blocked_at_native_public_output_binding |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 123566.101 | 1207.818 | 14246.984 | 153.379 | blocked | 90755 | blocked |
| Citeseer | 453.773 | blocked | 1418.744 | blocked | 153.837 | blocked | 90763 | blocked |
| Pubmed | 2962.933 | blocked | 5604.074 | blocked | 175.454 | blocked | 90760 | blocked |
| PPI | 3563.973 | blocked | 7111.447 | blocked | 424.710 | blocked | 21360 | blocked |
| ogbn-arxiv | 18071.562 | blocked | 66540.600 | blocked | 417.380 | blocked | 21361 | blocked |

## Blocked/Failed Reasons

- Cora: blocked / native_full_gat_forward_constraints_incomplete; step=prove_time_ms; gate=native.public_output_binding; region=native.full_model.layer_1.output.public_output_binding; row=0; relation=strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, exact inverse, aggregation, ELU, output average
- Citeseer: blocked / not_run_because_cora_full_blocked_at_native_public_output_binding
- Pubmed: blocked / not_run_because_cora_full_blocked_at_native_public_output_binding
- PPI: blocked / not_run_because_cora_full_blocked_at_native_public_output_binding
- ogbn-arxiv: blocked / not_run_because_cora_full_blocked_at_native_public_output_binding
