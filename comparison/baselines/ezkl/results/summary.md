# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Verified ezkl full-dataset rows: 3/5.
- Strict public input equivalent is true only for rows with a verified full native GAT proof.
- No output-binding workaround, precomputed logits, identity, digest, folding, subset, shard, or sample route was used.

## Dataset Status

| Dataset | status | route_id | strict | full GAT forward in proof | final logits derivation | block_reason |
| --- | --- | --- | --- | --- | --- | --- |
| Cora | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| Citeseer | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| Pubmed | verified | ezkl_gat_native_compressed_csr_attention_full | True | True | True |  |
| PPI | failed | ezkl_gat_native_compressed_csr_attention_full | False | False | True | native_constraint_synthesis_failed |
| ogbn-arxiv | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | True | native_prove_resource_failure |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 44691.611 | 1207.818 | 22186.999 | 153.379 | 397.249 | 90755 | 1281018 |
| Citeseer | 453.773 | 44597.467 | 1418.744 | 22231.393 | 153.837 | 508.609 | 90763 | 1348419 |
| Pubmed | 2962.933 | 90476.599 | 5604.074 | 47989.941 | 175.454 | 958.592 | 90760 | 3974097 |
| PPI | 3563.973 | 1428500.900 | 7111.447 | 98325.890 | 424.710 |  | 21360 |  |
| ogbn-arxiv | 18071.562 | 1458744.766 | 66540.600 | 606168.838 | 417.380 | blocked | 21361 | blocked |

## Blocked/Failed Reasons

- Cora: verified / 
- Citeseer: verified / 
- Pubmed: verified / 
- PPI: failed / native_constraint_synthesis_failed; step=prove_time_ms; gate=native.full_model.output.head_0.segment_sum_inverse; region=native.full_model.output.head_0.segment_sum_inverse; row=35543; relation=MockProver must satisfy every enabled native diagnostic gate before KZG proving; values=required_logrows=25, node_count=56944, edge_count=850576, run_dir=., output_head_index=0, output_node_index=17771, output_sum_fp64=326.70716581639925, output_sum_q=5227, output_inv_q=0, inverse_table_range_size=4096, output_sum_q_in_inverse_table_range=false, segment_begin=234519, segment_end=235101, segment_degree=582
- ogbn-arxiv: blocked / native_prove_resource_failure
