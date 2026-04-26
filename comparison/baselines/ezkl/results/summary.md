# GAT-ZKML vs ezkl Full-Dataset Baseline Summary

- Verified ezkl full-dataset rows: 0/5.
- Strict public input equivalent is true only for rows with a verified full native GAT proof.
- No output-binding workaround, precomputed logits, identity, digest, folding, subset, shard, or sample route was used.

## Dataset Status

| Dataset | status | route_id | strict | full GAT forward in proof | block_reason |
| --- | --- | --- | --- | --- | --- |
| Cora | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_gat_forward_constraints_incomplete |
| Citeseer | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_gat_forward_constraints_incomplete |
| Pubmed | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_full_gat_forward_constraints_incomplete |
| PPI | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_keygen_resource_failure |
| ogbn-arxiv | blocked | ezkl_gat_native_compressed_csr_attention_full | False | False | native_keygen_resource_failure |

## Four-Metric Comparison

| Dataset | main commitment_time_ms | ezkl commitment_time_ms | main prove_time_ms | ezkl prove_time_ms | main verify_time_ms | ezkl verify_time_ms | main proof_size_bytes | ezkl proof_size_bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 283.786 | 132921.294 | 1207.818 | 14752.499 | 153.379 | blocked | 90755 | blocked |
| Citeseer | 453.773 | 134072.710 | 1418.744 | 14817.840 | 153.837 | blocked | 90763 | blocked |
| Pubmed | 2962.933 | 544980.662 | 5604.074 | 62015.504 | 175.454 | blocked | 90760 | blocked |
| PPI | 3563.973 | 1608590.874 | 7111.447 | blocked | 424.710 | blocked | 21360 | blocked |
| ogbn-arxiv | 18071.562 | 1671729.993 | 66540.600 | blocked | 417.380 | blocked | 21361 | blocked |

## Blocked/Failed Reasons

- Cora: blocked / native_full_gat_forward_constraints_incomplete; step=prove_time_ms; gate=native.public_output_binding; region=native.full_model.layer_1.output.public_output_binding; row=0; relation=strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, ELU, output average; values=diagnostic proof binds CSR/table/metadata scoped relations only, scope=full_model, full_gat_forward_in_ezkl_proof=false, strict_public_input_equivalent=false, run_dir=.; node=0; degree=4; sum_fp32=3.9303939786677424; inv_fp32=0.25442742010788516; policy=quantized_inverse_lookup; sum_q=63; inv_q=4; expected_inv_q=4; product=252; inverse_ok=True; alpha_ok=True
- Citeseer: blocked / native_full_gat_forward_constraints_incomplete; step=prove_time_ms; gate=native.public_output_binding; region=native.full_model.layer_1.output.public_output_binding; row=0; relation=strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, ELU, output average; values=diagnostic proof binds CSR/table/metadata scoped relations only, scope=full_model, full_gat_forward_in_ezkl_proof=false, strict_public_input_equivalent=false, run_dir=.; node=0; degree=2; sum_fp32=1.9950954385290314; inv_fp32=0.5012291545998884; policy=quantized_inverse_lookup; sum_q=32; inv_q=8; expected_inv_q=8; product=256; inverse_ok=True; alpha_ok=True
- Pubmed: blocked / native_full_gat_forward_constraints_incomplete; step=prove_time_ms; gate=native.public_output_binding; region=native.full_model.layer_1.output.public_output_binding; row=0; relation=strict_public_input_equivalent requires public output bindings to be derived from every quantized full GAT forward relation: projection, attention score, LReLU lookup, segment max, Exp lookup, quantized inverse lookup, alpha rescale, aggregation, ELU, output average; values=diagnostic proof binds CSR/table/metadata scoped relations only, scope=full_model, full_gat_forward_in_ezkl_proof=false, strict_public_input_equivalent=false, run_dir=.; node=0; degree=6; sum_fp32=5.995988523252744; inv_fp32=0.16677817112590357; policy=quantized_inverse_lookup; sum_q=96; inv_q=3; expected_inv_q=3; product=288; inverse_ok=True; alpha_ok=True
- PPI: blocked / native_keygen_resource_failure; step=setup_ms; gate=native.full_model.keygen; region=native.full_model.setup.create_keys; row=0; relation=graph-gat-setup must complete KZG key generation and write pk.key/vk.key after quantized inverse and alpha preflight succeeds; values=dataset=ppi_full_formal, node_count=56944, edge_count=850576, total_rows=43372416, required_logrows=26, max_logrows=26, setup_ms=1608590.8740110025, pk_key_exists=False, vk_key_exists=False, pk_srs_bytes=8589934852, vk_srs_bytes=8589934852, inverse_preflight_status=ready, inverse_all_nodes_accepted=True, alpha_all_edges_accepted=True; node=0; degree=1; sum_fp32=1; inv_fp32=1; policy=quantized_inverse_lookup; sum_q=16; inv_q=16; expected_inv_q=16; product=256; inverse_ok=True; alpha_ok=True
- ogbn-arxiv: blocked / native_keygen_resource_failure; step=setup_ms; gate=native.full_model.keygen; region=native.full_model.setup.create_keys; row=0; relation=graph-gat-setup must complete KZG key generation and write pk.key/vk.key after quantized inverse and alpha preflight succeeds; values=dataset=ogbn_arxiv_full, node_count=169343, edge_count=1335586, total_rows=59159176, required_logrows=26, max_logrows=26, setup_ms=1671729.9925629923, pk_key_exists=False, vk_key_exists=False, pk_srs_bytes=8589934852, vk_srs_bytes=8589934852, inverse_preflight_status=ready, inverse_all_nodes_accepted=True, alpha_all_edges_accepted=True; node=0; degree=290; sum_fp32=63.34412927528665; inv_fp32=0.015786782633858767; policy=quantized_inverse_lookup; sum_q=1014; inv_q=0; expected_inv_q=0; product=0; inverse_ok=True; alpha_ok=True
