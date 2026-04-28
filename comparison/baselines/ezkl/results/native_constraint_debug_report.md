# Native Constraint Debug Report

- Route: `ezkl_gat_native_compressed_csr_attention_full`
- Diagnostic policy: explicit `--native-debug-scope` runs are available for development, but final benchmark collection uses only the verified full-model manifests.
- No output-binding workaround, precomputed logits, subset, shard, sample, digest, folding, or identity route is used.

## Full-Dataset Quantized Inverse Preflight Evidence

| Dataset | node | degree | policy | sum_fp32 | inv_fp32 | sum_q | inv_q | expected_inv_q | product | inverse_ok | alpha_ok |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Cora | 0 | 4 | quantized_inverse_lookup | 3.9303939786677424 | 0.25442742010788516 | 63 | 4 | 4 | 252 | True | True |
| Citeseer | 0 | 2 | quantized_inverse_lookup | 1.9950954385290314 | 0.5012291545998884 | 32 | 8 | 8 | 256 | True | True |
| Pubmed | 0 | 6 | quantized_inverse_lookup | 5.995988523252744 | 0.16677817112590357 | 96 | 3 | 3 | 288 | True | True |
| PPI | 0 | 1 | quantized_inverse_lookup | 1 | 1 | 16 | 16 | 16 | 256 | True | True |
| ogbn-arxiv | 0 | 290 | quantized_inverse_lookup | 63.34412927528665 | 0.015786782633858767 | 1014 | 0 | 0 | 0 | True | True |
