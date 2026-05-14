# PPI vs ogbn-arxiv Strict Audit

| dataset | N | E | G_batch | num_classes | final_logits_shape | public_instance_total_length | proof_size_bytes | commitment_time_ms | prove_time_ms | verify_time_ms | setup_ms | gen_witness_ms | compressed_row_estimate.total_rows_estimated | assigned_row_upper_bound | logrows | circuit_rows | pk.key size | formal_tables.bin size | node_features.f64.bin size | csr_segment_layout.bin size | peak_rss_bytes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppi_full_formal | 56944 | 850576 | 295 | 121 | [56944, 121] | 6890224 | None | 1322255.0203840074 | 1384312.234527024 | None | 1322255.0203840074 | 2222.4812769854907 | 29591968 | 29591968 | 24 | 16777216 | 51564774651 | 1155136 | 22777634 | 14525132 | 119402041344 |
| ogbn_arxiv_full | 169343 | 1335586 | 1 | 40 | [169343, 40] | 6773720 | 453849473 | 1276740.8295759815 | 1941818.2253029954 | 24279.1048999934 | 1276740.8295759815 | 3076.232806983171 | 45611736 | 45611736 | 24 | 16777216 | 51564774651 | 184557632 | 173407266 | 24078972 | 121800486912 |

## Interpretation

- PPI status is `blocked` and ogbn-arxiv status is `verified` in this clean run.
- PPI prove elapsed before resource termination differs from ogbn-arxiv verified prove time by 557505.991 ms; this is not a verified PPI benchmark comparison.
- The two large attempts use the same CPU strict route, same Halo2/KZG backend family, and large chunked public-instance columns.
- `public_instance_total_length`, proof size, PK/SRS scale, fixed logrows/circuit row class, and class-count differences are the primary controls to inspect before treating close proving times as suspicious.
- `compressed_row_estimate.total_rows_estimated` is a static planner estimate. `logrows` and `circuit_rows` come from the instantiated Halo2 circuit profile. They need not be equal because the compressed native backend uses a separate diagnostic/full-model layout bound by the selected logrows.
- No manual continuation, forbidden fallback, or precomputed-logits bypass is present. PPI intentionally retains the real failed command and resource block reason because it is blocked, not verified.
