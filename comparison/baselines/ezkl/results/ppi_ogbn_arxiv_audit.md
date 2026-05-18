# PPI vs ogbn-arxiv Strict Audit

| dataset | status | N | E | G_batch | num_classes | final_logits_shape | public_instance_total_length | proof_size_bytes | commitment_time_ms | prove_time_ms | verify_time_ms | setup_ms | gen_witness_ms | compressed_row_estimate.total_rows_estimated | assigned_row_upper_bound | logrows | circuit_rows | pk.key size | formal_tables.bin size | node_features.f64.bin size | csr_segment_layout.bin size | peak_rss_bytes | final_judgment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppi_full_formal | verified | 56944 | 850576 | 295 | 121 | [56944, 121] | 6890224 | 461655281 | 1280185.848116991 | 1956645.759498002 | 24571.38031598879 | 1280185.848116991 | 2439.6639710175805 | 29591968 | 29591968 | 24 | 16777216 | 51564774651 | 1155136 | 22777634 | 14525132 | 120375009280 | verified benchmark row |
| ogbn_arxiv_full | verified | 169343 | 1335586 | 1 | 40 | [169343, 40] | 6773720 | 453849491 | 1287653.770031 | 1947900.980831997 | 25975.18372000195 | 1287653.770031 | 3255.5468939826824 | 45611736 | 45611736 | 24 | 16777216 | 51564774651 | 184557632 | 173407266 | 24078972 | 123849797632 | verified benchmark row |

## Interpretation

- PPI status is `verified` and ogbn-arxiv status is `verified` in this clean run.
- PPI/ogbn-arxiv prove-time delta: 8744.779 ms.
- PPI is a verified benchmark row: `proof.json`, `verify.json`, final logits binding, formal tables, run manifest, and memory profiles all pass strict audit.
- ogbn-arxiv is a verified benchmark row after this clean rerun and remains the large-graph regression/control case.
- The two large attempts use the same CPU strict route, same Halo2/KZG backend family, and large chunked public-instance columns.
- `public_instance_total_length`, proof size, PK/SRS scale, fixed logrows/circuit row class, and class-count differences are the primary controls to inspect before treating close proving times as suspicious.
- `compressed_row_estimate.total_rows_estimated` is a static planner estimate. `logrows` and `circuit_rows` come from the instantiated Halo2 circuit profile. They need not be equal because the compressed native backend uses a separate diagnostic/full-model layout bound by the selected logrows.
- No manual continuation, forbidden fallback, or precomputed-logits bypass is present. Verified rows must have empty failed-command/block fields and proof sizes matching the real proof files.
