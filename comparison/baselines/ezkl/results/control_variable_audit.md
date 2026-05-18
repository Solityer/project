# Control Variable Audit

- Route: `ezkl_gat_native_compressed_csr_attention_full`
- Backend: `cpu` for every final row.
- Verified benchmark rows require strict public input equivalence, complete full GAT forward proof, and no precomputed logits bypass.
- The collector records real blocked resource runs as audit rows, but excludes them from benchmark metrics and returns a non-zero BLOCKED status.

| Dataset | status | backend | route_id | clean_run | check_mode | proof_generated | verify_passed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora_full | verified | cpu | ezkl_gat_native_compressed_csr_attention_full | true | safe | true | true |
| citeseer_full | verified | cpu | ezkl_gat_native_compressed_csr_attention_full | true | safe | true | true |
| pubmed_full | verified | cpu | ezkl_gat_native_compressed_csr_attention_full | true | safe | true | true |
| ppi_full_formal | verified | cpu | ezkl_gat_native_compressed_csr_attention_full | true | safe | true | true |
| ogbn_arxiv_full | verified | cpu | ezkl_gat_native_compressed_csr_attention_full | true | safe | true | true |

Only `manifest_strict_check=passed` rows are included as final benchmark rows; blocked rows are retained only as audit evidence.
