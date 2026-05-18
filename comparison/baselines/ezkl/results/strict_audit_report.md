# Strict Reproducibility Audit

- Generated at: `2026-05-18T03:07:05Z`
- Required route: `ezkl_gat_native_compressed_csr_attention_full`
- The collector writes an audit report for blocked resource runs, but returns a non-zero BLOCKED status unless every dataset is a verified benchmark row.

| Dataset | status | proof_file_size_bytes | block_reason | failure_step | failed_checks |
| --- | --- | ---: | --- | --- | --- |
| cora_full | passed | 1280317 |  | None |  |
| citeseer_full | passed | 1347633 |  | None |  |
| pubmed_full | passed | 3973375 |  | None |  |
| ppi_full_formal | passed | 461655281 |  | None |  |
| ogbn_arxiv_full | passed | 453849491 |  | None |  |

## Enforced Rules

- Verified benchmark rows require `status=verified`, `verify_passed=true`, and `proof_generated=true`.
- Verified benchmark rows require `proof_size_bytes` to equal the real `proof.json` file size.
- Verified benchmark rows require `gen_settings_ms`, `compile_ms`, `setup_ms`, `gen_witness_ms`, `prove_time_ms`, and `verify_time_ms` to be numeric.
- `failed_command`, `failure_step`, and `block_reason` must be empty for verified rows.
- Blocked rows must keep `verify_passed=false`, `proof_generated=false`, non-empty `block_reason`, non-empty `failure_step`, and the real failed command.
- `stdout_tail`, `stderr_tail`, and stage tails must not contain `manual continuation`.
- `proof.json`, `verify.json`, `run_manifest.json`, model metadata, formal table metadata, preflight, and memory profiles must exist.
