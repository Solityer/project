# zkGAT Security Claims And Limitations

`security_profile_id = zkgat_secure_profile_v1`

## Claims

- The secure main path accepts only `trusted_srs` or `mpc_srs` manifests.
- `PublicMetadata` binds `protocol_name`, `proof_task`, `srs_id`, `srs_type`, `srs_degree_bound`, `privacy_policy`, `input_privacy`, `output_mode`, `opening_mode`, `binding_mode`, model architecture, model parameters, static tables, quantization config, domains, dimensions, encoding, padding, and degree bound.
- The verifier rejects secure-profile mismatches before accepting a proof.
- `ProofSchema` classifies committed objects as `Public`, `PublicOutput`, or `PrivateWitness`; private witness values require masked evaluation semantics.
- `private_plain_value_count` is computed by the verifier from typed proof payload tags and must be zero for secure runs.
- The exporter refuses non-secure, failed, partial, or missing-field runs.
- `proof_size_bytes` is the proof transport size and excludes SRS files, manifest file bodies, local caches, raw witness, checkpoint files, datasets, logs, and run manifest attachments.

## Limitations

The public graph and public feature boundary is intentional. The current output mode is public logits, not label-only. Training is outside the proof statement. Side channels, timing leakage, memory-access leakage, GPU side channels, and compromised prover hosts are outside the implementation claim. The repository binds an SRS manifest but does not implement a complete ceremony transcript.

## Required Run Flags

Each paper result row must include:

- `dataset`
- `backend`
- `srs_type`
- `srs_id`
- `privacy_policy`
- `input_privacy`
- `output_mode`
- `opening_mode`
- `binding_mode`
- `proof_task`
- `commitment_time_ms`
- `prove_time_ms`
- `verify_time_ms`
- `proof_size_bytes`
- `verifier_reads_witness = false`
- `private_plain_value_count = 0`, as returned by verifier payload audit
- `reference_consistency_passed = true`
- `tamper_tests_passed = true`
- `full_dataset = true`
