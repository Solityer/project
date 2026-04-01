# Test Plan

## Full-Cora main tests
- `full_cora_edges_are_dst_sorted_and_self_looped`
  - Checks topology ordering and self-loop guarantees on the real Cora graph.
- `full_cora_bias_matches_reference`
  - Checks the dense attention mask against the project-local reference export.
- `formal_full_cora_builds_multihead_context_with_cat_and_C_domains`
  - Checks real multi-head bundle loading and explicit `H_cat/H_C` domains.
- `reference_style_multihead_forward_shapes`
  - Checks hidden/output head shapes on full Cora.
- `reference_style_multihead_forward_matches_reference_artifacts`
  - Checks project forward against reference artifacts for hidden heads, concat, and output head.

## Formal regression tests
- `transcript_order_consistency`
- `agg_witness_commitment_opening_consistency`
- `prove_verify_round_trip`
- `tampered_witness_fails`
- `metadata_mismatch_fails`
- `proof_block_order_mismatch_fails`
- `wrong_H_C_domain_reuse_fails`

## Current missing tests
- Full-Cora formal `H_cat/H_cat_star` consistency
- Full-Cora formal `Y/Y_star/PSQ_out` consistency
- Tamper rejection for `H_cat_star`, `Y_star`, and `PSQ_out`
- Full-Cora multi-head formal prove/verify round-trip
