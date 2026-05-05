# Large Public Output Policy Review

Reviewed sources:

- `/home/pzh/project/comparison/baselines/GAT-ZKML-ezkl.md`
- `/home/pzh/project/GAT-ZKML-多层多头.md`

## Policy Answers

| Question | Conclusion |
| --- | --- |
| Must every final logit be a Halo2/ezkl public instance cell in this strict run? | This route keeps every final logit as a public instance cell, so no commitment-only substitute is used. |
| Are public logits as an artifact plus commitment/hash allowed by the note? | Only conditionally, when the full public artifact is verifier-visible and a collision-resistant encoding hash is proof-public/bound. That route was not used here. |
| Can a commitment route be marked `strict_public_input_equivalent=true` in these results? | Not in this run set, because the implemented strict route is element-wise public logits. No digest, folding, output-binding replacement, or external-only public output shortcut was enabled. |
| Was shape-only setup allowed? | Yes. Setup/keygen consumes public-instance shape metadata instead of loading public values, but prove and verify still bind the full row-major final logits public instances. |

## Current Route

The current route is `ezkl_gat_native_compressed_csr_attention_full`. It keeps element-wise public final logits as native public instances, uses chunked public instance columns for large outputs, and reports `strict_public_input_equivalent=true` only after strict KZG proof verification succeeds.

## Large Dataset Evidence

| Dataset | status | public_instance_length | layout | columns | chunk_size | setup loads public values | lookup_table_columns | pk_size_bytes | proof_size_bytes | prove_peak_rss_bytes |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| PPI | verified | 6890224 | chunked_columns | 2 | 4194304 | no, shape metadata only | 3 | 51564774651 | 461655303 | 123818070016 |
| ogbn-arxiv | verified | 6773720 | chunked_columns | 2 | 4194304 | no, shape metadata only | 3 | 51564774651 | 453849446 | 123764703232 |

## Resource Conclusion

The earlier large-public-output keygen OOM was eliminated by shape-only setup plus chunked public instance columns. The later PPI inverse-table blocker was resolved by a deterministic public-metadata inverse range policy, not by witness-specific table construction. The later ogbn-arxiv prove SIGKILL was resolved by prove-time payload deduplication, low-memory execution settings, skipping large full-dataset diagnostic MockProver precheck, and consolidating native lookup tables into one tagged table.

No public-output commitment route was enabled in the strict main comparison. PPI and ogbn-arxiv both completed strict KZG prove and verify with full public final logits retained.
