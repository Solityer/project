# Large Public Output Policy Review

Reviewed sources:

- `/home/pzh/project/comparison/baselines/GAT-ZKML-ezkl.md`
- `/home/pzh/project/GAT-ZKML-多层多头.md`

## Policy Answers

| Question | Conclusion |
| --- | --- |
| A. Must every final logit be a Halo2/ezkl public instance cell? | No absolute requirement in the ezkl note. It defines `inst_i` separately from `public_i`. The current route still uses element-wise public final logits as native public instances. |
| B. Are public logits as an artifact plus commitment/hash allowed? | Conditionally yes. Public fields outside `inst_i` must be bound by `binding_digest_i` in `inst_i` or by aggregation public input. |
| C. May verifier externally read the full public logits artifact and verify a circuit-public commitment/hash? | Conditionally yes. Section 4.4 says that when output is public, the verifier reads `y` and checks `Hash(encoding(y, visibility_cfg, encoding_id)) = output_hash_i`. |
| D. Can such a commitment route still be `strict_public_input_equivalent=true`? | Only if the full logits artifact remains public, the commitment/hash is collision-resistant and encoding-stable, the digest is proof-public/bound, final logits are derived in-circuit before binding, and metrics consume the same full artifact. FNV64 or off-circuit-only digests are not strict-compatible. |
| E. If commitment were not allowed, is element-wise public instance OOM a hard boundary? | Not the policy conclusion here, because the note conditionally allows external public fields when bound. The current main route did not implement or use that commitment route. |

## Current Route

The current route is `ezkl_gat_native_compressed_csr_attention_full`. It keeps element-wise public final logits and does not use digest/folding/output compression as a substitute for logits. `strict_public_input_equivalent=true` is retained only for verified element-wise public-logit runs.

## Large Dataset Evidence

| Dataset | status | next blocker | public_instance_length | layout | columns | chunk_size | setup loads public values | setup peak RSS bytes | prove peak RSS bytes |
| --- | --- | --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| PPI | failed | `native_constraint_synthesis_failed`: output head inverse lookup has `output_sum_q=5227` outside `T_inv` range 4096 at node 17771 | 6890224 | chunked_columns | 2 | 4194304 | no, shape metadata only | 104225325056 | n/a |
| ogbn-arxiv | blocked | `native_prove_resource_failure`: prove killed by signal 9 | 6773720 | chunked_columns | 2 | 4194304 | no, shape metadata only | 104264855552 | 126824886272 |

## Resource Conclusion

The original `native_keygen_resource_failure` was eliminated for both PPI and ogbn-arxiv by shape-only setup plus 4M chunked public instance columns. Both setup/keygen runs returned 0 and wrote `pk.key`/`vk.key`.

No public-output commitment route was enabled in the strict main comparison. PPI is blocked after keygen by a real output-head inverse-table coverage failure under the unchanged quantization/table policy. ogbn-arxiv is blocked after keygen by proof-generation RSS reaching 123,852,428 KB and being killed by signal 9.
