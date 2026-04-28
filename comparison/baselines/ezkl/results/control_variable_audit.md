# Control Variable Audit

Route: `ezkl_gat_native_compressed_csr_attention_full`

## Scope

All reported runs use the full datasets:

| Dataset | run_manifest |
| --- | --- |
| cora_full | `runs/cora_full/run_manifest.json` |
| citeseer_full | `runs/citeseer_full/run_manifest.json` |
| pubmed_full | `runs/pubmed_full/run_manifest.json` |
| ppi_full_formal | `runs/ppi_full_formal/run_manifest.json` |
| ogbn_arxiv_full | `runs/ogbn_arxiv_full/run_manifest.json` |

No toy, smoke, subset, sample, shard, partial, debug dataset, or scaled graph route was used for the final table.

## Preserved Semantics

| Control variable | Audit result |
| --- | --- |
| Checkpoint bundle | Reused from the existing full-dataset configuration. |
| `model_arch_id` | Preserved by adapter-exported model metadata and backend manifest checks. |
| `model_param_id` | Preserved by adapter-exported model metadata and backend manifest checks. |
| `quant_cfg_id` | Preserved; inverse range policy is derived from the public quant/table metadata rather than witness sums. |
| `hidden_profile` | Preserved by dataset/model metadata. |
| `d_in_profile` | Preserved by dataset/model metadata. |
| `K_out` | Preserved by dataset/model metadata. |
| `output_average_rule` | Preserved; final logits remain derived inside the native GAT relation. |
| Graph preprocessing | Preserved by the dataset export route. |
| Self-loop rule | Preserved by graph metadata and compressed CSR attention input. |
| Edge sort rule | Preserved by graph metadata and compressed CSR attention input. |
| Batching rule | Preserved by full-dataset single-run compressed CSR execution. |
| Final logits semantics | Full row-major node-by-class logits are public instances and are bound to in-circuit derivation. |

## Allowed Adaptations

| Adaptation | Statement impact |
| --- | --- |
| Shape-only setup for large public outputs | Setup avoids loading public values; prove/verify still bind the full public logits. |
| Chunked public instance columns | Splits the same row-major public logits across instance columns without changing values or order. |
| Deterministic inverse table range policy | Extends `T_inv` from public degree/quant/table bounds; `witness_specific_table=false`. |
| Tagged lookup table columns | Reuses one deterministic lookup table for inverse/LReLU/Exp/ELU with kind tags; lookup semantics are unchanged. |
| Prove-time payload deduplication and low-memory environment | Reduces duplicate host memory only; constraints and public statement are unchanged. |
| Skipping large full-dataset diagnostic MockProver precheck | Removes a diagnostic-only precheck for resource reasons; strict KZG proof and verify remain required. |

## Strictness Check

| Dataset | status | strict_public_input_equivalent | full_gat_forward_in_ezkl_proof | final_logits_derivation_complete |
| --- | --- | --- | --- | --- |
| cora_full | verified | true | true | true |
| citeseer_full | verified | true | true | true |
| pubmed_full | verified | true | true | true |
| ppi_full_formal | verified | true | true | true |
| ogbn_arxiv_full | verified | true | true | true |

No output-binding replacement, digest-only output, folding, identity route, subset, shard, sample, or precomputed-logits bypass appears in the final verified route.
