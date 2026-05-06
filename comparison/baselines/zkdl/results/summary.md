# zkDL Full GAT Baseline Summary

Generated at: `2026-05-06T05:50:39.092987+00:00`

Correct baseline directory: `comparison/baselines/zkdl`.

The full-dataset statement is attempted for `cora_full`, `citeseer_full`, `pubmed_full`, `ppi_full_formal`, and `ogbn_arxiv_full`. No row is marked verified unless a real zkDL proof and verifier acceptance path succeeds.

`semantic_precheck_passed=true` only means the baseline recomputed the full-dataset forward and local semantic checker accepted it. It is not a zkDL proof and never contributes to `proof_size_bytes`.

CPU runs also emit `runs/<dataset>/cpu/debug_all_relations.json`, an all-relations diagnostic covering the full GAT relation payload in one pass. That diagnostic is not a proof and is not used for `status=verified`.

Relation payload files named `full_gat_forward.zkdl_relation_payload.json` are deliberately typed as diagnostics. A transferable `zkdl_proof` artifact path remains empty until the native zkDL verifier can accept the composed full-GAT statement.

Formal commitment/prove/verify/proof-size metrics are populated only for `status=verified`. Failed proof-attempt timings are debug fields in each run manifest and are excluded from the comparison tables.

## Main Repo Method Porting Map

| item | main_repo_reference | zkdl_baseline_file | status | evidence |
| --- | --- | --- | --- | --- |
| config parsing | src/util/config.cpp::load_config | src/zkdl_gat/baseline.cpp::load_dataset_config | rewritten | same formal keys are required by the zkDL cfg files; no main src/include is linked |
| dataset cache reading | src/data/loader.cpp::load_dataset/load_cached_dataset | src/zkdl_gat/baseline.cpp::load_dataset_meta/load_feature_matrix/load_edges | rewritten | reads /home/pzh/project/data/cache/* directly in the same cache layout |
| feature/edge/label | src/data/loader.cpp::load_features_matrix/load_labels/edges parser | src/zkdl_gat/baseline.cpp::validate_feature_source/validate_label_source/load_edges | ported | feature shape, label count, and 2/3-column edge formats are checked before proving |
| PPI batch | src/data/loader.cpp::selected_graph_ids/normalize_graph_input | src/zkdl_gat/baseline.cpp::normalize_public_input | ported | batch_graphs=295 and node_ptr from cache_node_ptr are consumed for ppi_full_formal |
| ogbn_arxiv | scripts/prepare_ogbn_arxiv.py output + src/data/loader.cpp npy path | src/zkdl_gat/baseline.cpp::load_feature_matrix | ported | features.npy float32 path is supported without calling main scripts |
| self-loop | src/data/loader.cpp::normalize_graph_input self-loop insertion | src/zkdl_gat/baseline.cpp::normalize_public_input | ported | adds one per-node self-loop after local remapping with deduplication |
| edge sort | src/data/loader.cpp stable graph_id,dst sort | src/zkdl_gat/baseline.cpp::normalize_public_input | ported | stable_sort by graph_id then dst then stable_index |
| node_ptr/edge_ptr | src/data/loader.cpp::normalize_graph_input | src/zkdl_gat/baseline.cpp::normalize_public_input | ported | manifest gate compares node_ptr and edge_ptr against main run_manifest |
| checkpoint | src/model/gat.cpp::inspect/load_checkpoint_bundle_parameters | src/zkdl_gat/baseline.cpp::load_model_params/load_attention_head | ported | manifest.json and tensors.txt are parsed by tensor name and shape |
| quant config | src/util/config.cpp + checkpoint manifest quant_cfg_id | src/zkdl_gat/baseline.cpp::fixed_point_policy/build_manifest | ported | range_bits/quant_cfg_id are gate-checked; fixed-point scale and signed encoding are recorded |
| GAT forward | src/model/gat.cpp::forward_note_style | src/zkdl_gat/baseline.cpp::compute_full_forward | ported | hidden heads, concat, output attention, and output head average are computed locally |
| output bias/head average | src/model/gat.cpp::attention_head_forward_impl/average_matrices | src/zkdl_gat/baseline.cpp::compute_head_forward/compute_full_forward | ported | output heads add bias before arithmetic average; hidden heads do not add output bias |
| public input | src/protocol/challenges.cpp and run_manifest public topology fields | src/zkdl_gat/baseline.cpp::hash_normalized_public_input/build_manifest | rewritten | public topology hash is derived after gate-equivalent node/edge normalization |
| final logits | src/model/gat.cpp::forward_note_style | src/zkdl_gat/baseline.cpp::hash_final_logits/run_full_gat_semantic_precheck | rewritten | semantic precheck hashes locally derived final logits; not accepted as proof |
| metric scope | src/main.cpp metrics and src/protocol/prover.cpp::proof_size_bytes | scripts/run_full_baseline.py + src/zkdl_gat/baseline.cpp metrics gating | rewritten | formal four metrics remain null unless status=verified with real zkDL proof/verifier |

## zkDL Proof Retention Map

| item | zkdl_source_file_or_api | proof_section | verifier_check | status |
| --- | --- | --- | --- | --- |
| proof object | original zkDL has no full-GAT serializable proof object API; diagnostic payload lives in src/zkdl_gat/baseline.cpp | full_gat_forward relation payload | none; payload is explicitly not accepted | blocked |
| commitment | commitment.cuh::Commitment::commit | tensor commitments in original Hyrax-style primitive path | Commitment::open is available only as an opening computation/counting path | retained primitive only |
| opening | commitment.cuh::Commitment::open/me_open | MLE opening for committed tensors | no standalone serialized verifier API for transported openings | retained primitive only |
| transcript | proof.cuh::random_vec | sumcheck challenge sampling | not a Fiat-Shamir transcript bound to public input/proof bytes | missing for full GAT |
| sumcheck | proof.cuh::inner_product_sumcheck/hadamard_product_sumcheck/binary_sumcheck; zkmatmul.cuh; zkprod.cuh; zkrelu.cuh | dense/product/binary/ReLU-style primitive sections | no composed full-GAT verifier acceptance API | retained primitive only |
| lookup/range | zkrelu.cuh binary auxiliary; no static lookup table primitive for LeakyReLU/ELU/Exp | activation and exp table requirements | no verifier-backed static table membership check | blocked |
| inverse/binding | no scaled softmax inverse or final logits public binding API in original zkDL files | softmax denominator inverse and public output binding | no native verifier check | blocked |
| verifier acceptance | no verify(...) counterpart for zkMatMul/zkProd/zkReLU/full statement in comparison/baselines/zkdl/*.cuh | composed full GAT statement | not invoked; status cannot be verified | blocked |

## Equivalence Map

| dataset | N | E | feature_dim | class_count | layer/head config | checkpoint hash | quant_cfg_id | public_input_hash | final_logits_shape | final_logits_hash | main_manifest_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cora_full | 2708 | 13264 | 1433 | 7 | L=2; hidden=8x8; K_out=1 | d560eb43dacf6e7adbd8fd4dad7f6691ee6e6989c4cf49f73cfc421ee2a05e7a | range_bits=12 | 0x0562d892cb30cdd4 | 2708x7 | 0x35e4d324b481436b | true |
| citeseer_full | 3327 | 12431 | 3703 | 6 | L=2; hidden=8x8; K_out=1 | 744e1992064040aa381bd40db8713aefa83ea49abe725a5f6f9351195030f4b1 | range_bits=12 | 0x4a693c769c72bd6e | 3327x6 | 0xe1d7839f7af258ba | true |
| pubmed_full | 19717 | 108365 | 500 | 3 | L=2; hidden=8x8; K_out=1 | c8c1b5d627cbbb211feb1f13d825e3a4d904dc2820f876c362ad09bdaf0f6c24 | range_bits=12 | 0x9f71cc9e0822d146 | 19717x3 | 0xac429a4a0a4c876c | true |
| ppi_full_formal | 56944 | 850576 | 50 | 121 | L=2; hidden=1x8; K_out=1 | 31f086a3821e5536cd3bb67db47185b96aad4303c431d664baed42511c5cff75 | fp32_bundle_export | 0x235f94d80c4f81fc | 56944x121 | 0x0f8f4cde2f8ec1da | true |
| ogbn_arxiv_full | 169343 | 1335586 | 128 | 40 | L=2; hidden=1x8; K_out=1 | c250e54df0b0fd85106edac032630446d3cd499be53b62ed8bdf59027b017751 | fp32_bundle_export | 0x17a59427c8a2496a | 169343x40 | 0xce9f3974030ed588 | true |

## CPU

| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes |
| --- | --- | --- | --- | --- | --- |
| cora_full | blocked |  |  |  |  |
| citeseer_full | blocked |  |  |  |  |
| pubmed_full | blocked |  |  |  |  |
| ppi_full_formal | blocked |  |  |  |  |
| ogbn_arxiv_full | blocked |  |  |  |  |

## GPU

| dataset | status | commitment_time_ms | prove_time_ms | verify_time_ms | proof_size_bytes | gpu_backend_active | cpu_fallback_used |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cora_full | blocked |  |  |  |  | True | False |
| citeseer_full | blocked |  |  |  |  | True | False |
| pubmed_full | blocked |  |  |  |  | True | False |
| ppi_full_formal | blocked |  |  |  |  | True | False |
| ogbn_arxiv_full | blocked |  |  |  |  | True | False |

## Blocked Items

| dataset | route_id | failure_step | block_reason |
| --- | --- | --- | --- |
| cora_full | zkdl_gat_full_forward_cpu | sparse_source_target_route_sumcheck_verify | native_zkdl_relation_coverage_blocked: original zkDL exposes dense/product/binary/ReLU-style sumcheck primitives but no verifier-backed sparse route, static lookup, scaled inverse, rescale range, or final logits public binding relations required by the full-dataset GAT statement; relation payload and all-relations diagnostic were generated but are not accepted as a zkDL proof |
| citeseer_full | zkdl_gat_full_forward_cpu | sparse_source_target_route_sumcheck_verify | native_zkdl_relation_coverage_blocked: original zkDL exposes dense/product/binary/ReLU-style sumcheck primitives but no verifier-backed sparse route, static lookup, scaled inverse, rescale range, or final logits public binding relations required by the full-dataset GAT statement; relation payload and all-relations diagnostic were generated but are not accepted as a zkDL proof |
| pubmed_full | zkdl_gat_full_forward_cpu | sparse_source_target_route_sumcheck_verify | native_zkdl_relation_coverage_blocked: original zkDL exposes dense/product/binary/ReLU-style sumcheck primitives but no verifier-backed sparse route, static lookup, scaled inverse, rescale range, or final logits public binding relations required by the full-dataset GAT statement; relation payload and all-relations diagnostic were generated but are not accepted as a zkDL proof |
| ppi_full_formal | zkdl_gat_full_forward_cpu | sparse_source_target_route_sumcheck_verify | native_zkdl_relation_coverage_blocked: original zkDL exposes dense/product/binary/ReLU-style sumcheck primitives but no verifier-backed sparse route, static lookup, scaled inverse, rescale range, or final logits public binding relations required by the full-dataset GAT statement; relation payload and all-relations diagnostic were generated but are not accepted as a zkDL proof |
| ogbn_arxiv_full | zkdl_gat_full_forward_cpu | sparse_source_target_route_sumcheck_verify | native_zkdl_relation_coverage_blocked: original zkDL exposes dense/product/binary/ReLU-style sumcheck primitives but no verifier-backed sparse route, static lookup, scaled inverse, rescale range, or final logits public binding relations required by the full-dataset GAT statement; relation payload and all-relations diagnostic were generated but are not accepted as a zkDL proof |
| cora_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| citeseer_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| pubmed_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ppi_full_formal | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
| ogbn_arxiv_full | zkdl_gat_full_forward_gpu | gpu_composed_gat_prover | zkdl_cuda_primitive_probe_passed_but_cuda_composed_full_gat_forward_prover_verifier_is_not_implemented_without_cpu_fallback |
