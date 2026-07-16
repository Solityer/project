# zkGAT Secure Main Experiment

This repository is being brought into conformance with the secure zkGAT protocol for full-graph node classification. The target main-path profile is fixed to:

- `protocol_name = zkgat`
- `proof_task = node_classification_full_gat`
- `privacy_policy = model_and_trace_hidden`
- `input_privacy = public_graph_public_features`
- `output_mode = public_logits`
- `opening_mode = point_grouped_batch_opening`
- `binding_mode = zk_subproof_for_crpc`
- `srs_type = trusted_srs`

The intended public input boundary is public graph structure and public node features. Model parameters, intermediate trace values, attention scores, softmax weights, lookup witness, route witness, PSQ state, folding vectors, private polynomial evaluations, and blinding randomness are private witness material. Final logits are public outputs.

## Current implementation status (2026-07-16)

Protocol-correction phase is **not complete**, so the code is not yet eligible for paper-result generation or performance claims.

What has been revalidated from the current source:

- the lightweight CPU regression suite passes, including KZG opening, compact/padded evaluation, MSM equivalence, SRS mismatch, metadata/schema, and public multi-graph boundary checks;
- public batch boundaries are now validated (`node_ptr`, `edge_ptr`, graph-local node/edge ownership, graph-local edges, destination ordering, unique node IDs, and nonzero indegree);
- a representative real PPI batch made from two complete source graphs passes the boundary validator without cropping nodes or intra-graph edges;
- batch-aware graph/group selectors and public graph IDs are committed to the transcript;
- proof-specific trace, lookup, quotient, challenge, and verifier-result reuse has been disabled on the fresh-proof path;
- a separate transcript-derived KZG opening binds the complete public-logit vector to `P_Y`;
- only final `P_Y` is classified as `PublicOutput`; pre-bias and per-output-head logits remain private payloads;
- the strict wire-v2 serializer/parser covers all nine tagged proof blocks, rejects non-canonical/trailing data, and computes `proof_size_bytes` from the exact serialized byte vector;
- the protocol compiler emits authenticated ProofSchema/ObservationLedger/RandomIdentityLedger/SecurityBudgetLedger templates; the current instance artifact includes the complete length-delimited `RelationMeta` registry and a canonical typed dependency DAG, enforces that every private schema object is exactly one independent blinder owner or one derived target, and enumerates the active feature, hidden, output, LogUp, SignedRoundDiv, quotient, public-output, and point-group random checks; all `RelationMeta` identity/source/target/domain/length/quotient fields enter the transcript, and the compiler has a strict semantic/gap/tail selector plus fixed edge-chunk layout constructor;
- the output-bias `Com_basis` path now materializes the registered coefficient tail, fixed edge chunks, verifier-local Gamma powers, one accumulator and fixed terminal per chunk, Taylor-shifted basis-link commitments, strict wire/parser fields, global point-group openings, and verifier equations; the `unit_test_srs_v21_toy` CPU and CUDA-hotspot paths both complete generation/load/prove/parse/verify and tamper rejection (`212749` CPU / `212748` GPU proof bytes in the latest cold runs);
- `unit_test_srs_v15` uses manifest v7, compiler artifact v3, the strict maximum-instance-envelope v2 (`N_total`, concat width, range bits, coefficient-object length and object-family count included), and has passed generation, loading, proof/verification self-tests, SRS/manifest/domain tamper rejection, obsolete-v14 rejection, and the real PPI representative-batch boundary regression;
- requested CUDA algebra kernels now fail closed instead of silently falling back to CPU.

Both the CPU and CUDA builds currently compile and pass the lightweight regression suite. The small trusted-SRS formal proof passes on both backends, and the real two-graph PPI boundary regression passes. This remains development evidence only; the blocked full Cora proof has not passed on either backend.

The full Cora security regression currently stops at SRS loading: all locally present large SRS directories use superseded, rejected manifest formats, while the current v7 compiler/manifest rules have only been exercised with the small trusted test SRS. A new formal large SRS must not be generated until the remaining protocol-compiler and `Pi_bind` rules are frozen.

The following protocol requirements are still open and block phase completion:

- completion of the canonical symbolic ObservationLedger RREF/image-inclusion certificates and exact `BadCRS_obs` factor accounting for the full derived dependency DAG;
- expansion of the now-closed output-bias `Com_basis` machinery to every model and dynamic matrix relation, including canonical node-domain columns, row-major indexed views, coefficient tails, per-chunk basis links, CRPC identities, and aggregation bindings; the current proof-local model reblinding and output-bias relation remain only a partial closure;
- expansion of the maximum compiler artifact/envelope to the complete maximum ProofSchema and all three maximum ledgers before generating the formal large SRS;
- verifier-only model/static verification-key separation and complete leakage/tamper coverage;
- successful final Cora CPU/CUDA and representative real PPI-batch regressions.

Consequently, structure governance and performance optimization have not started. Existing run artifacts and performance reports are historical only and are not evidence for the current source.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j
```

## Security model / threat model

The formal threat model is in `docs/security/threat_model.md`. The matching claims and limitations are in `docs/security/security_claims.md`.

The verifier already applies several metadata, ordering, commitment, opening, public-output, and transcript checks, but the complete security claim is pending the open protocol items listed above. A payload tagged `MaskedPrivateValue` is not considered masked unless the committed polynomial is actually blinded.

The trusted setup assumption is explicit. The old large SRS directories are not accepted by the current loader and are not evidence for the current protocol. The proof size excludes SRS files, SRS manifest files, local caches, raw witness, checkpoint files, dataset files, logs, and run manifest attachments.

## Secure configs

The five intended full-dataset configs are present, but full formal runs are not authorized during protocol development:

- `configs/cora_secure.cfg`
- `configs/citeseer_secure.cfg`
- `configs/pubmed_secure.cfg`
- `configs/ppi_secure.cfg`
- `configs/ogbn_arxiv_secure.cfg`

Each config points to the same SRS manifest. `allow_synthetic_model=true` is not used in paper-result configs. The presence of secure-looking metadata does not establish protocol conformance.

## Development execution boundary

During protocol correction, end-to-end work is limited to Cora and a representative real PPI batch. Do not run full formal end-to-end jobs for the other datasets, do not run the full PPI formal experiment, and do not generate or overwrite the ten CPU/GPU paper-result runs. Those runs require explicit project-owner authorization after all protocol, structure, performance, and final-review gates pass.

The CUDA path is currently a hybrid implementation: only selected field-evaluation and histogram hotspots have CUDA kernels; witness construction, most commitments/MSMs, quotient work, KZG openings, transcript work, and verification remain CPU-side. It must not be described as a full-GPU proof path. A requested CUDA kernel failure is fatal rather than a silent CPU fallback.

## Export paper results

```bash
python3 scripts/export_paper_secure.py \
  --cpu cora=runs/cora_secure/warm \
  --cpu citeseer=runs/citeseer_secure/warm \
  --cpu pubmed=runs/pubmed_secure/warm \
  --cpu ppi=runs/ppi_secure/warm \
  --cpu ogbn-arxiv=runs/ogbn_arxiv_secure/warm \
  --gpu cora=runs/cora_secure/cuda_formal/warm \
  --gpu citeseer=runs/citeseer_secure/cuda_formal/warm \
  --gpu pubmed=runs/pubmed_secure/cuda_formal/warm \
  --gpu ppi=runs/ppi_secure/cuda_formal/warm \
  --gpu ogbn-arxiv=runs/ogbn_arxiv_secure/cuda_formal/warm \
  --output-dir results/paper_secure
```

Do not invoke this command until the project owner authorizes the final ten runs. The exporter is expected to fail closed for incomplete or nonconforming manifests, but exporter checks do not repair an insecure proof implementation.

## Tests and security scan

```bash
cmake --build build --target gatzk_tests -j
./build/gatzk_tests
GATZK_RUN_FULL_SECURITY_TESTS=1 ./build/gatzk_tests
GATZK_RUN_PPI_BOUNDARY_TESTS=1 ./build/gatzk_tests
python3 tools/security/static_scan_no_tau.py
```

The full-security command currently cannot pass without a current formal large SRS; a green lightweight suite must not be reported as a successful Cora proof. The static scan checks prohibited setup material and SRS hashes, but is not a proof of zero knowledge.

`proof_size_bytes` is the exact size of the strict canonical serialized proof bytes, including metadata, commitments, quotient commitments, opening proofs, typed claimed values, public outputs, block tags, family identities, and length prefixes, while excluding SRS and local artifacts. No current paper-result proof-size claim is authorized because phase one is incomplete.
