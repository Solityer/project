# zkGAT Secure Main Experiment

This repository is being brought into conformance with the secure zkGAT protocol for full-graph node classification. The target main-path profile is fixed to:

- `protocol_name = zkgat`
- `proof_task = node_classification_full_gat`
- `privacy_policy = model_and_trace_hidden`
- `input_privacy = public_graph_public_features`
- `output_mode = public_logits`
- `opening_mode = domain_grouped_batch_opening`
- `binding_mode = zk_subproof_for_crpc`
- `srs_type = trusted_srs` or `mpc_srs`

The intended public input boundary is public graph structure and public node features. Model parameters, intermediate trace values, attention scores, softmax weights, lookup witness, route witness, PSQ state, folding vectors, private polynomial evaluations, and blinding randomness are private witness material. Final logits are public outputs.

## Current implementation status (2026-07-11)

Protocol-correction phase is **not complete**, so the code is not yet eligible for paper-result generation or performance claims.

What has been revalidated from the current source:

- the lightweight CPU regression suite passes, including KZG opening, compact/padded evaluation, MSM equivalence, SRS mismatch, metadata/schema, and public multi-graph boundary checks;
- public batch boundaries are now validated (`node_ptr`, `edge_ptr`, graph-local node/edge ownership, graph-local edges, destination ordering, unique node IDs, and nonzero indegree);
- a representative real PPI batch made from two complete source graphs passes the boundary validator without cropping nodes or intra-graph edges;
- batch-aware graph/group selectors and public graph IDs are committed to the transcript;
- proof-specific trace, lookup, quotient, challenge, and verifier-result reuse has been disabled on the fresh-proof path;
- a separate transcript-derived KZG opening binds the complete public-logit vector to `P_Y`;
- only final `P_Y` is classified as `PublicOutput`; pre-bias and per-output-head logits remain private payloads;
- requested CUDA algebra kernels now fail closed instead of silently falling back to CPU.

Both the CPU and CUDA builds currently compile and pass the lightweight regression suite. This is a build/unit-test result only; the blocked full Cora proof has not passed on either backend.

The full Cora security regression currently stops before proof generation with:

```text
hidden aggregate escaped the public ELU table; quantization/rescale is not proven
```

This is an intentional fail-closed gate. The current checkpoint declares only `quant_cfg_id=fp32_bundle_export`; it does not materialize the public softmax/aggregation scales and rounding rules required by the protocol. More importantly, the formal relations `Sum * inv = 1` and `alpha = U * inv` produce a general field element for the hidden aggregate, while the implemented finite ELU table is keyed by signed fixed-point integers. The former implementation hid this mismatch by feeding an independently rounded floating-point aggregate into ELU. That disconnected path has been removed and must not be restored.

The following protocol requirements are still open and block phase completion:

- a protocol-compatible, fully specified and proven quantization/rescale relation between softmax aggregation and ELU (and the corresponding output-head scale alignment);
- actual domain-vanishing blinding for every private polynomial and quotient, with corrected quotient/opening equations;
- real zero-knowledge `Pi_bind` subproofs for CRPC bindings; the present shared-dimension trace constraints are not a substitute;
- a strict byte serializer/parser for all nine proof blocks and a complete per-object `ProofSchema` with layer/head/domain identity;
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

The trusted setup assumption is explicit: the main experiment loads and binds an SRS manifest from `artifacts/srs/zkgat_secure_main_srs_v1/srs_manifest.json`. The proof size excludes SRS files, SRS manifest files, local caches, raw witness, checkpoint files, dataset files, logs, and run manifest attachments.

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

The full-security command currently fails at the quantization/ELU gate described above; a green lightweight suite must not be reported as a successful Cora proof. The static scan checks prohibited setup material and SRS hashes, but is not a proof of zero knowledge.

`proof_size_bytes` is currently an in-memory field-size estimate because the formal byte serializer/parser is not implemented. It must not be used as the final transmitted proof-size result. The eventual byte count must include transmitted metadata, commitments, quotient commitments, opening proofs, typed claimed values, public outputs, block tags, family identities, and length prefixes, while excluding SRS and local artifacts.
