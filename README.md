# zkGAT Secure Main Experiment

This repository implements the secure zkGAT main experiment for full-graph node classification. The submitted main path is fixed to:

- `protocol_name = zkgat`
- `proof_task = node_classification_full_gat`
- `privacy_policy = model_and_trace_hidden`
- `input_privacy = public_graph_public_features`
- `output_mode = public_logits`
- `opening_mode = domain_grouped_batch_opening`
- `binding_mode = zk_subproof_for_crpc`
- `srs_type = trusted_srs` or `mpc_srs`

The public input boundary is public graph structure and public node features. Model parameters, intermediate trace values, attention scores, softmax weights, lookup witness, route witness, PSQ state, folding vectors, private polynomial evaluations, and blinding randomness are private witness material. Final logits are public outputs and are bound by the proof.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

cmake -S . -B build_cuda -DCMAKE_BUILD_TYPE=Release -DGATZK_ENABLE_CUDA_BACKEND=ON
cmake --build build_cuda -j
```

## Security Model / Threat Model

The formal threat model is in `docs/security/threat_model.md`. The matching claims and limitations are in `docs/security/security_claims.md`.

The verifier accepts only secure metadata matching the fixed profile above. It rejects mismatched `srs_id`, `srs_type`, privacy policy, input privacy, output mode, opening mode, binding mode, proof task, metadata order, commitment order, commitment point, quotient commitment, opening values, external masked values, typed payload leakage, and transcript challenges. The verifier does not read private witness objects.

The trusted setup assumption is explicit: the main experiment loads and binds an SRS manifest from `artifacts/srs/zkgat_secure_main_srs_v1/srs_manifest.json`. The proof size excludes SRS files, SRS manifest files, local caches, raw witness, checkpoint files, dataset files, logs, and run manifest attachments.

## Secure Configs

The five full-dataset main configs are:

- `configs/cora_secure.cfg`
- `configs/citeseer_secure.cfg`
- `configs/pubmed_secure.cfg`
- `configs/ppi_secure.cfg`
- `configs/ogbn_arxiv_secure.cfg`

Each config sets the secure fields listed above and points to the same SRS manifest. `allow_synthetic_model=true` is not used in paper-result configs.

## Run Secure Main Suite

CPU:

```bash
./build/gatzk_run --config configs/cora_secure.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/citeseer_secure.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/pubmed_secure.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/ppi_secure.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
./build/gatzk_run --config configs/ogbn_arxiv_secure.cfg --benchmark-mode warm --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on
```

GPU:

```bash
./build_cuda/gatzk_run --config configs/cora_secure.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/citeseer_secure.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/pubmed_secure.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ppi_secure.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
./build_cuda/gatzk_run --config configs/ogbn_arxiv_secure.cfg --benchmark-mode warm --compute-backend cuda_hotspots --fast-msm on --parallel-fft on --fft-backend-upgrade on --fft-kernel-upgrade on --trace-layout-upgrade on --fast-verify-pairing on --export-tag cuda_formal
```

The GPU path fails if CUDA support or runtime availability is missing; it is not silently counted as a CPU run.

## Export Paper Results

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

The exporter hard-fails if any run is not verified, not full-dataset, not secure-profile compliant, missing required timing fields, has `private_plain_value_count != 0`, has `proof_payload_audit_status != passed`, has `verifier_reads_witness != false`, or lacks reference/tamper pass flags.

## Tests And Security Scan

```bash
cmake --build build --target gatzk_tests -j
./build/gatzk_tests
python3 tools/security/static_scan_no_tau.py
```

`proof_size_bytes` includes transmitted proof metadata, commitments, quotient commitments, opening proofs, claimed opening values, public outputs, and proof block tags. It does not include SRS files, the manifest file body, local caches, raw witness, checkpoint files, datasets, logs, or benchmark report attachments.
