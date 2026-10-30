# Formal GAT timing and proof-size audit

This report checks the three completed cells in retained formal session `6202ad2d504f4208a70ac380c3a04525`. They report `VERIFY_OK`; their result records bind them to the historical GPU binary SHA-256 `07b7eca4d5a58c6bd1bedda5850ce0c805e3121050b6565cad4279a85b9130d0` and source tree SHA-256 `3674e441c32f5d78ffd24a931f8104e99e46536e00d7b877e7e0a0f777a7d980`. These are measurements from that build, not from later source changes.

| Dataset | Commitment (ms) | Prove core (ms) | Prove command wall (ms) | Verify core (ms) | Verify phase wall (ms) | `proof.json` (bytes) | Raw transcript (bytes) | Public instance count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 14664.965575 | 3927509.427623 | 4525964.880641 | 13206.411477 | 586233.921 | 82462 | 40736 | 3928761 |
| CiteSeer | 57603.621338 | 9009979.412674999 | 11438596.4909 | 41967.028942 | 2408752.237 | 46560 | 22784 | 12368037 |
| PubMed | 79475.26587100001 | 20944137.421901 | 23404861.126875 | 34372.960081000005 | 2345734.978 | 108128 | 53568 | 10154103 |

The core fields and outer phase fields measure different intervals. These three cells use the formal SHPLONK path with `QUERY_INSTANCE=false`. Each `native_proof_metrics.json` records `instance_commitment_count: 0`; its `commitment_time_ms` therefore measures phase-0 advice KZG commitments plus affine normalization only. The phase-0 advice commitment counts are 79 for Cora, 49 for CiteSeer, and 91 for PubMed. Public-instance values are absorbed into the proof transcript during Prove, so their processing belongs to the Prove core and is not counted by this commitment hook. `prove_time_ms` is the native create-proof residual plus authenticated proving-key materialization. `verify_time_ms` is proof decoding plus the native Halo2 verifier call. It excludes the formal CLI's evidence and resource authentication, artifact validation, public-instance reconstruction, SRS and verification-key loading, and settings loading. The complete `prove_command_wall_ms` and `phase_timings_ms.verify` values include command and preparation work around those core operations. Do not substitute one set for the other.

In these old runs, Verify core took 13.206–41.967 seconds while the runner's full Verify phase took 586.234–2408.752 seconds. The retained logs did not time every preparation substage, so the difference cannot be assigned to proof-file reading or any one validation step. New Verify stage logs were added in `src/execute.rs` to measure evidence/SRS authentication, external statement and resource checks, artifact-file authentication, circuit loading, public-instance reconstruction and proof decoding, SRS parameter loading, verification-key loading, and the native proof check. They will make later command-wall time attributable; they do not rewrite or reinterpret the historical values.

The small `proof.json` sizes do not indicate a truncated transcript. In each retained artifact, the complete compact-v2 envelope's `proof_transcript.byte_length` matches the decoded transcript length; its lowercase hex field has exactly twice that many characters. The recorded `proof_size_bytes` equals the on-disk `proof.json` size, and the artifact SHA-256 matches `result.json`. The compact envelope stores an `instances_ref` digest and count, not the full public-instance vector. The public statement and public output are separate inputs to Verify and are outside `proof_size_bytes`. A succinct KZG proof is not expected to scale with the millions of public-instance values as if they were serialized into the transcript.

The retained records and artifacts are directly inspectable here:

| Dataset | Result | Native proof metrics | Native Verify metrics | Complete proof artifact |
| --- | --- | --- | --- | --- |
| Cora | [result.json](../../../runs/formal/cora/result.json) | [native_proof_metrics.json](../../../runs/formal/cora/native_proof_metrics.json) | [native_verify_metrics.json](../../../runs/formal/cora/native_verify_metrics.json) | [proof.json](../../../runs/formal/cora/proof.json) |
| CiteSeer | [result.json](../../../runs/formal/citeseer/result.json) | [native_proof_metrics.json](../../../runs/formal/citeseer/native_proof_metrics.json) | [native_verify_metrics.json](../../../runs/formal/citeseer/native_verify_metrics.json) | [proof.json](../../../runs/formal/citeseer/proof.json) |
| PubMed | [result.json](../../../runs/formal/pubmed/result.json) | [native_proof_metrics.json](../../../runs/formal/pubmed/native_proof_metrics.json) | [native_verify_metrics.json](../../../runs/formal/pubmed/native_verify_metrics.json) | [proof.json](../../../runs/formal/pubmed/proof.json) |

This audit does not establish measurements for PPI or OGBN-Arxiv, and it does not measure the later optimized source. No formal experiment or independent Verify was run to produce this report.
