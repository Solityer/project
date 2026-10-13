# zkDL upstream provenance and local boundary

This file records what is and is not authenticated for the restored
baseline-local snapshot.  It is deliberately conservative: a filename that
looks like an upstream component is not evidence of its repository, release,
or commit.

## Source identity

- Workspace Git commit at the time this record was refreshed:
  `af33ba4517c478039edf6d7e25ddc8a831b1611f`.
- `comparison/zkdl` is not a separate Git repository; its source identity is
  therefore the workspace commit plus the working-tree hashes below.
- The official upstream is `https://github.com/SafeAILab/zkDL`, branch `main`,
  commit `b40427cabbd7e1e8ddc7fde8b560b64ff874c7f9`.  This identity was
  checked with `git ls-remote` and a depth-one clone on 2026-09-13.
- The upstream README advertises v1.0.0, but the repository has no matching
  tag and its GitHub Releases page has no releases or release assets.  The
  upstream tree contains no GATZKG1C frames, IPA G/H/U basis, SRS ceremony
  transcript/seed, or authoritative digest for such material.  Consequently
  formal resource authenticity is established by the local trusted setup and authenticated manifests described below, rather than an upstream ceremony artifact.
- The copied dataset/checkpoint origin is also not encoded in the restored
  baseline.  `asset_manifest.py` authenticates the bytes present under this
  baseline root, but it cannot prove where an earlier copy was obtained.

## Working-tree hashes of official-core paths

These are historical working-tree SHA-256 hashes recorded when the upstream snapshot was restored, not current build identities or claims about an upstream release. Formal builds generate a fresh `build_manifest.json` covering current native sources, adapter, scripts, quantization configuration, pinned blst sources and binaries. That manifest is authenticated through each run asset manifest and public statement.

```text
256975986f1076df731d32dd3dc80442932699b55c42f0b7e2410e748c8df43e  bls12-381.cuh
f8ee3ec70433d856e433f2023232c5bc12a053ba65671024916e379a61a0b5c7  fr-tensor.cuh
d11724cdd2311662f877359a330e9a1fac6020d6c8d35d2077b4ebef05aa8e04  g1-tensor.cuh
368c23d589396ebfae09c6d9dba7768f046719bf57ac15d4e2169cd07020734f  commitment.cuh
e1b5bb128c23c01c22aeccb92be18e50aa2cf5bf2d965653af58e07c482ef9b2  proof.cuh
29f808afc4b4ed161247b3140002a79daf39b76a90c412f5c084a15a1274b4ed  zkmatmul.cuh
d68d802a47f395d55e7fb301e5fb0c5d2c86ea4ffa20b257518fe4a10216ee7a  zkprod.cuh
12ee5563c3ccb60c6eddc5094057faf6fac2261c675fbeaf87d7886f8f9e38c2  zkrelu.cuh
57438ef2d2d32554c75f7df01242ef1952e9fee6abdd0421410f0c860499c94e  zkfc.cuh
747111e1cfc848a389f3a93b495c6fedb831b8c17f258b8f17181b161c008111  demo.cu
```

Reproduction:

```bash
cd /home/pzh/project/comparison/zkdl
sha256sum bls12-381.cuh fr-tensor.cuh g1-tensor.cuh commitment.cuh proof.cuh \
  zkmatmul.cuh zkprod.cuh zkrelu.cuh zkfc.cuh demo.cu
```

## Modification boundary

GAT-specific code belongs in `gat_adapter/`.  The baseline-local build adds
the adapter translation units and include/link rules, plus the local
`zkrelation-proof.cuh` relation-engine adaptation.  Its guarded folded-state
cache and native consuming-prover path change temporary resource ownership;
they preserve the existing relation, transcript, and serialized proof format.
Configurations, copied assets, schemas, scripts, provenance, and test-only
fixtures remain under `comparison/zkdl/`.

The local `commitment.cuh`, `fr-tensor.cuh`, `g1-tensor.cuh`, and `proof.cuh`
files differ from the pinned upstream snapshot and are not treated as
authenticated upstream provenance.  The local `zkmatmul.cuh`, `zkprod.cuh`,
and `zkrelation-proof.cuh` files are baseline adaptation additions, not
upstream files.
Formal use must review these changes against the pinned upstream source
before declaring the baseline ready.

## Formal build and resource identity

The formal entry is `bash scripts/run_five.sh` with no arguments; `bash scripts/run_five_detached.sh` is the no-argument detached wrapper for the same runner. It uses the native CPU commitment, full-relation prover and independent verifier. It force-builds the baseline-local targets before preparing resources, records current source and binary hashes, and binds the build manifest through the run asset manifest to the proof statement. The collector rejects changed sources, binaries or authenticated resources.

The native SRS helper preserves independent nonzero OS-CSPRNG scalar multiplication on BLS12-381 and the GATZKG1C frame format. This is an honest local trusted setup with destruction of secret scalar and entropy buffers, not transparent setup. Public frames and their complete manifests are authenticated before use. The pinned baseline-local blst dependency supplies equivalent generic CPU group arithmetic and strict subgroup/canonical point decoding; it does not replace the zkDL proof protocol. Its upstream identity, license and imported source hashes are recorded in `third_party/BLST_PROVENANCE.txt` and the vendored dependency: blst v0.3.16, commit `e7f90de551e8df682f3cc99067d204d8b90d27ad`, archive SHA-256 `e04805b7d6ef9e1d89b7f511a5b86136c57b455d97924d7324da2305a864673f`.

Formal sessions freeze their binaries, configs, public input/metadata and resources independently; each dataset receives fresh registered-model blinding randomness. No registered-model multi-session privacy guarantee is claimed. Private checkpoint bytes and sidecar opening material belong to the prover authentication manifest and are excluded from independent verifier/collector access.

Formal results require actual native proof artifacts and independent verifier success for all five complete datasets. Development builds and static checks do not establish five-dataset runtime completion.

## 2026-10-12 bounded resource and CPU-path changes

The current local adapter adds an ANALYSIS_ONLY `gat_runner resource-plan` path and a Python report wrapper. The native path streams normalized graph edges and static-table element counts, validates the checkpoint manifest tensor index against the configured architecture, counts selector descriptor shapes without building full selector arrays, and emits typed proof, resident-memory, SRS, family-file and sorted-scratch bounds. The wrapper passes those records through the existing formal-phase host/disk equations. It does not prepare datasets, create SRS resources, run formal admission, produce or verify proofs, or collect results. Its output is capacity evidence only.

The descriptor builder interns exact-content selector maps and masks within one canonical relation-plan build. Index-map and mask storage use copy-on-write ownership; operation references and the expanded canonical wire encoding remain unchanged. The verification-side decoded proof descriptors are still counted as an unshared tree. The CPU commitment path uses a bounded persistent worker pool over fixed Pippenger chunks, immutable prepared affine basis data and exact receipts; ordered reduction preserves the serial group result. The Hyrax prover keeps the legacy expanded projective-G fold and uses paired BLST point arithmetic plus checked, chunked Montgomery inner products for its fold updates. Canonical input rejection precedes parallel dispatch, zero terms skip arithmetic only after that scan, and partial field sums reduce in fixed chunk order. The independent verifier's fold-coefficient derivation is unchanged. A sorted-prefix final-block boundary correction was also made. These are baseline-local implementation changes, not upstream restoration or equivalence claims; they do not change the zkDL relation, proof protocol, serialized descriptor bytes or validation conditions.

The bounded CPU differential fixture passed for n=65,553 (P=131,072) with full-width deterministic scalars and transcript-derived challenges: every LR point, folded G point, `a/b` value and final Ghat matched the generic reference, and a tiny opening independently verified while a tampered copy was rejected. This is synthetic CPU-path evidence only; it is not a full-dataset proof or a formal runtime claim.

For over-budget segmented Sumchecks, the prover now keeps the 32 GiB in-memory folded-state path and uses anonymous, preallocated direct-I/O folded-state files under the existing `.private_native` directory only when the state exceeds that budget. It preserves round-zero provider evaluation, field operations, pair order, Fiat–Shamir challenges and terminal values; the proof and verifier wire paths are unchanged. The ANALYSIS_ONLY resource profile counts the simultaneous old/next file extents, provider scans and spill I/O from native shape metrics. A bounded round-by-round differential fixture is included in source and remains to be built and run; no five-dataset formal run is claimed.

The historical hash block above remains a restoration-time record, not a current binary/source manifest. The current formal build regenerates and authenticates its own source and binary manifest. A bounded production build attempted during this update exceeded its 180-second limit, so no build-pass or formal-run claim is made here. No full dataset preparation, admission, proof, verification or result collection was run.
