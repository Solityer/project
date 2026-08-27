# Offline source lock

The local path dependencies below are copied from the exact cached checkouts
used by this baseline.  They are not generated replacements or mocks.

| Local path | Upstream | Commit | Upstream tree | License files |
| --- | --- | --- | --- | --- |
| `halo2/halo2_proofs`, `halo2/halo2_gadgets` | `https://github.com/zkonduit/halo2` | `01c88842679b4308e43ae5ed91c4183e861669bd` | `not available in the cached checkout` | `halo2/LICENSE-MIT`, `halo2/LICENSE-APACHE` |
| `halo2curves/halo2curves` | `https://github.com/privacy-scaling-explorations/halo2curves` | `b753a832e92d5c86c5c997327a9cf9de86a18851` | `1ffadfcc2dc955367343b06b5f0b155d8060dacd` | `halo2curves/halo2curves/LICENSE-MIT`, `halo2curves/halo2curves/LICENSE-APACHE` |
| `icicle/icicle`, `icicle/rust` | `https://github.com/ingonyama-zk/icicle` | `012e00694f4cf399fe7a42d9cfbfa6cd7a60f876` | `not available in the cached checkout` | `icicle/LICENSE` |
| `mopro-msm` | `https://github.com/zkonduit/metal-msm-gpu-acceleration` | `be5f647b1a6c1a6ea9024390744a2b4d87f5d002` | `not available in the cached checkout` | `mopro-msm/LICENSE-MIT.txt`, `mopro-msm/LICENSE-APACHE.txt` |
| `poseidon` | `https://github.com/privacy-scaling-explorations/poseidon` | `807f8f555313f726ca03bdf941f798098f488ba4` | `bb7dcb779c64f878dcf3ac2eabc8264ef4082ce3` | `poseidon/LICENSE-MIT`, `poseidon/LICENSE-APACHE` |

The `halo2curves` and `poseidon` entries above are ordinary files in this
baseline, not Git submodules.  Their recorded upstream commit and tree are
the provenance of the copied source.  The `poseidon/Cargo.toml` manifest has
one intentional baseline-local patch: its `halo2curves` dependency is a path
dependency to the adjacent copied tree, so offline Cargo cannot consult a
network Git source.  The local file contents are included in the formal source
and vendor digests emitted by the build and result manifests.

The ICICLE source is deliberately path-patched as a complete exact commit
snapshot.  This removes the upstream `emir/gate_eval_2` branch selector from
the transitive Halo2 manifest while retaining the native ICICLE build scripts
and CPU/CUDA behavior.  The build identity includes a deterministic digest of
the local vendor tree and rejects branch, tag, non-revision query, absolute,
or escaping path selectors.
