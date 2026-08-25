# Offline source lock

The local path dependencies below are copied from the exact cached checkouts
used by this baseline.  They are not generated replacements or mocks.

| Local path | Upstream | Commit | License files |
| --- | --- | --- | --- |
| `halo2/halo2_proofs`, `halo2/halo2_gadgets` | `https://github.com/zkonduit/halo2` | `01c88842679b4308e43ae5ed91c4183e861669bd` | `halo2/LICENSE-MIT`, `halo2/LICENSE-APACHE` |
| `halo2curves/halo2curves` | `https://github.com/privacy-scaling-explorations/halo2curves` | `b753a832e92d5c86c5c997327a9cf9de86a18851` | package metadata: MIT/Apache-2.0 |
| `icicle/icicle`, `icicle/rust` | `https://github.com/ingonyama-zk/icicle` | `012e00694f4cf399fe7a42d9cfbfa6cd7a60f876` | `icicle/LICENSE` |
| `mopro-msm` | `https://github.com/zkonduit/metal-msm-gpu-acceleration` | `be5f647b1a6c1a6ea9024390744a2b4d87f5d002` | `mopro-msm/LICENSE-MIT.txt`, `mopro-msm/LICENSE-APACHE.txt` |

The ICICLE source is deliberately path-patched as a complete exact commit
snapshot.  This removes the upstream `emir/gate_eval_2` branch selector from
the transitive Halo2 manifest while retaining the native ICICLE build scripts
and CPU/CUDA behavior.  The build identity includes a deterministic digest of
the local vendor tree and rejects branch, tag, non-revision query, absolute,
or escaping path selectors.
