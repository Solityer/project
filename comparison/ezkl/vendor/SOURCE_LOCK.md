# Offline source lock

The local path dependencies below are copied from the exact cached checkouts
used by this baseline.  They are not generated replacements or mocks.

| Local path | Upstream | Commit | Upstream tree | License files |
| --- | --- | --- | --- | --- |
| `halo2/halo2_proofs`, `halo2/halo2_gadgets` | `https://github.com/zkonduit/halo2` | `01c88842679b4308e43ae5ed91c4183e861669bd` | `not available in the cached checkout` | `halo2/LICENSE-MIT`, `halo2/LICENSE-APACHE` |
| `halo2curves/halo2curves` | `https://github.com/privacy-scaling-explorations/halo2curves` | `b753a832e92d5c86c5c997327a9cf9de86a18851` | `1ffadfcc2dc955367343b06b5f0b155d8060dacd` | `halo2curves/halo2curves/LICENSE-MIT`, `halo2curves/halo2curves/LICENSE-APACHE` |
| `halo2wrong/{halo2wrong,maingate,integer,ecc}` | `https://github.com/scroll-tech/halo2wrong` | `07005dd59a76d64895f27f114141c1798aec8c9f` | `not available in the cached checkout` | `pinned checkout contains no license file and its package metadata omits a license` |
| `icicle/icicle`, `icicle/rust` | `https://github.com/ingonyama-zk/icicle` | `012e00694f4cf399fe7a42d9cfbfa6cd7a60f876` | `not available in the cached checkout` | `icicle/LICENSE` |
| `poseidon` | `https://github.com/privacy-scaling-explorations/poseidon` | `807f8f555313f726ca03bdf941f798098f488ba4` | `bb7dcb779c64f878dcf3ac2eabc8264ef4082ce3` | `poseidon/LICENSE-MIT`, `poseidon/LICENSE-APACHE` |
| `snark-verifier/snark-verifier` | `https://github.com/privacy-scaling-explorations/snark-verifier` | `508d96b20874d575bb489cb5f92f8bb116e9d44a` | `not available in the cached checkout` | `snark-verifier/LICENSE-MIT`, `snark-verifier/LICENSE-APACHE` |
| `taskflow` | `https://github.com/taskflow/taskflow` | `d8c49c64b4ee5015a3f1c0a42748fa7a2bf5529` (v3.8.0) | `not available in the cached checkout` | `taskflow/LICENSE` |

The `halo2curves` and `poseidon` entries above are ordinary files in this
baseline, not Git submodules.  Their recorded upstream commit and tree are
the provenance of the copied source.  The `poseidon/Cargo.toml` manifest has
one intentional baseline-local patch: its `halo2curves` dependency is a path
dependency to the adjacent copied tree, so offline Cargo cannot consult a
network Git source.  The local file contents are included in the formal source
and vendor digests emitted by the build and result manifests.

The ICICLE source is deliberately path-patched as a complete exact commit
snapshot.  The locked public snapshot does not contain `backend/cuda`; the
formal CUDA cell therefore requires the user to stage the four ABI-matched
ICICLE CUDA DSOs under the baseline-local `resources/icicle_cuda_backend`
manifest.  The runtime accepts only that exact authenticated bundle and never
searches `/opt`, `LD_LIBRARY_PATH`, or another project directory.  The build
identity and cell result bind the staged bundle's commit, ABI, relative paths,
sizes, and SHA-256 digests.  Taskflow is vendored as the exact v3.8.0 header
closure because the ICICLE CPU backend uses it; no CMake FetchContent or
network path is permitted.  The build identity includes a deterministic digest
of the explicit local vendor inventory and rejects branch, tag, non-revision
query, absolute, or escaping path selectors.

The inventory intentionally contains only the production dependency closure.
Upstream benchmark/example/test/regression trees and pure test-only modules
removed from the cached checkouts are not part of the formal source identity;
the baseline's contract tests remain under `tests/` and never supply formal
proof inputs.
