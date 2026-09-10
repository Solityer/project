# zkDL upstream provenance and local boundary

This file records what is and is not authenticated for the restored
baseline-local snapshot.  It is deliberately conservative: a filename that
looks like an upstream component is not evidence of its repository, release,
or commit.

## Source identity

- Workspace Git commit at the time this record was written:
  `07d35a2edf69085bc2047dcf138bf1768ba8e671`.
- `comparison/zkdl` is not a separate Git repository; its source identity is
  therefore the workspace commit plus the working-tree hashes below.
- The restored tree contains no authenticated upstream URL, release tag,
  archive digest, or upstream commit identifier.  The upstream project,
  version, and archive origin are consequently **UNCONFIRMED**.  This record
  does not invent them.
- The copied dataset/checkpoint origin is also not encoded in the restored
  baseline.  `asset_manifest.py` authenticates the bytes present under this
  baseline root, but it cannot prove where an earlier copy was obtained.

## Working-tree hashes of official-core paths

These are SHA-256 hashes of the files currently present in this working tree,
not claims about an upstream release.  Recompute them after any source change
with the command shown below.

```text
256975986f1076df731d32dd3dc80442932699b55c42f0b7e2410e748c8df43e  bls12-381.cuh
9af0b75d66ad00f55369ecd004513983ed6f65b3b57b39a48f106266e1c2cad0  fr-tensor.cuh
49d43e67bbbbf9ed13ae53d00b5f43dfd9ca76c28b43e9098b4f1a3cce0f4f16  g1-tensor.cuh
a32d28aac23638ffc729af932cd78d1a1dfbfcf620f0819dd18a6cfe853b9ef6  commitment.cuh
ce6f2bcc814a0ac6ac12db0ed77483b54cd7ad2719c53dc5e824297f94d2086f  proof.cuh
08caa51355a679db80c2ec62030f94b8291eb9fbcf0fc5814de53e30fec4afaa  zkmatmul.cuh
25dd312b2cd2f1461bffc0549a9f364c564b69d05b987f38b8db804362cf07e5  zkprod.cuh
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
only the adapter translation units and include/link rules.  Configurations,
copied assets, schemas, scripts, provenance, and test-only fixtures remain
under `comparison/zkdl/`.

The current workspace status also contains pre-existing uncommitted changes
to `commitment.cuh`, `fr-tensor.cuh`, `g1-tensor.cuh`, and `proof.cuh`.  They
were not modified while writing this provenance record and are not treated as
authenticated upstream provenance.  Formal use must review those changes
against the actual upstream source before declaring the baseline ready.

## Proof readiness

`gat_native_bridge.cu` fails closed because the restored core does not expose
the complete GAT table/range, sparse-route, FAC4DNN multi-view,
SignedRoundDiv, authenticated-SRS, and independent-verifier interfaces.  No
artifact in this tree should be called a formal verified proof until those
interfaces and source identities are supplied and checked.
