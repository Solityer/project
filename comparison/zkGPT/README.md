# zkGPT comparison baseline

This directory retains the upstream zkGPT C++ source, paper, license, and
submodule declaration as reference material. That source is not a formal GAT
experiment route.

## Current status

- Exact GAT relation: `unsupported_exact_relation`
- Required privacy profile: `unsupported_privacy_profile`
- CPU formal path: `unsupported`
- GPU formal path: `unsupported`

The removed GAT graft did not provide a complete native Lasso proof, exact
`SignedRoundDiv` constraints at every required site, or complete
commitment-bound GKR and matrix-multiplication proofs. Its row blinding was
deterministically derived instead of freshly sampled, and its GPU component
only reduced fields from a proof produced on the CPU. Those paths therefore
could not be represented as faithful CPU or GPU proving backends.

There are no formal build, run, prove, verify, collector, or exporter commands
in this baseline. No prior result is retained. Formal results must wait for a
faithful implementation and then be regenerated from scratch.

The declared `mcl` submodule is not populated in this checkout. It was not
downloaded or built in this cleanup, so the retained upstream reference source
remains pending formal-environment validation. This environment item does not
account for the unsupported protocol capabilities above.
