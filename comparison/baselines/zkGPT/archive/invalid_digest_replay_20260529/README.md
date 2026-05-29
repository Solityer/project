# Invalid Digest Replay Artifacts

This archive contains the `results/` tree that existed before the real zkGPT
backend work on 2026-05-29.

The archived artifacts are invalid for official reporting because the runner used:

- SHA-256 payload digests as commitments.
- Residual checks as Lasso, matrix multiplication, and GKR proof blocks.
- Local replay of Python objects as the verifier.
- A CUDA digest helper rather than a GPU proof hot path.

They are preserved only for audit history.
