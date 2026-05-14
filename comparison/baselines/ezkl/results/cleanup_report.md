# Cleanup Report

- Final run directories are regenerated with `--clean --run` for every dataset.
- The collector accepts only final strict CPU run directories under `comparison/baselines/ezkl/runs/<dataset>`.
- Debug/scratch/manual/old/bak/tmp/orig artifacts are not part of the final result set.
- Large binary/proof/key payloads remain reproducible run artifacts and are ignored by `.gitignore`; JSON manifests and audit files record their measured sizes and hashes where available.
