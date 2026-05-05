# zkDL GAT Data Sources

This baseline reads full datasets and checkpoint bundles from the main repository as immutable inputs:

- `/home/pzh/project/data/cache/*`
- `/home/pzh/project/artifacts/checkpoints/*`
- `/home/pzh/project/configs/*`
- `/home/pzh/project/runs/*/warm/run_manifest.json`
- `/home/pzh/project/runs/*/cuda_formal/warm/run_manifest.json`

The baseline code does not include or execute `/home/pzh/project/src`, `/home/pzh/project/include`, or `/home/pzh/project/scripts`. Per-run manifests record every source path and SHA-256 digest used for the statement check.
