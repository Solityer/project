# zkDL baseline-local resources

The formal runner consumes only resources below this directory and the
baseline's generated `artifacts/` directory.  It never falls back to the main
experiment's `data/` or `artifacts/` paths.

Before a user's full-data run, stage each dataset cache and checkpoint bundle
with:

```bash
python3 scripts/prepare_resources.py --source-root /path/to/project
```

The source root is used only by this explicit preparation command.  The formal
runner subsequently checks the authenticated `resource_manifest.json` and
requires the staged files to exist under `resources/datasets/` and
`resources/models/`.

The PP is generated once by the baseline-local setup step and stored at
`artifacts/setup/pp.bin`; it is never imported from another baseline.
