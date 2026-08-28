# zkDL baseline-local resources

The formal runner consumes only resources below this directory and the
baseline's generated `artifacts/` directory. It never falls back to the main
experiment's `data/` or `artifacts/` paths. `resource_manifest.json` is an
immutable source policy only: it contains canonical relative paths and the PP
security/max-vector bounds, never a fake PP digest or placeholder `pp_id`.

Before a user's full-data run, create the canonical directories, place each
dataset cache and checkpoint bundle directly in this baseline, and check them:

```bash
python3 scripts/prepare_resources.py --init
# Place resources directly under resources/datasets/ and resources/models/.
python3 scripts/prepare_resources.py --check
python3 scripts/setup_resources.py
```

`prepare_resources.py` accepts no external path.  It only creates/checks the
baseline-local layout.  The setup command invokes the
native CPU binary, authenticates the baseline-local PP, hashes every local
cache/checkpoint file, and atomically writes
`staged_resource_manifest.json`. Every production preflight, prover,
collector, and validator requires that authenticated manifest; missing or
stale local resources fail closed.

The PP is generated once by the baseline-local setup step and stored at
`artifacts/setup/pp.bin`; it is never imported from another baseline. The
staged manifest is regenerated after any `--replace` import or PP replacement.
