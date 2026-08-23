# zkGPT formal experiment status

No formal CPU/GPU experiment is currently authorized from this directory.
`configs/formal_capabilities.json` is the authoritative status and
`scripts/formal_preflight.py` is the authoritative executable gate.

Previously generated tables from the retained Python/CUDA research graft are
not formal results.  That graft does not implement the protocol's native
Lasso, ML-friendly GKR, exact integer GAT relation, four-segment
`X|W|Aux|L` input binding, repaired final `com_L` opening, native BN254 Hyrax,
formal CUDA prover path, or the required `masked_full` multi-session privacy
extension.  It must not be used to populate comparison tables.

The only current reproduction commands are fail-closed checks:

```bash
python3 scripts/formal_preflight.py
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

The collector writes explicit blocked rows (`proof_generated=false`,
`verified=false`); the preflight and validator return code 2.  Formal run
commands remain the same only so automation fails safely until all blockers are
removed and the capability manifest is deliberately upgraded.
