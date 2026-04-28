# Code Cleanliness Audit

Created: 2026-04-28

## Strict Route Cleanup

- Final benchmark route is restricted to `ezkl_gat_native_compressed_csr_attention_full`.
- `configs/ezkl_routes.json` contains only the final route and explicitly lists forbidden fallback classes.
- `scripts/run_gat_ezkl_baseline.py` accepts only the final route for benchmark execution.
- The C++ exporter rejects `analysis-only` and non-final routes for this comparison.
- Stale ONNX/segment/output-binding route reports were deleted from `results/`.
- Adapter "placeholder" wording was removed; generated ONNX marker text is now a `blocked_artifact` marker only when relevant to non-final blocked exports.

## Grep Command

```bash
grep -R "TODO\\|FIXME\\|HACK\\|XXX\\|stub\\|dummy\\|fake\\|placeholder\\|temporary\\|debug-only\\|not implemented\\|unimplemented!\\|todo!\\|panic!(\"TODO" -n src cpp scripts configs || true
```

## Remaining Hits

| Count | Path | Classification | Strict route impact |
| ---: | --- | --- | --- |
| 4 | `src/bin/py_stub_gen.rs` | Upstream Python typing stub generator. | None. |
| 40 | `src/bindings/python.rs` | Upstream `pyo3_stub_gen` annotations and generated Python binding metadata. | None. |
| 1 | `src/circuit/modules/planner.rs` | Upstream generic planner TODO comment. | None. |
| 1 | `src/circuit/modules/poseidon/spec.rs` | Upstream generic unimplemented Poseidon spec branch. | None. |
| 40 | `src/circuit/ops/chip.rs` | Upstream dummy layout/configuration phase used for sizing. | None. |
| 4 | `src/circuit/ops/chip/einsum/layouts.rs` | Upstream dummy-region guard. | None. |
| 13 | `src/circuit/ops/chip/einsum/mod.rs` | Upstream dummy einsum config and dummy-phase comments. | None. |
| 163 | `src/circuit/ops/layouts.rs` | Upstream examples and dummy-region documentation for generic ops. | None. |
| 2 | `src/circuit/ops/mod.rs` | Upstream generic TODO/unimplemented op branch. | None. |
| 12 | `src/circuit/ops/region.rs` | Upstream dummy-region assignment machinery. | None. |
| 2 | `src/commands.rs` | Upstream clap placeholder styling and dummy SRS command help text. | None. |
| 1 | `src/eth.rs` | Upstream VKA digest TODO. | None. |
| 2 | `src/execute.rs` | Upstream registered-VKA lookup TODO comments. | None. |
| 5 | `src/graph/mod.rs` | Upstream placeholder/dummy public-input comments for generic graph loading. | None. |
| 14 | `src/graph/model.rs` | Upstream dummy layout phase used for generic circuit sizing. | None. |
| 1 | `src/graph/vars.rs` | Upstream dummy vars helper. | None. |
| 5 | `src/tensor/var.rs` | Upstream placeholder/dummy tensor helpers for generic layout/testing. | None. |

## Conclusion

No grep hits remain in `cpp/gat_ezkl_adapter/`, `scripts/`, `configs/`, or the strict native GAT route implementation. Remaining hits are upstream ezkl generic infrastructure and are not fake proof, dummy verify, mock setup, output-binding replacement, digest/folding route, identity route, subset/shard/sample route, or a precomputed-logits bypass.
