# Fixed-Point Rescale Diagnostic

Generated: 2026-06-27T09:59:43Z

Scope: main secure experiment only. `comparison/**` was not used.

## Result

OGBN-arxiv LReLU lookup overflow is reproducible under the current raw fixed-point projection encoding. A diagnostic-only rescale probe shows the overflow collapses to the existing large-graph table capacity if each fixed-point product sum is rounded back to the configured scale 16 after multiplication.

This probe is not retained in the proving path. `MAIN_EXPERIMENT_INVARIANTS.md` forbids changing proof relation, verifier equation, transcript order, proof schema, and quantization semantics without explicit approval.

## Evidence

Current raw encoding:

```text
label,index,edge_count,unique_s,min_s_raw,max_s_raw,span_s_raw,min_z_raw,max_z_raw
hidden,0,1335586,18816,-4578405,108122576,112700982,-915681,108122576
output,0,1335586,32500,-44524211,20775666,65299878,-8904842,20775666
```

Diagnostic rescale of each product sum by scale 16:

```text
label,index,edge_count,unique_s,min_s_raw,max_s_raw,span_s_raw,min_z_raw,max_z_raw
hidden,0,1335586,626,-17885,422354,440240,-3577,422354
output,0,1335586,1013,-173924,81155,255080,-34785,81155
```

Files:

- `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_current_raw_20260627T095943Z.csv`
- `results/performance_engineering/diagnostics/ogbn_lrelu_range_diag_rescale_products_20260627T095943Z.csv`

## Interpretation

The current raw projection path introduced in `5c70428` computes fixed-point dot products as field sums of raw products. With scale 16 inputs and weights, the product sum is scale 256, but the LReLU table decodes raw inputs at scale 16. That makes OGBN attention scores appear roughly one scale factor too large and causes the contiguous table span to exceed the fixed edge Lagrange domain.

Changing the proof path to rescale products would require synchronizing the binding quotient equations that currently enforce `Acc(omega)-Acc = a*b`. That would modify verifier equations and proof relation, so it is not an allowed retained optimization under the current invariants.

## Status

No final matrix row was produced from this probe. No paper result was exported.
