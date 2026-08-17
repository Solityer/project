# zkDL baseline

This directory is reserved for a faithful zkDL-style GAT inference baseline. The governing protocol note is [GAT-ZKML-zkDL.md](GAT-ZKML-zkDL.md).

The previous full-chain implementation did not implement the exact relation or privacy profile and has been removed fail-closed. Its CPU and GPU paths, collectors, exporters, build entry points, and run entry points are all unsupported. This directory intentionally exposes no formal build, run, prove, verify, or export command.

[`configs/capabilities.json`](configs/capabilities.json) records the unsupported capabilities. The files retained under `src/core/` are unattached native arithmetic primitives only; they are not a formal proof implementation or runnable baseline.

`results/` and `runs/` are reserved for future formal execution and contain no current results.

代码和实验逻辑已整理，正式实验结果等待忠实实现后重新运行。
