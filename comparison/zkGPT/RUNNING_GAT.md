# Running the zkGPT GAT comparison

Start the complete five-dataset experiment in the background:

```bash
cd /home/pzh/project/comparison/zkGPT
./scripts/run_formal_detached.sh
```

To run only selected datasets, pass `--datasets` and their names. For example:

```bash
./scripts/run_formal_detached.sh --datasets ppi ogbn_arxiv
```

The launcher creates a fresh session directory, builds the CPU prover and verifier, runs resource and capacity admission, then processes the requested datasets serially. A dataset failure is recorded and later selected datasets are still attempted. Each successful cell produces a full-GAT proof and runs the independent native verifier once. Only an exact five-dataset run with five verified cells is reported as complete; a subset result always records `full_five_complete=false`. GPU is unsupported for the formal `masked_full` relation.

Follow the latest launch, or take one read-only snapshot:

```bash
./scripts/formal_progress.sh
./scripts/formal_progress.sh --once
```

Stop the latest launch safely, or specify its exact launch ID printed by the launcher:

```bash
python3 scripts/stop_formal.py
python3 scripts/stop_formal.py --launch-id <launch_id>
```

Session logs and aggregate results are under `results/formal/<session_id>/`; per-dataset proof artifacts, native results, and verifier reports are under `baseline-local/runs/formal/<session_id>/<dataset>/`. The detached console and status are under `baseline-local/formal-detached/<launch_id>/`.
