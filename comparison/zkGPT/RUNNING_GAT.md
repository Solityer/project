# Running the zkGPT GAT comparison

Run commands from this directory.  The resource step authenticates only
`baseline-local/`; it does not read or copy resources from the main experiment.

```bash
python3 scripts/prepare_resources.py --check
./build.sh --cpu
python3 scripts/formal_preflight.py --backend cpu
python3 scripts/run_five_datasets.py --backend cpu
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

For the GPU matrix, use the explicit CUDA build and backend:

```bash
./build.sh --gpu
python3 scripts/formal_preflight.py --backend gpu
python3 scripts/run_five_datasets.py --backend gpu
python3 scripts/collect_results.py
python3 scripts/validate_final_results.py
```

Model registration is a separate native setup operation and may be run before
preflight for each dataset, for example:

```bash
python3 scripts/register_model.py --dataset cora --backend cpu
```

Preflight is intentionally fail-closed.  If it reports an unsupported
`masked_full` verifier, do not treat a build or smoke test as a formal result;
the runner will refuse to start proving until the native capability is
available.
