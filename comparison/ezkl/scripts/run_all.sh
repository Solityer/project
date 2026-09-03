#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONDONTWRITEBYTECODE=1

# This is the sole production entry point.  Partial selectors are deliberately
# rejected so a successful invocation always means the canonical 5x2 matrix.
if [[ -n "${EZKL_DEVICES:-}" || -n "${EZKL_DATASETS:-}" || -n "${EZKL_FINALIZE:-}" ]]; then
  echo "run_all.sh does not accept partial matrix selectors" >&2
  exit 2
fi
devices=(cpu cuda)
datasets=(cora_full citeseer_full pubmed_full ppi_full_formal ogbn_arxiv_full)

# Resource policy and build identity are part of the formal entry, before any
# cell can materialize a circuit or start a prover.
python3 scripts/prepare_k28_srs.py --ensure
python3 scripts/prepare_resources.py --check --backend cpu
python3 scripts/prepare_resources.py --check --backend cuda
python3 scripts/preflight.py --require-staged --backend cpu
python3 scripts/preflight.py --require-staged --backend cuda
EZKL_BUILD_GPU=1 scripts/build.sh

for dataset in "${datasets[@]}"; do
  for device in "${devices[@]}"; do
    python3 scripts/run_gat_ezkl_baseline.py \
      --dataset "${dataset}" --device "${device}" --clean --run
  done
done

python3 scripts/collect_results.py --datasets "${datasets[@]}" --devices cpu cuda
python3 scripts/validate_final_results.py
