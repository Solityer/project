#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

datasets=(cora_full citeseer_full pubmed_full ppi_full_formal ogbn_arxiv_full)
for dataset in "${datasets[@]}"; do
  if [[ "${dataset}" == "ppi_full_formal" || "${dataset}" == "ogbn_arxiv_full" ]]; then
    resume_from="$(python3 - "${dataset}" <<'PY'
import json
import subprocess
import sys

dataset = sys.argv[1]
completed = subprocess.run(
    ["python3", "scripts/run_gat_ezkl_baseline.py", "--dataset", dataset, "--check-resume-state"],
    text=True,
    capture_output=True,
    check=True,
)
state = json.loads(completed.stdout)
print("prove" if state.get("setup_reusable") else "gen-settings")
PY
)"
    if [[ "${resume_from}" == "prove" ]]; then
      python3 scripts/run_gat_ezkl_baseline.py --dataset "${dataset}" --resume --resume-from prove --no-clean-existing-keys --run
    else
      python3 scripts/run_gat_ezkl_baseline.py --dataset "${dataset}" --clean --run
    fi
  else
    python3 scripts/run_gat_ezkl_baseline.py --dataset "${dataset}" --clean --run
  fi
done

python3 scripts/collect_results.py
