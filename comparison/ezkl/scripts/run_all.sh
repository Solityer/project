#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# Formal matrix entry.  CUDA is included by default and fails closed if the
# distinct gpu-accelerated binary or runtime device is unavailable.
read -r -a devices <<< "${EZKL_DEVICES:-cpu cuda}"
if [[ -n "${EZKL_DATASETS:-}" ]]; then
  IFS=',' read -r -a datasets <<< "${EZKL_DATASETS}"
else
  datasets=(cora_full citeseer_full pubmed_full ppi_full_formal ogbn_arxiv_full)
fi

if [[ "${#devices[@]}" -eq 0 || "${#datasets[@]}" -eq 0 ]]; then
  echo "EZKL_DEVICES and EZKL_DATASETS must select at least one cell" >&2
  exit 2
fi
for device in "${devices[@]}"; do
  case "${device}" in
    cpu|cuda) ;;
    *) echo "unsupported EZKL device: ${device}" >&2; exit 2 ;;
  esac
done
for dataset in "${datasets[@]}"; do
  case "${dataset}" in
    cora_full|citeseer_full|pubmed_full|ppi_full_formal|ogbn_arxiv_full) ;;
    *) echo "unsupported EZKL dataset: ${dataset}" >&2; exit 2 ;;
  esac
done

contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    [[ "${item}" == "${needle}" ]] && return 0
  done
  return 1
}

full_matrix=true
[[ "${#devices[@]}" -eq 2 ]] || full_matrix=false
contains cpu "${devices[@]}" || full_matrix=false
contains cuda "${devices[@]}" || full_matrix=false
for required_dataset in cora_full citeseer_full pubmed_full ppi_full_formal ogbn_arxiv_full; do
  contains "${required_dataset}" "${datasets[@]}" || full_matrix=false
done

finalize="${EZKL_FINALIZE:-0}"
if [[ "${finalize}" != "0" && "${finalize}" != "1" ]]; then
  echo "EZKL_FINALIZE must be 0 or 1" >&2
  exit 2
fi

for dataset in "${datasets[@]}"; do
  for device in "${devices[@]}"; do
    python3 scripts/run_gat_ezkl_baseline.py \
      --dataset "${dataset}" --device "${device}" --clean --run
  done
done

collect_command=(python3 scripts/collect_results.py)
if [[ "${full_matrix}" == true || "${finalize}" == "1" ]]; then
  # Strict collection is only a matrix-level operation.  --finalize is an
  # explicit request and checks all ten canonical cells even after a partial
  # run selection.
  collect_command+=(--datasets cora_full citeseer_full pubmed_full ppi_full_formal ogbn_arxiv_full)
  collect_command+=(--devices cpu cuda)
else
  collect_command+=(--datasets "${datasets[@]}" --devices "${devices[@]}" --allow-incomplete)
fi
"${collect_command[@]}"

if [[ "${full_matrix}" == true || "${finalize}" == "1" ]]; then
  python3 scripts/validate_final_results.py
else
  echo "partial matrix collected; final validation deferred until all 10 cells exist (or EZKL_FINALIZE=1)"
fi
