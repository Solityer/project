#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
config="${1:-}"
backend="${2:-}"
run_dir="${3:-}"
if [[ -z "$config" || "$backend" != "cpu" ]]; then
    echo "usage: $0 configs/<dataset>_secure.cfg cpu [runs/<name>]" >&2
    exit 2
fi
if [[ "${ZKDL_ALLOW_FULL_DATASET_RUN:-0}" != "1" ]]; then
    echo "run-single: refusing full-dataset proving; set ZKDL_ALLOW_FULL_DATASET_RUN=1 after review" >&2
    exit 3
fi
case "$config" in
    /*) config_path="$config" ;;
    *) config_path="$root/$config" ;;
esac
config_path="$(realpath -- "$config_path")"
case "$config_path" in
    "$root/configs/"*.cfg) ;;
    *)
        echo "run-single: config must be under baseline-local configs/" >&2
        exit 2
        ;;
esac
if [[ -z "$run_dir" ]]; then
    mkdir -p "$root/runs"
    run_dir="$(mktemp -d "$root/runs/single_${backend}_XXXXXXXX")"
fi
case "$run_dir" in
    /*) run_path="$run_dir" ;;
    *) run_path="$root/$run_dir" ;;
esac
run_path="$(realpath -m -- "$run_path")"
case "$run_path" in
    "$root"/*) ;;
    *) echo "run-single: run directory escapes baseline root" >&2; exit 2 ;;
esac
if [[ -e "$run_path/proof.bin" || -e "$run_path/run.record" || -e "$run_path/statement.txt" ]]; then
    echo "run-single: run directory already contains formal artifacts; choose a fresh directory" >&2
    exit 2
fi
mkdir -p -- "$run_path"
python3 "$script_dir/formal_session.py" --session "$run_path" --config "$config_path"
config_path="$root/$(cat -- "$run_path/formal_config.path")"
cell_path="$root/$(cat -- "$run_path/formal_run.path")"
python3 "$script_dir/formal_provenance.py" --output "$run_path/build_manifest.json"
export ZKDL_BUILD_MANIFEST="$run_path/build_manifest.json"
"$script_dir/setup_register.sh" "$config_path"
"$script_dir/prepare_local.sh" "$config_path" "$cell_path"
exec "$run_path/build/gat_runner" prove --config "$config_path" --backend cpu --run-dir "$cell_path"
