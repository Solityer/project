#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
config="${1:-}"
bundle="${ZKDL_CEREMONY_BUNDLE:-}"
blinder_seed="${ZKDL_MODEL_BLINDER_SEED_FILE:-}"
if [[ "${1:-}" == "--bundle" ]]; then
  bundle="${2:-}"
  shift 2
fi
if [[ "${1:-}" == "--model-blinder-seed" ]]; then
  blinder_seed="${2:-}"
  shift 2
fi
config="${1:-}"
if [[ -z "$config" || -z "$bundle" || -z "$blinder_seed" ]]; then
  echo "usage: $0 --bundle /path/to/zkdl-ceremony-bundle --model-blinder-seed /path/to/owner-seed configs/<dataset>_secure.cfg" >&2
  exit 2
fi
case "$config" in
  /*) config_path="$config" ;;
  *) config_path="$root/$config" ;;
esac
config_path="$(realpath -- "$config_path")"
case "$config_path" in
  "$root/configs/"*.cfg) ;;
  *) echo "setup-register: config must be under baseline-local configs/" >&2; exit 2 ;;
esac
python3 "$script_dir/prepare_form_resources.py" \
  --config "$config_path" \
  --ceremony-bundle "$bundle" \
  --model-blinder-seed-file "$blinder_seed" \
  --full-dataset
python3 "$script_dir/validate_resources.py" --config "$config_path"
echo "setup-register: resources authenticated; native zkDL registration is ready"
