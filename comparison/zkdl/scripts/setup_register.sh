#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
root="$(cd -- "$script_dir/.." && pwd -P)"
config="${1:-}"
if [[ -z "$config" ]]; then
  echo "usage: $0 configs/<dataset>_secure.cfg" >&2
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
python3 "$script_dir/validate_resources.py" --config "$config_path"
echo "setup-register: resources authenticated; native zkDL registration is ready"
echo "setup-register: no SRS, registry, commitment, or blinder was generated"
