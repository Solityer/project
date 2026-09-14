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
if [[ -z "${ZKDL_CEREMONY_SEED_FILE:-}" || -z "${ZKDL_CEREMONY_TRANSCRIPT_FILE:-}" ]]; then
  echo "setup-register: BLOCKED: set ZKDL_CEREMONY_SEED_FILE and ZKDL_CEREMONY_TRANSCRIPT_FILE" >&2
  echo "setup-register: the ceremony transcript must contain seed_sha256=<sha256(seed)>" >&2
  exit 3
fi
python3 "$script_dir/prepare_form_resources.py" \
  --config "$config_path" \
  --ceremony-seed-file "$ZKDL_CEREMONY_SEED_FILE" \
  --ceremony-transcript-file "$ZKDL_CEREMONY_TRANSCRIPT_FILE" \
  --full-dataset
python3 "$script_dir/validate_resources.py" --config "$config_path"
echo "setup-register: resources authenticated; native zkDL registration is ready"
