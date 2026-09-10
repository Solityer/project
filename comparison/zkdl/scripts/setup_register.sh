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
srs_rel="$(awk -F= '$1 ~ /^[[:space:]]*srs_path[[:space:]]*$/ {gsub(/[[:space:]]/,"",$2); print $2}' "$config_path")"
if [[ -z "$srs_rel" ]]; then
  echo "setup-register: srs_path is missing" >&2
  exit 3
fi
case "$srs_rel" in
  /*|*..*) echo "setup-register: SRS path must be baseline-local" >&2; exit 3 ;;
esac
srs="$root/$srs_rel"
if [[ ! -f "$srs" ]]; then
  echo "setup-register: authenticated SRS artifact is absent: $srs" >&2
  echo "setup-register: refusing to generate or register an untrusted replacement" >&2
  exit 3
fi
echo "setup-register: official SRS authentication/registry API is not exposed by restored core" >&2
exit 3
