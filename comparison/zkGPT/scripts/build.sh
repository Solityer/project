#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
case "${1:---all}" in
  --cpu|--gpu|--all) ;;
  *) echo "usage: $0 [--cpu|--gpu|--all]" >&2; exit 2 ;;
esac
python3 scripts/formal_preflight.py --source-only
