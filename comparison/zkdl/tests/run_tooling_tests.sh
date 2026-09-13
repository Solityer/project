#!/usr/bin/env bash
set -euo pipefail

root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 -m py_compile "$root"/scripts/*.py "$root"/tests/test_tooling.py
bash -n "$root"/scripts/*.sh "$root"/tests/run_tooling_tests.sh
python3 "$root"/tests/test_tooling.py
