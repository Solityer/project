#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/third_party/mcl"

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "mcl already present at ${TARGET_DIR}"
  exit 0
fi

git clone --depth=1 https://github.com/herumi/mcl.git "${TARGET_DIR}"
echo "mcl fetched to ${TARGET_DIR}"
