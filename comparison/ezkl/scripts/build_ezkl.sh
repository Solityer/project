#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"

usage() {
  echo "usage: $0 --cpu|--gpu" >&2
  exit 2
}

[[ $# -eq 1 ]] || usage
backend=""
case "$1" in
  --cpu) backend=cpu ;;
  --gpu) backend=gpu ;;
  *) usage ;;
esac

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 - "$backend" <<'PY'
import sys
from formal_common import validate_resources

backend = sys.argv[1]
validate_resources()
PY

if [[ "$backend" == gpu ]]; then
  bash "$ROOT/setup-gpu.sh" --yes
  source "$ROOT/artifacts/build/icicle-cuda/env.sh"
  python3 -c "from formal_common import validate_gpu_environment; validate_gpu_environment()"
fi

features=(ezkl)
if [[ "$backend" == gpu ]]; then
  features+=(gpu-accelerated)
fi

echo "building official ezkl Cargo target: backend=$backend features=${features[*]}" >&2
cargo build --offline --release --target-dir "$ROOT/target/$backend" --features "$(IFS=,; echo "${features[*]}")"

binary="$ROOT/target/$backend/release/ezkl"
[[ -f "$binary" && ! -L "$binary" && -x "$binary" ]] || {
  echo "build prerequisite failed: official ezkl binary is missing or not executable: $binary" >&2
  exit 2
}

mkdir -p "$ROOT/artifacts/gat_adapter"
echo "generating Rust-owned formal CLI contract: $ROOT/artifacts/gat_adapter/formal_cli.json" >&2
"$binary" gat-contract --output artifacts/gat_adapter/formal_cli.json

export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"
python3 - "$backend" "$binary" <<'PY'
import sys
from formal_common import load_cli_contract, sha256

backend, binary_text = sys.argv[1:]
contract = load_cli_contract()
if contract["native_backend"] != "halo2-kzg-shplonk":
    raise SystemExit("post-build adapter contract has no native Halo2/KZG/SHPLONK backend")
if backend == "gpu" and contract["gpu_backend"] != "icicle-cuda":
    raise SystemExit("post-build adapter contract has no ICICLE CUDA backend")
print(
    f"verified adapter contract {contract['contract_id']} "
    f"relation={contract['relation_id']} binary_sha256={sha256(__import__('pathlib').Path(binary_text))}"
)
PY

mkdir -p "$ROOT/artifacts/build/$backend"
python3 - "$backend" "$binary" "$ROOT/artifacts/build/$backend/build_receipt.json" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from formal_common import CONTRACT, ROOT, load_cli_contract, native_backend_identity, resource_manifest_hash, sha256, source_tree_hash

backend, binary_text, receipt_text = sys.argv[1:]
binary = Path(binary_text)
receipt = Path(receipt_text)
contract = load_cli_contract()
def command_text(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()

payload = {
    "schema_version": 1,
    "baseline": "ezkl",
    "backend": backend,
    "features": ["ezkl"] + (["gpu-accelerated"] if backend == "gpu" else []),
    "binary_path": binary.relative_to(ROOT).as_posix(),
    "binary_sha256": sha256(binary),
    "source_tree_sha256": source_tree_hash(),
    "resource_manifest_sha256": resource_manifest_hash(),
    "adapter_contract_id": contract["contract_id"],
    "adapter_contract_sha256": sha256(CONTRACT),
    "relation_id": contract["relation_id"],
    "rustc": command_text("rustc", "--version"),
    "cargo": command_text("cargo", "--version"),
}
if backend == "gpu":
    payload.update(native_backend_identity())
receipt.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {receipt}")
PY
