#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
backend=""
python_bin=""
cargo_bin=""
rustc_bin=""
cmake_bin=""
make_bin=""
nvcc_bin=""
cc_bin=""
cxx_bin=""
nvidia_smi_bin=""
selected_device_json=""
build_environment_json=""

usage() {
  echo "usage: $0 --cpu|--gpu --python ABS [GPU identity/tool arguments for --gpu]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) [[ -z "$backend" ]] || usage; backend=cpu; shift ;;
    --gpu) [[ -z "$backend" ]] || usage; backend=gpu; shift ;;
    --python|--cargo|--rustc|--cmake|--make|--nvcc|--cc|--cxx|--nvidia-smi|--selected-device-json|--build-environment-json)
      [[ $# -ge 2 ]] || usage
      key=$1
      value=$2
      case "$key" in
        --python) python_bin=$value ;;
        --cargo) cargo_bin=$value ;;
        --rustc) rustc_bin=$value ;;
        --cmake) cmake_bin=$value ;;
        --make) make_bin=$value ;;
        --nvcc) nvcc_bin=$value ;;
        --cc) cc_bin=$value ;;
        --cxx) cxx_bin=$value ;;
        --nvidia-smi) nvidia_smi_bin=$value ;;
        --selected-device-json) selected_device_json=$value ;;
        --build-environment-json) build_environment_json=$value ;;
      esac
      shift 2
      ;;
    *) usage ;;
  esac
done
[[ -n "$backend" && "$python_bin" = /* && -x "$python_bin" ]] || usage
export CARGO_HOME="$ROOT/.cargo_local"
unset RUSTUP_TOOLCHAIN CARGO_BUILD_TARGET CARGO_TARGET_DIR RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$backend" == gpu ]]; then
  for entry in "cargo_bin:$cargo_bin" "rustc_bin:$rustc_bin" "cmake_bin:$cmake_bin" "make_bin:$make_bin" "nvcc_bin:$nvcc_bin" "cc_bin:$cc_bin" "cxx_bin:$cxx_bin" "nvidia_smi_bin:$nvidia_smi_bin"; do
    name=${entry%%:*}
    path=${entry#*:}
    [[ "$path" = /* && -x "$path" ]] || { echo "formal $name must be an absolute executable path: $path" >&2; exit 2; }
  done
  [[ -n "$selected_device_json" && -n "$build_environment_json" ]] || { echo "formal GPU selected-device and build-environment identities are required" >&2; exit 2; }
  export CARGO_TARGET_DIR="$ROOT/target/gpu"
  export RUSTC="$rustc_bin"
  export CC="$cc_bin"
  export CXX="$cxx_bin"
  export CUDA_HOME="$(dirname "$(dirname "$nvcc_bin")")"
  export CUDACXX="$nvcc_bin"
  export CUDAHOSTCXX="$cxx_bin"
  export CMAKE_CUDA_COMPILER="$nvcc_bin"
  export CMAKE_GENERATOR="Unix Makefiles"
  export CMAKE_MAKE_PROGRAM="$make_bin"
  export CMAKE_C_COMPILER="$cc_bin"
  export CMAKE_CXX_COMPILER="$cxx_bin"
  echo "[ezkl-formal] GPU_BUILD_STAGE=prerequisite-authentication" >&2
fi

if [[ "$backend" == gpu ]]; then
  set +e
  "$python_bin" - <<'PY'
from formal_common import ContractError, validate_gpu_environment
try:
    validate_gpu_environment()
except ContractError as exc:
    print(f"native ICICLE prerequisites need setup: {exc}")
    raise SystemExit(73)
PY
  backend_status=$?
  set -e
  if [[ $backend_status -ne 0 ]]; then
    echo "[ezkl-formal] GPU_BUILD_STAGE=prepare-native-cuda-backend" >&2
    bash "$ROOT/setup-gpu.sh" \
      --yes \
      --python "$python_bin" \
      --nvcc "$nvcc_bin" \
      --cmake "$cmake_bin" \
      --make "$make_bin" \
      --cc "$cc_bin" \
      --cxx "$cxx_bin" \
      --nvidia-smi "$nvidia_smi_bin" \
      --selected-device-json "$selected_device_json"
  fi
  source "$ROOT/artifacts/build/icicle-cuda/env.sh"
  "$python_bin" - <<'PY'
from formal_common import validate_gpu_environment
validate_gpu_environment()
PY
fi

if [[ "$backend" == gpu ]]; then
  features=(formal-gpu)
  default_feature_mode=disabled
else
  features=(ezkl)
  default_feature_mode=enabled
fi

echo "[ezkl-formal] BUILD_STAGE=rust-release-build backend=$backend default_features=$default_feature_mode features=${features[*]}" >&2
if [[ "$backend" == gpu ]]; then
  "$cargo_bin" build --offline --release --no-default-features --target-dir "$ROOT/target/gpu" --features "${features[0]}"
else
  cargo_bin=$(command -v cargo)
  [[ "$cargo_bin" = /* && -x "$cargo_bin" ]] || { echo "cargo is not an executable absolute path" >&2; exit 2; }
  "$cargo_bin" build --offline --release --target-dir "$ROOT/target/cpu" --features "$(IFS=,; echo "${features[*]}")"
fi

binary="$ROOT/target/$backend/release/ezkl"
[[ -f "$binary" && ! -L "$binary" && -x "$binary" ]] || {
  echo "build prerequisite failed: official ezkl binary is missing or not executable: $binary" >&2
  exit 2
}

mkdir -p "$ROOT/artifacts/gat_adapter"
echo "[ezkl-formal] BUILD_STAGE=verify-native-cli-contract binary=$binary" >&2
"$binary" gat-contract --output artifacts/gat_adapter/formal_cli.json

if [[ "$backend" == gpu ]]; then
  "$python_bin" -u - "$backend" "$binary" "$ROOT/artifacts/build/$backend/build_receipt.json" "$selected_device_json" "$build_environment_json" <<'PY'
import json
import sys
from pathlib import Path

from formal_common import CONTRACT, FORMAL_GPU_FEATURES, ROOT, canonical_json_digest, load_cli_contract, native_backend_identity, resource_manifest_hash, sha256, source_tree_hash

backend, binary_text, receipt_text, selected_text, environment_text = sys.argv[1:]
binary = Path(binary_text)
receipt = Path(receipt_text)
selected = json.loads(selected_text)
environment = json.loads(environment_text)
environment_digest = environment.get("build_environment_sha256")
without_digest = dict(environment)
without_digest.pop("build_environment_sha256", None)
if environment_digest != canonical_json_digest(without_digest):
    raise SystemExit("formal GPU build environment digest is invalid")
if environment.get("selected_cuda_device") != selected or environment.get("features") != list(FORMAL_GPU_FEATURES):
    raise SystemExit("formal GPU build environment does not match selected device/features")
contract = load_cli_contract()
native = native_backend_identity()
if native["selected_cuda_device"] != selected:
    raise SystemExit("native ICICLE setup identity does not match the selected device")
payload = {
    "schema_version": 2,
    "baseline": "ezkl",
    "backend": backend,
    "features": list(FORMAL_GPU_FEATURES),
    "binary_path": binary.relative_to(ROOT).as_posix(),
    "binary_sha256": sha256(binary),
    "source_tree_sha256": source_tree_hash(),
    "resource_manifest_sha256": resource_manifest_hash(),
    "adapter_contract_id": contract["contract_id"],
    "adapter_contract_sha256": sha256(CONTRACT),
    "relation_id": contract["relation_id"],
    "rustc": environment["rustc"],
    "cargo": environment["cargo"],
    "build_environment": environment,
    "build_environment_sha256": environment_digest,
    **native,
}
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"[ezkl-formal] wrote current GPU build receipt {receipt}")
PY
else
  "$python_bin" -u - "$backend" "$binary" "$ROOT/artifacts/build/$backend/build_receipt.json" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from formal_common import CONTRACT, ROOT, load_cli_contract, resource_manifest_hash, sha256, source_tree_hash

backend, binary_text, receipt_text = sys.argv[1:]
binary = Path(binary_text)
receipt = Path(receipt_text)
contract = load_cli_contract()
payload = {
    "schema_version": 1,
    "baseline": "ezkl",
    "backend": backend,
    "features": ["ezkl"],
    "binary_path": binary.relative_to(ROOT).as_posix(),
    "binary_sha256": sha256(binary),
    "source_tree_sha256": source_tree_hash(),
    "resource_manifest_sha256": resource_manifest_hash(),
    "adapter_contract_id": contract["contract_id"],
    "adapter_contract_sha256": sha256(CONTRACT),
    "relation_id": contract["relation_id"],
    "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
    "cargo": subprocess.check_output(["cargo", "--version"], text=True).strip(),
}
receipt.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {receipt}")
PY
fi
echo "[ezkl-formal] BUILD_STAGE=complete backend=$backend binary=$binary" >&2
