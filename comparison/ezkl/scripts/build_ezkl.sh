#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
cd "$ROOT"
backend=""
python_bin=""
cargo_bin=""
rustc_bin=""
rustdoc_bin=""
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
    --python|--cargo|--rustc|--rustdoc|--cmake|--make|--nvcc|--cc|--cxx|--nvidia-smi|--selected-device-json|--build-environment-json)
      [[ $# -ge 2 ]] || usage
      key=$1
      value=$2
      case "$key" in
        --python) python_bin=$value ;;
        --cargo) cargo_bin=$value ;;
        --rustc) rustc_bin=$value ;;
        --rustdoc) rustdoc_bin=$value ;;
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
for entry in "cargo_bin:$cargo_bin" "rustc_bin:$rustc_bin" "rustdoc_bin:$rustdoc_bin"; do
  name=${entry%%:*}
  path=${entry#*:}
  [[ "$path" = /* && -x "$path" ]] || { echo "formal $name must be an absolute executable path: $path" >&2; exit 2; }
done
export CARGO_HOME="$ROOT/.cargo_local"
export CARGO_BUILD_JOBS=2
export RUSTUP_TOOLCHAIN=nightly-2025-12-01
export RUSTC="$rustc_bin"
export RUSTDOC="$rustdoc_bin"
unset CARGO_BUILD_TARGET CARGO_TARGET_DIR RUSTFLAGS CARGO_ENCODED_RUSTFLAGS
export PYTHONPATH="$ROOT/scripts${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$backend" == gpu ]]; then
  for entry in "cmake_bin:$cmake_bin" "make_bin:$make_bin" "nvcc_bin:$nvcc_bin" "cc_bin:$cc_bin" "cxx_bin:$cxx_bin" "nvidia_smi_bin:$nvidia_smi_bin"; do
    name=${entry%%:*}
    path=${entry#*:}
    [[ "$path" = /* && -x "$path" ]] || { echo "formal $name must be an absolute executable path: $path" >&2; exit 2; }
  done
  [[ -n "$selected_device_json" && -n "$build_environment_json" ]] || { echo "formal GPU selected-device and build-environment identities are required" >&2; exit 2; }
  export CARGO_TARGET_DIR="$ROOT/target/gpu"
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
  IFS=$'\t' read -r selected_device_uuid selected_device_arch < <("$python_bin" - "$selected_device_json" <<'PY'
import json
import sys
device = json.loads(sys.argv[1])
expected = {"index", "uuid", "name", "driver_version", "compute_capability", "logical_device_id"}
if not isinstance(device, dict) or set(device) != expected or device.get("logical_device_id") != 0:
    raise SystemExit("formal selected physical CUDA device contract is invalid")
if not isinstance(device.get("uuid"), str) or not device["uuid"] or not isinstance(device.get("compute_capability"), str):
    raise SystemExit("formal selected physical CUDA device contract is incomplete")
print(device["uuid"] + "\t" + device["compute_capability"].replace(".", ""))
PY
  )
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$selected_device_uuid"
  export ICICLE_DEVICE_ID=0
  export CUDA_ARCH="$selected_device_arch"
  export EZKL_FORMAL_SELECTED_CUDA_DEVICE="$selected_device_json"
fi

set_native_gpu_environment() {
  local native_dir="$ROOT/artifacts/build/icicle-cuda"
  local manifest="$native_dir/native_backend_manifest.json"
  export ICICLE_BACKEND_INSTALL_DIR="$native_dir"
  export ICICLE_NATIVE_LIB_DIR="$native_dir/lib"
  if [[ -f "$manifest" && ! -L "$manifest" ]]; then
    export ICICLE_NATIVE_MANIFEST_SHA256=$("$python_bin" - "$manifest" <<'PY'
import hashlib
import sys
from pathlib import Path
print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
PY
    )
  else
    unset ICICLE_NATIVE_MANIFEST_SHA256
  fi
}

# The isolated cargo home may contain deliberately shared source symlinks to
# ~/.cargo. After a crate is pruned there, a dangling source symlink prevents
# Cargo from extracting its locked, cached archive. Preserve such links in a
# quarantine and let Cargo unpack/checksum the official archive itself.
"$python_bin" - "$ROOT" "$CARGO_HOME" <<'PY'
import os
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
cargo_home = Path(sys.argv[2])
lock = tomllib.loads((root / "Cargo.lock").read_text())
registry_roots = list((cargo_home / "registry" / "src").glob("*"))
cache_roots = list((cargo_home / "registry" / "cache").glob("*"))
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
quarantine = cargo_home / "recovery-quarantine" / timestamp
moved = []
for package in lock.get("package", []):
    if not str(package.get("source", "")).startswith("registry+"):
        continue
    package_dir = f"{package['name']}-{package['version']}"
    archive_name = package_dir + ".crate"
    if not any((cache / archive_name).is_file() for cache in cache_roots):
        continue
    for registry_root in registry_roots:
        source = registry_root / package_dir
        if source.is_symlink() and not source.exists():
            quarantine.mkdir(parents=True, exist_ok=True)
            destination = quarantine / f"{registry_root.name}--{package_dir}"
            os.replace(source, destination)
            moved.append((source, destination))
if moved:
    print(f"[ezkl-formal] preserved {len(moved)} dangling baseline-local Cargo source links in {quarantine}", flush=True)
    for source, destination in moved:
        print(f"[ezkl-formal] Cargo cache recovery: {source.name} -> {destination}", flush=True)
PY

if [[ "$backend" == gpu ]]; then
  set_native_gpu_environment
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
  set_native_gpu_environment
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="$selected_device_uuid"
  export ICICLE_DEVICE_ID=0
  export CUDA_ARCH="$selected_device_arch"
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
  default_feature_mode=disabled
fi

echo "[ezkl-formal] BUILD_STAGE=rust-release-build backend=$backend default_features=$default_feature_mode features=${features[*]}" >&2
if [[ "$backend" == gpu ]]; then
  "$cargo_bin" build --offline --locked --release --no-default-features --target-dir "$ROOT/target/gpu" --features "${features[0]}"
else
  "$cargo_bin" build --offline --locked --release --no-default-features --target-dir "$ROOT/target/cpu" --features "$(IFS=,; echo "${features[*]}")"
fi

binary="$ROOT/target/$backend/release/ezkl"
[[ -f "$binary" && ! -L "$binary" && -x "$binary" ]] || {
  echo "build prerequisite failed: official ezkl binary is missing or not executable: $binary" >&2
  exit 2
}

mkdir -p "$ROOT/artifacts/gat_adapter"
echo "[ezkl-formal] BUILD_STAGE=verify-native-cli-contract binary=$binary" >&2
source_tree_sidecar=$(mktemp "$ROOT/artifacts/gat_adapter/native_source_tree.XXXXXXXX")
source_tree_sidecar_relative="artifacts/gat_adapter/${source_tree_sidecar##*/}"
echo "[ezkl-formal] BUILD_STAGE=verify-native-source-identity sidecar=$source_tree_sidecar" >&2
EZKL_WORKING_DIR="$ROOT" "$binary" gat-contract \
  --output artifacts/gat_adapter/formal_cli.json \
  --source-tree-sha256-output "$source_tree_sidecar_relative"

if [[ "$backend" == gpu ]]; then
"$python_bin" -u - "$backend" "$binary" "$ROOT/artifacts/build/$backend/build_receipt.json" "$selected_device_json" "$build_environment_json" "$rustc_bin" "$cargo_bin" "$rustdoc_bin" "$source_tree_sidecar" <<'PY'
import json
import sys
from pathlib import Path

from formal_common import CONTRACT, FORMAL_GPU_FEATURES, ROOT, canonical_json_digest, load_cli_contract, native_backend_identity, resource_manifest_hash, sha256, verify_native_source_tree_hash

backend, binary_text, receipt_text, selected_text, environment_text, rustc_path, cargo_path, rustdoc_path, source_tree_sidecar_text = sys.argv[1:]
binary = Path(binary_text)
receipt = Path(receipt_text)
source_tree_sha256 = verify_native_source_tree_hash(Path(source_tree_sidecar_text))
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
    "source_tree_sha256": source_tree_sha256,
    "resource_manifest_sha256": resource_manifest_hash(),
    "adapter_contract_id": contract["contract_id"],
    "adapter_contract_sha256": sha256(CONTRACT),
    "relation_id": contract["relation_id"],
    "rustc_path": rustc_path,
    "rustc": environment["rustc"],
    "cargo_path": cargo_path,
    "cargo": environment["cargo"],
    "rustdoc_path": rustdoc_path,
    "rustdoc": environment["rustdoc"],
    "build_environment": environment,
    "build_environment_sha256": environment_digest,
    **native,
}
receipt.parent.mkdir(parents=True, exist_ok=True)
receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(f"[ezkl-formal] wrote current GPU build receipt {receipt}")
PY
else
  "$python_bin" -u - "$backend" "$binary" "$ROOT/artifacts/build/$backend/build_receipt.json" "$rustc_bin" "$cargo_bin" "$rustdoc_bin" "$source_tree_sidecar" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from formal_common import CONTRACT, ROOT, load_cli_contract, resource_manifest_hash, sha256, verify_native_source_tree_hash

backend, binary_text, receipt_text, rustc_path, cargo_path, rustdoc_path, source_tree_sidecar_text = sys.argv[1:]
binary = Path(binary_text)
receipt = Path(receipt_text)
source_tree_sha256 = verify_native_source_tree_hash(Path(source_tree_sidecar_text))
contract = load_cli_contract()
payload = {
    "schema_version": 2,
    "baseline": "ezkl",
    "backend": backend,
    "features": ["ezkl"],
    "binary_path": binary.relative_to(ROOT).as_posix(),
    "binary_sha256": sha256(binary),
    "source_tree_sha256": source_tree_sha256,
    "resource_manifest_sha256": resource_manifest_hash(),
    "adapter_contract_id": contract["contract_id"],
    "adapter_contract_sha256": sha256(CONTRACT),
    "relation_id": contract["relation_id"],
    "rustc_path": rustc_path,
    "rustc": subprocess.check_output([rustc_path, "--version"], text=True).strip(),
    "cargo_path": cargo_path,
    "cargo": subprocess.check_output([cargo_path, "--version"], text=True).strip(),
    "rustdoc_path": rustdoc_path,
    "rustdoc": subprocess.check_output([rustdoc_path, "--version"], text=True).strip(),
}
receipt.parent.mkdir(parents=True, exist_ok=True)
receipt.write_text(json.dumps(payload, indent=2) + "\n")
print(f"wrote {receipt}")
PY
fi
echo "[ezkl-formal] BUILD_STAGE=complete backend=$backend binary=$binary" >&2
