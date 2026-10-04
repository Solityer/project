#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
python_bin=""
nvcc_path=""
cmake_path=""
make_path=""
cc_path=""
cxx_path=""
nvidia_smi_path=""
selected_device_json=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -y|--yes) shift ;;
    --python|--nvcc|--cmake|--make|--cc|--cxx|--nvidia-smi|--selected-device-json)
      [[ $# -ge 2 ]] || { echo "$1 requires a value" >&2; exit 2; }
      key=$1
      value=$2
      case "$key" in
        --python) python_bin=$value ;;
        --nvcc) nvcc_path=$value ;;
        --cmake) cmake_path=$value ;;
        --make) make_path=$value ;;
        --cc) cc_path=$value ;;
        --cxx) cxx_path=$value ;;
        --nvidia-smi) nvidia_smi_path=$value ;;
        --selected-device-json) selected_device_json=$value ;;
      esac
      shift 2
      ;;
    *) echo "usage: $0 --yes --python ABS --nvcc ABS --cmake ABS --cc ABS --cxx ABS --nvidia-smi ABS --selected-device-json JSON" >&2; exit 2 ;;
  esac
done
for entry in "python_bin:$python_bin" "nvcc_path:$nvcc_path" "cmake_path:$cmake_path" "make_path:$make_path" "cc_path:$cc_path" "cxx_path:$cxx_path" "nvidia_smi_path:$nvidia_smi_path"; do
  name=${entry%%:*}
  path=${entry#*:}
  [[ "$path" = /* && -x "$path" ]] || { echo "formal $name must be an absolute executable path: $path" >&2; exit 2; }
done
[[ -n "$selected_device_json" ]] || { echo "formal selected CUDA device identity is required" >&2; exit 2; }

device_fields=$("$python_bin" - "$nvidia_smi_path" "$selected_device_json" <<'PYTHON'
import csv
import io
import json
import subprocess
import sys

nvidia_smi, raw = sys.argv[1:]
device = json.loads(raw)
keys = {"index", "uuid", "name", "driver_version", "compute_capability", "logical_device_id"}
if not isinstance(device, dict) or set(device) != keys or device.get("logical_device_id") != 0:
    raise SystemExit("invalid selected CUDA device contract")
if device.get("index", -1) < 0 or any(not isinstance(device.get(key), str) or not device[key] for key in ("uuid", "name", "driver_version", "compute_capability")):
    raise SystemExit("incomplete selected CUDA device contract")
if __import__("os").environ.get("CUDA_VISIBLE_DEVICES") != device["uuid"] or __import__("os").environ.get("ICICLE_DEVICE_ID") != "0":
    raise SystemExit("formal CUDA_VISIBLE_DEVICES/ICICLE_DEVICE_ID do not match selected device")
probe = subprocess.run(
    [nvidia_smi, "--query-gpu=index,uuid,name,driver_version,compute_cap", "--format=csv,noheader,nounits"],
    check=True, capture_output=True, text=True, timeout=10,
)
rows = []
for row in csv.reader(io.StringIO(probe.stdout), skipinitialspace=True):
    fields = [field.strip() for field in row]
    if len(fields) != 5:
        raise SystemExit(f"invalid nvidia-smi device row: {row}")
    rows.append({"index": int(fields[0]), "uuid": fields[1], "name": fields[2], "driver_version": fields[3], "compute_capability": fields[4], "logical_device_id": 0})
live = next((entry for entry in rows if entry["uuid"] == device["uuid"]), None)
if live != device:
    raise SystemExit("selected physical CUDA device no longer matches the live device inventory")
print("\t".join((str(device["index"]), device["uuid"], device["name"], device["driver_version"], device["compute_capability"], "0")))
PYTHON
)
IFS=$'\t' read -r device_index device_uuid device_name driver_version compute_capability logical_device_id <<< "$device_fields"
detected_arch=${compute_capability//./}
[[ "$detected_arch" =~ ^[0-9]+$ && "$logical_device_id" == 0 ]] || { echo "cannot derive CUDA architecture/logical device from the selected physical GPU" >&2; exit 2; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$device_uuid"
export ICICLE_DEVICE_ID=0
export CUDA_ARCH="$detected_arch"
export CUDA_HOME="$(dirname "$(dirname "$nvcc_path")")"
export CUDACXX="$nvcc_path"
export CC="$cc_path"
export CXX="$cxx_path"
export CMAKE_CUDA_COMPILER="$nvcc_path"
export CMAKE_CUDA_ARCHITECTURES="$detected_arch"
export CMAKE_GENERATOR="Unix Makefiles"
export CMAKE_MAKE_PROGRAM="$make_path"
export CMAKE_C_COMPILER="$cc_path"
export CMAKE_CXX_COMPILER="$cxx_path"

BUILD_DIR="$ROOT/artifacts/build/icicle-cuda"
for path in "$ROOT/artifacts" "$ROOT/artifacts/build"; do
  [[ ! -L "$path" ]] || { echo "native ICICLE build path must not be symlinked: $path" >&2; exit 2; }
done
if [[ -e "$BUILD_DIR" || -L "$BUILD_DIR" ]]; then
  [[ -d "$BUILD_DIR" && ! -L "$BUILD_DIR" ]] || { echo "native ICICLE build directory must be a regular directory: $BUILD_DIR" >&2; exit 2; }
  echo "[ezkl-formal] GPU_SETUP_STAGE=reset-stale-native-build-directory path=$BUILD_DIR" >&2
  rm -rf -- "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"
echo "[ezkl-formal] GPU_SETUP_STAGE=configure selected_device=index:$device_index uuid:$device_uuid logical:0 architecture=sm_$detected_arch compiler=$nvcc_path" >&2
"$cmake_path" \
  -S "$ROOT/vendor/icicle/icicle" \
  -B "$BUILD_DIR" \
  -DCURVE=bn254 \
  -DCUDA_ARCH="$detected_arch" \
  -G "Unix Makefiles" \
  -DCMAKE_CUDA_COMPILER="$nvcc_path" \
  -DCMAKE_CUDA_ARCHITECTURES="$detected_arch" \
  -DCMAKE_MAKE_PROGRAM="$make_path" \
  -DCMAKE_C_COMPILER="$cc_path" \
  -DCMAKE_CXX_COMPILER="$cxx_path" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTS=OFF \
  -DBUILD_BENCHMARKS=OFF
echo "[ezkl-formal] GPU_SETUP_STAGE=compile selected_device_uuid=$device_uuid" >&2
"$cmake_path" --build "$BUILD_DIR" --target icicle_curve --parallel 4
echo "[ezkl-formal] GPU_SETUP_STAGE=record_identity selected_device_uuid=$device_uuid" >&2
"$python_bin" - "$ROOT" "$BUILD_DIR" "$nvcc_path" "$cmake_path" "$make_path" "$cc_path" "$cxx_path" "$nvidia_smi_path" "$device_index" "$device_uuid" "$device_name" "$driver_version" "$compute_capability" <<'PYTHON'
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

root, build_dir, nvcc, cmake, make, cc, cxx, nvidia_smi, index, uuid, name, driver, capability = sys.argv[1:]
root = Path(root)
directory = Path(build_dir)
files = {}
for native_library_name in ("libingo_field_bn254.a", "libingo_curve_bn254.a"):
    path = directory / "lib" / native_library_name
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    files[path.relative_to(root).as_posix()] = digest.hexdigest()
source_hash = hashlib.sha256()
for path in sorted((root / "vendor/icicle/icicle").rglob("*")):
    if path.is_file():
        relative = path.relative_to(root / "vendor/icicle/icicle").as_posix().encode()
        file_hash = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                file_hash.update(block)
        source_hash.update(len(relative).to_bytes(8, "big"))
        source_hash.update(relative)
        source_hash.update(file_hash.hexdigest().encode())

def tool_text(path, *args):
    result = subprocess.run([path, *args], check=True, capture_output=True, text=True, timeout=10)
    return result.stdout.strip()

selected = {
    "index": int(index),
    "uuid": uuid,
    "name": name,
    "driver_version": driver,
    "compute_capability": capability,
    "logical_device_id": 0,
}
manifest = {
    "schema_version": 2,
    "native_source_sha256": source_hash.hexdigest(),
    "halo2_commit": "f441c920be45f8f05d2c06a173d82e8885a5ed4d",
    "frontend_commit": "5dfe006a0f1bc62ea82ca297709bbf3d22a2ca25",
    "abi": "icicle-2.8.0-bn254-static-cuda",
    "device": name,
    "driver_version": driver,
    "selected_cuda_device": selected,
    "architecture": "sm_" + capability.replace(".", ""),
    "runtime": tool_text(nvcc, "--version"),
    "cuda_compiler_path": str(Path(nvcc).resolve()),
    "cuda_toolkit_root": str(Path(nvcc).resolve().parent.parent),
    "cmake_path": str(Path(cmake).resolve()),
    "cmake": tool_text(cmake, "--version").splitlines()[0],
    "cmake_generator": "Unix Makefiles",
    "make_path": str(Path(make).resolve()),
    "make": tool_text(make, "--version").splitlines()[0],
    "cc_path": str(Path(cc).resolve()),
    "cc": tool_text(cc, "--version").splitlines()[0],
    "cxx_path": str(Path(cxx).resolve()),
    "cxx": tool_text(cxx, "--version").splitlines()[0],
    "nvidia_smi_path": str(Path(nvidia_smi).resolve()),
    "nvidia_smi": tool_text(nvidia_smi, "--version").splitlines()[0],
    "libraries": files,
}
manifest_path = directory / "native_backend_manifest.json"
manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
environment = {
    "ICICLE_BACKEND_INSTALL_DIR": str(directory),
    "ICICLE_NATIVE_LIB_DIR": str(directory / "lib"),
    "ICICLE_NATIVE_MANIFEST_SHA256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "CUDA_ARCH": capability.replace(".", ""),
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    "CUDA_VISIBLE_DEVICES": uuid,
    "ICICLE_DEVICE_ID": "0",
}
(directory / "env.sh").write_text("".join("export " + key + "=" + shlex.quote(value) + "\n" for key, value in environment.items()))
print("[ezkl-formal] native ICICLE GPU backend identity recorded:", manifest_path)
PYTHON
