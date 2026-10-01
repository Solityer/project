#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
[[ $# -eq 0 || ( $# -eq 1 && ( "$1" == -y || "$1" == --yes ) ) ]] || { echo "usage: $0 [--yes]" >&2; exit 2; }
CUDA_ARCH=${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')}
[[ "$CUDA_ARCH" =~ ^[0-9]+$ ]] || { echo "cannot determine CUDA architecture" >&2; exit 2; }
BUILD_DIR="$ROOT/artifacts/build/icicle-cuda"
cmake -S "$ROOT/vendor/icicle/icicle" -B "$BUILD_DIR" -DCURVE=bn254 -DCUDA_ARCH="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF
cmake --build "$BUILD_DIR" --target icicle_curve --parallel "${CARGO_BUILD_JOBS:-4}"
python3 - "$ROOT" "$CUDA_ARCH" <<'PYTHON'
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);directory=root/'artifacts/build/icicle-cuda'
files={}
for name in ['libingo_field_bn254.a','libingo_curve_bn254.a']:
 path=directory/'lib'/name
 files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
source_hash=hashlib.sha256()
for path in sorted((root/'vendor/icicle/icicle').rglob('*')):
 if path.is_file():
  relative=path.relative_to(root/'vendor/icicle/icicle').as_posix().encode();source_hash.update(len(relative).to_bytes(8,'big'));source_hash.update(relative);source_hash.update(hashlib.sha256(path.read_bytes()).hexdigest().encode())
manifest={'schema_version':1,'native_source_sha256':source_hash.hexdigest(),'halo2_commit':'f441c920be45f8f05d2c06a173d82e8885a5ed4d','frontend_commit':'5dfe006a0f1bc62ea82ca297709bbf3d22a2ca25','abi':'icicle-2.8.0-bn254-static-cuda','device':subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip(),'architecture':'sm_'+sys.argv[2],'runtime':subprocess.check_output(['nvcc','--version'],text=True).strip(),'libraries':files}
(directory/'native_backend_manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
environment={'ICICLE_BACKEND_INSTALL_DIR':str(directory),'ICICLE_NATIVE_LIB_DIR':str(directory/'lib'),'CUDA_ARCH':sys.argv[2],'ICICLE_NATIVE_MANIFEST_SHA256':hashlib.sha256((directory/'native_backend_manifest.json').read_bytes()).hexdigest()}
import shlex
(directory/'env.sh').write_text(''.join('export '+key+'='+shlex.quote(value)+'\n' for key,value in environment.items()))
print('native ICICLE CUDA backend prepared:',directory)
PYTHON
