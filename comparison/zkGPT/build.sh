#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
backend="cpu"
build_type="Release"
build_dir="${ROOT_DIR}/build/cpu"
developer_tests=0

case "${1:-}" in
    --cpu)
        backend="cpu"
        build_dir="${ROOT_DIR}/build/cpu"
        ;;
    --cpu-tests)
        backend="cpu"
        build_dir="${ROOT_DIR}/build/cpu"
        developer_tests=1
        ;;
    --gpu)
        backend="gpu"
        build_dir="${ROOT_DIR}/build/gpu"
        ;;
    Debug|Release|RelWithDebInfo|MinSizeRel)
        build_type="$1"
        build_dir="${ROOT_DIR}/cmake-build-${1,,}"
        ;;
    "")
        ;;
    --help|-h)
        echo "Usage: $0 --cpu|--gpu|--cpu-tests"
        echo "       --cpu-tests explicitly builds the isolated development proof check"
        echo "       $0 [Debug|Release|RelWithDebInfo|MinSizeRel] (upstream CPU compatibility)"
        exit 0
        ;;
    *)
        echo "error: expected --cpu or --gpu" >&2
        exit 2
        ;;
esac

if [[ "${backend}" == gpu ]]; then
    cuda_option=ON
else
    cuda_option=OFF
fi

cmake_args=(
    -S "${ROOT_DIR}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE="${build_type}"
    -DZKGPT_BUILD_TESTS=OFF
    -DZKGPT_ENABLE_CUDA="${cuda_option}"
    -DMCL_TEST_WITH_GMP=OFF
)
cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --target zkgpt_gat_prover zkgpt_gat_verifier --parallel 2

if [[ "${developer_tests}" -eq 1 ]]; then
    smoke_build_dir="${ROOT_DIR}/build/cpu-tests"
    smoke_cmake_args=(
        -S "${ROOT_DIR}"
        -B "${smoke_build_dir}"
        -DCMAKE_BUILD_TYPE="${build_type}"
        -DZKGPT_BUILD_TESTS=ON
        -DZKGPT_ENABLE_CUDA=OFF
        -DMCL_TEST_WITH_GMP=OFF
    )
    cmake "${smoke_cmake_args[@]}"
    cmake --build "${smoke_build_dir}" --target gat_adapter_smoke --parallel 2
    echo "built isolated gat_adapter_smoke (cpu tests) at ${smoke_build_dir}/tests/"
fi

if [[ "${build_dir}" == "${ROOT_DIR}/build/${backend}" ]]; then
    PYTHONPATH="${ROOT_DIR}/scripts" python3 - "${backend}" <<'PYBUILD'
import sys
from common import baseline_root, write_build_provenance
write_build_provenance(baseline_root(), sys.argv[1])
PYBUILD
fi
echo "built zkgpt_gat_prover and zkgpt_gat_verifier (${backend}) at ${build_dir}/gat_adapter/"
