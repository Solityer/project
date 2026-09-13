#!/usr/bin/env bash
set -euo pipefail

# The upstream demo accepted a build-type positional argument. Formal GAT
# runs use an explicit backend so a requested GPU build can never silently
# become a CPU build. Keep the old build-type spelling as a CPU-only
# compatibility entry for the upstream demo, but the production comparison
# path is `./build.sh --cpu` or `./build.sh --gpu`.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
backend="cpu"
build_type="Release"
build_dir="${ROOT_DIR}/build/cpu"

case "${1:-}" in
    --cpu)
        backend="cpu"
        build_dir="${ROOT_DIR}/build/cpu"
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
        echo "Usage: $0 --cpu|--gpu"
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
cmake --build "${build_dir}" --target zkgpt_gat_prover zkgpt_gat_verifier --parallel

echo "built zkgpt_gat_prover and zkgpt_gat_verifier (${backend}) at ${build_dir}/gat_adapter/"
