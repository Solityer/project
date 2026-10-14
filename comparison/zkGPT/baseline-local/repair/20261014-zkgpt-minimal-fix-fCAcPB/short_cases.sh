#!/usr/bin/env bash
set -euo pipefail
check_root=/home/pzh/project/comparison/zkGPT/baseline-local/repair/20261014-zkgpt-minimal-fix-fCAcPB
repo=/home/pzh/project/comparison/zkGPT
start_epoch=$(date +%s)
if ZKGPT_NATIVE_SCRATCH_DIR=baseline-local/repair/20261014-zkgpt-minimal-fix-fCAcPB/native-capacity-scratch timeout 20s "$check_root/cmake/tests/native_storage_capacity_smoke" > "$check_root/native-storage-capacity.log" 2>&1; then
    echo 'STAGE=native_storage_capacity_smoke EXIT=0'
else
    status=$?
    echo "STAGE=native_storage_capacity_smoke EXIT=$status"
    cat "$check_root/native-storage-capacity.log"
    exit "$status"
fi
c++ -DMCL_BINT_ASM_X64=0 -DMCL_FP_BIT=384 -DMCL_FR_BIT=256 -DMCL_USE_LLVM=1 -DZKGPT_HAS_CUDA=0 -I"$repo/src" -I"$repo/3rd" -I"$repo/3rd/mcl/include" -I"$repo/gat_adapter" -pthread -O3 -DNDEBUG -std=gnu++17 -UNDEBUG "$check_root/scratch_page_smoke.cpp" -o "$check_root/scratch_page_smoke" > "$check_root/scratch-page-compile.log" 2>&1
if ZKGPT_NATIVE_SCRATCH_DIR=baseline-local/repair/20261014-zkgpt-minimal-fix-fCAcPB/scratch-page-work timeout 5s "$check_root/scratch_page_smoke" > "$check_root/scratch-page-run.log" 2>&1; then
    cat "$check_root/scratch-page-run.log"
    echo 'STAGE=scratch_page_boundary EXIT=0'
else
    status=$?
    echo "STAGE=scratch_page_boundary EXIT=$status"
    cat "$check_root/scratch-page-run.log"
    exit "$status"
fi
if timeout 5s python3 "$check_root/secret_field_cases.py" > "$check_root/secret-field-cases.log" 2>&1; then
    cat "$check_root/secret-field-cases.log"
    echo 'STAGE=secret_canonical_fields EXIT=0'
else
    status=$?
    echo "STAGE=secret_canonical_fields EXIT=$status"
    cat "$check_root/secret-field-cases.log"
    exit "$status"
fi
end_epoch=$(date +%s)
echo "SHORT_CASES_EXIT=0 ELAPSED_SECONDS=$((end_epoch - start_epoch))"
