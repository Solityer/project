# Native scratch empty-tail false rejection repair

## Failure addressed

The 2026-10-22 PPI proof failed after native GKR when scratch reclamation had no complete page to punch. The live prefix was 147,095,538,208 bytes and the backing file was 147,095,538,240 bytes, leaving 32 bytes. The empty range was sent to an extent check that treated it as unproven; the allocated-block estimate also included 40,960 bytes above the bound. The proof was rejected after roughly 12,821.8 seconds. The detached runner then advanced to OGBN-Arxiv, which was stopped and preserved separately in `baseline-local/repair/20261022-latest-ppi-failure/`.

## Change

`gat_adapter/native_storage.hpp` now determines whether the page-aligned live prefix leaves any complete reclaimable page. Existing mapping identity, live-prefix limit, and alignment overflow checks run first. A no-complete-page case skips hole punching and extent inspection, reads the actual allocated-byte count, updates the existing live/allocated mapping fields, and returns the real before/after byte counts. It does not lower or estimate allocated bytes. The nonempty-tail punch and FIEMAP fail-closed path remains in place.

`tests/native_storage_capacity_smoke.cpp` adds no-op cases for a zero-byte tail, a one-byte tail, the incident's 32-byte near-full tail, and a tail smaller than one page. It checks that the allocation count and live/allocated accounting remain truthful, retained data is preserved, and the existing invalid-live-prefix rejection remains covered. The existing complete-page reclamation case now checks that live data survives reclaim.

## Build/check availability

The dedicated `native_storage_capacity_smoke` CTest target is declared in `tests/CMakeLists.txt`. `build/cpu-tests` exists. Production `zkgpt_gat_prover` and `zkgpt_gat_verifier` targets are declared in `gat_adapter/CMakeLists.txt`, and their CPU build directory and binaries exist under `build/cpu/gat_adapter/`. This repair turn deliberately did not build or execute either target; root will run the single approved bounded check group after static review.

## Bounded verification

The single approved check group passed:

1. `cmake --build build/cpu --target zkgpt_gat_prover zkgpt_gat_verifier --parallel 2` — exit 0, real 48.42 s; both production CPU CLIs linked.
2. `cmake --build build/cpu-tests --target native_storage_capacity_smoke --parallel 2` — exit 0, real 45.36 s.
3. `ctest --test-dir build/cpu-tests -R '^native_storage_capacity_smoke$' --output-on-failure` — exit 0, 1/1 passed, real 1.77 s.

Exact command output and exit codes are preserved in `baseline-local/bounded-check/repair-2026-10-22/native-scratch-check/`. No full dataset or benchmark was run; no further source changes or checks were made after this group passed.
