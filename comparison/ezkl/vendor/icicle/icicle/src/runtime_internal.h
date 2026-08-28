#pragma once

#include "icicle/errors.h"

// This is a private ABI between the baseline-local Rust runtime and its
// baseline-local ICICLE loader bridge.  It intentionally lives under src/
// rather than include/icicle/, so arbitrary-path backend loading is not part
// of the public ICICLE runtime API.  The bridge is linked into the formal
// Rust executable and is not an exported symbol of the ICICLE device DSO.
#if defined(_WIN32)
  #define ICICLE_RUNTIME_INTERNAL_EXPORT
#elif defined(__GNUC__) || defined(__clang__)
  #define ICICLE_RUNTIME_INTERNAL_EXPORT __attribute__((visibility("hidden")))
#else
  #define ICICLE_RUNTIME_INTERNAL_EXPORT
#endif

extern "C" ICICLE_RUNTIME_INTERNAL_EXPORT icicle::eIcicleError
icicle_load_backend_file_internal(const char* path);
