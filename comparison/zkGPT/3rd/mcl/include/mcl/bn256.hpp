#pragma once

// Compatibility shim for the historical zkGPT include path.  Recent mcl
// releases expose the C++ BN/BN254 API from bn.hpp and fold the former
// mclbn256 library into the main mcl target.  Keep the old include stable so
// the upstream zkGPT sources continue to compile against the copied official
// dependency without changing their proof code.
#include <mcl/bn.hpp>
