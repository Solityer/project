
#include "gat_native_bridge.hpp"

namespace zkdl_gat {

bool zkdl_gat_native_core_prove(const NativeProveRequest &request, NativeProofResult *result, std::string *error);

bool zkdl_gat_native_core_verify(const NativeVerifyRequest &request, std::string *error);

}

namespace zkdl_gat {

bool zkdl_gat_native_prove(const NativeProveRequest &request, NativeProofResult *result, std::string *error) {
    return zkdl_gat_native_core_prove(request, result, error);
}

bool zkdl_gat_native_verify(const NativeVerifyRequest &request, std::string *error) { return zkdl_gat_native_core_verify(request, error); }

}
