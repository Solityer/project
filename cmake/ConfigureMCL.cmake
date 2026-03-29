set(GATZK_USE_MCL ON)
set(GATZK_SELECTED_BACKEND mcl)
set(GATZK_MCL_INCLUDE_DIRS "")
set(GATZK_MCL_LIBRARIES "")

function(gatzk_configure_vendored_mcl)
    set(GATZK_VENDOR_MCL_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/mcl")
    set(GATZK_VENDOR_MCL_INCLUDE_DIR "${GATZK_VENDOR_MCL_SOURCE_DIR}/include")
    set(GATZK_VENDOR_MCL_LIBRARY "${GATZK_VENDOR_MCL_SOURCE_DIR}/lib/libmcl.a")

    if(NOT EXISTS "${GATZK_VENDOR_MCL_INCLUDE_DIR}/mcl/bn.hpp")
        set(GATZK_VENDOR_MCL_FOUND OFF PARENT_SCOPE)
        return()
    endif()

    find_program(GATZK_HOST_MAKE NAMES gmake make REQUIRED)

    add_custom_command(
        OUTPUT "${GATZK_VENDOR_MCL_LIBRARY}"
        COMMAND "${GATZK_HOST_MAKE}" -C "${GATZK_VENDOR_MCL_SOURCE_DIR}" lib/libmcl.a MCL_USE_LLVM=0 MCL_BINT_ASM=0 MCL_MSM=0 -j4
        WORKING_DIRECTORY "${GATZK_VENDOR_MCL_SOURCE_DIR}"
        COMMENT "Building vendored mcl backend"
        VERBATIM
    )
    add_custom_target(gatzk_mcl_build DEPENDS "${GATZK_VENDOR_MCL_LIBRARY}")

    if(NOT TARGET gatzk_mcl)
        add_library(gatzk_mcl STATIC IMPORTED GLOBAL)
        set_target_properties(
            gatzk_mcl
            PROPERTIES
                IMPORTED_LOCATION "${GATZK_VENDOR_MCL_LIBRARY}"
                INTERFACE_INCLUDE_DIRECTORIES "${GATZK_VENDOR_MCL_INCLUDE_DIR}"
        )
        add_dependencies(gatzk_mcl gatzk_mcl_build)
    endif()

    set(GATZK_VENDOR_MCL_FOUND ON PARENT_SCOPE)
    set(GATZK_VENDOR_MCL_INCLUDE_DIR "${GATZK_VENDOR_MCL_INCLUDE_DIR}" PARENT_SCOPE)
    set(GATZK_VENDOR_MCL_LIBRARY_TARGET gatzk_mcl PARENT_SCOPE)
endfunction()

if(GATZK_CRYPTO_BACKEND STREQUAL "auto")
    message(WARNING "GAT-ZKML: GATZK_CRYPTO_BACKEND=auto is deprecated; defaulting to mcl")
    set(GATZK_CRYPTO_BACKEND mcl)
endif()

if(NOT GATZK_CRYPTO_BACKEND STREQUAL "mcl")
    message(FATAL_ERROR "GAT-ZKML: reference backend has been removed; only -DGATZK_CRYPTO_BACKEND=mcl is supported")
endif()

gatzk_configure_vendored_mcl()
if(GATZK_VENDOR_MCL_FOUND)
    set(GATZK_MCL_INCLUDE_DIRS "${GATZK_VENDOR_MCL_INCLUDE_DIR}")
    set(GATZK_MCL_LIBRARIES gatzk_mcl)
    message(STATUS "GAT-ZKML: using vendored mcl backend from third_party/mcl")
else()
    find_path(GATZK_SYSTEM_MCL_INCLUDE_DIR
        NAMES mcl/bn.hpp
        HINTS
            /usr/local/include
            /usr/include
    )
    find_library(GATZK_SYSTEM_MCL_LIBRARY
        NAMES mclbn384_256 mclbn256 mcl
        HINTS
            /usr/local/lib
            /usr/lib
    )

    if(GATZK_SYSTEM_MCL_INCLUDE_DIR AND GATZK_SYSTEM_MCL_LIBRARY)
        set(GATZK_MCL_INCLUDE_DIRS "${GATZK_SYSTEM_MCL_INCLUDE_DIR}")
        set(GATZK_MCL_LIBRARIES "${GATZK_SYSTEM_MCL_LIBRARY}")
        message(STATUS "GAT-ZKML: using system mcl backend")
    else()
        message(FATAL_ERROR "GAT-ZKML: no vendored or system mcl installation was found")
    endif()
endif()
