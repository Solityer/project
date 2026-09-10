add_executable(gat_adapter_smoke gat_adapter_smoke.cpp)
target_link_libraries(gat_adapter_smoke PRIVATE gat_adapter)
add_test(NAME gat_adapter_smoke COMMAND gat_adapter_smoke)

add_executable(gat_proof_plan_smoke gat_proof_plan_smoke.cpp)
target_link_libraries(gat_proof_plan_smoke PRIVATE gat_adapter)
add_test(NAME gat_proof_plan_smoke COMMAND gat_proof_plan_smoke)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
    add_test(NAME gat_script_schema
             COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test_scripts_schema.py)
    set_tests_properties(gat_script_schema PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()
