# CMake generated Testfile for 
# Source directory: /home/pzh/project/comparison/zkGPT/tests
# Build directory: /home/pzh/project/comparison/zkGPT/cmake-build-gat-prod/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(gat_adapter_smoke "/home/pzh/project/comparison/zkGPT/cmake-build-gat-prod/tests/gat_adapter_smoke")
set_tests_properties(gat_adapter_smoke PROPERTIES  _BACKTRACE_TRIPLES "/home/pzh/project/comparison/zkGPT/tests/CMakeLists.txt;3;add_test;/home/pzh/project/comparison/zkGPT/tests/CMakeLists.txt;0;")
add_test(gat_proof_plan_smoke "/home/pzh/project/comparison/zkGPT/cmake-build-gat-prod/tests/gat_proof_plan_smoke")
set_tests_properties(gat_proof_plan_smoke PROPERTIES  _BACKTRACE_TRIPLES "/home/pzh/project/comparison/zkGPT/tests/CMakeLists.txt;7;add_test;/home/pzh/project/comparison/zkGPT/tests/CMakeLists.txt;0;")
add_test(gat_script_schema "/usr/bin/python3" "/home/pzh/project/comparison/zkGPT/tests/test_scripts_schema.py")
set_tests_properties(gat_script_schema PROPERTIES  WORKING_DIRECTORY "/home/pzh/project/comparison/zkGPT" _BACKTRACE_TRIPLES "/home/pzh/project/comparison/zkGPT/tests/CMakeLists.txt;11;add_test;/home/pzh/project/comparison/zkGPT/tests/CMakeLists.txt;0;")
