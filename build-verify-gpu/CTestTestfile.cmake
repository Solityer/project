# CMake generated Testfile for 
# Source directory: /home/pzh/project
# Build directory: /home/pzh/project/build-verify-gpu
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(gatzk_tests "/home/pzh/project/build-verify-gpu/gatzk_tests")
set_tests_properties(gatzk_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/pzh/project/CMakeLists.txt;334;add_test;/home/pzh/project/CMakeLists.txt;0;")
add_test(gatzk_family_scaling_tests "/home/pzh/project/build-verify-gpu/gatzk_family_scaling_tests")
set_tests_properties(gatzk_family_scaling_tests PROPERTIES  _BACKTRACE_TRIPLES "/home/pzh/project/CMakeLists.txt;342;add_test;/home/pzh/project/CMakeLists.txt;0;")
add_test(static_secure_scan "/usr/bin/python3" "/home/pzh/project/tools/security/static_scan_no_tau.py")
set_tests_properties(static_secure_scan PROPERTIES  _BACKTRACE_TRIPLES "/home/pzh/project/CMakeLists.txt;345;add_test;/home/pzh/project/CMakeLists.txt;0;")
