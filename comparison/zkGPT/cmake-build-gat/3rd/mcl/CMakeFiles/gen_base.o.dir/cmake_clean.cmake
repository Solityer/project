file(REMOVE_RECURSE
  "base64.o"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/gen_base.o.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
