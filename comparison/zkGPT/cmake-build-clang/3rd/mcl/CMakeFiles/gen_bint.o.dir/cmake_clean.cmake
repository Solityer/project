file(REMOVE_RECURSE
  "bint64.o"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/gen_bint.o.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
