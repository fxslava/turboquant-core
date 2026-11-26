# Toolkit-free gates for the sealed Wave-3 source. Never configure its native
# probe CMakeLists: that project unconditionally enables device compilation.
foreach(gate IN ITEMS decode_cpu_check layout_check)
  add_executable(e8_${gate} ${gate}.cpp)
  target_link_libraries(e8_${gate} PRIVATE tq_strict)
  target_compile_options(e8_${gate} PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
  target_include_directories(e8_${gate} SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/frozen")
  add_test(NAME e8_${gate} COMMAND e8_${gate})
  set_tests_properties(e8_${gate} PROPERTIES LABELS "host;e8")
endforeach()
