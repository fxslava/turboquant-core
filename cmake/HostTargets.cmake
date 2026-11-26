# Default build: strict C++17 CPU gates without device library dependencies.
set(TQ_GOLDEN_DUMP_ROOT "${CMAKE_BINARY_DIR}/golden-dumps" CACHE PATH
    "Complete CPU-generated fixtures used by host and device golden tests")
set(TQ_HOST_TEST_TARGETS)
foreach(gate IN ITEMS contract_gates layout_bridge telemetry_dump qwen35_decode
                      decode_reduce e8_splitk qwen35_telemetry golden_decode_oracle turboquant_production_host)
  if(gate STREQUAL "turboquant_production_host")
    set(source tests/cpu/test_turboquant_production.cpp)
  else()
    set(source "tests/cpu/test_${gate}.cpp")
  endif()
  add_executable(test_${gate} "${source}")
  target_link_libraries(test_${gate} PRIVATE turboquant_host)
  list(APPEND TQ_HOST_TEST_TARGETS test_${gate})
  if(gate STREQUAL "golden_decode_oracle")
    add_test(NAME ${gate} COMMAND test_${gate} --dump-root "${TQ_GOLDEN_DUMP_ROOT}")
    set_tests_properties(${gate} PROPERTIES LABELS "host;golden")
  else()
    add_test(NAME ${gate} COMMAND test_${gate})
    set_tests_properties(${gate} PROPERTIES LABELS host)
  endif()
endforeach()
target_compile_options(test_turboquant_production_host PRIVATE -fno-fast-math -ffp-contract=off)
target_include_directories(test_turboquant_production_host SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/tests/cpu/e8/frozen")
target_include_directories(test_e8_splitk SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/tests/cpu/e8/frozen")
target_compile_options(test_e8_splitk PRIVATE -fno-fast-math -ffp-contract=off)
add_subdirectory(tests/cpu/e8)
list(APPEND TQ_HOST_TEST_TARGETS e8_decode_cpu_check e8_layout_check)
add_custom_target(tq_host_tests DEPENDS ${TQ_HOST_TEST_TARGETS})
add_custom_target(tq_check_host
    COMMAND "${CMAKE_CTEST_COMMAND}" -C "$<CONFIG>" --output-on-failure -L host
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS tq_host_tests USES_TERMINAL VERBATIM)

# Explicit preparation keeps torch optional for the C++ build itself.
find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_custom_target(tq_generate_goldens
      COMMAND "${CMAKE_COMMAND}" -E env TORCH_DEVICE_BACKEND_AUTOLOAD=0
              "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/golden/generate_dumps.py"
              "${TQ_GOLDEN_DUMP_ROOT}"
      USES_TERMINAL VERBATIM)
endif()
if(NOT EXISTS "${TQ_GOLDEN_DUMP_ROOT}/decode_real_ctx1055/manifest.txt")
  message(STATUS "Golden fixtures need preparation: build tq_generate_goldens with a CPU-torch Python, or set TQ_GOLDEN_DUMP_ROOT to a complete fixture directory. See tests/README.md.")
endif()
