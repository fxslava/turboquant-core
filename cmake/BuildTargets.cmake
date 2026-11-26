# Public groups build artifacts; only tq_check_host executes tests.
add_custom_target(tq_tests DEPENDS tq_host_tests)
add_custom_target(tq_kernels)
set(tq_device_help "Device targets are off. Configure -DTQ_BUILD_KERNELS=ON -DRUN_MODE=npu for hardware, or -DRUN_MODE=sim for CAModel.")
if(TQ_BUILD_E8_KERNELS)
  add_dependencies(tq_kernels tq_e8_kernels)
endif()
if(TQ_BUILD_KERNELS)
  add_dependencies(tq_tests tq_${RUN_MODE}_tests)
  add_dependencies(tq_kernels tq_${RUN_MODE}_kernel_libs)
  add_custom_target(tq_benchmarks DEPENDS tq_${RUN_MODE}_benchmarks)
  set(tq_device_help "Device runtime: ${RUN_MODE}. tq_tests builds host and ${RUN_MODE} tests. tq_kernels builds libraries only. tq_benchmarks builds ${RUN_MODE} benchmark executables in bin/.")

endif()
file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/turboquant-targets.txt" CONTENT
"TurboQuant build targets
RUN_MODE=${RUN_MODE}
SOC_VERSION=${SOC_VERSION}
ASCEND_SOC_VERSION=${ASCEND_SOC_VERSION}
TQ_BUILD_KERNELS=${TQ_BUILD_KERNELS}
TQ_BUILD_E8_KERNELS=${TQ_BUILD_E8_KERNELS}
  tq_host_tests        Build all CPU C++ tests (bin/)
  tq_check_host        Build and run host-labeled CTests
  tq_generate_goldens  Generate fixtures using Python3_EXECUTABLE with CPU torch
  tq_tests             Build all tests configured in this build directory
  tq_kernels           Build configured device libraries (lib/)
  tq_${RUN_MODE}_kernels    Build selected-runtime device libraries (when enabled)
  tq_${RUN_MODE}_tests      Build selected-runtime correctness runners (when enabled)
  tq_benchmarks        Build device benchmarks when TQ_BUILD_KERNELS=ON
${tq_device_help}
Device CTest registration: ${TQ_REGISTER_DEVICE_TESTS}
E8 device libraries: ${TQ_BUILD_E8_KERNELS}
Golden fixture root: ${TQ_GOLDEN_DUMP_ROOT}
Guides: docs/BUILD.md, tests/README.md and benchmarks/README.md
")
add_custom_target(tq_help
    COMMAND "${CMAKE_COMMAND}" -E cat "${CMAKE_BINARY_DIR}/turboquant-targets.txt"
    VERBATIM)
message(STATUS "TurboQuant target guide: cmake --build ${CMAKE_BINARY_DIR} --target tq_help")
