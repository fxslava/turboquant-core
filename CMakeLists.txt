cmake_minimum_required(VERSION 3.16)
project(turboquant_core LANGUAGES CXX)

# Two kinds of target live here, and the split is the point of this file.
#
#   THE CPU GATES are toolkit-free: they include the layout contract, the host tiling and the
#   kernels' host-visible geometry, and nothing else. They build in seconds with any C++17
#   compiler, in every RUN_MODE, and they are what `cmake --build build` builds.
#
#   THE DEVICE LEGS compile the three operators through BiSheng, for CAModel (RUN_MODE=sim) or
#   for the physical Ascend 950PR / dav-3510 target (RUN_MODE=npu). One ascendc_library expands
#   into nested ExternalProjects and compiles the whole translation unit five to seven times, and
#   every one of them carries BUILD_ALWAYS, so the three kernels together are tens of minutes of
#   build even with a warm compiler cache.
#
#   THEY ARE THEREFORE OFF BY DEFAULT, behind TQ_BUILD_KERNELS, and not merely EXCLUDE_FROM_ALL:
#   ascendc_library does not honour EXCLUDE_FROM_ALL. Its nested ExternalProjects are registered
#   against the default target whatever the executables that link them are marked, so a
#   configure that merely declared them put the whole BiSheng build back into `cmake --build`.
#   The option is the only thing that actually keeps a CPU-gate build fast.
#
#       cmake -B build -DRUN_MODE=sim && cmake --build build -j16          # the CPU gates
#       cmake -B build -DRUN_MODE=sim -DTQ_BUILD_KERNELS=ON                # + CAModel kernels
#       cmake --build build --target tq_sim_kernels -j16
#       cmake -B build-npu -DRUN_MODE=npu -DTQ_BUILD_KERNELS=ON
#       cmake --build build-npu --target tq_npu_kernels -j16
#
# Strict flags are not negotiable on our own sources: -std=c++17 -Wall -Wextra -Werror
# everywhere, with CANN's own headers reclassified as system headers so they cannot spend the
# budget (cmake/CannToolchain.cmake).

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

set(CMAKE_BUILD_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

set(RUN_MODE "sim" CACHE STRING "Device leg to configure: sim (CAModel) or npu (dav-3510)")
set_property(CACHE RUN_MODE PROPERTY STRINGS sim npu)
if(NOT RUN_MODE STREQUAL "sim" AND NOT RUN_MODE STREQUAL "npu")
  message(FATAL_ERROR "RUN_MODE must be sim or npu, not '${RUN_MODE}'")
endif()

include(cmake/CannToolchain.cmake)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(tq_strict INTERFACE)
target_compile_options(tq_strict INTERFACE -Wall -Wextra -Werror)
target_include_directories(tq_strict INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# ------------------------------------------------------------------ the host tiling engine
# One translation unit, linked by every gate and every runner: the fused decode's descriptors and
# dispatch records are read by the device BY WORD OFFSET, and nothing in the toolchain diagnoses
# two sides disagreeing about which word is which, so there is exactly one definition of them.
add_library(turboquant_host STATIC src/host/turboquant_tiling.cpp)
target_link_libraries(turboquant_host PUBLIC tq_strict)

# ------------------------------------------------------------------ the CPU gates
enable_testing()

add_executable(test_contract_gates tests/cpu/test_contract_gates.cpp)
target_link_libraries(test_contract_gates PRIVATE turboquant_host)

add_executable(test_layout_bridge tests/cpu/test_layout_bridge.cpp)
target_link_libraries(test_layout_bridge PRIVATE turboquant_host)

add_test(NAME contract_gates COMMAND test_contract_gates)
add_test(NAME layout_bridge COMMAND test_layout_bridge)

# ------------------------------------------------------------------ the device legs
option(TQ_BUILD_KERNELS "Configure the BiSheng device legs (tens of minutes to build)" OFF)
if(NOT TQ_BUILD_KERNELS)
  message(STATUS "Device leg: NOT CONFIGURED. Re-run with -DTQ_BUILD_KERNELS=ON to compile the "
                 "three operators for ${RUN_MODE}; only the CPU gates are built otherwise.")
elseif(NOT TQ_HAVE_ASCENDC)
  message(FATAL_ERROR "TQ_BUILD_KERNELS=ON but no ascendc.cmake under ${TQ_ASCENDC_CMAKE_DIR}. "
                      "Point ASCEND_HOME_PATH at a CANN toolkit, or leave the device legs off.")
elseif(RUN_MODE STREQUAL "sim")
  include(cmake/SimTargets.cmake)
  add_turboquant_sim_targets()
  message(STATUS "Device leg: CAModel ${SOC_VERSION}, target tq_sim_kernels.")
else()
  include(cmake/NpuTargets.cmake)
  add_turboquant_npu_targets()
  message(STATUS "Device leg: physical ${SOC_VERSION}, target tq_npu_kernels.")
endif()
