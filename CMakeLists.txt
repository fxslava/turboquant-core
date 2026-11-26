cmake_minimum_required(VERSION 3.16)
project(turboquant_core LANGUAGES CXX)

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

# Set and synchronize the SoC before loading any CANN CMake module.
set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "Ascend device SoC")
if(SOC_VERSION STREQUAL "dav-3510")
  set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "Ascend device SoC" FORCE)
endif()
set(ASCEND_SOC_VERSION "${SOC_VERSION}" CACHE STRING "CANN SoC selection" FORCE)
string(TOLOWER "${SOC_VERSION}" TQ_SOC_FAMILY)

option(TQ_BUILD_KERNELS "Configure native device kernels and runners" OFF)
option(TQ_BUILD_E8_KERNELS "Configure the sealed E8 reader and dequantizer" ${TQ_BUILD_KERNELS})
option(TQ_REGISTER_DEVICE_TESTS "Register device-executing CTests (explicit opt-in)" OFF)
if(TQ_REGISTER_DEVICE_TESTS AND NOT TQ_BUILD_KERNELS)
  message(FATAL_ERROR "TQ_REGISTER_DEVICE_TESTS requires TQ_BUILD_KERNELS=ON")
endif()

# Toolkit discovery does not initialize ASC or add runtime dependencies here.
include(cmake/CannToolchain.cmake)
if((TQ_BUILD_KERNELS OR TQ_BUILD_E8_KERNELS) AND NOT TQ_HAVE_ASCENDC)
  message(FATAL_ERROR "Device targets require asc-config.cmake under ${TQ_ASCENDC_CMAKE_DIR}. Set ASCEND_HOME_PATH to CANN 9.2.1, or turn both TQ_BUILD_KERNELS and TQ_BUILD_E8_KERNELS OFF.")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

add_library(tq_strict INTERFACE)
target_compile_options(tq_strict INTERFACE -Wall -Wextra -Werror)
target_include_directories(tq_strict INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

# ------------------------------------------------------------------ the host tiling engine
# One translation unit, linked by every gate and every runner: the fused decode's descriptors and
# dispatch records are read by the device BY WORD OFFSET, and nothing in the toolchain diagnoses
# two sides disagreeing about which word is which, so there is exactly one definition of them.
# turboquant_telemetry.cpp rides the same library: the probe records are the other GM layout the
# device writes by word offset and the host parses by the same, so they share the one definition.
add_library(turboquant_host STATIC src/turboquant/common/turboquant_tiling.cpp src/turboquant/common/turboquant_telemetry.cpp)
target_link_libraries(turboquant_host PUBLIC tq_strict)

enable_testing()
include(cmake/HostTargets.cmake)
include(cmake/TestRegistration.cmake)

if(TQ_BUILD_KERNELS OR TQ_BUILD_E8_KERNELS)
  add_subdirectory(cmake/device)
endif()
include(cmake/BuildTargets.cmake)
