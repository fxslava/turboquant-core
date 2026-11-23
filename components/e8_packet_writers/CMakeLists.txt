cmake_minimum_required(VERSION 3.22)
project(turboquant_e8_packet_components LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
enable_testing()
function(e8_packet_strict target)
  target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
endfunction()
add_library(e8_packet_contract INTERFACE)
target_include_directories(e8_packet_contract INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/slot02/src")
add_library(turboquant::e8_packet_contract ALIAS e8_packet_contract)
add_executable(e8_packet_host slot02/src/test_e8_packet_v1.cpp)
e8_packet_strict(e8_packet_host)
target_include_directories(e8_packet_host SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reference")
add_test(NAME e8_packet_host COMMAND e8_packet_host "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_executable(e8_batch_ownership slot07/src/ownership_model.cpp)
e8_packet_strict(e8_batch_ownership)
add_test(NAME e8_batch_ownership COMMAND e8_batch_ownership "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
add_executable(e8_packet_device_bytes_leaf_cpu slot02/analysis/device_leaf.cpp)
e8_packet_strict(e8_packet_device_bytes_leaf_cpu)
target_include_directories(e8_packet_device_bytes_leaf_cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/slot02/src")
target_include_directories(e8_packet_device_bytes_leaf_cpu SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/reference")
add_test(NAME e8_packet_device_bytes_leaf_cpu COMMAND e8_packet_device_bytes_leaf_cpu
    "${CMAKE_CURRENT_SOURCE_DIR}/evidence/slot02/run" "${CMAKE_CURRENT_SOURCE_DIR}/fixtures")
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME e8_packet_provenance COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/verify_package.py")

# New optional component targets only. Default CPU configuration never invokes CANN.
# The child projects are the byte-identical configurations used for the verified runs.
option(TQ_E8_CONFIGURE_CAMODEL "Expose isolated CANN component build targets" OFF)
if(TQ_E8_CONFIGURE_CAMODEL)
  if(NOT DEFINED ASCEND_HOME_PATH OR NOT EXISTS "${ASCEND_HOME_PATH}/set_env.sh")
    message(FATAL_ERROR "Set ASCEND_HOME_PATH to the CANN9.2+ toolkit for optional CAModel targets")
  endif()
  include(ExternalProject)
  ExternalProject_Add(e8_packet_scatter_camodel
      SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/slot02/src"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/device-slot02"
      CMAKE_ARGS "-DASCEND_HOME_PATH=${ASCEND_HOME_PATH}" -DTQ_COMPILER_CACHE=OFF -DCMAKE_BUILD_TYPE=Release
      BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target packet_runner
      INSTALL_COMMAND "" EXCLUDE_FROM_ALL TRUE)
  ExternalProject_Add(e8_packet_batch_camodel
      SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/slot07/src"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/device-slot07"
      CMAKE_ARGS "-DASCEND_HOME_PATH=${ASCEND_HOME_PATH}" -DTQ_COMPILER_CACHE=OFF -DCMAKE_BUILD_TYPE=Release
      BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --target batch_runner
      INSTALL_COMMAND "" EXCLUDE_FROM_ALL TRUE)
endif()
