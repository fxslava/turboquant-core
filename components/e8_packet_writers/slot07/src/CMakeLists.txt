cmake_minimum_required(VERSION 3.22)
project(checkpoint2_e8_encoded_batch LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
include(cmake/CannToolchain.cmake)
include(cmake/CannWrapperWarnings.cmake)
set(RUN_MODE sim)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
ascendc_library(e8_batch SHARED e8_packet_v1_batch_encoded.cpp)
ascendc_compile_definitions(e8_batch PRIVATE ASCEND_MICRO_CANN_LAUNCH)
ascendc_compile_options(e8_batch PRIVATE
    -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
    -forward-options-to-host-compiler -Wall -Wextra -Werror
    ${TQ_CANN_SYSTEM_OPTIONS} -DASCEND_MICRO_CANN_LAUNCH)
tq_scope_cann_wrapper_warnings(e8_batch)
set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_3510/lib")
add_executable(batch_runner batch_runner.cpp)
target_include_directories(batch_runner PRIVATE "${ASCEND_HOME_PATH}/include")
target_compile_options(batch_runner PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
target_link_libraries(batch_runner PRIVATE e8_batch ascendcl
    "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
target_link_options(batch_runner PRIVATE "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
set_target_properties(batch_runner PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/../lib;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64;${ASCEND_HOME_PATH}/x86_64-linux/devlib")

