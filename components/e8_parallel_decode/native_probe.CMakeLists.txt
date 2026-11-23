cmake_minimum_required(VERSION 3.22)
project(e8_fused_reader LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
include(cmake/CannToolchain.cmake)
include(cmake/CannWrapperWarnings.cmake)
set(RUN_MODE sim)
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
ascendc_library(e8_fused_decode SHARED e8_fused_decode.cpp)
ascendc_compile_definitions(e8_fused_decode PRIVATE
    ASCEND_MICRO_CANN_LAUNCH ENABLE_CV_COMM_VIA_SSBUF=1)
ascendc_compile_options(e8_fused_decode PRIVATE
    -std=c++17 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off
    ${TQ_CANN_SYSTEM_OPTIONS}
    -forward-options-to-host-compiler -Wall -Wextra -Werror
    ${TQ_CANN_SYSTEM_OPTIONS} -DASCEND_MICRO_CANN_LAUNCH
    -DENABLE_CV_COMM_VIA_SSBUF=1)
tq_scope_cann_wrapper_warnings(e8_fused_decode)
set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_3510/lib")
set(WRITER_DIR "/home/dmin/work/checkpoint2_20261123/slot02/build-r2/lib")
add_executable(e8_reader reader_runner.cpp)
target_include_directories(e8_reader PRIVATE "${ASCEND_HOME_PATH}/include")
target_include_directories(e8_reader SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/frozen")
target_compile_options(e8_reader PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
target_link_libraries(e8_reader PRIVATE e8_fused_decode "${WRITER_DIR}/libe8_scatter.so" ascendcl
    "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
target_link_options(e8_reader PRIVATE "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
set_target_properties(e8_reader PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/../lib;${WRITER_DIR};${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64;${ASCEND_HOME_PATH}/x86_64-linux/devlib")
add_executable(e8_reader_fixture reader_runner.cpp)
target_include_directories(e8_reader_fixture SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/frozen")
target_compile_definitions(e8_reader_fixture PRIVATE E8_READER_HOST_ONLY=1)
target_compile_options(e8_reader_fixture PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)

ascendc_library(e8_parallel_dequant SHARED parallel_component.cpp)
ascendc_compile_definitions(e8_parallel_dequant PRIVATE ASCEND_MICRO_CANN_LAUNCH)
ascendc_compile_options(e8_parallel_dequant PRIVATE
    -std=c++17 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off
    ${TQ_CANN_SYSTEM_OPTIONS}
    -forward-options-to-host-compiler -Wall -Wextra -Werror
    ${TQ_CANN_SYSTEM_OPTIONS} -DASCEND_MICRO_CANN_LAUNCH)
tq_scope_cann_wrapper_warnings(e8_parallel_dequant)
add_executable(e8_parallel_component_runner component_runner.cpp)
target_include_directories(e8_parallel_component_runner PRIVATE "${ASCEND_HOME_PATH}/include")
target_include_directories(e8_parallel_component_runner SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/frozen")
target_compile_options(e8_parallel_component_runner PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
target_link_libraries(e8_parallel_component_runner PRIVATE e8_parallel_dequant ascendcl
    "${SIM_LIB_DIR}/libnpu_drv_camodel.so" "${SIM_LIB_DIR}/libmodel_top.so")
target_link_options(e8_parallel_component_runner PRIVATE "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
set_target_properties(e8_parallel_component_runner PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "$ORIGIN/../lib;${SIM_LIB_DIR};${ASCEND_HOME_PATH}/lib64;${ASCEND_HOME_PATH}/x86_64-linux/devlib")
