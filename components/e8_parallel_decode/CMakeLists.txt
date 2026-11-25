# Toolkit-free gates for the sealed Wave-3 source. Never configure its native
# probe CMakeLists: that project unconditionally enables device compilation.
foreach(gate IN ITEMS decode_cpu_check layout_check)
  add_executable(e8_${gate} ${gate}.cpp)
  target_compile_options(e8_${gate} PRIVATE -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off)
  target_include_directories(e8_${gate} SYSTEM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/frozen")
  add_test(NAME e8_${gate} COMMAND e8_${gate})
endforeach()

# Explicit opt-in for a deployment checkout. The CPU suite never configures
# these libraries, and no raw BF16 E8 encoder is included.
option(TQ_BUILD_E8_KERNELS "Configure the verified E8 reader and dequantizer device libraries" OFF)
if(TQ_BUILD_E8_KERNELS)
  if(NOT TQ_HAVE_ASCENDC)
    message(FATAL_ERROR "TQ_BUILD_E8_KERNELS requires the CANN toolchain")
  endif()
  include("${PROJECT_SOURCE_DIR}/cmake/CannWrapperWarnings.cmake")
  ascendc_library(tq_e8_reader SHARED e8_fused_decode.cpp)
  ascendc_library(tq_e8_dequant SHARED parallel_component.cpp)
  foreach(kernel IN ITEMS tq_e8_reader tq_e8_dequant)
    ascendc_compile_definitions(${kernel} PRIVATE ASCEND_MICRO_CANN_LAUNCH ENABLE_CV_COMM_VIA_SSBUF=1)
    ascendc_compile_options(${kernel} PRIVATE
        -std=c++17 -Wall -Wextra -Werror -fno-fast-math -ffp-contract=off ${TQ_CANN_SYSTEM_OPTIONS}
        -forward-options-to-host-compiler -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
        -DASCEND_MICRO_CANN_LAUNCH -DENABLE_CV_COMM_VIA_SSBUF=1)
    tq_scope_cann_wrapper_warnings(${kernel})
  endforeach()
  add_custom_target(tq_e8_kernels DEPENDS tq_e8_reader tq_e8_dequant)
endif()
