# Discover toolkit headers without enabling ASC or linking host tests to CANN.
# Kernel helpers consume this shared SYSTEM include list once per target.

set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.2.1" CACHE PATH "CANN toolkit root")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(TQ_CANN_SYSTEM_DIRS "${ASCEND_HOME_PATH}/include")
foreach(path IN ITEMS asc asc/include asc/include/basic_api asc/include/simt_api
    asc/impl/basic_api asc/impl/simt_api tikcpp/tikcfw tikcpp/tikcfw/interface tikcpp/tikcfw/impl)
  list(APPEND TQ_CANN_SYSTEM_DIRS "${ASCEND_HOME_PATH}/x86_64-linux/${path}")
  list(APPEND TQ_CANN_SYSTEM_DIRS "${ASCEND_HOME_PATH}/${path}")
endforeach()

# The Cube GEMM reduction of the fused decode's Split-K reduction phase. ON by default: the
# reducer's UB comes from a TBufPool aliased onto the attention phase's dead transient pool, so it
# costs no headroom. Defined here, once, rather than in either target module, because both legs
# compile the identical kernel source and an option visible to one leg only would build the Cube
# reduction on the simulator and the vector reducer on the device -- two different operators behind
# one flag, with whichever leg a given host can build deciding which one it gets.
option(TQ_GEMM_REDUCE "Drive the fused decode's reduction with the Cube GEMM" ON)
set(TQ_REDUCE_DEFINES "")
if(TQ_GEMM_REDUCE)
  set(TQ_REDUCE_DEFINES -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1)
endif()

# The fused decode's attention telemetry probe. ON by default: the kernel records the
# per-(token, head) argmax context position and its logical page into the Split-K workspace's
# dead lanes (src/turboquant/common/turboquant_layout.h), piggybacking the existing buffer, and
# the runners' workspace contract verifies the records instead of requiring a pristine
# workspace. The define goes to BOTH the kernel legs and the runners so a build never checks
# for records the kernel was not compiled to write; layout.h carries the same ON default so a
# downstream build without this option (the torch binding) gets the probe compiled in too.
option(TQ_DECODE_TELEMETRY "Record the per-(token, head) attention argmax into the Split-K workspace" ON)
set(TQ_TELEMETRY_DEFINES "")
if(NOT TQ_DECODE_TELEMETRY)
  set(TQ_TELEMETRY_DEFINES -DASCEND_TQ_DECODE_TELEMETRY=0)
endif()

# Cache host compilation when a supported launcher is installed.
option(TQ_COMPILER_CACHE "Route BiSheng and host compiles through ccache/sccache when present" ON)
if(TQ_COMPILER_CACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(TQ_COMPILER_CACHE_PROGRAM NAMES ccache sccache)
  if(TQ_COMPILER_CACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${TQ_COMPILER_CACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${TQ_COMPILER_CACHE_PROGRAM}")
    message(STATUS "Compiler cache: ${TQ_COMPILER_CACHE_PROGRAM}")
  else()
    message(STATUS "Compiler cache: none found")
  endif()
endif()

set(TQ_ASCENDC_CMAKE_DIR "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
set(TQ_HAVE_ASCENDC OFF)
if(EXISTS "${TQ_ASCENDC_CMAKE_DIR}/asc-config.cmake")
  set(TQ_HAVE_ASCENDC ON)
endif()
set(TQ_KERNEL_REDUCE_DEFS)
if(TQ_GEMM_REDUCE)
  set(TQ_KERNEL_REDUCE_DEFS ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1)
endif()
set(TQ_KERNEL_TELEMETRY_DEFS)
if(NOT TQ_DECODE_TELEMETRY)
  set(TQ_KERNEL_TELEMETRY_DEFS ASCEND_TQ_DECODE_TELEMETRY=0)
endif()
