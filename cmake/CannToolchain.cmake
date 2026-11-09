# The CANN toolkit: where it is, which SoC the device legs target, and the one thing that lets
# -Werror stay on for our own sources.
#
# CANN 9.2 adds its own headers with -I and emits hundreds of warnings under -Wall. Reclassifying
# those same directories as SYSTEM headers for BiSheng is what makes -Werror survivable for the
# kernels. The list is built once, here, because both the simulator leg and the physical leg
# compile the identical kernel sources: a path added to one list and not the other turns -Werror
# on in one leg only, and which leg starts failing then depends on which one a given host can
# actually build.

set(ASCEND_HOME_PATH "/usr/local/Ascend/cann-9.2.0-beta.2"
    CACHE PATH "CANN toolkit root")
set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_HOME_PATH}")
set(SOC_VERSION "Ascend950PR_9599" CACHE STRING "Device SoC / CAModel variant")

set(TQ_CANN_SYSTEM_OPTIONS "")
foreach(path IN ITEMS
    asc/impl/adv_api asc/impl/basic_api asc/impl/c_api
    asc/impl/basic_api/reg_compute asc/impl/simt_api asc/impl/utils
    asc asc/include asc/include/adv_api asc/include/basic_api
    asc/include/aicpu_api asc/include/c_api
    asc/include/basic_api/reg_compute asc/include/simt_api asc/include/utils
    tikcpp/tikcfw tikcpp/tikcfw/interface tikcpp/tikcfw/impl)
  list(APPEND TQ_CANN_SYSTEM_OPTIONS
      "-isystem${ASCEND_HOME_PATH}/x86_64-linux/${path}")
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

# CANN's legacy pipeline expands each ascendc_library into nested ExternalProjects -- precompile,
# preprocess, aic_device, aiv_device and host -- so one kernel is compiled five to seven times and
# every one of those projects carries BUILD_ALWAYS. A compiler cache is the difference between
# minutes and tens of minutes per kernel.
option(TQ_COMPILER_CACHE "Route BiSheng and host compiles through ccache/sccache when present" ON)
if(TQ_COMPILER_CACHE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
  find_program(TQ_COMPILER_CACHE_PROGRAM NAMES ccache sccache)
  if(TQ_COMPILER_CACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${TQ_COMPILER_CACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${TQ_COMPILER_CACHE_PROGRAM}")
    message(STATUS "Compiler cache: ${TQ_COMPILER_CACHE_PROGRAM}")
  else()
    message(STATUS "Compiler cache: none found. CANN recompiles each kernel's preprocess core "
                   "objects on every build; install ccache to make those hits.")
  endif()
endif()

set(TQ_ASCENDC_CMAKE_DIR "${ASCEND_HOME_PATH}/tools/tikcpp/ascendc_kernel_cmake")
set(TQ_HAVE_ASCENDC OFF)
if(EXISTS "${TQ_ASCENDC_CMAKE_DIR}/ascendc.cmake")
  include("${TQ_ASCENDC_CMAKE_DIR}/ascendc.cmake")
  set(TQ_HAVE_ASCENDC ON)
  # CANN disables RPATH globally. Restore it for the runners, which load the adjacent kernel
  # libraries from $ORIGIN without a build-directory search path.
  set(CMAKE_SKIP_RPATH FALSE)
endif()
