# The physical-device leg: the same three operators compiled by BiSheng for the Ascend 950PR /
# dav-3510 target, and the same two runners against the device runtime.
#
# The kernel sources and every compile option but the runtime selection are identical to
# SimTargets.cmake, deliberately: the CAModel evidence is only about the device binary if the two
# legs compile the same translation unit with the same definitions. RUN_MODE is set inside the
# function for the reason given there.
function(add_turboquant_npu_targets)
  set(RUN_MODE npu)
  set(SRC "${CMAKE_CURRENT_SOURCE_DIR}")

  ascendc_library(tq_prefill_npu SHARED src/prefill/turboquant_reshape_batch.cpp)
  set_property(SOURCE src/prefill/turboquant_reshape_batch.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/turboquant_reshape_batch.cpp)
  ascendc_compile_definitions(tq_prefill_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_prefill_npu PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # The same writer with the write-time NZ-tiled ingress on; see SimTargets.cmake for why it is
  # its own library rather than an option on the one above.
  ascendc_library(tq_prefill_nz_npu SHARED src/prefill/turboquant_reshape_batch.cpp)
  set_property(SOURCE src/prefill/turboquant_reshape_batch.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/turboquant_reshape_batch.cpp)
  ascendc_compile_definitions(tq_prefill_nz_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_INGRESS_NZ=1)
  ascendc_compile_options(tq_prefill_nz_npu PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH -DTURBOQUANT_INGRESS_NZ=1)

  ascendc_library(tq_scatter_npu SHARED src/scatter/turboquant_reshape_scatter.cpp)
  set_property(SOURCE src/scatter/turboquant_reshape_scatter.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/scatter/turboquant_reshape_scatter.cpp)
  ascendc_compile_definitions(tq_scatter_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_scatter_npu PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # The same scatter writer with the NZ-tiled ingress on; see SimTargets.cmake.
  ascendc_library(tq_scatter_nz_npu SHARED src/scatter/turboquant_reshape_scatter.cpp)
  set_property(SOURCE src/scatter/turboquant_reshape_scatter.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/scatter/turboquant_reshape_scatter.cpp)
  ascendc_compile_definitions(tq_scatter_nz_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_INGRESS_NZ=1)
  ascendc_compile_options(tq_scatter_nz_npu PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH -DTURBOQUANT_INGRESS_NZ=1)

  ascendc_library(tq_decode_npu SHARED src/decode/turboquant_fused_decode.cpp)
  set_property(SOURCE src/decode/turboquant_fused_decode.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/decode/turboquant_fused_decode.cpp)
  ascendc_compile_definitions(tq_decode_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_decode_npu PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS} ${TQ_REDUCE_DEFINES}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
      "-I${SRC}/src/decode" "-I${SRC}/src/common" "-I${SRC}/include")

  # The runner link must resolve libruntime.so's own NEEDED entries -- libascend_hal.so and
  # the drv/hdc providers (drvHdcSessionConnect, halGetDeviceInfo, drvGetDevNum, ...) -- which
  # live in the DRIVER package, not the toolkit. Without -rpath-link pointing there, ld fails
  # with "libascend_hal.so ... not found" plus a screen of undefined drvHdc* references even
  # though nothing the runner calls needs them. The driver's install root is registered in
  # /etc/ascend_install.info (the same file set_env.sh reads); default to the standard layout.
  set(TQ_DRIVER_HOME "/usr/local/Ascend/driver")
  if(EXISTS "/etc/ascend_install.info")
    file(STRINGS "/etc/ascend_install.info" _tq_drv_line
         REGEX "^[Dd]river_[Ii]nstall_[Pp]ath_[Pp]aram=")
    if(_tq_drv_line)
      string(REGEX REPLACE "^[^=]*=" "" _tq_drv_root "${_tq_drv_line}")
      if(EXISTS "${_tq_drv_root}/driver")
        set(TQ_DRIVER_HOME "${_tq_drv_root}/driver")
      endif()
    endif()
  endif()
  set(TQ_DRIVER_RPATH_LINKS "")
  foreach(_tq_drv_lib "${TQ_DRIVER_HOME}/lib64" "${TQ_DRIVER_HOME}/lib64/driver")
    if(EXISTS "${_tq_drv_lib}")
      list(APPEND TQ_DRIVER_RPATH_LINKS "-Wl,-rpath-link,${_tq_drv_lib}")
    endif()
  endforeach()
  if(NOT TQ_DRIVER_RPATH_LINKS)
    message(WARNING "No Ascend driver lib dir found under ${TQ_DRIVER_HOME}; NPU runner links "
                    "may fail on libruntime.so's indirect driver dependencies.")
  endif()

  function(add_npu_runner name source)
    add_executable(${name} EXCLUDE_FROM_ALL "${source}")
    target_include_directories(${name} PRIVATE "${SRC}" "${ASCEND_HOME_PATH}/include")
    target_compile_options(${name} PRIVATE -std=c++17 -Wall -Wextra -Werror)
    target_link_libraries(${name} PRIVATE ${ARGN} turboquant_host ascendcl)
    target_link_options(${name} PRIVATE ${TQ_DRIVER_RPATH_LINKS})
    set_target_properties(${name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ASCEND_HOME_PATH}/lib64")
  endfunction()

  add_npu_runner(test_live_write_npu tests/npu/test_live_write.cpp
      tq_prefill_npu tq_scatter_npu)
  add_npu_runner(test_live_write_nz_npu tests/npu/test_live_write.cpp
      tq_scatter_nz_npu)
  target_compile_definitions(test_live_write_nz_npu PRIVATE TQ_SCATTER_NZ=1)
  add_npu_runner(test_nz_ingress_canary_npu tests/npu/test_nz_ingress_canary.cpp
      tq_prefill_nz_npu)
  add_npu_runner(test_fused_decode_e2e_npu tests/npu/test_fused_decode_e2e.cpp tq_decode_npu)

  # The golden-dump comparison runners on the physical device: the same source and the same
  # per-kernel split as the CAModel ones in SimTargets.cmake. The dumps are host-generated
  # (golden/generate_dumps.py), so a hardware run is: generate the dumps, then
  # `build/tests/test_golden_*_npu --dump-root golden/dumps` from the repo root.
  add_npu_runner(test_golden_prefill_strip_npu tests/npu/test_golden_dump.cpp tq_prefill_npu)
  target_compile_definitions(test_golden_prefill_strip_npu PRIVATE
      TQ_GOLDEN_KERNEL_PREFILL=1 TQ_GOLDEN_PREFILL_NZ=0)
  add_npu_runner(test_golden_prefill_nz_npu tests/npu/test_golden_dump.cpp tq_prefill_nz_npu)
  target_compile_definitions(test_golden_prefill_nz_npu PRIVATE
      TQ_GOLDEN_KERNEL_PREFILL=1 TQ_GOLDEN_PREFILL_NZ=1)
  add_npu_runner(test_golden_scatter_npu tests/npu/test_golden_dump.cpp tq_scatter_npu)
  target_compile_definitions(test_golden_scatter_npu PRIVATE TQ_GOLDEN_KERNEL_SCATTER=1)
  add_npu_runner(test_golden_decode_npu tests/npu/test_golden_dump.cpp tq_decode_npu)
  target_compile_definitions(test_golden_decode_npu PRIVATE TQ_GOLDEN_KERNEL_DECODE=1)

  add_custom_target(tq_npu_kernels
      DEPENDS tq_prefill_npu tq_prefill_nz_npu tq_scatter_npu tq_scatter_nz_npu tq_decode_npu
              test_live_write_npu test_live_write_nz_npu test_nz_ingress_canary_npu
              test_fused_decode_e2e_npu
              test_golden_prefill_strip_npu test_golden_prefill_nz_npu
              test_golden_scatter_npu test_golden_decode_npu)
endfunction()
