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

  ascendc_library(tq_decode_npu SHARED src/decode/turboquant_fused_decode.cpp)
  set_property(SOURCE src/decode/turboquant_fused_decode.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/decode/turboquant_fused_decode.cpp)
  ascendc_compile_definitions(tq_decode_npu PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_decode_npu PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS} ${TQ_REDUCE_DEFINES}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
      "-I${SRC}/src/decode" "-I${SRC}/src/common" "-I${SRC}/include")

  function(add_npu_runner name source)
    add_executable(${name} EXCLUDE_FROM_ALL "${source}")
    target_include_directories(${name} PRIVATE "${SRC}" "${ASCEND_HOME_PATH}/include")
    target_compile_options(${name} PRIVATE -std=c++17 -Wall -Wextra -Werror)
    target_link_libraries(${name} PRIVATE ${ARGN} turboquant_host ascendcl)
    set_target_properties(${name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ASCEND_HOME_PATH}/lib64")
  endfunction()

  add_npu_runner(test_live_write_npu tests/npu/test_live_write.cpp
      tq_prefill_npu tq_scatter_npu)
  add_npu_runner(test_nz_ingress_canary_npu tests/npu/test_nz_ingress_canary.cpp
      tq_prefill_nz_npu)
  add_npu_runner(test_fused_decode_e2e_npu tests/npu/test_fused_decode_e2e.cpp tq_decode_npu)

  add_custom_target(tq_npu_kernels
      DEPENDS tq_prefill_npu tq_prefill_nz_npu tq_scatter_npu tq_decode_npu
              test_live_write_npu test_nz_ingress_canary_npu test_fused_decode_e2e_npu)
endfunction()
