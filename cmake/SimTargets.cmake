# The simulator leg: the three operators compiled for CAModel, and the runners that drive them.
#
# RUN_MODE is set inside the function rather than globally, because CANN reads it when each target
# is created: a module included at directory scope would pick up whichever mode happened to be
# global and would give one build of a kernel instead of the simulator / device pair.
#
# This module is included only under -DTQ_BUILD_KERNELS=ON, and that option -- not the
# EXCLUDE_FROM_ALL on the runners below -- is what keeps a CPU-gate build fast: ascendc_library
# registers its nested ExternalProjects against the default target regardless, so merely
# declaring these kernels would put five to seven BiSheng compiles of each translation unit back
# into `cmake --build`. The fused decode is the expensive one, because it pulls the cube, vector
# and reduction services in through templates. The aggregate target is `tq_sim_kernels`.
function(add_turboquant_sim_targets)
  set(RUN_MODE sim)
  set(SRC "${CMAKE_CURRENT_SOURCE_DIR}")
  set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")

# ---------------------------------------------------------------- the kernels
  ascendc_library(tq_prefill_sim SHARED src/prefill/turboquant_reshape_batch.cpp)
  set_property(SOURCE src/prefill/turboquant_reshape_batch.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/turboquant_reshape_batch.cpp)
  ascendc_compile_definitions(tq_prefill_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_prefill_sim PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  # The same writer with the write-time NZ-tiled ingress on. Its own library, not an option on
  # the one above: the two layouts are two operators behind one entry point, and the strip gate
  # (test_live_write) has to keep running against the strip writer in the same configure. Both
  # export the same symbol, so each is linked by its own runner and never by both.
  ascendc_library(tq_prefill_nz_sim SHARED src/prefill/turboquant_reshape_batch.cpp)
  set_property(SOURCE src/prefill/turboquant_reshape_batch.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/prefill/turboquant_reshape_batch.cpp)
  ascendc_compile_definitions(tq_prefill_nz_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_INGRESS_NZ=1)
  ascendc_compile_options(tq_prefill_nz_sim PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH -DTURBOQUANT_INGRESS_NZ=1)

  ascendc_library(tq_scatter_sim SHARED src/scatter/turboquant_reshape_scatter.cpp)
  set_property(SOURCE src/scatter/turboquant_reshape_scatter.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/scatter/turboquant_reshape_scatter.cpp)
  ascendc_compile_definitions(tq_scatter_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_scatter_sim PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH)

  ascendc_library(tq_decode_sim SHARED src/decode/turboquant_fused_decode.cpp)
  set_property(SOURCE src/decode/turboquant_fused_decode.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/decode/turboquant_fused_decode.cpp)
  ascendc_compile_definitions(tq_decode_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_decode_sim PRIVATE
      -std=c++17 -Wall -Werror ${TQ_CANN_SYSTEM_OPTIONS} ${TQ_REDUCE_DEFINES}
      -forward-options-to-host-compiler -DASCEND_MICRO_CANN_LAUNCH
      "-I${SRC}/src/decode" "-I${SRC}/src/common" "-I${SRC}/include")

  # ---------------------------------------------------------------- the runners
  # add_sim_runner(<name> <source> <kernel targets...>)
  # The CAModel libraries depend on further simulator DSOs; -rpath-link lets the link resolve
  # those indirect dependencies, and the install RPATH lets the runner find them at run time.
  function(add_sim_runner name source)
    add_executable(${name} EXCLUDE_FROM_ALL "${source}")
    target_include_directories(${name} PRIVATE "${SRC}" "${ASCEND_HOME_PATH}/include")
    target_compile_options(${name} PRIVATE -std=c++17 -Wall -Wextra -Werror)
    target_link_libraries(${name} PRIVATE ${ARGN} turboquant_host ascendcl
        "${SIM_LIB_DIR}/libnpu_drv_camodel.so"
        "${SIM_LIB_DIR}/libmodel_top.so")
    target_link_options(${name} PRIVATE
        "-Wl,-rpath-link,${SIM_LIB_DIR}" "-Wl,--disable-new-dtags")
    set_target_properties(${name} PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN;$ORIGIN/../lib;${ASCEND_HOME_PATH}/lib64")
  endfunction()

  add_sim_runner(test_live_write_sim tests/npu/test_live_write.cpp
      tq_prefill_sim tq_scatter_sim)
  add_sim_runner(test_nz_ingress_canary_sim tests/npu/test_nz_ingress_canary.cpp
      tq_prefill_nz_sim)
  add_sim_runner(test_fused_decode_e2e_sim tests/npu/test_fused_decode_e2e.cpp tq_decode_sim)

  # The golden-dump comparison runners: one source, one binary per kernel, the
  # prefill twice -- the strip build and the NZ build export the same symbol
  # and are therefore linked by separate runners, exactly like test_live_write
  # and test_nz_ingress_canary above.
  add_sim_runner(test_golden_prefill_strip_sim tests/npu/test_golden_dump.cpp tq_prefill_sim)
  target_compile_definitions(test_golden_prefill_strip_sim PRIVATE
      TQ_GOLDEN_KERNEL_PREFILL=1 TQ_GOLDEN_PREFILL_NZ=0)
  add_sim_runner(test_golden_prefill_nz_sim tests/npu/test_golden_dump.cpp tq_prefill_nz_sim)
  target_compile_definitions(test_golden_prefill_nz_sim PRIVATE
      TQ_GOLDEN_KERNEL_PREFILL=1 TQ_GOLDEN_PREFILL_NZ=1)
  add_sim_runner(test_golden_scatter_sim tests/npu/test_golden_dump.cpp tq_scatter_sim)
  target_compile_definitions(test_golden_scatter_sim PRIVATE TQ_GOLDEN_KERNEL_SCATTER=1)
  add_sim_runner(test_golden_decode_sim tests/npu/test_golden_dump.cpp tq_decode_sim)
  target_compile_definitions(test_golden_decode_sim PRIVATE TQ_GOLDEN_KERNEL_DECODE=1)

  add_custom_target(tq_golden_sim
      DEPENDS test_golden_prefill_strip_sim test_golden_prefill_nz_sim
              test_golden_scatter_sim test_golden_decode_sim)

  add_custom_target(tq_sim_kernels
      DEPENDS tq_prefill_sim tq_prefill_nz_sim tq_scatter_sim tq_decode_sim
              test_live_write_sim test_nz_ingress_canary_sim test_fused_decode_e2e_sim)
endfunction()
