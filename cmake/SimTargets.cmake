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
  include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/CannWrapperWarnings.cmake")
  set(RUN_MODE sim)
  set(SRC "${CMAKE_CURRENT_SOURCE_DIR}")
  set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/tools/simulator/${SOC_VERSION}/lib")
  if(NOT EXISTS "${SIM_LIB_DIR}/libmodel_top.so" AND TQ_SOC_FAMILY MATCHES "^ascend950")
    set(SIM_LIB_DIR "${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_3510/lib")
  endif()
  # The CAModel runner needs the HAL devlib and the simulator DSOs on its loader path; the
  # install RPATH already carries $ORIGIN and the toolkit lib64, not these. ctest therefore runs
  # the canary with the same environment `source set_env.sh` would have produced.
  set(TQ_SIM_TEST_ENV
      "LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/x86_64-linux/devlib:${ASCEND_HOME_PATH}/lib64:${SIM_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
      "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")

# ---------------------------------------------------------------- the kernels
  function(add_sim_prefill_pair narrow wide)
    ascendc_library(${wide} SHARED src/turboquant/ingress/native/turboquant_reshape_batch_wide.cpp)
    ascendc_library(${narrow} SHARED src/turboquant/ingress/native/turboquant_reshape_batch.cpp)
    set(entry turboquant_reshape_and_cache_bf16)
    set(${narrow}_exports "launch_${entry}; aclrtlaunch_${entry};")
    set(${wide}_exports "aclrtlaunch_${entry}_wide;")
    set(layout_options)
    foreach(definition IN LISTS ARGN)
      list(APPEND layout_options "-D${definition}")
    endforeach()
    foreach(kernel IN ITEMS ${narrow} ${wide})
      ascendc_compile_definitions(${kernel} PRIVATE
          ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_LEGACY_CANN_LAUNCH=1 ${ARGN})
      ascendc_compile_options(${kernel} PRIVATE
          -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
          -forward-options-to-host-compiler -Wall -Wextra -Werror
          ${TQ_CANN_SYSTEM_OPTIONS} -DASCEND_MICRO_CANN_LAUNCH -DTURBOQUANT_LEGACY_CANN_LAUNCH=1
          ${layout_options})
      target_sources(${kernel} PRIVATE src/host/turboquant_sim_binary_lifetime.cpp)
      target_compile_options(${kernel} PRIVATE -Wall -Wextra -Werror)
      set(export_map "${CMAKE_CURRENT_BINARY_DIR}/${kernel}.exports.map")
      file(GENERATE OUTPUT "${export_map}"
          CONTENT "{ global: ${${kernel}_exports} local: *; };\n")
      set_property(TARGET ${kernel} APPEND PROPERTY LINK_DEPENDS "${export_map}")
      # The generated registration constructor uses these SDK providers before main.
      target_link_libraries(${kernel} PRIVATE
          "-Wl,--push-state,--no-as-needed"
          "${ASCEND_HOME_PATH}/lib64/libascendcl.so"
          "${SIM_LIB_DIR}/libruntime_camodel.so"
          "${ASCEND_HOME_PATH}/lib64/libprofapi.so"
          "-Wl,--pop-state")
      # CANN's profiler can retain callbacks after dlclose. Retain each simulator module
      # until process exit so the callbacks and both binary owners share that lifetime.
      target_link_options(${kernel} PRIVATE
          "LINKER:--version-script=${export_map}" "LINKER:--wrap=RegisterAscendBinary"
          "LINKER:-z,nodelete")
      # The SDK replays its link without shell expansion.
      set_target_properties(${kernel} PROPERTIES
          BUILD_RPATH_USE_ORIGIN FALSE BUILD_RPATH "${CMAKE_BINARY_DIR}/lib")
    endforeach()

    # CANN supplies its own launch adapter to each host build. The public dispatcher also
    # calls the wide entry, so its host build needs the adapter from the companion library.
    ExternalProject_Add_StepDependencies(${narrow}_host configure ${wide}_preprocess)
    set_property(TARGET ${narrow}_interface APPEND PROPERTY HOST_COMPILE_OPTIONS
        "SHELL:-include ${CMAKE_BINARY_DIR}/include/${wide}/aclrtlaunch_triple_chevrons_func.h")
    # Link the companion without inheriting its SDK runner link interface.
    target_link_libraries(${narrow} PRIVATE "$<TARGET_LINKER_FILE:${wide}>")
    add_dependencies(${narrow} ${wide})
  endfunction()

  set_property(SOURCE src/turboquant/ingress/native/turboquant_reshape_batch.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/turboquant/ingress/native/turboquant_reshape_batch.cpp)
  add_sim_prefill_pair(tq_prefill_sim tq_prefill_wide_sim)
  # STRIP and NZ have the same public symbol and each runner links only its own layout.
  add_sim_prefill_pair(tq_prefill_nz_sim tq_prefill_nz_wide_sim TURBOQUANT_INGRESS_NZ=1)

  ascendc_library(tq_scatter_sim SHARED src/turboquant/ingress/native/turboquant_reshape_scatter.cpp)
  set_property(SOURCE src/turboquant/ingress/native/turboquant_reshape_scatter.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/turboquant/ingress/native/turboquant_reshape_scatter.cpp)
  ascendc_compile_definitions(tq_scatter_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_LEGACY_CANN_LAUNCH=1)
  ascendc_compile_options(tq_scatter_sim PRIVATE
      -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -Wall -Wextra -Werror
      ${TQ_CANN_SYSTEM_OPTIONS} -DASCEND_MICRO_CANN_LAUNCH -DTURBOQUANT_LEGACY_CANN_LAUNCH=1)

  # The same scatter writer with the NZ-tiled ingress on. Its own library for the reason given
  # at tq_prefill_nz_sim: the two layouts are two operators behind one entry point, and the
  # strip gate has to keep running against the strip writer in the same configure.
  ascendc_library(tq_scatter_nz_sim SHARED src/turboquant/ingress/native/turboquant_reshape_scatter.cpp)
  set_property(SOURCE src/turboquant/ingress/native/turboquant_reshape_scatter.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/turboquant/ingress/native/turboquant_reshape_scatter.cpp)
  ascendc_compile_definitions(tq_scatter_nz_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_LEGACY_CANN_LAUNCH=1 TURBOQUANT_INGRESS_NZ=1)
  ascendc_compile_options(tq_scatter_nz_sim PRIVATE
      -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -Wall -Wextra -Werror
      ${TQ_CANN_SYSTEM_OPTIONS} -DASCEND_MICRO_CANN_LAUNCH -DTURBOQUANT_LEGACY_CANN_LAUNCH=1 -DTURBOQUANT_INGRESS_NZ=1)

  ascendc_library(tq_decode_sim SHARED src/turboquant/decode/native/turboquant_fused_decode.cpp)
  set_property(SOURCE src/turboquant/decode/native/turboquant_fused_decode.cpp PROPERTY OBJECT_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/turboquant/decode/native/turboquant_fused_decode.cpp)
  ascendc_compile_definitions(tq_decode_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH TURBOQUANT_LEGACY_CANN_LAUNCH=1)
  ascendc_compile_options(tq_decode_sim PRIVATE
      -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS} ${TQ_REDUCE_DEFINES} ${TQ_TELEMETRY_DEFINES}
      -forward-options-to-host-compiler -Wall -Wextra -Werror
      ${TQ_CANN_SYSTEM_OPTIONS} -DASCEND_MICRO_CANN_LAUNCH -DTURBOQUANT_LEGACY_CANN_LAUNCH=1
      "-I${SRC}/src/turboquant")

  # ---------------------------------------------------------------- the runners
  # add_sim_runner(<name> <source> <kernel targets...>)
  # The CAModel libraries depend on further simulator DSOs; -rpath-link lets the link resolve
  # those indirect dependencies, and the install RPATH lets the runner find them at run time.
  tq_scope_cann_wrapper_warnings(
      tq_prefill_sim tq_prefill_wide_sim tq_prefill_nz_sim tq_prefill_nz_wide_sim
      tq_scatter_sim tq_scatter_nz_sim tq_decode_sim)

  function(add_sim_runner name source)
    add_executable(${name} EXCLUDE_FROM_ALL "${source}")
    target_include_directories(${name} PRIVATE "${SRC}" "${ASCEND_HOME_PATH}/include")
    target_compile_options(${name} PRIVATE -std=c++17 -Wall -Wextra -Werror ${TQ_TELEMETRY_DEFINES})
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
  # The NZ scatter gate: the same source, the NZ scatter library, and the NZ scatter rig
  # selected by TQ_SCATTER_NZ=1. The NZ prefill stays gated by test_nz_ingress_canary_sim.
  add_sim_runner(test_live_write_nz_sim tests/npu/test_live_write.cpp
      tq_scatter_nz_sim)
  target_compile_definitions(test_live_write_nz_sim PRIVATE TQ_SCATTER_NZ=1)
  add_sim_runner(test_nz_ingress_canary_sim tests/npu/test_nz_ingress_canary.cpp
      tq_prefill_nz_sim)
  add_sim_runner(test_fused_decode_e2e_sim tests/npu/test_fused_decode_e2e.cpp tq_decode_sim)

  set(TQ_CYCLE_MODEL_DIR "${ASCEND_HOME_PATH}/x86_64-linux/simulator/dav_3510/camodel")
  ascendc_library(tq_qwen35_reduce_canary_sim SHARED tests/npu/qwen35_reduce_canary.cpp)
  ascendc_compile_definitions(tq_qwen35_reduce_canary_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_qwen35_reduce_canary_sim PRIVATE
      -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -DASCEND_MICRO_CANN_LAUNCH)
  tq_scope_cann_wrapper_warnings(tq_qwen35_reduce_canary_sim)
  ascendc_library(tq_qwen35_tail_canary_sim SHARED tests/npu/qwen35_tail_canary.cpp)
  ascendc_compile_definitions(tq_qwen35_tail_canary_sim PRIVATE ASCEND_MICRO_CANN_LAUNCH)
  ascendc_compile_options(tq_qwen35_tail_canary_sim PRIVATE
      -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -DASCEND_MICRO_CANN_LAUNCH)
  tq_scope_cann_wrapper_warnings(tq_qwen35_tail_canary_sim)
  foreach(canary IN ITEMS bench_qwen35_decode_sim test_qwen35_reduce_canary_sim test_qwen35_tail_canary_sim)
    if(canary STREQUAL "bench_qwen35_decode_sim")
      set(canary_source tests/npu/bench_qwen35_decode.cpp)
      set(canary_kernel tq_decode_sim)
    elseif(canary STREQUAL "test_qwen35_tail_canary_sim")
      set(canary_source tests/npu/test_qwen35_tail_canary.cpp)
      set(canary_kernel tq_qwen35_tail_canary_sim)
    else()
      set(canary_source tests/npu/test_qwen35_reduce_canary.cpp)
      set(canary_kernel tq_qwen35_reduce_canary_sim)
    endif()
    add_executable(${canary} EXCLUDE_FROM_ALL ${canary_source})
    target_include_directories(${canary} PRIVATE "${SRC}" "${ASCEND_HOME_PATH}/include")
    target_compile_options(${canary} PRIVATE -std=c++17 -Wall -Wextra -Werror)
    target_compile_definitions(${canary} PRIVATE TQ_BENCH_CAMODEL=1)
    target_link_directories(${canary} BEFORE PRIVATE "${TQ_CYCLE_MODEL_DIR}")
    target_link_libraries(${canary} PRIVATE ${canary_kernel} turboquant_host ascendcl
        "${TQ_CYCLE_MODEL_DIR}/libnpu_drv_camodel.so")
    target_link_options(${canary} PRIVATE "-Wl,-rpath-link,${TQ_CYCLE_MODEL_DIR}")
    set_target_properties(${canary} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "$ORIGIN/../lib;${TQ_CYCLE_MODEL_DIR};${ASCEND_HOME_PATH}/lib64")
  endforeach()

  # DIRTY-DATA (D2/D3): `--tail` uses zero live K/V and finite spikes through the page end,
  # including past align16(context), with exact-zero checks and a one-spike positive control.
  # Each launch has 1-2 live tiles; --blocktable checks an invalid entry beyond the 64-page cache.
  add_sim_runner(test_dirty_data_canary_sim tests/npu/test_dirty_data_canary.cpp tq_decode_sim)
  add_test(NAME dirty_tail_canary COMMAND test_dirty_data_canary_sim --tail)
  add_test(NAME dirty_blocktable_canary COMMAND test_dirty_data_canary_sim --blocktable)
  set_tests_properties(dirty_tail_canary dirty_blocktable_canary
      PROPERTIES ENVIRONMENT "${TQ_SIM_TEST_ENV}")

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
      DEPENDS tq_prefill_sim tq_prefill_nz_sim tq_scatter_sim tq_scatter_nz_sim tq_decode_sim
              test_live_write_sim test_live_write_nz_sim test_nz_ingress_canary_sim
              test_fused_decode_e2e_sim test_dirty_data_canary_sim)
endfunction()
