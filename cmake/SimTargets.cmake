# Simulator libraries and bounded correctness runners; no hardware timing claim.
function(add_turboquant_sim_targets)
  set(TQ_SIM_TEST_ENV
      "LD_LIBRARY_PATH=${ASCEND_HOME_PATH}/x86_64-linux/devlib:${ASCEND_HOME_PATH}/lib64:${TQ_SIM_LIB_DIR}:$ENV{LD_LIBRARY_PATH}"
      "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}")
  tq_add_native_kernels(sim)
  tq_add_standard_runners(sim)
  foreach(kind IN ITEMS reduce tail)
    tq_add_ascend_kernel(TARGET tq_qwen35_${kind}_canary_sim RUN_MODE sim
        SOURCES "${PROJECT_SOURCE_DIR}/src/host/asc_qwen35_${kind}.cpp")
    tq_add_device_runner(test_qwen35_${kind}_canary_sim
        "${PROJECT_SOURCE_DIR}/tests/npu/test_qwen35_${kind}_canary.cpp" tq_qwen35_${kind}_canary_sim)
  endforeach()
  tq_add_device_runner(bench_qwen35_decode_sim "${PROJECT_SOURCE_DIR}/tests/npu/bench_qwen35_decode.cpp" tq_decode_sim)
  target_compile_definitions(bench_qwen35_decode_sim PRIVATE TQ_BENCH_CAMODEL=1)
  tq_register_device_test(dirty_tail_canary test_dirty_data_canary_sim ARGS --tail LABELS canary)
  tq_register_device_test(dirty_blocktable_canary test_dirty_data_canary_sim ARGS --blocktable LABELS canary)
  tq_register_device_test(sim_production test_turboquant_production_sim ARGS --list --mode all --stage all LABELS production)
  add_custom_target(tq_golden_sim DEPENDS test_golden_prefill_strip_sim
      test_golden_prefill_nz_sim test_golden_scatter_sim test_golden_decode_sim)
  add_custom_target(tq_sim_tests DEPENDS test_live_write_sim test_live_write_nz_sim
      test_nz_ingress_canary_sim test_fused_decode_e2e_sim test_dirty_data_canary_sim
      tq_golden_sim test_qwen35_reduce_canary_sim test_qwen35_tail_canary_sim
      bench_qwen35_decode_sim test_turboquant_production_sim)
  add_custom_target(tq_sim_benchmarks DEPENDS bench_qwen35_decode_sim bench_turboquant_production_sim)
  add_custom_target(tq_sim_kernels DEPENDS tq_prefill_sim tq_prefill_nz_sim
      tq_scatter_sim tq_scatter_nz_sim tq_decode_sim
      tq_qwen35_reduce_canary_sim tq_qwen35_tail_canary_sim)
  if(TQ_BUILD_E8_KERNELS)
    add_dependencies(tq_sim_kernels tq_e8_kernels)
  endif()
  add_custom_target(tq_sim_kernel_libs DEPENDS tq_sim_kernels)
  tq_register_device_test(sim_live_write test_live_write_sim ARGS --op prefill --slices 1 LABELS ingress)
  tq_register_device_test(sim_scatter test_live_write_sim ARGS --op scatter --mode canary LABELS ingress)
  tq_register_device_test(sim_live_write_nz test_live_write_nz_sim ARGS --probe LABELS ingress)
  tq_register_device_test(sim_nz_ingress test_nz_ingress_canary_sim LABELS ingress)
  tq_register_device_test(sim_fused_decode test_fused_decode_e2e_sim ARGS --camodel-smoke LABELS decode)
  tq_register_device_test(sim_qwen35_decode bench_qwen35_decode_sim ARGS --canary --device 0 LABELS canary)
  tq_register_device_test(sim_qwen35_reduce_s1 test_qwen35_reduce_canary_sim ARGS 1 LABELS canary)
  tq_register_device_test(sim_qwen35_reduce_s2 test_qwen35_reduce_canary_sim ARGS 2 LABELS canary)
  tq_register_device_test(sim_qwen35_tail test_qwen35_tail_canary_sim LABELS canary)
  foreach(gate IN ITEMS prefill_strip prefill_nz scatter decode)
    set(golden_args --dump-root "${TQ_GOLDEN_DUMP_ROOT}")
    if(gate STREQUAL "decode")
      list(APPEND golden_args --case "${TQ_GOLDEN_DUMP_ROOT}/decode_ctx128")
    endif()
    tq_register_device_test(sim_golden_${gate} test_golden_${gate}_sim ARGS ${golden_args} LABELS golden)
  endforeach()
endfunction()
