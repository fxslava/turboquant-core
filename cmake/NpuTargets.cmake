# Physical device libraries and correctness/performance runners.
function(add_turboquant_npu_targets)
  tq_add_native_kernels(npu)
  tq_add_standard_runners(npu)
  tq_add_device_runner(bench_qwen35_decode "${PROJECT_SOURCE_DIR}/tests/npu/bench_qwen35_decode.cpp" tq_decode_npu)
  add_custom_target(tq_npu_tests DEPENDS
      test_live_write_npu test_live_write_nz_npu test_nz_ingress_canary_npu
      test_fused_decode_e2e_npu test_dirty_data_canary_npu
      test_golden_prefill_strip_npu test_golden_prefill_nz_npu
      test_golden_scatter_npu test_golden_decode_npu test_turboquant_production_npu)
  add_custom_target(tq_npu_benchmarks DEPENDS bench_qwen35_decode bench_turboquant_production_npu)
  add_custom_target(tq_npu_production DEPENDS test_turboquant_production_npu bench_turboquant_production_npu)
  add_custom_target(tq_npu_kernels DEPENDS tq_prefill_npu tq_prefill_nz_npu
      tq_scatter_npu tq_scatter_nz_npu tq_decode_npu)
  if(TQ_BUILD_E8_KERNELS)
    add_dependencies(tq_npu_kernels tq_e8_kernels)
  endif()
  add_custom_target(tq_npu_kernel_libs DEPENDS tq_npu_kernels)
  tq_register_device_test(npu_live_write test_live_write_npu LABELS ingress)
  tq_register_device_test(npu_live_write_nz test_live_write_nz_npu LABELS ingress)
  tq_register_device_test(npu_nz_ingress test_nz_ingress_canary_npu LABELS ingress)
  tq_register_device_test(npu_fused_decode test_fused_decode_e2e_npu LABELS decode)
  tq_register_device_test(npu_dirty_tail test_dirty_data_canary_npu ARGS --tail LABELS canary)
  tq_register_device_test(npu_dirty_blocktable test_dirty_data_canary_npu ARGS --blocktable LABELS canary)
  foreach(gate IN ITEMS prefill_strip prefill_nz scatter decode)
    tq_register_device_test(npu_golden_${gate} test_golden_${gate}_npu
        ARGS --dump-root "${TQ_GOLDEN_DUMP_ROOT}" LABELS golden)
  endforeach()
  if(TARGET test_turboquant_production_npu)
    tq_register_device_test(npu_production test_turboquant_production_npu LABELS production)
  endif()
endfunction()
