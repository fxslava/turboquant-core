# Vendored-decode probe: compile C:\vllm-ascend's decode/native/turboquant_fused_decode.cpp
# (the wheel's entry-point adaptation) with the wheel's device define set, through the
# harness's own BiSheng pipeline. The result, swapped in for libtq_decode_npu.so on the
# NPU box, lets test_golden_decode_npu arbitrate:
#
#   PASS  -> the vendored source and its entry adaptation are healthy; the wheel's build
#            product (toolchain or stale objects) is the defect -> clean rebuild earned.
#   FAIL  -> the entry adaptation itself breaks the kernel on real hardware -> fix the
#            entry section, not the build.
#
# Off by default. Configure with:
#   cmake -B build-npu -DRUN_MODE=npu -DTQ_BUILD_KERNELS=ON -DTQ_VENDORED_DECODE_PROBE=ON \
#         [-DTQ_VENDORED_DECODE_SRC=/path/to/turboquant_fused_decode.cpp]

option(TQ_VENDORED_DECODE_PROBE "Build the vendored fused-decode probe library" OFF)
set(TQ_VENDORED_DECODE_SRC
    "/mnt/c/vllm-ascend/csrc/attention/turboquant/decode/native/turboquant_fused_decode.cpp"
    CACHE FILEPATH "Vendored fused_decode.cpp to compile into the probe library")

function(add_vendored_decode_probe)
  if(NOT EXISTS "${TQ_VENDORED_DECODE_SRC}")
    message(FATAL_ERROR "TQ_VENDORED_DECODE_PROBE=ON but no source at ${TQ_VENDORED_DECODE_SRC}")
  endif()

  tq_add_ascend_kernel(TARGET tq_decode_vendored_npu RUN_MODE npu
      SOURCES "${TQ_VENDORED_DECODE_SRC}"
      DEFS ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 TURBOQUANT_INGRESS_NZ=1)

  # Same library name as the harness decode, into a probe directory: swapping
  # probe/lib/libtq_decode_npu.so in for build-npu/lib/libtq_decode_npu.so on the
  # NPU box is the whole experiment, and the runners need no relink.
  set_target_properties(tq_decode_vendored_npu PROPERTIES
      OUTPUT_NAME "tq_decode_npu"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/probe/lib")
endfunction()
