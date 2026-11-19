# Vendored-decode probe: compile C:\vllm-ascend's op_kernel/turboquant_fused_decode.cpp
# (the wheel's entry-point adaptation) with the wheel's exact define set, through the
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
    "/mnt/c/vllm-ascend/csrc/attention/turboquant/op_kernel/turboquant_fused_decode.cpp"
    CACHE FILEPATH "Vendored fused_decode.cpp to compile into the probe library")

function(add_vendored_decode_probe)
  if(NOT EXISTS "${TQ_VENDORED_DECODE_SRC}")
    message(FATAL_ERROR "TQ_VENDORED_DECODE_PROBE=ON but no source at ${TQ_VENDORED_DECODE_SRC}")
  endif()
  get_filename_component(_vendored_op_kernel "${TQ_VENDORED_DECODE_SRC}" DIRECTORY)
  get_filename_component(_vendored_tree "${_vendored_op_kernel}" DIRECTORY)

  ascendc_library(tq_decode_vendored_npu SHARED "${TQ_VENDORED_DECODE_SRC}")
  set_property(SOURCE "${TQ_VENDORED_DECODE_SRC}" PROPERTY OBJECT_DEPENDS "${TQ_VENDORED_DECODE_SRC}")
  # The wheel's define set, verbatim from vllm-ascend's CMakeLists for
  # vllm_ascend_turboquant: same two definitions as the harness PLUS the NZ ingress
  # define (inert for this translation unit, carried here so the probe compiles
  # exactly what the wheel compiles).
  ascendc_compile_definitions(tq_decode_vendored_npu PRIVATE
      ASCEND_MICRO_CANN_LAUNCH ASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1 TURBOQUANT_INGRESS_NZ=1)
  ascendc_compile_options(tq_decode_vendored_npu PRIVATE
      -std=c++17 -Wall -Wextra -Werror ${TQ_CANN_SYSTEM_OPTIONS}
      -forward-options-to-host-compiler
      -DASCEND_MICRO_CANN_LAUNCH
      -DASCEND_TQ_FUSED_DECODE_GEMM_REDUCE=1
      -DTURBOQUANT_INGRESS_NZ=1
      "-I${_vendored_op_kernel}"
      "-I${_vendored_op_kernel}/common"
      "-I${_vendored_tree}/include")

  # Same library name as the harness decode, into a probe directory: swapping
  # probe/lib/libtq_decode_npu.so in for build-npu/lib/libtq_decode_npu.so on the
  # NPU box is the whole experiment, and the runners need no relink.
  set_target_properties(tq_decode_vendored_npu PROPERTIES
      OUTPUT_NAME "tq_decode_npu"
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/probe/lib")
endfunction()
