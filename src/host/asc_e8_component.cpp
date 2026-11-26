// Generate ASC host launch glue while retaining the canonical device source.
#if defined(ASCEND_MICRO_CANN_LAUNCH) && !defined(__CCE_AICORE__)
#include "kernel_operator.h"
// BiSheng injects unused launch-config parameters into this empty host stub.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
extern "C" __global__ __aicore__ void e8_parallel_component(
    __gm__ uint16_t*, __gm__ float*, __gm__ uint16_t*, __gm__ int8_t*) {}
#pragma clang diagnostic pop
#endif
#include "../turboquant/decode/e8/parallel_component.cpp"
