#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../src/turboquant/decode/native/cube/turboquant_gemm_reduce_service.h"

struct CanarySelect {
    __aicore__ inline bool NeedsCombine(uint32_t) const { return true; }
};

extern "C" __global__ __aicore__ void qwen35_reduce_canary(GM_ADDR workspace, GM_ADDR output,
    uint32_t splits)
{

    AscendC::InitSocState();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::A1> attention_l1;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> prefix;
    AscendC::TBufPool<AscendC::TPosition::VECCALC> shared;
    pipe.InitBuffer(attention_l1, ascend_micro::gemm_reduce::kAttentionL1Bytes);
    pipe.InitBuffer(prefix, 43200);
    pipe.InitBufPool(shared, ascend_micro::fused_decode_opt::kTransientPoolBytes);
    ascend_micro::gemm_reduce::TurboQuantBatchedGemmReducer<bfloat16_t> reducer(&pipe);
    reducer.Init(workspace, output, 6, 256, splits, shared);
    CanarySelect select;
    vllm_ascend::turboquant::TurboQuantRotatedOutput finisher;
    reducer.Reduce(6, 3, select, finisher);
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void qwen35_reduce_canary(GM_ADDR, GM_ADDR, uint32_t);
extern "C" void launch_qwen35_reduce_canary(void* stream, void* workspace, void* output,
    uint32_t splits)
{
    qwen35_reduce_canary<<<2, nullptr, stream>>>(static_cast<uint8_t*>(workspace),
        static_cast<uint8_t*>(output), splits);
}
#endif
