#include <cstdint>

#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
#include "../../include/turboquant/turboquant_common.h"
#include "../../src/decode/vector/routine_vector_cube_softmax.hpp"

extern "C" __global__ __aicore__ void qwen35_tail_canary(GM_ADDR output)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    AscendC::InitSocState();
    AscendC::TPipe pipe;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> arena_buffer, state_buffer, tails_buffer;
    pipe.InitBuffer(arena_buffer, 8192);
    pipe.InitBuffer(state_buffer, 192);
    pipe.InitBuffer(tails_buffer, 192);
    if ASCEND_IS_AIV {
        using Layout = vllm_ascend::turboquant::TelemetryPredicateLayout;
        const auto arena = arena_buffer.Get<float>();
        const auto state = state_buffer.Get<float>();
        const auto tails = tails_buffer.Get<float>();
        AscendC::GlobalTensor<float> result;
        result.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        AscendC::ArithProgression(arena[128].ReinterpretCast<int32_t>(), 0, 1, 64);
        AscendC::Cast(arena[128], arena[128].ReinterpretCast<int32_t>(), AscendC::RoundMode::CAST_NONE, 64);
        AscendC::Duplicate(state, 1.25f, 24);
        AscendC::Duplicate(state[24], 64.0f, 24);
        AscendC::Duplicate(arena[256], 0.0f, 24);
        if (AscendC::GetSubBlockIdx() != 0) AscendC::Duplicate(arena[264], 1.0f, 8);
        for (uint32_t fixed = 0; fixed < 2; ++fixed) {
            const auto position = arena[Layout::kPosition].ReinterpretCast<uint8_t>();
            const auto page = arena[Layout::kPage].ReinterpretCast<uint8_t>();
            AscendC::Compares(position, arena[128], 1.0f, AscendC::CMPMODE::EQ, 16);
            AscendC::Compares(page, arena[128], 2.0f, AscendC::CMPMODE::EQ, 16);
            const uint32_t scratch_offset = fixed ? Layout::kArgmax : Layout::kPosition;
            AscendC::Compares(arena[scratch_offset].ReinterpretCast<uint8_t>(), arena[256],
                0.0f, AscendC::CMPMODE::GT, 24);
            ascend_micro::vector_cube_pack_partial_tails(tails, state, state[24], 3, 16, 0, 8);
            for (uint32_t head = 0; head < 3; ++head) {
                AscendC::Duplicate(arena[192], 37.0f + static_cast<float>(head), 64);
                AscendC::Select(tails[head * 16], position, arena[192], tails[head * 16],
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 16);
                AscendC::Duplicate(arena[192], 2.0f, 64);
                AscendC::Select(tails[head * 16], page, arena[192], tails[head * 16],
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, 16);
            }
            vllm_ascend::turboquant::SyncVectorToMte3();
            AscendC::DataCopy(result[fixed * 96 + AscendC::GetSubBlockIdx() * 48], tails, 48);
            vllm_ascend::turboquant::SyncMte3ToVector();
        }
    }
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void qwen35_tail_canary(GM_ADDR);
extern "C" void launch_qwen35_tail_canary(void* stream, void* output)
{
    qwen35_tail_canary<<<1, nullptr, stream>>>(static_cast<uint8_t*>(output));
}
#endif