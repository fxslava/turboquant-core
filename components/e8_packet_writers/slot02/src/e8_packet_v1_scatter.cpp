#include <cstdint>
#include "e8_packet_layout.hpp"
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
extern "C" __global__ __aicore__ void e8_packet_v1_scatter(
    __gm__ uint8_t* inputLabels, __gm__ uint8_t* inputScale, __gm__ uint8_t* outputPacket, uint32_t row) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::InitSocState();
    if (row >= e8_packet_abi::kRows) return;
    // One AIV, one encoded D256 token/plane; caller owns (packet,row).
    // No scalar tensor loads, quantization, arithmetic, VF or Cube work.
    constexpr std::uint32_t kUbBytes = 96;
    static_assert(kUbBytes <= 216u * 1024u);
    AscendC::GlobalTensor<std::uint16_t> labelsIn;
    AscendC::GlobalTensor<float> scaleIn;
    AscendC::GlobalTensor<std::uint16_t> labelsOut;
    AscendC::GlobalTensor<float> scalesOut;
    labelsIn.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint16_t*>(inputLabels));
    scaleIn.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(inputScale));
    labelsOut.SetGlobalBuffer(reinterpret_cast<__gm__ std::uint16_t*>(outputPacket));
    scalesOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(outputPacket + e8_packet_abi::kScaleOffset));
    // @ascend-layout: name=labels pos=VECIN offset=0 count=32 dtype=uint16_t
    const AscendC::LocalTensor<std::uint16_t> labels{AscendC::TPosition::VECIN, 0, 32};
    // @ascend-layout: name=scale pos=VECIN offset=64 count=8 dtype=float
    const AscendC::LocalTensor<float> scale{AscendC::TPosition::VECIN, 64, 8};
    AscendC::DataCopy(labels, labelsIn, 32u);
    const AscendC::DataCopyExtParams scaleCopy{1, 4, 0, 0, 0};
    const AscendC::DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    AscendC::DataCopyPad(scale, scaleIn, scaleCopy, noPad);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(0);
    AscendC::DataCopy(labelsOut[row * 32u], labels, 32u);
    // Exactly four bytes at scale slot; preserve neighbouring tokens.
    AscendC::DataCopyPad(scalesOut[row], scale, scaleCopy);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void e8_packet_v1_scatter(
    __gm__ uint8_t*, __gm__ uint8_t*, __gm__ uint8_t*, uint32_t);
extern "C" void launch_e8_packet_v1_scatter(void* stream, std::uint8_t* labels,
    std::uint8_t* scale, std::uint8_t* packet, std::uint32_t row) {
    e8_packet_v1_scatter<<<1, nullptr, stream>>>(labels, scale, packet, row);
}
#endif

