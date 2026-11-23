#include <cstdint>
#include "e8_packet_layout.hpp"
#if defined(__CCE_AICORE__)
#include "kernel_operator.h"
namespace e8_encoded_batch {
constexpr uint32_t kDepth = 4;
constexpr uint32_t kStrips = kDepth + 2;
constexpr uint32_t kRowsPerStrip = 16;
constexpr uint32_t kLabelsPerStrip = kRowsPerStrip * 32;
constexpr uint32_t kLabelStrideElements = 528; // 1024B payload +32B skew.
constexpr uint32_t kScaleStrideElements = 24;  // 64B payload +32B skew.
constexpr uint32_t kUbBytes = 9216;
static_assert(kDepth == 4 && kStrips == 6);
static_assert(8832 + kDepth * kScaleStrideElements * sizeof(float) == kUbBytes);
static_assert(kUbBytes <= 216u * 1024u);
// Input mode2 means already-encoded E8 labels/scales. It is not a BF16 quantizer.
constexpr uint32_t kEncodedE8Packet = 2;
}
extern "C" __global__ __aicore__ void e8_packet_v1_batch_encoded(
  __gm__ uint8_t* keyLabels, __gm__ uint8_t* keyScales,
  __gm__ uint8_t* valueLabels, __gm__ uint8_t* valueScales,
  __gm__ uint8_t* keyPackets, __gm__ uint8_t* valuePackets,
  uint32_t strips, uint32_t inputMode) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  AscendC::InitSocState();
  using namespace e8_encoded_batch;
  // Probe sizing and mode dispatch happen once, outside the uniform grinder.
  if (strips != kStrips || inputMode != kEncodedE8Packet) return;
  AscendC::GlobalTensor<uint16_t> keyIn, valueIn, keyOut, valueOut;
  AscendC::GlobalTensor<float> keyScaleIn, valueScaleIn, keyScaleOut, valueScaleOut;
  keyIn.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(keyLabels));
  valueIn.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(valueLabels));
  keyScaleIn.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keyScales));
  valueScaleIn.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(valueScales));
  keyOut.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(keyPackets));
  valueOut.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(valuePackets));
  keyScaleOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(keyPackets));
  valueScaleOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(valuePackets));
  // Four disjoint, skewed rings. Scalar instructions touch addresses only.
  // @ascend-layout: name=keyLabelRing pos=VECIN offset=0 count=2112 dtype=uint16_t
  const AscendC::LocalTensor<uint16_t> keyLabelRing{AscendC::TPosition::VECIN,0,2112};
  // @ascend-layout: name=keyScaleRing pos=VECIN offset=4224 count=96 dtype=float
  const AscendC::LocalTensor<float> keyScaleRing{AscendC::TPosition::VECIN,4224,96};
  // @ascend-layout: name=valueLabelRing pos=VECIN offset=4608 count=2112 dtype=uint16_t
  const AscendC::LocalTensor<uint16_t> valueLabelRing{AscendC::TPosition::VECIN,4608,2112};
  // @ascend-layout: name=valueScaleRing pos=VECIN offset=8832 count=96 dtype=float
  const AscendC::LocalTensor<float> valueScaleRing{AscendC::TPosition::VECIN,8832,96};
#pragma clang loop unroll(disable)
  for (uint32_t slot=0;slot<kDepth;++slot)
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
  // Identical branch-free strip grinder, instantiated with constant ring IDs.
  // Source preprocessing exposes each token edge to AKA; no category suppressions.
#define E8_ENCODED_STRIP(STRIP, SLOT) { \
    constexpr uint32_t labelSlot=(SLOT)*kLabelStrideElements; \
    constexpr uint32_t scaleSlot=(SLOT)*kScaleStrideElements; \
    constexpr uint32_t packet=(STRIP)/4u; \
    constexpr uint32_t row=((STRIP)%4u)*kRowsPerStrip; \
    constexpr uint32_t labelOut=(packet*e8_packet_abi::kPacketBytes+row*64u)/2u; \
    constexpr uint32_t scaleOut=(packet*e8_packet_abi::kPacketBytes+4096u)/4u+row; \
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(SLOT); \
    AscendC::DataCopy(keyLabelRing[labelSlot],keyIn[(STRIP)*kLabelsPerStrip],kLabelsPerStrip); \
    AscendC::DataCopy(keyScaleRing[scaleSlot],keyScaleIn[(STRIP)*kRowsPerStrip],kRowsPerStrip); \
    AscendC::DataCopy(valueLabelRing[labelSlot],valueIn[(STRIP)*kLabelsPerStrip],kLabelsPerStrip); \
    AscendC::DataCopy(valueScaleRing[scaleSlot],valueScaleIn[(STRIP)*kRowsPerStrip],kRowsPerStrip); \
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(SLOT); \
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(SLOT); \
    AscendC::DataCopy(keyOut[labelOut],keyLabelRing[labelSlot],kLabelsPerStrip); \
    AscendC::DataCopy(keyScaleOut[scaleOut],keyScaleRing[scaleSlot],kRowsPerStrip); \
    AscendC::DataCopy(valueOut[labelOut],valueLabelRing[labelSlot],kLabelsPerStrip); \
    AscendC::DataCopy(valueScaleOut[scaleOut],valueScaleRing[scaleSlot],kRowsPerStrip); \
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(SLOT); \
  }
  E8_ENCODED_STRIP(0,0)
  E8_ENCODED_STRIP(1,1)
  E8_ENCODED_STRIP(2,2)
  E8_ENCODED_STRIP(3,3)
  E8_ENCODED_STRIP(4,0)
  E8_ENCODED_STRIP(5,1)
#undef E8_ENCODED_STRIP
#pragma clang loop unroll(disable)
  for (uint32_t slot=0;slot<kDepth;++slot)
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
  AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(0);
  AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(0);
}
#elif defined(ASCEND_MICRO_CANN_LAUNCH)
extern "C" __global__ __aicore__ void e8_packet_v1_batch_encoded(
  __gm__ uint8_t*,__gm__ uint8_t*,__gm__ uint8_t*,__gm__ uint8_t*,
  __gm__ uint8_t*,__gm__ uint8_t*,uint32_t,uint32_t);
extern "C" void launch_e8_packet_v1_batch_encoded(void* stream,uint8_t* kl,uint8_t* ks,
 uint8_t* vl,uint8_t* vs,uint8_t* kp,uint8_t* vp,uint32_t strips,uint32_t mode) {
 e8_packet_v1_batch_encoded<<<1,nullptr,stream>>>(kl,ks,vl,vs,kp,vp,strips,mode);
}
#endif
