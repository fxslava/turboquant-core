/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// TurboQuant BF16 prefill reshape-and-cache. Ascend 950PR / arch35 / dav-3510 / CANN 9.2,
// MIX_AIC_1_2: one Cube core and both Vector cores in a four-stage modulo pipeline.
//
//   engine        owner        work
//   ------------  -----------  ----------------------------------------------------
//   MTE2          AIC          GM -> L1 page, one strided ND->NZ burst per chunk
//   MTE1/M/FIX    AIC          L1 -> L0, 64 BF16 Mmads, one Fixpipe -> AIV UB
//   Vector        AIV0 + AIV1  absmax, power-of-two scale, INT4 nibble pack
//   MTE3          AIV0 + AIV1  packed page + scale slice -> GM
//
// THE ROTATION IS ONE CUBE PASS. Per head of kHeadSize channels the transform is the
// randomised Hadamard
//
//   Y = M . x,   M = (H8 (x) I32) . diag(piSigns) . (I8 (x) H32)
//
// normalised by invSqrtLen = 1 / sqrt(headSize). Writing the channel index as
// ch = c1 * kGroup + c0, the entries of M are
//
//   M[c1*32 + c0][g*32 + i] = H8[c1][g] . piSigns[g*32 + c0] . H32[c0][i]              (1)
//
// All of (1) is folded into the Cube. The contraction is split over the eight source groups
// g, and the outer factor's sign H8[c1][g] is folded into the resident operand: for output
// group c1 and source group g the Cube multiplies the page's group-g fractal by the 32x32
// block
//
//   T[s][g][c0][i] = (s ? -1 : +1) . piSigns[g*32 + c0] . H32[c0][i],
//   s = parity(c1 & g) = (H8[c1][g] < 0)                                               (2)
//
// and accumulates the eight products for one c1 into one L0C region. There are only
// 2 * kGroups = 16 distinct blocks, because the dependence on c1 is a single sign; `tables`
// carries them and they stay resident in both L0A and L0B for the whole kernel. No part of
// the orthogonal transform runs on the Vector cores: no sign diagonal, no butterfly cascade.
//
// Both factors are exact in FP32 for integer-valued BF16 activations: each row of M has 256
// signed entries, a BF16 significand is 8 bits, so a product of two integer inputs is exact
// and a sum of 256 of them is an integer below 2^24. The rotation is therefore bit-exact
// against an integer oracle for that class of input.
//
// kGroup is 32 because a 32-channel group is a whole number of fractals at either width: the
// arch-3510 BF16 fractal is 16 x 16, so a group is two fractals along the channel axis and a
// 32 x 32 resident block is four of them.
//
// ONE L1 IMAGE, TWO OPERAND ROLES. The page is NZ(kTokens, kHeadSize), so its only
// fractal-grid axis is the channel axis; the group-G slice read as ZN(kGroup, kTokens) asks
// for the same bytes with the second index renamed. So one GM -> L1 ND -> NZ copy per chunk
// produces a page that serves as
//
//   A operand, NZ(16, 32)  -> C[t][c0out]   for the key chunks, and
//   B operand, ZN(32, 16)  -> C[c0out][t]   for the value chunks,
//
// which is the orientation each cache layout needs. The same symmetry covers the resident
// blocks: NZ(32, 32) of W and ZN(32, 32) of W-as-B are the same image. What a 16-bit element
// width does NOT allow is treating a row-major 32 x 32 block as that image -- at C0 = 16 both
// grid axes are live and the required image is four 16 x 16 row-major sub-blocks ordered
// column-block major. So `tables` stays a flat row-major blob in GM and is converted ND -> NZ
// on the way into L1 by routine_stage_page_zn_bf16.hpp. MTE1 cannot repair it downstream: the
// L1 -> L0 copy unit is the 512-byte fractal and its only rearrangement is an intra-fractal
// transpose, so ingress is the last stage that can reorder sub-fractals.
//
// LAYOUT CONTRACTS (RFC 3.1; the normative statement is turboquant_layout.h).
//   keyCache   key-major. Within the (block, head) page the nibble index is
//              key * headSize + d.
//   valueCache channel-major INSIDE A STRIP. Within the (block, head) page the nibble index
//              is (key / kTokens) * (kTokens * headSize) + ch * kTokens + (key % kTokens).
//   scaleCache per (kv, block, head) a contiguous run of blockSize FP32, one scale per token:
//              ((kv * numBlocks + block) * numKvHeads + head) * blockSize + key.
//   Nibbles    two per byte, low nibble first, signed INT4 in [-8, 7]. This is what a
//              hardware Cast to int4b_t produces directly.
//   tables     kTableBlocks = 16 row-major 32 x 32 BF16 blocks, block s*8 + g holding
//              T[s][g] of (2).
//
// WORK SHARING: ONE CHUNK PER VECTOR CORE, NOT ONE TOKEN HALF. Channel-major means a
// channel's blockSize nibbles are adjacent, so one channel of a 16-key page is 8 bytes; a
// core holding only 8 of the 16 keys owns 4 bytes of every such group and can never assemble
// a 32-byte MTE3 burst, which is the egress granularity. Splitting by chunk -- chunk c is
// owned by Vector core c & 1 -- gives each core whole 16-token chunks, so both cache layouts
// come out as one contiguous burst, and adjacent chunks still sit on different cores, so the
// two Vector cores are in flight together.
//
// THE QUANTISER IS POWER-OF-TWO, AND THAT IS A VERIFICATION DECISION. With
// 2^e <= absmax < 2^(e+1) the scale is 2^(e-2) and the quantised value is
// min(rint(Y * 2^(2-e)), 7), clamped only from above because |Y * 2^(2-e)| < 8 and rint can
// reach +8. 2^(2-e) is a power of two, so the product is exact in FP32 and round-to-nearest-
// even on it is exact: the device and an integer host oracle agree bit for bit. A 7/absmax
// scale would put the rounding of a hardware Div inside the verification contract, where a
// 1-ULP reciprocal can flip a tie and change a nibble. The exponent is extracted with integer
// ops on the FP32 bit pattern, so no division or logarithm runs. invSqrtLen folds into the
// stored scale, not into the quantised value, since scaling Y and absmax together leaves
// every nibble unchanged.
//
// THE SEAMLESS CHUNK PIPELINE. A block owns tokensPerCore tokens, which is
// strips = tokensPerCore / kTokens strips, and runs them as ONE chunk stream of
// total_chunks = strips * kChunks columns rather than as a strip loop. Column j stages chunk
// j and multiplies chunk j - kIngressLookahead, and both indices run across strip boundaries:
// the prologue ingress of strip s + 1 is issued inside the Mmads of strip s, so MTE2 never
// drains at the end of a 16-token strip. The ring periods divide kChunks (asserted below), so
// a strip boundary does not shift which slot a chunk owns and every slot trait stays a
// function of the chunk index alone.
//
// SLOT MAPPING CONTRACT. Strip i of a block fills one whole STRIP of a physical page:
// slotMapping[i * kTokens + k] must equal slotMapping[i * kTokens] + k, and
// slotMapping[i * kTokens] must be a multiple of kTokens. The kernel reads one slotMapping
// entry per strip, on the scalar unit, before that strip's first egress; that is a scalar read
// of caller-supplied Global Memory metadata, not of computed Unified Buffer data.
//
// SYNCHRONISATION. A marked graph: one token per free slot, every feedback cycle primed
// before the schedule and drained after it. Inside a core the edges are HardEvent tokens on
// the route whose consumer queue must stall; across cores they are the three mode-2 channels
// of turboquant_wave_protocol.hpp. Every cross-core set that means "my pipe work has finished"
// is preceded by a HardEvent::x_S drain of that pipe, because the cross-core set issues on the
// scalar stream and the pipe argument does not order it.
//
// WAVES. A grid wider than the physical block count runs in waves, and the scheduler hands a
// physical block straight to its next wave's block without draining anything, so a burst still
// in flight would land in the next wave's ring slot -- at the same compile-time address. Each
// core therefore ends with an inter-wave epilogue drain of every route it used, and the Cube's
// kSync2PageDrained broadcast is the join that keeps a Vector core from entering the next wave
// while a Fixpipe of this one is still in flight.
//
// Target: Ascend950PR_9599, __NPU_ARCH__ == 3510, CANN 9.2.0-beta.2.

#include <cstdint>
#include <utility>

#if defined(__CCE_AICORE__)
#include "tensor_api/tensor.h"
#include "kernel_operator.h"
// The resident-block ingress: at 16 bits a row-major 32 x 32 block is not its own NZ image,
// so the blocks are converted on the way into L1 rather than copied flat.
#include "routine_stage_page_zn_bf16.hpp"
#endif

// After kernel_operator.h on purpose: this header's TQ_WAVE_FN marks every function in it
// `__aicore__ inline` for a device pass, and `__aicore__` is defined by kernel_operator.h.
// Including it earlier leaves turboquant_wave::TotalStrips host-attributed in the device
// object and fails the device link.
#include "../../include/turboquant/turboquant_wave_protocol.hpp"

namespace turboquant_bf16 {

// ------------------------------------------------------------ strip geometry
constexpr std::uint32_t kTokens = 16;   // tokens per strip; blockSize is a multiple of it
constexpr std::uint32_t kHeads = 4;     // numKvHeads
constexpr std::uint32_t kHeadSize = 256;
constexpr std::uint32_t kGroup = 32;    // channels per Hadamard group
constexpr std::uint32_t kGroups = kHeadSize / kGroup;  // 8, order of the outer factor
constexpr std::uint32_t kChunks = 2u * kHeads;         // 4 value chunks then 4 key chunks
constexpr std::uint32_t kPanel = kTokens * kHeadSize;  // 4096 elements per chunk
constexpr std::uint32_t kPackedBytes = kPanel / 2u;    // 2048 bytes of nibbles
constexpr std::uint32_t kScaleFloats = kTokens;        // one scale per token

// ----------------------------------------------------- element width, arch35
// An element count and a byte count are different numbers at 16 bits: raw L1 and Unified
// Buffer addresses are BYTE offsets, while an offset added to a __cbuf__/__ca__/__cb__
// bfloat16_t* advances ELEMENTS. Every constant below says which it is in its name.
constexpr std::uint32_t kElemBytes = 2u;  // sizeof(bfloat16_t)
//
// The fractal is 16 x (32B / sizeof(T)), 512 bytes, so K0 is 16 at 16 bits and a 32-wide group
// is TWO fractals. kGroup stays 32 regardless: it is the width of the ALGORITHM's Hadamard
// group, not of the hardware's contraction step, so header equation (1), the 16 resident blocks
// and the host oracle are unchanged and a chunk is still 64 Mmads of k = 32. The layout
// machinery derives C0 from the element type, so a 32-wide group tile is described the same way
// at either width and simply resolves to two fractals.
constexpr std::uint32_t kFractalK = 32u / kElemBytes;            // 16
constexpr std::uint32_t kFractalsPerGroup = kGroup / kFractalK;  // 2
static_assert(kFractalsPerGroup * kFractalK == kGroup,
              "a Hadamard group must be a whole number of hardware fractals");

// ------------------------------------------------------- rotation table shape
// One 32x32 bfloat16_t block per (sign, source group) pair: block s * kGroups +
// g is T[s][g] of equation (2) in the header. The sign is the only way the
// outer factor's H8[c1][g] enters, so 16 blocks cover all 64 (output group,
// source group) pairs of a chunk. The entries are +-1, exactly representable.
constexpr std::uint32_t kBlockElems = kGroup * kGroup;              // 1024
constexpr std::uint32_t kTableBlocks = 2u * kGroups;                // 16
constexpr std::uint32_t kTableElems = kTableBlocks * kBlockElems;   // 16384
constexpr std::uint32_t kTableBytes = kTableElems * kElemBytes;     // 32768
constexpr std::uint32_t kTableRows = kTableBlocks * kGroup;     // 512, as one ND blob

// Ring depths. The L1 ring is the pipeline depth; the per-core UB rings are
// half of it because each core owns every other chunk.
constexpr std::uint32_t kStageDepth = 4;
constexpr std::uint32_t kLandDepth = 2;
constexpr std::uint32_t kPackDepth = 2;

static_assert(kHeadSize == kGroups * kGroup);
static_assert(kTableRows * kGroup == kTableElems,
              "the table blob must be a whole number of 32-wide rows");
static_assert(kChunks / 2u == kLandDepth * 2u,
              "each core must wrap its landing ring at least twice");

}  // namespace turboquant_bf16

#if defined(__CCE_AICORE__)

// Mode-2 cross-core channels and a Fixpipe that picks a Vector core both need a
// non-1:1 AIC:AIV ratio. At 1:1 there is one Vector core per Cube core, the
// sub-block index is meaningless and the library reroutes cross-core traffic.
#if defined(__MIX_CORE_AIC_RATION__) && (__MIX_CORE_AIC_RATION__ == 1)
#error "MIX_AIC_1_2 is required: a 1:1 AIC:AIV ratio has no second Vector core"
#endif

namespace turboquant_bf16 {

// --------------------------------------------------------------- chunk traits
// Chunks 0..3 are the value heads, 4..7 the key heads, which is the order the
// four-stage schedule runs them in. Every trait is a compile-time value, so no
// column of the schedule contains a runtime predicate or a modulo.
template <std::uint32_t C>
constexpr bool kIsKey = (C >= kHeads);
template <std::uint32_t C>
constexpr std::uint32_t kChunkHead = C % kHeads;
template <std::uint32_t C>
constexpr std::uint32_t kOwner = C & 1u;
template <std::uint32_t C>
constexpr std::uint32_t kL1Slot = C % kStageDepth;
template <std::uint32_t C>
constexpr std::uint32_t kLandSlot = (C / 2u) % kLandDepth;
template <std::uint32_t C>
constexpr std::uint32_t kPackSlot = (C / 2u) % kPackDepth;

// H8[c1][g] = (-1)^popcount(c1 & g), so the block for (c1, g) is the one whose
// sign flag is that parity. Resolved at compile time, once per Mmad site.
constexpr std::uint32_t ParityOf(std::uint32_t v)
{
    std::uint32_t parity = 0;
    while (v != 0u) {
        parity ^= (v & 1u);
        v >>= 1;
    }
    return parity;
}
template <std::uint32_t C1, std::uint32_t G>
constexpr std::uint32_t kSignBlock = ParityOf(C1 & G) * kGroups + G;

// --------------------------------------------------------------- UB arena
// Rule AKA3003: every operand base 32-byte aligned, one DataBlock of bank skew
// between live slots, and VECTOR_REG_WIDTH of addressable UB behind the last
// operand because a level-2 vector op reads a whole vector register per source
// unmasked. One map serves both Vector cores, and the Cube forms its Fixpipe
// destination from the same constants.
// The constants and both helpers come from turboquant_wave_protocol.hpp, so
// the arena algebra the device build uses is the same declaration the host
// than a second copy of it that can drift.
using turboquant_wave::kBankSkewBytes;
using turboquant_wave::kBlock;
using turboquant_wave::kReadPadBytes;
using turboquant_wave::kUbWindow;
using turboquant_wave::NextSlot;
using turboquant_wave::RoundUpBlock;
constexpr std::uint32_t kLanesPerBlock = kBlock / sizeof(float);  // 8

constexpr std::uint32_t kPanelBytes = kPanel * 4u;        // 16384, FP32 or INT32
constexpr std::uint32_t kLandSlotBytes = kPanelBytes + kBankSkewBytes;
constexpr std::uint32_t kPackSlotBytes =
    RoundUpBlock(kPackedBytes + kScaleFloats * 4u) + kBankSkewBytes;

// The Fixpipe landing ring stays at UB 0 so the Cube forms a raw __ubuf__
// pointer without depending on any Vector-side allocation.
constexpr std::uint32_t kUbLandRing = 0u;
constexpr std::uint32_t kUbLandBytes = kLandDepth * kLandSlotBytes;

// Working panels. Single buffered: reuse across chunks is a same-pipe Vector
// dependency, which the Vector queue orders in issue order, not a cross-engine
// hazard: the Vector queue orders a same-pipe dependency in issue order.
//   kUbCast   FP32 image of the INT32 landing panel, later the INT32 quantised
//             panel, which is the same 16 KiB read as 4-byte lanes
//   kUbWork   |Y|, then the value-layout fold in place, then Y * inv
// No butterfly ping-pong panels and no replicated sign panels: the rotation finishes in L0C.
constexpr std::uint32_t kUbCast = NextSlot(kUbLandRing, kUbLandBytes);
constexpr std::uint32_t kUbWork = NextSlot(kUbCast, kPanelBytes);

// Row-wise max ladder for the key layout, where the reduction runs along the
// fast axis. Two halving stages at the full 64-lane mask take 256 partials per
// token down to 64, and two BlockReduceMax passes take 64 down to 8 and 8 down
// to 1 - a DataBlock at a time, so no stage wastes most of a vector register
// the way the old mask-32/16/8 stages did. The value layout folds in place and
// needs none of these.
constexpr std::uint32_t kUbFold128 = NextSlot(kUbWork, kPanelBytes);
constexpr std::uint32_t kUbFold64 = NextSlot(kUbFold128, kTokens * 128u * 4u);
constexpr std::uint32_t kUbFold8 = NextSlot(kUbFold64, kTokens * 64u * 4u);

// FP16 staging for the nibble pack: the hardware Cast to int4b_t takes a
// 2-byte source. The staged values are exact integers in [-8, 7], so the
// INT32 -> FP16 -> INT4 chain rounds once, in the INT32 cast, and the two
// narrowing steps are exact. The INT16 source form compiles to a deprecated
// vcvt_s162s4 intrinsic, which is the only reason this goes through FP16.
constexpr std::uint32_t kUbHalf = NextSlot(kUbFold8, kTokens * 8u * 4u);

// Logically scalar state, one lane-replicated DataBlock each where the API
// reads whole blocks. kScaleFloats is 16 == two DataBlocks.
constexpr std::uint32_t kUbAbsMax = NextSlot(kUbHalf, kPanel * 2u);
constexpr std::uint32_t kUbExpBits = NextSlot(kUbAbsMax, kScaleFloats * 4u);
constexpr std::uint32_t kUbPow = NextSlot(kUbExpBits, kScaleFloats * 4u);
constexpr std::uint32_t kUbInv = NextSlot(kUbPow, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvRep = NextSlot(kUbInv, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvBcast = NextSlot(kUbInvRep, 4u * kScaleFloats * 4u);
constexpr std::uint32_t kUbOne = NextSlot(kUbInvBcast, kTokens * kLanesPerBlock * 4u);
constexpr std::uint32_t kUbExpMask = NextSlot(kUbOne, kScaleFloats * 4u);
constexpr std::uint32_t kUbInvSeed = NextSlot(kUbExpMask, kScaleFloats * 4u);

// Packed egress ring: 2 KiB of nibbles plus the 64-byte scale slice, so one
// back edge covers both MTE3 bursts of a chunk.
constexpr std::uint32_t kUbPackRing = NextSlot(kUbInvSeed, kScaleFloats * 4u);
constexpr std::uint32_t kArenaEnd = kUbPackRing + kPackDepth * kPackSlotBytes;

static_assert(kUbLandRing % kBlock == 0u && kUbCast % kBlock == 0u &&
                  kUbWork % kBlock == 0u && kUbFold128 % kBlock == 0u &&
                  kUbFold64 % kBlock == 0u && kUbFold8 % kBlock == 0u &&
                  kUbHalf % kBlock == 0u && kUbAbsMax % kBlock == 0u &&
                  kUbExpBits % kBlock == 0u && kUbPow % kBlock == 0u &&
                  kUbInv % kBlock == 0u &&
                  kUbInvRep % kBlock == 0u && kUbInvBcast % kBlock == 0u &&
                  kUbOne % kBlock == 0u && kUbExpMask % kBlock == 0u &&
                  kUbInvSeed % kBlock == 0u && kUbPackRing % kBlock == 0u,
              "every UB operand base must be 32-byte aligned");
static_assert(kLandSlotBytes % kBlock == 0u && kPackSlotBytes % kBlock == 0u,
              "every ring slot stride must be a whole number of DataBlocks");
static_assert(kLandSlotBytes % 256u != 0u && kPackSlotBytes % 256u != 0u,
              "a 256-byte ring stride puts every slot on the same bank phase");
static_assert(kArenaEnd + kReadPadBytes <= kUbWindow,
              "UB arena leaves no VECTOR_REG_WIDTH read pad behind the state");
static_assert(kUbInvSeed + kReadPadBytes <= kUbWindow,
              "the last state block must have a whole vector register behind it");
static_assert(kPackedBytes % kBlock == 0u && kScaleFloats * 4u % kBlock == 0u,
              "both MTE3 bursts must be whole 32-byte bursts");
static_assert(kTokens * kLanesPerBlock * 4u == kTokens * kBlock,
              "a Brcb repeat writes eight DataBlocks per source element");

template <std::uint32_t Slot>
constexpr std::uint32_t kLandBase = kUbLandRing + Slot * kLandSlotBytes;
template <std::uint32_t Slot>
constexpr std::uint32_t kPackBase = kUbPackRing + Slot * kPackSlotBytes;
template <std::uint32_t Slot>
constexpr std::uint32_t kScaleBase = kPackBase<Slot> + kPackedBytes;

// --------------------------------------------------------------- L1 arena
// The 16 rotation blocks are page invariant and reach L1 once, as one flat
// 16 KiB blob. The page ring has one slot per pipeline stage. L1 operands need
// 32-byte alignment; the 512-byte rule applies to L0A and L0B, and the fractal
// offsets inside a slot are multiples of 512 from a 32-byte aligned base,
// which is the layout the fused attention kernel validated.
constexpr std::uint32_t kL1Table = 0u;
constexpr std::uint32_t kL1PageElems = kTokens * kHeadSize;          // 4096
constexpr std::uint32_t kL1PageBytes = kL1PageElems * kElemBytes;    // 8192
constexpr std::uint32_t kL1Ring = NextSlot(kL1Table, kTableBytes);
constexpr std::uint32_t kL1SlotBytes = kL1PageBytes + kBankSkewBytes;
constexpr std::uint32_t kL1End = kL1Ring + kStageDepth * kL1SlotBytes;
// Element stride for a typed L1 pointer, and byte stride for a raw L1 address.
// Both are needed: the group's L1 base is computed as an integer byte address
// before the reinterpret_cast, while an L0 operand offset is added to a typed
// pointer and therefore counts elements.
constexpr std::uint32_t kFractalElems = kTokens * kGroup;            // 512
constexpr std::uint32_t kFractalBytes = kFractalElems * kElemBytes;  // 1024

static_assert(kL1Table % kBlock == 0u && kL1Ring % kBlock == 0u &&
                  kL1SlotBytes % kBlock == 0u,
              "every L1 operand base must be 32-byte aligned");
static_assert(kL1SlotBytes % 256u != 0u, "L1 slot stride collapses the bank phase");
// 32 KiB of resident blocks plus four 8 KiB page slots is about 64 KiB of the 512 KiB L1.
static_assert(kL1End <= 512u * 1024u, "L1 ring exceeds the L1 buffer");
static_assert(kFractalElems * kGroups == kL1PageElems,
              "the page must be a whole number of Hadamard groups");

template <std::uint32_t Slot>
constexpr std::uint32_t kL1PageBase = kL1Ring + Slot * kL1SlotBytes;

// --------------------------------------------------------------- L0 arena
// The 16 rotation blocks stay resident in both L0A and L0B: the key chunks
// take them as the B operand and the value chunks as the A operand, and
// nothing writes that region after the prologue. The page fractals land behind
// it. 16 KiB of table plus 4 KiB of page is 20 KiB of the 64 KiB each of L0A
// and L0B holds on arch35.
constexpr std::uint32_t kL0TableElems = kTableElems;              // 16384
constexpr std::uint32_t kL0PageElems = kGroups * kFractalElems;   // 4096
constexpr std::uint32_t kL0Elems = kL0TableElems + kL0PageElems;  // 20480 elements
constexpr std::uint32_t kL0CElems = kGroups * kTokens * kGroup;   // 4096 FP32
// The 512-byte rules are about bytes, so they take the element width now.
// 32 KiB of resident blocks plus 8 KiB of page is 40 KiB of the 64 KiB each of
// L0A and L0B holds, against 20 KiB for INT8.
static_assert(kL0TableElems * kElemBytes % 512u == 0u,
              "the L0 page region must stay 512-byte aligned");
static_assert(kBlockElems * kElemBytes % 512u == 0u,
              "every resident block must be 512-byte aligned");
static_assert(kL0Elems * kElemBytes <= 64u * 1024u, "L0A/L0B hold 64 KiB on arch35");

// --------------------------------------------------- cross-core rendezvous
// The channel map, the id budget and the pipe rules are single-sourced in
// turboquant_wave_protocol.hpp; this block is only the mapping from the schedule's chunk
// index onto it.
//
// The rendezvous is per landing GENERATION. Generation g owns chunks 2g and 2g + 1 -- one per
// Vector core, by kOwner -- and landing slot g % kLandDepth in each core's Unified Buffer. The
// generation's first chunk carries the quorum wait and its second the release broadcast, so
// one rendezvous covers both Fixpipes.
using turboquant_wave::kSync2AicAnnounce;
using turboquant_wave::kSync2AicWait;
using turboquant_wave::kSync2AivAnnounce;
using turboquant_wave::kSync2AivWait;
using turboquant_wave::kSync2PageDrained;
using turboquant_wave::kSync2PanelFree;
using turboquant_wave::kSync2PanelFreePrime;
using turboquant_wave::kSync2PanelReady;

// 1-safety of the landing ring: neither side can run more than kLandDepth generations ahead
// of the other, so no mode-2 counter can reach the 0..15 wrap.
static_assert(turboquant_wave::Mode2LandingBudget<kLandDepth>::kFits,
              "a mode-2 flag counter overflow is a reported error, not a wrap");

constexpr std::uint32_t kGenerations = kChunks / 2u;
template <std::uint32_t C>
constexpr bool kOpensGeneration = (C % 2u) == 0u;
template <std::uint32_t C>
constexpr bool kClosesGeneration = (C % 2u) == 1u;

static_assert(kGenerations * 2u == kChunks, "a landing generation is one chunk per Vector core");
static_assert(kGenerations % kLandDepth == 0u,
              "the landing ring must close a whole number of times per strip, or a strip "
              "boundary would shift which slot a generation owns and kLandSlot would stop "
              "being a function of the chunk alone");
static_assert(kChunks % kStageDepth == 0u,
              "the L1 page ring must close a whole number of times per strip, for the same "
              "reason kLandSlot must");
static_assert(kGenerations % kPackDepth == 0u,
              "the packed egress ring must close a whole number of times per strip");
static_assert(kTokens == turboquant_wave::kStripTokens,
              "the strip height is the wave protocol's, or the boundary guard and the "
              "pipeline disagree about what a strip is");

// Intra-core HardEvent ids. The table-copy barriers finish before slot 0 is primed, so their
// route-local id 0 cannot collide with the ring's slot 0.
constexpr std::int32_t kEvtL0 = 0;
constexpr std::int32_t kEvtEpilogue = 0;
// The inter-wave epilogue drain rides route-local id 0 on every x_S route. Nothing else is
// live on those routes by then: the epilogue is the last thing each core issues, after every
// ring credit has reached a terminal sink.
constexpr std::int32_t kEvtDrain = 0;
// The drain standing immediately behind a mode-2 rendezvous wait, on its own route-local id so
// it cannot be confused with a ring slot's credit. Not needed for correctness -- a cross-core
// wait gates issuance of the pipe it names -- but it is what makes the wait's RELEASE tick
// appear in a trace, and that tick is what tells a wait that blocked from a wait that did not.
constexpr std::int32_t kEvtRendezvous = 2;

// ------------------------------------------------- compile-time column list
using ChunkSeq = std::make_integer_sequence<std::uint32_t, kChunks>;
using GroupSeq = std::make_integer_sequence<std::uint32_t, kGroups>;

// =========================================================== AIC: Cube core
// MTE2 ingress of the raw INT8 page into the L1 ring, then per chunk eight
// L1 -> L0 fractal loads, 64 accumulating INT8 Mmads and one Fixpipe into the
// owning Vector core's landing panel.
struct CubeEngine {
    __gm__ const bfloat16_t* key;
    __gm__ const bfloat16_t* value;
    std::uint32_t row_pitch;   // numKvHeads * headSize, elements
    std::uint32_t head_size;   // headSize, elements
};

// MTE2: one strided ND -> NZ burst per chunk. The source is the headSize-wide window of head h
// inside the kTokens x (numKvHeads * headSize) activation block of strip `strip_elems`, so the
// row pitch is a runtime stride and the window is expressed with MakePatternLayout rather than
// MakeFrameLayout, whose NDExt row pitch is pinned to the column count.
//
// `strip_elems` is the element offset of the strip inside this block's share: strip i of the
// share is at i * kTokens * row_pitch. It is the only runtime quantity the ingress needs, which
// is what lets one chunk stream cross strip boundaries without a second code path.
template <std::uint32_t Chunk>
__aicore__ inline void StagePage(CubeEngine& engine, std::uint32_t strip_elems)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t slot = kL1Slot<Chunk>;
    constexpr std::int32_t evt = static_cast<std::int32_t>(slot);

    __gm__ const bfloat16_t* base = kIsKey<Chunk> ? engine.key : engine.value;
    auto* src_ptr = const_cast<__gm__ bfloat16_t*>(base) + strip_elems +
                    kChunkHead<Chunk> * engine.head_size;
    auto* dst_ptr = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot>);

    auto shape = MakeShape(MakeShape(_1{}, kTokens), MakeShape(_1{}, kHeadSize));
    auto stride = MakeStride(MakeStride(_0{}, engine.row_pitch), MakeStride(_0{}, _1{}));
    auto src = MakeTensor(MakeMemPtr(src_ptr),
                          MakePatternLayout<NDExtLayoutPtn,
                                            LayoutTrait<AscendC::Std::ignore_t, _1>>(shape, stride));
    auto dst = MakeTensor(MakeMemPtr(dst_ptr),
                          MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kHeadSize));

    // The L1 slot is free once MTE1 has read the page it held kStageDepth
    // chunks ago. WaitFlag<MTE1_MTE2> stalls the MTE2 queue, which is the
    // queue that must not run ahead here: nothing else gates it.
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(evt);
    Copy(MakeCopy(CopyGM2L1{}), dst, src);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(evt);
}

// MTE1 for one channel group. Key and value chunks differ only in which L0
// buffer the page fractal belongs to, because the page is the A operand of the
// key chunks and the B operand of the value chunks.
template <std::uint32_t Chunk, std::uint32_t G>
__aicore__ inline void GemmGroupLoad(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t slot = kL1Slot<Chunk>;
    auto* page = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1PageBase<slot> + G * kFractalBytes);

    if constexpr (kIsKey<Chunk>) {
        // A = the page group, NZ(kTokens, kGroup): A[t][i] = x[t][G*32 + i].
        auto src = MakeTensor(MakeMemPtr(page),
                              MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
        auto dst = MakeTensor(MakeMemPtr(l0a + kL0TableElems + G * kFractalElems),
                              MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
        Copy(MakeCopy(CopyL12L0A{}), dst, src);
    } else {
        // B = the page group, ZN(kGroup, kTokens): B[i][t] = x[t][G*32 + i],
        // the same bytes read with the transposed meaning.
        auto src = MakeTensor(MakeMemPtr(page),
                              MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
        auto dst = MakeTensor(MakeMemPtr(l0b + kL0TableElems + G * kFractalElems),
                              MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
        Copy(MakeCopy(CopyL12L0B{}), dst, src);
    }
}

// One accumulating Mmad: output channel group C1 takes the contribution of
// source channel group G through the resident block kSignBlock<C1, G>. G == 0
// initialises the L0C region and the other seven accumulate into it, so one
// L0C region ends up holding
//
//   sum_G  sum_i  T[s(C1,G)][G][c0][i] . x[t][G*32 + i]
//
// which is row C1*32 + c0 of M . x by equation (1). cmatrixInitVal is the
// compile-time flag that picks "C = A B" over "C += A B"; the Cube pipe is
// in-order, so nothing else orders the eight accumulations.
template <std::uint32_t Chunk, std::uint32_t C1, std::uint32_t G>
__aicore__ inline void GemmBlock(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                 __cc__ float* l0c)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t block = kSignBlock<C1, G> * kBlockElems;
    constexpr bool init = (G == 0u);
    auto* c_ptr = l0c + C1 * kTokens * kGroup;

    if constexpr (kIsKey<Chunk>) {
        auto a = MakeTensor(MakeMemPtr(l0a + kL0TableElems + G * kFractalElems),
                            MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTokens, kGroup));
        auto b = MakeTensor(MakeMemPtr(l0b + block),
                            MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kGroup));
        auto c = MakeTensor(MakeMemPtr(c_ptr),
                            MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kGroup));
        MmadParams mp(static_cast<std::uint16_t>(kTokens), static_cast<std::uint16_t>(kGroup),
                      static_cast<std::uint16_t>(kGroup), 0, init);
        Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
    } else {
        auto a = MakeTensor(MakeMemPtr(l0a + block),
                            MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kGroup, kGroup));
        auto b = MakeTensor(MakeMemPtr(l0b + kL0TableElems + G * kFractalElems),
                            MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTokens));
        auto c = MakeTensor(MakeMemPtr(c_ptr),
                            MakeFrameLayout<NZLayoutPtn, 16>(kGroup, kTokens));
        MmadParams mp(static_cast<std::uint16_t>(kGroup), static_cast<std::uint16_t>(kTokens),
                      static_cast<std::uint16_t>(kGroup), 0, init);
        Mmad(MakeMmad(MmadOperation{}, MmadTraitDefault{}).with(mp), c, a, b);
    }
}

// One Fixpipe per chunk. The whole rotation is finished in L0C, so the panel leaves as a single
// L0C -> UB transfer: NZ(16, 256) -> ND for the token-major key layout and NZ(256, 16) -> ND for
// the channel-major value layout.
template <std::uint32_t Chunk>
__aicore__ inline void FixpipeChunk(__cc__ float* l0c)
{
    using namespace AscendC::Te;
    constexpr std::uint32_t land = kLandSlot<Chunk>;
    auto* ub = reinterpret_cast<__ubuf__ float*>(kLandBase<land>);

    FixpipeParams fp;
    // No unit flag: M_FIX carries the L0C dependency explicitly, because a
    // group of 64 accumulating Mmads feeds this one transfer.
    fp.unitFlag = 0;
    fp.subBlockId = (kOwner<Chunk> == 1u);

    if constexpr (kIsKey<Chunk>) {
        auto c = MakeTensor(MakeMemPtr(l0c),
                            MakeFrameLayout<NZLayoutPtn, 16>(kTokens, kHeadSize));
        auto y = MakeTensor(MakeMemPtr(ub),
                            MakeFrameLayout<NDExtLayoutPtn>(kTokens, kHeadSize));
        Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
    } else {
        auto c = MakeTensor(MakeMemPtr(l0c),
                            MakeFrameLayout<NZLayoutPtn, 16>(kHeadSize, kTokens));
        auto y = MakeTensor(MakeMemPtr(ub),
                            MakeFrameLayout<NDExtLayoutPtn>(kHeadSize, kTokens));
        Copy(MakeCopy(CopyL0C2UB{}).with(fp), y, c);
    }
}

template <std::uint32_t Chunk, std::uint32_t... Gs>
__aicore__ inline void GemmGroupLoads(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                      std::integer_sequence<std::uint32_t, Gs...>)
{
    (GemmGroupLoad<Chunk, Gs>(l0a, l0b), ...);
}

template <std::uint32_t Chunk, std::uint32_t C1, std::uint32_t... Gs>
__aicore__ inline void GemmBlockRow(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                    __cc__ float* l0c,
                                    std::integer_sequence<std::uint32_t, Gs...>)
{
    (GemmBlock<Chunk, C1, Gs>(l0a, l0b, l0c), ...);
}

template <std::uint32_t Chunk, std::uint32_t... C1s>
__aicore__ inline void GemmBlockRows(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                     __cc__ float* l0c,
                                     std::integer_sequence<std::uint32_t, C1s...>)
{
    (GemmBlockRow<Chunk, C1s>(l0a, l0b, l0c, GroupSeq{}), ...);
}

template <std::uint32_t Chunk>
__aicore__ inline void GemmChunk(__ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                 __cc__ float* l0c)
{
    constexpr std::uint32_t slot = kL1Slot<Chunk>;

    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(static_cast<std::int32_t>(slot));
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    GemmGroupLoads<Chunk>(l0a, l0b, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
    // Issued on MTE1 and behind the eight fractal loads, so the page slot is
    // announced free only once they have read it.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(static_cast<std::int32_t>(slot));

    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
    AscendC::SetMMRowMajor();
    GemmBlockRows<Chunk>(l0a, l0b, l0c, GroupSeq{});
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::SetFlag<AscendC::HardEvent::M_FIX>(kEvtL0);

    // The landing panel is waited here rather than before the Mmads: only the
    // Fixpipe writes Unified Buffer, so the 64 multiplies for this chunk run
    // while the owning Vector core is still finishing the previous one.
    //
    // ONE quorum wait per GENERATION, on the chunk that opens it: it blocks
    // until BOTH Vector cores have copied the generation that last held this
    // landing slot out of it. The generation's other Fixpipe is covered by the
    // same wait -- the quorum already means "both cores", and the Fixpipe queue
    // is in order, so gating its issuance here gates the pair.
    if constexpr (kOpensGeneration<Chunk>) {
        AscendC::CrossCoreWaitFlag<2, kSync2AicWait>(kSync2PanelFree);
        // Not needed to order the Fixpipe -- the wait gates PIPE_FIX and the
        // Fixpipe is on PIPE_FIX -- but it puts the wait's RELEASE on the
        // scalar stream, where the trace can see it.
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtRendezvous);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtRendezvous);
    }
    AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(kEvtL0);
    FixpipeChunk<Chunk>(l0c);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

    // A cross-core set runs on the scalar stream, so drain the Fixpipe into it
    // first; without this the announcement overtakes the transfer it reports
    // One broadcast per generation, on the chunk
    // that closes it, so it reports both Fixpipes; FIX is in order, so this
    // drain stands behind the even chunk's transfer as well as the odd one's.
    if constexpr (kClosesGeneration<Chunk>) {
        AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
        AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtL0);
        AscendC::CrossCoreSetFlag<2, kSync2AicAnnounce>(kSync2PanelReady);
    }
}

// How far MTE2 ingress runs ahead of the Cube, in chunks. Column j stages chunk j and
// multiplies chunk j - kIngressLookahead, so at 1 the two are adjacent and at kStageDepth the
// L1 ring is full before the first Mmad.
//
// It is 1 less than the ring depth, and both bounds are load bearing:
//
//   * at most kStageDepth, because StagePage(c) waits the MTE1_MTE2 credit that
//     GemmChunk(c - kStageDepth) releases, and that Gemm is issued kStageDepth -
//     kIngressLookahead steps earlier. The credit therefore exists by the time StagePage(c)
//     issues only while kIngressLookahead <= kStageDepth, and the prologue's kStageDepth
//     primed credits cover the first wrap;
//   * above 1, because that is the point. GemmChunk contains a CrossCoreWaitFlag, which stalls
//     the SCALAR stream, which is what dispatches into the MTE2 queue. At a lookahead of 1 the
//     ingress of chunk j + 1 sits behind the Cube's wait for a landing panel for chunk j - 1:
//     MTE2 cannot run ahead even with three L1 slots free and its own credits unconsumed.
//
// What is NOT established by the lookahead alone is any wall-clock claim: source order and
// balanced events do not establish overlap, and a CAModel trace with intersecting MTE2/Cube
// intervals is what would.
constexpr std::uint32_t kIngressLookahead = kStageDepth - 1u;
static_assert(kIngressLookahead >= 1u && kIngressLookahead <= kStageDepth,
              "ingress may not run further ahead than the L1 ring is deep");
static_assert(kIngressLookahead < kChunks,
              "the lookahead must stay inside one strip's chunk list, or a step would stage "
              "two strips ahead");

// Steady-state step J of strip s: stage the chunk kIngressLookahead ahead of the Mmads, then
// multiply chunk J. For the last kIngressLookahead steps the staged chunk belongs to strip
// s + 1, which is what keeps MTE2 from draining at a strip boundary -- the prologue ingress of
// the next strip is issued inside this strip's Mmads. StageNext is false on the last strip,
// where there is no next strip to prime.
template <std::uint32_t J, bool StageNext>
__aicore__ inline void StripStepCube(CubeEngine& engine, std::uint32_t strip_elems,
                                     __ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                     __cc__ float* l0c)
{
    constexpr std::uint32_t ahead = J + kIngressLookahead;
    constexpr bool crosses = ahead >= kChunks;
    if constexpr (!crosses || StageNext) {
        StagePage<ahead % kChunks>(
            engine, crosses ? strip_elems + kTokens * engine.row_pitch : strip_elems);
    }
    GemmChunk<J>(l0a, l0b, l0c);
}

template <bool StageNext, std::uint32_t... Js>
__aicore__ inline void StripStepsCube(CubeEngine& engine, std::uint32_t strip_elems,
                                      __ca__ bfloat16_t* l0a, __cb__ bfloat16_t* l0b,
                                      __cc__ float* l0c,
                                      std::integer_sequence<std::uint32_t, Js...>)
{
    (StripStepCube<Js, StageNext>(engine, strip_elems, l0a, l0b, l0c), ...);
}

// The pipeline prologue: the first kIngressLookahead chunks of strip 0, with no Mmad behind
// them. Every later chunk is staged inside a step that also multiplies.
template <std::uint32_t... Js>
__aicore__ inline void PrologueCube(CubeEngine& engine,
                                    std::integer_sequence<std::uint32_t, Js...>)
{
    (StagePage<Js>(engine, 0u), ...);
}

// kLandDepth quorum waits, one per landing generation whose release nothing in the schedule
// consumed. Written as a fold so the count follows kLandDepth rather than a hand-written list.
template <std::uint32_t... Ks>
__aicore__ inline void EpilogueQuorumDrains(std::integer_sequence<std::uint32_t, Ks...>)
{
    (((void)Ks, AscendC::CrossCoreWaitFlag<2, kSync2AicWait>(kSync2PanelFree)), ...);
}

__aicore__ inline void run_cube_core(__gm__ const bfloat16_t* key,
                                     __gm__ const bfloat16_t* value,
                                     __gm__ const std::int32_t* tables,
                                     std::uint32_t num_kv_heads,
                                     std::uint32_t head_size,
                                     std::uint32_t strips)
{
    using namespace AscendC::Te;
    __ca__ bfloat16_t l0_a[kL0Elems] __attribute__((aligned(512)));
    __cb__ bfloat16_t l0_b[kL0Elems] __attribute__((aligned(512)));
    __cc__ float l0_c[kL0CElems] __attribute__((aligned(1024)));

    CubeEngine engine{key, value, num_kv_heads * head_size, head_size};

    // The 16 rotation blocks reach L1 once, converted ND -> NZ one block at a time, and each of
    // L0A and L0B then takes the whole blob once as a byte identity. Block b lands at
    // b * kBlockElems, so every operand base below and the GM contract for `tables` are the
    // row-major blob the host builds. The two L1 -> L0 copies are identities because source and
    // destination carry the same layout descriptor over the same kTableElems elements, so
    // element i moves to element i whatever order the bytes are in. Inside the blob, block b
    // then reads as NZ(32, 32) from L0A and as ZN(32, 32) from L0B -- the same image, since
    // those two differ only in the name of the second index -- which are the two operand roles
    // equation (1) needs.
    auto* l1_table = reinterpret_cast<__cbuf__ bfloat16_t*>(kL1Table);
    ascend_micro::stage_blocks_nz<kTableBlocks, kGroup>(
        reinterpret_cast<__gm__ const bfloat16_t*>(tables), kL1Table);
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(kEvtEpilogue);

    auto ta_src = MakeTensor(MakeMemPtr(l1_table),
                             MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTableRows, kGroup));
    auto ta_dst = MakeTensor(MakeMemPtr(l0_a),
                             MakeFrameLayout<NZLayoutPtn, bfloat16_t>(kTableRows, kGroup));
    Copy(MakeCopy(CopyL12L0A{}), ta_dst, ta_src);
    auto tb_src = MakeTensor(MakeMemPtr(l1_table),
                             MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTableRows));
    auto tb_dst = MakeTensor(MakeMemPtr(l0_b),
                             MakeFrameLayout<ZNLayoutPtn, bfloat16_t>(kGroup, kTableRows));
    Copy(MakeCopy(CopyL12L0B{}), tb_dst, tb_src);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(kEvtEpilogue);

    // Feedback cycle priming: every L1 page slot starts free, and L0A/L0B and L0C are free for
    // the first chunk.
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::SetFlag<AscendC::HardEvent::FIX_M>(kEvtL0);

    // One chunk stream over the block's strips. The last strip is peeled rather than guarded,
    // so no step of the steady state carries a runtime predicate.
    PrologueCube(engine, std::make_integer_sequence<std::uint32_t, kIngressLookahead>{});
    const std::uint32_t strip_pitch = kTokens * engine.row_pitch;
    std::uint32_t strip_elems = 0u;
    for (std::uint32_t strip = 0u; strip + 1u < strips; ++strip) {
        StripStepsCube<true>(engine, strip_elems, l0_a, l0_b, l0_c, ChunkSeq{});
        strip_elems += strip_pitch;
    }
    StripStepsCube<false>(engine, strip_elems, l0_a, l0_b, l0_c, ChunkSeq{});

    // Drain every primed token and every panel-free token the Vector cores handed back, so no
    // event bit and no intra-block flag leaks into the next launch. These waits also retire the
    // last Fixpipe before the kernel ends.
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(1);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(3);
    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(kEvtL0);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(kEvtL0);
    // The last kLandDepth generations' releases: nothing in the schedule waits for them,
    // because the schedule's g-th wait consumes generation g - kLandDepth. One quorum wait each.
    EpilogueQuorumDrains(std::make_integer_sequence<std::uint32_t, kLandDepth>{});

    // ---- the mandatory inter-wave epilogue drain ---------------------------------------
    // The waits above retire this core's event bits; they do not retire its TRANSFERS. Above
    // the physical block count the scheduler hands this block straight to the next wave's block
    // without draining anything, so an MTE2 burst still in flight would land in wave 1's L1 ring
    // slot and a Fixpipe still in flight in wave 1's landing panel -- and both are addressed
    // from compile-time constants, so they are the SAME addresses. Each route is drained into
    // the scalar stream with one SET/WAIT pair, which is two instructions against a
    // whole-pipeline stall for PipeBarrier<PIPE_ALL>. MTE1 has no scalar route; it is covered by
    // the M_MTE1 wait above, which sits behind the Mmads that consumed its loads.
    AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
    AscendC::SetFlag<AscendC::HardEvent::M_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::M_S>(kEvtDrain);
    AscendC::SetFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
    AscendC::WaitFlag<AscendC::HardEvent::FIX_S>(kEvtDrain);
    // Only now is "the share is retired" true, so now both Vector cores are told. The set
    // issues on the scalar stream, which the three drains above have just emptied, so this
    // announcement cannot overtake the work it reports. One broadcast reaches both cores.
    AscendC::CrossCoreSetFlag<2, kSync2AicAnnounce>(kSync2PageDrained);
}

// ======================================================= AIV: Vector cores
// Both cores run this body; the chunk list is filtered by owner, so each core
// handles two value chunks and two key chunks and nothing else. The rotation
// arrives finished, so the stage is absmax, a power-of-two scale, and the INT4
// pack - nothing else.
struct VectorEngine {
    AscendC::GlobalTensor<std::int8_t> key_cache;
    AscendC::GlobalTensor<std::int8_t> value_cache;
    AscendC::GlobalTensor<float> scale_cache;
    std::uint32_t page_bytes;    // byte base of (block, head 0) in either cache
    std::uint32_t head_bytes;    // blockSize * headSize / 2
    std::uint32_t key_scale;     // float base of (kv 0, block, head 0)
    std::uint32_t value_scale;   // float base of (kv 1, block, head 0)
    std::uint32_t head_scale;    // blockSize
    float inv_sqrt_len;
};

constexpr auto kUb = AscendC::TPosition::VECCALC;

template <class T>
__aicore__ inline AscendC::LocalTensor<T> View(std::uint32_t base, std::uint32_t count)
{
    return AscendC::LocalTensor<T>(kUb, base, count);
}

// Row-wise max ladder for the key layout: fold N partials per token down to
// N/2, from a SrcPitch-strided source into a DstPitch-strided destination.
// Both uses run at the full 64-lane mask.
template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch,
          std::uint32_t... Cs>
__aicore__ inline void KeyFoldEmit(const AscendC::LocalTensor<float>& dst,
                                   const AscendC::LocalTensor<float>& src,
                                   std::integer_sequence<std::uint32_t, Cs...>)
{
    constexpr std::uint32_t half = N / 2u;
    constexpr std::int32_t mask = static_cast<std::int32_t>(half > 64u ? 64u : half);
    AscendC::BinaryRepeatParams rp;
    rp.dstBlkStride = 1;
    rp.src0BlkStride = 1;
    rp.src1BlkStride = 1;
    rp.dstRepStride = static_cast<std::uint8_t>(DstPitch / kLanesPerBlock);
    rp.src0RepStride = static_cast<std::uint8_t>(SrcPitch / kLanesPerBlock);
    rp.src1RepStride = rp.src0RepStride;
    (AscendC::Max<float>(dst[Cs * mask], src[Cs * mask], src[half + Cs * mask], mask,
                         static_cast<std::uint8_t>(kTokens), rp),
     ...);
}

template <std::uint32_t N, std::uint32_t SrcPitch, std::uint32_t DstPitch>
__aicore__ inline void KeyFold(const AscendC::LocalTensor<float>& dst,
                               const AscendC::LocalTensor<float>& src)
{
    constexpr std::uint32_t half = N / 2u;
    constexpr std::uint32_t mask = half > 64u ? 64u : half;
    static_assert(mask == 64u, "both key folds must run at the full vector mask");
    static_assert(DstPitch >= kLanesPerBlock, "a fold destination pitch is a block count");
    KeyFoldEmit<N, SrcPitch, DstPitch>(
        dst, src, std::make_integer_sequence<std::uint32_t, half / mask>{});
}

// In-place halving fold for the value layout, where the reduction axis is the
// slow one and every step is one contiguous level-2 op.
template <std::uint32_t... Steps>
__aicore__ inline void ValueFoldEmit(const AscendC::LocalTensor<float>& buf,
                                     std::integer_sequence<std::uint32_t, Steps...>)
{
    // Step s folds kPanel >> s elements onto kPanel >> (s + 1).
    (AscendC::Max(buf, buf, buf[kPanel >> (Steps + 1u)],
                  static_cast<std::int32_t>(kPanel >> (Steps + 1u))),
     ...);
}

// One level-0 op per vector register of a token row: the key panel has the
// token index outermost, so the per-token inverse block advances once per
// repeat (src1RepStride 1) and is read by every DataBlock of that repeat
// (src1BlkStride 0).
template <std::uint32_t... Qs>
__aicore__ inline void KeyQuantiseEmit(const AscendC::LocalTensor<float>& dst,
                                       const AscendC::LocalTensor<float>& src,
                                       const AscendC::LocalTensor<float>& inv_bcast,
                                       const AscendC::BinaryRepeatParams& rp,
                                       std::integer_sequence<std::uint32_t, Qs...>)
{
    (AscendC::Mul<float>(dst[Qs * 64u], src[Qs * 64u], inv_bcast, 64,
                         static_cast<std::uint8_t>(kTokens), rp),
     ...);
}

// --- the Vector stage --------------------------------------------------------
template <std::uint32_t Chunk>
__aicore__ inline void RotateAndQuantise(VectorEngine& engine)
{
    constexpr std::uint32_t land = kLandSlot<Chunk>;
    constexpr std::uint32_t pack = kPackSlot<Chunk>;
    constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
    constexpr std::int32_t all = static_cast<std::int32_t>(kPanel);

    auto landing = View<float>(kLandBase<land>, kPanel);
    auto cast = View<float>(kUbCast, kPanel);
    auto work = View<float>(kUbWork, kPanel);

    // The generation's release broadcast: both landing panels of this
    // generation hold their products. This core reads only the one chunk of the
    // generation it owns, so one wait per generation is one wait per chunk here.
    AscendC::CrossCoreWaitFlag<2, kSync2AivWait>(kSync2PanelReady);
    // As on the Cube: not needed to order the Muls below -- the wait gates
    // PIPE_V and the Muls is on PIPE_V -- but it is what puts the wait's
    // RELEASE tick in the trace, which is the only way the audit can tell a
    // wait that blocked from a wait that stopped blocking.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtRendezvous);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtRendezvous);
    // BF16 x BF16 accumulates to FP32 in L0C, so the panel lands as FP32 and no type cast is
    // needed. What is needed is to get the panel OUT of the landing ring slot in one pass, so
    // the slot can be released immediately and the Cube can refill it; this is that copy,
    // expressed as a multiply by one so it stays a single contiguous Vector op.
    //
    // Exactness is preserved for integer-valued BF16 activations, which is what
    // the fixture supplies: a BF16 significand is 8 bits, so integers up to 256
    // are exact, a product of two is at most 2^16 and exact in FP32, and a sum
    // of 256 such products is an integer below 2^24 -- exact whatever order the
    // Cube accumulates in. With general BF16 inputs that order is unspecified
    // and no integer oracle could reproduce it bit for bit.
    AscendC::Muls(cast, landing, 1.0f, all);
    // The landing panel is dead from here on, so release it before the rest of
    // the stage runs and let the Cube refill it immediately. The drain makes
    // the scalar-stream set wait for the Vector queue to retire the copy.
    // Half of the generation's quorum. The Cube's counter advances only once
    // the other Vector core has announced too, which is exactly the
    // precondition its next Fixpipe into this slot has.
    AscendC::SetFlag<AscendC::HardEvent::V_S>(pack_evt);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(pack_evt);
    AscendC::CrossCoreSetFlag<2, kSync2AivAnnounce>(kSync2PanelFree);

    // Row-wise absolute maximum, one value per token, left contiguous so the
    // scale slice is a single 64-byte MTE3 burst.
    auto abs_max = View<float>(kUbAbsMax, kScaleFloats);
    auto one = View<float>(kUbOne, kScaleFloats);
    AscendC::Abs(work, cast, all);
    if constexpr (kIsKey<Chunk>) {
        auto f128 = View<float>(kUbFold128, kTokens * 128u);
        auto f64 = View<float>(kUbFold64, kTokens * 64u);
        auto f8 = View<float>(kUbFold8, kTokens * 8u);
        KeyFold<256u, 256u, 128u>(f128, work);
        KeyFold<128u, 128u, 64u>(f64, f128);
        // 64 partials per token remain, eight DataBlocks each, and
        // BlockReduceMax writes one result per block contiguously: 64 -> 8 at
        // one repeat per token, then 8 -> 1 over the two registers that holds.
        AscendC::BlockReduceMax<float>(f8, f64, static_cast<std::uint8_t>(kTokens), 64,
                                       1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::BlockReduceMax<float>(abs_max, f8, 2, 64, 1, 1, 8);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Max(abs_max, abs_max, one, static_cast<std::int32_t>(kScaleFloats));
    } else {
        ValueFoldEmit(work, std::make_integer_sequence<std::uint32_t, 8u>{});
        AscendC::PipeBarrier<PIPE_V>();
        // Floor at 1.0 and copy out of the fold buffer in one op: every value
        // is an integer, so 1.0 is the smallest non-zero magnitude and an
        // all-zero token gets scale 2^-2 rather than a division by zero.
        AscendC::Max(abs_max, work, one, static_cast<std::int32_t>(kScaleFloats));
    }

    // Power-of-two scale, by integer arithmetic on the FP32 bit pattern. A
    // float 2^k carries the exponent field k + 127, so with
    // 2^e <= absmax < 2^(e+1):
    //   exp_bits  = absmax & 0x7F800000       -> 2^e,       field e + 127
    //   pow_bits  = exp_bits - (2 << 23)      -> 2^(e-2),   field e + 125
    //   inv_bits  = (254 << 23) - pow_bits    -> 2^(2-e),   field 129 - e
    // The last line is the exact reciprocal of a power of two: the two fields
    // sum to 254, which is what (254 << 23) - x computes. No division, no
    // logarithm, and no hardware reciprocal rounding enters the contract.
    auto exp_bits = View<std::int32_t>(kUbExpBits, kScaleFloats);
    auto exp_mask = View<std::int32_t>(kUbExpMask, kScaleFloats);
    auto inv_seed = View<std::int32_t>(kUbInvSeed, kScaleFloats);
    auto pow_bits = View<std::int32_t>(kUbPow, kScaleFloats);
    auto pow_two = View<float>(kUbPow, kScaleFloats);
    auto inv_bits = View<std::int32_t>(kUbInv, kScaleFloats);
    auto inv = View<float>(kUbInv, kScaleFloats);
    constexpr std::int32_t lanes = static_cast<std::int32_t>(kScaleFloats);

    AscendC::And(exp_bits, abs_max.template ReinterpretCast<std::int32_t>(), exp_mask, lanes);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Adds(pow_bits, exp_bits, static_cast<std::int32_t>(-0x01000000), lanes);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(inv_bits, inv_seed, pow_bits, lanes);

    // Broadcast inv along the panel's fast axis and quantise. The value layout
    // has the token index innermost, so one level-0 op with a repeat-invariant
    // second operand covers the whole panel; the key layout has the token index
    // outermost, so the per-token block advances once per repeat and a repeat
    // covers a quarter of a token row.
    auto quant = View<std::int32_t>(kUbCast, kPanel);
    auto staged = View<half>(kUbHalf, kPanel);
    auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
    AscendC::BinaryRepeatParams bc;
    bc.dstBlkStride = 1;
    bc.src0BlkStride = 1;
    bc.dstRepStride = 8;
    bc.src0RepStride = 8;
    if constexpr (kIsKey<Chunk>) {
        auto inv_bcast = View<float>(kUbInvBcast, kTokens * kLanesPerBlock);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(inv_bcast, inv, 2, {1, 8});
        AscendC::PipeBarrier<PIPE_V>();
        bc.src1BlkStride = 0;
        bc.src1RepStride = 1;
        bc.dstRepStride = static_cast<std::uint8_t>(kHeadSize / kLanesPerBlock);
        bc.src0RepStride = bc.dstRepStride;
        KeyQuantiseEmit(work, cast, inv_bcast, bc,
                        std::make_integer_sequence<std::uint32_t, kHeadSize / 64u>{});
    } else {
        // Four copies of the 16-token block fill one vector register, and the
        // register repeats unchanged across the whole panel.
        auto inv_rep = View<float>(kUbInvRep, 4u * kScaleFloats);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep, inv, 1.0f, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep[kScaleFloats], inv_rep, 1.0f, lanes);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(inv_rep[2u * kScaleFloats], inv_rep, 1.0f,
                      static_cast<std::int32_t>(2u * kScaleFloats));
        AscendC::PipeBarrier<PIPE_V>();
        bc.src1BlkStride = 1;
        bc.src1RepStride = 0;
        AscendC::Mul<float>(work, cast, inv_rep, 64,
                            static_cast<std::uint8_t>(kPanel / 64u), bc);
    }

    // |Y * inv| < 8 and round-to-nearest-even can reach +8, which INT4 cannot
    // hold, so clamp from above only; -8 is in range. The INT32 cast is the
    // only rounding in the whole path, and it rounds an exactly representable
    // product, so the host oracle reproduces it in integer arithmetic.
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Mins(work, work, 7.0f, all);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(quant, work, AscendC::RoundMode::CAST_RINT, kPanel);

    // The packed slot and its scale slice are one ring slot, so one back edge
    // covers both: its previous occupant is still being read by MTE3 until now.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Cast(staged, quant, AscendC::RoundMode::CAST_RINT, kPanel);
    AscendC::PipeBarrier<PIPE_V>();
    auto nibbles = packed.template ReinterpretCast<AscendC::int4b_t>();
    AscendC::Cast(nibbles, staged, AscendC::RoundMode::CAST_RINT, kPanel);
    // The quantised panel is scale invariant - scaling Y and absmax together
    // leaves every nibble unchanged - so invSqrtLen folds into the stored scale
    // alone, and q * scale reconstructs the orthonormal rotation.
    auto scale = View<float>(kScaleBase<pack>, kScaleFloats);
    AscendC::Muls(scale, pow_two, engine.inv_sqrt_len, lanes);

    AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
}

// --- the MTE3 stage ----------------------------------------------------------
template <std::uint32_t Chunk>
__aicore__ inline void StoreChunk(VectorEngine& engine)
{
    constexpr std::uint32_t pack = kPackSlot<Chunk>;
    constexpr std::int32_t pack_evt = static_cast<std::int32_t>(pack);
    auto packed = View<std::int8_t>(kPackBase<pack>, kPackedBytes);
    auto scale = View<float>(kScaleBase<pack>, kScaleFloats);

    AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(pack_evt);
    const std::uint32_t page = engine.page_bytes + kChunkHead<Chunk> * engine.head_bytes;
    const std::uint32_t slice =
        (kIsKey<Chunk> ? engine.key_scale : engine.value_scale) +
        kChunkHead<Chunk> * engine.head_scale;
    if constexpr (kIsKey<Chunk>) {
        AscendC::DataCopy(engine.key_cache[page], packed, kPackedBytes);
    } else {
        AscendC::DataCopy(engine.value_cache[page], packed, kPackedBytes);
    }
    AscendC::DataCopy(engine.scale_cache[slice], scale, kScaleFloats);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(pack_evt);
}

// Steady-state step J of strip s on Vector core k: quantise the chunk one step ahead of the
// egress, then store chunk J -- each only if core k owns that chunk, which is decided at
// compile time, so a core's instruction stream contains only its own chunks and no predicate.
// For J = kChunks - 1 the quantised chunk is chunk 0 of strip s + 1, which is the Vector side
// of the seamless boundary: the next strip's first panel is taken out of the landing ring
// inside this strip's egress. QuantNext is false on the last strip.
template <std::uint32_t J, std::uint32_t Core, bool QuantNext>
__aicore__ inline void StripStepVector(VectorEngine& engine)
{
    constexpr std::uint32_t ahead = J + 1u;
    constexpr bool crosses = ahead >= kChunks;
    if constexpr ((!crosses || QuantNext) && kOwner<ahead % kChunks> == Core)
        RotateAndQuantise<ahead % kChunks>(engine);
    if constexpr (kOwner<J> == Core) StoreChunk<J>(engine);
}

template <std::uint32_t Core, bool QuantNext, std::uint32_t... Js>
__aicore__ inline void StripStepsVector(VectorEngine& engine,
                                        std::integer_sequence<std::uint32_t, Js...>)
{
    (StripStepVector<Js, Core, QuantNext>(engine), ...);
}

// The pipeline prologue: chunk 0 of strip 0, the one panel with no egress step ahead of it.
template <std::uint32_t Core>
__aicore__ inline void PrologueVector(VectorEngine& engine)
{
    if constexpr (kOwner<0u> == Core) RotateAndQuantise<0u>(engine);
}

// kLandDepth priming announcements on the readiness channel, one per landing slot. A fold
// rather than a hand-written pair so that a depth change cannot leave the ring half primed.
template <std::uint32_t... Slots>
__aicore__ inline void PrimeLandingRing(std::integer_sequence<std::uint32_t, Slots...>)
{
    (((void)Slots, AscendC::CrossCoreSetFlag<2, kSync2AivAnnounce>(kSync2PanelFreePrime)), ...);
}

// The strip's physical cache addresses, from the one slotMapping entry that strip owns.
//
// A page holds block_size / kTokens strips, and the in-page offset is kPackedBytes for BOTH
// planes: the key plane is token-major, so tokens 16s..16s+15 start at byte s * kPackedBytes of
// the (block, head) region, and the value plane's strip base (strip * kTokens * headSize
// nibbles) is the same byte. One expression serves both.
struct StripAddr {
    std::uint32_t page_bytes;
    std::uint32_t key_scale;
    std::uint32_t value_scale;
};

__aicore__ inline StripAddr strip_addr(__gm__ const std::int32_t* slot_mapping,
                                       std::uint32_t strip, std::uint32_t num_kv_heads,
                                       std::uint32_t head_bytes, std::uint32_t block_size,
                                       std::uint32_t num_blocks)
{
    // One scalar read of caller-supplied Global Memory metadata per strip. The physical page
    // and the strip inside it both come out of that one slot, so no stage reads the mapping
    // again.
    const std::uint32_t slot0 = static_cast<std::uint32_t>(slot_mapping[strip * kTokens]);
    const std::uint32_t block_idx = slot0 / block_size;
    const std::uint32_t strip_in_block = (slot0 % block_size) / kTokens;
    const std::uint32_t strip_bytes = strip_in_block * kPackedBytes;
    StripAddr a;
    // StoreChunk adds kChunkHead<Chunk> * head_bytes, so the byte it writes is
    // (block_idx * num_kv_heads + kChunkHead<Chunk>) * head_bytes + strip_bytes.
    a.page_bytes = block_idx * num_kv_heads * head_bytes + strip_bytes;
    // StoreChunk lands kScaleFloats == kTokens floats at `*_scale + head * head_scale`, so the
    // strip's own tokens are at + strip_in_block * kTokens:
    // ((kv * numBlocks + block) * numKvHeads + head) * blockSize + key.
    a.key_scale = block_idx * num_kv_heads * block_size + strip_in_block * kTokens;
    a.value_scale =
        (num_blocks + block_idx) * num_kv_heads * block_size + strip_in_block * kTokens;
    return a;
}

template <std::uint32_t Core>
__aicore__ inline void run_vector_core(__gm__ std::int8_t* key_cache,
                                       __gm__ std::int8_t* value_cache,
                                       __gm__ float* scale_cache,
                                       __gm__ const std::int32_t* slot_mapping,
                                       std::uint32_t num_kv_heads,
                                       std::uint32_t head_size,
                                       std::uint32_t block_size,
                                       std::uint32_t num_blocks,
                                       std::uint32_t strips,
                                       float inv_sqrt_len)
{
    const std::uint32_t head_bytes = block_size * head_size / 2u;

    VectorEngine engine;
    engine.key_cache.SetGlobalBuffer(key_cache);
    engine.value_cache.SetGlobalBuffer(value_cache);
    engine.scale_cache.SetGlobalBuffer(scale_cache);
    engine.head_bytes = head_bytes;
    engine.head_scale = block_size;
    engine.inv_sqrt_len = inv_sqrt_len;

    // ---- prologue: the quantiser's three immediate constants ---------------------------
    // Every constant is immediate sourced, so the Vector prologue reads no Global Memory and
    // carries no tensor hazard.
    AscendC::Duplicate(View<float>(kUbOne, kScaleFloats), 1.0f,
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::Duplicate(View<std::int32_t>(kUbExpMask, kScaleFloats),
                       static_cast<std::int32_t>(0x7F800000),
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::Duplicate(View<std::int32_t>(kUbInvSeed, kScaleFloats),
                       static_cast<std::int32_t>(0x7F000000),
                       static_cast<std::int32_t>(kScaleFloats));
    AscendC::PipeBarrier<PIPE_V>();

    // Feedback cycle priming: both packed page slots start free, and both landing panels are
    // free for the Cube to write. Each core announces kLandDepth times; the quorum advances the
    // Cube's counter once per PAIR, so the Cube starts with kLandDepth credits.
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(1);
    PrimeLandingRing(std::make_integer_sequence<std::uint32_t, kLandDepth>{});

    // One chunk stream over the block's strips. The addresses are refreshed once per strip,
    // before that strip's first egress; the last strip is peeled so no steady-state step
    // carries a runtime predicate.
    PrologueVector<Core>(engine);
    for (std::uint32_t strip = 0u; strip + 1u < strips; ++strip) {
        const StripAddr a =
            strip_addr(slot_mapping, strip, num_kv_heads, head_bytes, block_size, num_blocks);
        engine.page_bytes = a.page_bytes;
        engine.key_scale = a.key_scale;
        engine.value_scale = a.value_scale;
        StripStepsVector<Core, true>(engine, ChunkSeq{});
    }
    {
        const StripAddr a = strip_addr(slot_mapping, strips - 1u, num_kv_heads, head_bytes,
                                       block_size, num_blocks);
        engine.page_bytes = a.page_bytes;
        engine.key_scale = a.key_scale;
        engine.value_scale = a.value_scale;
        StripStepsVector<Core, false>(engine, ChunkSeq{});
    }

    // Drain the primed tokens. These waits also retire the last MTE3 bursts before the kernel
    // returns, so Global Memory is visible to the host.
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(1);

    // ---- the mandatory inter-wave epilogue drain ---------------------------------------
    // Await the Cube's drain first: the Cube is the only core that writes this core's landing
    // ring, and until it reports every queue retired, a Fixpipe of THIS wave could still be in
    // flight against a panel wave 1 is about to allocate at the same UB address. Then empty
    // this core's own queues into the scalar stream.
    //
    // The two cores drain the same three routes in different orders, each leading with the pipe
    // it is most likely to still have work on -- AIV0 the Vector queue it has just finished
    // casting on, AIV1 the MTE3 queue carrying its last cache bursts -- so neither spends its
    // first drain on the other's busiest engine. MTE2 is idle on both cores in this kernel: no
    // Vector-side stage reads Global Memory. The pair is issued anyway, because "this queue is
    // empty" is exactly what the wave boundary needs established, and an empty queue retires it
    // immediately.
    //
    // Both cores wait the SAME broadcast id, and both on PIPE_V: the hazard the join covers is a
    // Fixpipe landing in a panel the next wave is about to read, and PIPE_V is the pipe that
    // reads it.
    AscendC::CrossCoreWaitFlag<2, kSync2AivWait>(kSync2PageDrained);
    if constexpr (Core == 0u) {
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
    } else {
        AscendC::SetFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_S>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(kEvtDrain);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(kEvtDrain);
    }
}

}  // namespace turboquant_bf16

// The input pointers lose their const at this boundary and get it back one line into the body.
// CANN 9.2 generates host_stub.cpp with a `const void*` parameter for a `__gm__ const T*`
// kernel parameter and then assigns it to a `void*` field of the argument block, which does not
// compile. Dropping const on the parameters below is the only way to keep the published
// argument order and types; every device-side function still takes `__gm__ const`, so nothing
// inside the kernel can write an input.
namespace turboquant_bf16 {
__aicore__ inline void KernelBody(
    // The activations are bfloat16_t; the caches stay INT8, because they hold packed INT4
    // nibbles and the output contract does not move with the input precision.
    __gm__ bfloat16_t* __restrict__ key,
    __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache,
    __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache,
    __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns,
    __gm__ std::int32_t* __restrict__ tables,
    // Unqualified uint32_t, not std::uint32_t: CANN copies this parameter list verbatim into
    // aclrtlaunch_triple_chevrons_func.h, which has no using-declaration for std, so a
    // qualified scalar type breaks the host leg of the build while the device legs compile
    // cleanly. Pointer parameters are rewritten to void* and are unaffected.
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore,
    float invSqrtLen)
{
    // GetBlockIdx() is NOT the grid block index on a Vector core of a MIX launch. CANN 9.2
    // dav_3510 (GetBlockIdxImpl in
    // asc/impl/basic_api/dav_3510/kernel_operator_sys_var_impl.h) returns
    //
    //   AIC:  get_block_idx()
    //   AIV:  get_block_idx() * get_subblockdim() + get_subblockid()
    //
    // so at a 1:2 AIC:AIV ratio the AIC range is [0, numBlocks) and the AIV range is
    // [0, 2 * numBlocks). Taken raw, the two Vector cores of block b would address the shares of
    // blocks 2b and 2b + 1 while their Cube read block b's. Dividing by GetSubBlockNum() -- 1 on
    // the AIC, the sub-block count on an AIV -- is the one expression all three roles agree on.
    const uint32_t block =
        static_cast<uint32_t>(AscendC::GetBlockIdx() / AscendC::GetSubBlockNum());

    // ---- boundary and guard isolation --------------------------------------------------
    // The true task boundary, taken before there is anything to leak. A grid may be wider than
    // the context -- a caller rounding up to the physical block count, or a launcher padding to
    // a whole wave -- and a block past the end owns no strip, no physical page and no
    // slotMapping entry. Reading slotMapping there is an invalid load: it kills that AIV, and
    // the Cube then sits on its next CrossCoreWaitFlag for the rest of the launch, which is a
    // hang rather than a wrong answer.
    //
    // The return is taken here: before InitSocState, before a tensor or any SOC state exists,
    // before a feedback cycle is primed and before any Global Memory address is formed. All
    // three roles derive `block` and `strips` from the same two hardware registers and the same
    // two host scalars, so they reach the same decision, and a guarded group announces nothing
    // its peers could wait for.
    //
    // One scalar divide each, and TotalStrips absorbs a zero tokensPerCore rather than faulting
    // the scalar unit on it.
    const uint32_t total_blocks = turboquant_wave::TotalStrips(numTokens, tokensPerCore);
    if (turboquant_wave::BlockOutOfStrips(block, total_blocks)) return;
    // A share that is not a whole number of strips has no pipeline: the chunk stream is
    // strips * kChunks columns long and there is no partial-strip column.
    const uint32_t strips = turboquant_wave::StripsPerBlock(tokensPerCore);
    if (strips == 0u) return;

    AscendC::InitSocState();

    // Input and metadata bases advance together to the head of this block's share; the cache
    // addresses come from each strip's own first slot.
    const uint32_t first_token = block * tokensPerCore;
    key += first_token * numKvHeads * headSize;
    value += first_token * numKvHeads * headSize;
    slotMapping += first_token;
    // piSigns is folded into `tables` by whoever builds it -- the sign diagonal sits between the
    // two Hadamard factors, so it is a property of the resident rotation blocks (header
    // equation (2)) and not of a Vector-side multiply. The argument stays in the published
    // signature; the device does not read it.
    (void)piSigns;
    // The only role dispatch in the kernel, outside every pipeline stage.
    if ASCEND_IS_AIC {
        run_cube_core(key, value, tables, numKvHeads, headSize, strips);
    } else if (AscendC::GetSubBlockIdx() == 0) {
        run_vector_core<0u>(keyCache, valueCache, scaleCache, slotMapping, numKvHeads, headSize,
                            blockSize, numBlocks, strips, invSqrtLen);
    } else {
        run_vector_core<1u>(keyCache, valueCache, scaleCache, slotMapping, numKvHeads, headSize,
                            blockSize, numBlocks, strips, invSqrtLen);
    }
}
}  // namespace turboquant_bf16

// The only entry point. Three mode-2 channels whatever the schedule length, each one a counter
// that banks unconsumed announcements.
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16(
    __gm__ bfloat16_t* __restrict__ key,
    __gm__ bfloat16_t* __restrict__ value,
    __gm__ std::int8_t* __restrict__ keyCache,
    __gm__ std::int8_t* __restrict__ valueCache,
    __gm__ float* __restrict__ scaleCache,
    __gm__ std::int32_t* __restrict__ slotMapping,
    __gm__ float* __restrict__ piSigns,
    __gm__ std::int32_t* __restrict__ tables,
    uint32_t numTokens, uint32_t numKvHeads, uint32_t headSize,
    uint32_t blockSize, uint32_t numBlocks, uint32_t tokensPerCore,
    float invSqrtLen)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    turboquant_bf16::KernelBody(key, value, keyCache, valueCache, scaleCache, slotMapping,
                                piSigns, tables, numTokens, numKvHeads, headSize, blockSize,
                                numBlocks, tokensPerCore, invSqrtLen);
}

#elif defined(ASCEND_MICRO_CANN_LAUNCH)

#include <stdexcept>

// The activations are 2-byte here, and this leg has no CANN headers -- the device types live
// behind `#if defined(__CCE_AICORE__)` -- so `bfloat16_t` is not available. std::uint16_t stands
// in for it: the two legs are separate translation units, every pointer becomes void* in the
// generated argument block, and the host side holds BF16 as bit patterns anyway. The `tables`
// parameter keeps the std::int32_t* spelling and is reinterpreted device side.
extern "C" __global__ __aicore__ void turboquant_reshape_and_cache_bf16(
    __gm__ std::uint16_t*, __gm__ std::uint16_t*, __gm__ std::int8_t*,
    __gm__ std::int8_t*, __gm__ float*, __gm__ std::int32_t*,
    __gm__ float*, __gm__ std::int32_t*, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t, uint32_t, float);

// The launcher keeps the const-correct interface the caller sees and strips it at the
// triple-chevron call, which is where the generated argument block forces a non-const pointer.
extern "C" void launch_turboquant_reshape_and_cache_bf16(
    void* stream, const std::uint16_t* key, const std::uint16_t* value,
    std::int8_t* key_cache, std::int8_t* value_cache, float* scale_cache,
    const std::int32_t* slot_mapping, const float* pi_signs,
    const std::int32_t* tables, std::uint32_t num_tokens, std::uint32_t num_kv_heads,
    std::uint32_t head_size, std::uint32_t block_size, std::uint32_t num_blocks,
    std::uint32_t tokens_per_core, float inv_sqrt_len)
{
    // One monolithic grid, however many waves it spans. The kernel's boundary guard and
    // inter-wave epilogue drain are what make that safe, so there is no launch window here and
    // no staggered sub-launch.
    //
    // Arbitrary context length means any number of strips, over any number of waves. It does
    // not mean a partial strip: a strip is one 16-token page group, and TotalStrips rounds down,
    // so dispatching 1000 tokens at 16 a strip would run 62 blocks and silently never write
    // tokens 992..999 -- a short cache with every canary intact and every other token
    // bit-exact, which is the hardest kind of wrong answer to notice. There is no partial-page
    // path in the kernel, so the launcher refuses the geometry instead.
    if (!turboquant_wave::WholeStripShare(tokens_per_core))
        throw std::invalid_argument(
            "reshape_and_cache_bf16: tokensPerCore must be a positive multiple of the 16-token "
            "strip; the chunk pipeline has no partial-strip column");
    if (!turboquant_wave::WholeStrips(num_tokens, tokens_per_core))
        throw std::invalid_argument(
            "reshape_and_cache_bf16: numTokens must be a whole number of tokensPerCore shares; "
            "a partial trailing share would be dropped");
    const std::uint32_t grid = turboquant_wave::TotalStrips(num_tokens, tokens_per_core);
    turboquant_reshape_and_cache_bf16<<<grid, nullptr, stream>>>(
        const_cast<std::uint16_t*>(key), const_cast<std::uint16_t*>(value), key_cache,
        value_cache, scale_cache, const_cast<std::int32_t*>(slot_mapping),
        const_cast<float*>(pi_signs), const_cast<std::int32_t*>(tables), num_tokens,
        num_kv_heads, head_size, block_size, num_blocks, tokens_per_core, inv_sqrt_len);
}

#endif
